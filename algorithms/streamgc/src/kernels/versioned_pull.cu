#include "streamgc.cuh"

// Read ghost vertex's color atomically from the owning GPU
// Returns: whether the ghost was updated (true = pulled fresh value)
__device__ bool versioned_pull(
    uint32_t ghost_global_id,
    GPUPartition* local_partition,
    GPUPartition* remote_partition
) {
    uint32_t local_ghost_idx = get_ghost_index(ghost_global_id, local_partition);
    if (local_ghost_idx == HASH_EMPTY) return false;

    uint16_t local_ver = local_partition->ghost[local_ghost_idx].version;

    uint32_t remote_owned_idx = get_owned_index(ghost_global_id, remote_partition);
    if (remote_owned_idx == HASH_EMPTY) return false;

    // CRITICAL: single 32-bit load -- color and version read atomically
    // Hardware memory model on NVIDIA GPUs guarantees aligned 32-bit loads are atomic
    uint32_t remote_word = *reinterpret_cast<const uint32_t*>(
        &remote_partition->owned[remote_owned_idx]
    );

    uint16_t remote_color   = unpack_color(remote_word);
    uint16_t remote_version = unpack_version(remote_word);

    if (remote_version == local_ver) {
        return false;  // ghost is fresh
    }

    // Ghost is stale -- update with atomically-consistent value
    local_partition->ghost[local_ghost_idx].color   = remote_color;
    local_partition->ghost[local_ghost_idx].version = remote_version;

    // Φ instrumentation: peer has caught up. If the remote owner's phi_dirty
    // bit for this vertex is set, clear it and decrement the global inflight
    // counter. This correctly handles multiple peers catching up at different
    // times — only the first CAS-to-clean transition decrements the counter.
 
#ifdef STREAMGC_INSTRUMENT_PHI
    if (remote_partition->phi_dirty != nullptr && remote_partition->phi_inflight != nullptr) {
        uint32_t  word_idx = remote_owned_idx / 4;
        uint32_t  byte_off = remote_owned_idx % 4;
        uint32_t* word_ptr = reinterpret_cast<uint32_t*>(remote_partition->phi_dirty) + word_idx;
        uint32_t  byte_mask = 1u << (byte_off * 8);
        uint32_t  old = atomicAnd(word_ptr, ~byte_mask);
        if (((old >> (byte_off * 8)) & 0xFFu) != 0) {
            atomicSub(remote_partition->phi_inflight, 1);
        }
    }
#endif
    return true;
}

// Synchronize all ghost vertices with their owning GPUs via NVLink
__global__ void ghost_sync_kernel(
    GPUPartition*  local_partition,
    GPUPartition** all_partitions,
    uint32_t       num_gpus
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= local_partition->num_ghost) return;

    uint32_t ghost_global = local_partition->ghost_to_global[tid];
    uint8_t remote_gpu = get_owner(ghost_global, local_partition->d_partition_map);
    if (remote_gpu >= num_gpus) return;

    versioned_pull(ghost_global, local_partition, all_partitions[remote_gpu]);
}
