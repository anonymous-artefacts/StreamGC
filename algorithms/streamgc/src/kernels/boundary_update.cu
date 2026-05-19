#include "streamgc.cuh"
#include "color_select.cuh"

// Forward declarations
__device__ bool versioned_pull(uint32_t ghost_global_id, GPUPartition* local_partition, GPUPartition* remote_partition);
__device__ void recolor_vertex(uint32_t local_vertex_idx, GPUPartition* partition, uint32_t histogram_threshold);
__device__ void try_color_reduction(uint32_t local_vertex_idx, uint16_t freed_color, GPUPartition* partition, uint32_t histogram_threshold);

// Add a ghost neighbor to vertex's ghost adjacency list
// If no slot available, writes to overflow buffer (drained during epoch compaction)
__device__ void add_ghost_neighbor(
    uint32_t local_owned_idx,
    uint32_t ghost_global,
    GPUPartition* partition
) {
    uint32_t start = partition->ghost_row_ptr[local_owned_idx];
    uint32_t end   = partition->ghost_row_ptr[local_owned_idx + 1];

    for (uint32_t j = start; j < end; j++) {
        uint32_t old = atomicCAS(&partition->ghost_col_idx[j], CSR_INVALID, ghost_global);
        if (old == CSR_INVALID) return;      // successfully claimed slot
        if (old == ghost_global) return;      // already present
    }
    // No empty slot — write to overflow buffer for retry after compaction
#ifndef STREAMGC_ABLATION_NO_OVERFLOW
    uint32_t pos = atomicAdd(partition->overflow_ghost_count, 1);
    if (pos < OVERFLOW_CAPACITY) {
        partition->overflow_ghost[pos].local_vertex_idx = local_owned_idx;
        partition->overflow_ghost[pos].neighbor_global = ghost_global;
    }
#else
    (void)local_owned_idx; (void)ghost_global; (void)partition;
#endif
}

// Remove a ghost neighbor (lazy deletion)
__device__ void remove_ghost_neighbor(
    uint32_t local_owned_idx,
    uint32_t ghost_global,
    GPUPartition* partition
) {
    uint32_t start = partition->ghost_row_ptr[local_owned_idx];
    uint32_t end   = partition->ghost_row_ptr[local_owned_idx + 1];

    for (uint32_t j = start; j < end; j++) {
        if (partition->ghost_col_idx[j] == ghost_global) {
            partition->ghost_col_idx[j] = CSR_INVALID;
            return;
        }
    }
}

// Process a batch of boundary edges (endpoints on different GPUs)
// Called on both GPUs simultaneously for each boundary edge
__global__ void boundary_update_kernel(
    EdgeUpdate*    updates,
    uint32_t       num_updates,
    GPUPartition*  local_partition,
    GPUPartition** all_partitions,
    uint32_t       this_gpu_id,
    uint32_t       histogram_threshold
) {
    uint32_t warp_id    = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    uint32_t lane_id    = threadIdx.x % 32;
    uint32_t total_warps = (gridDim.x * blockDim.x) / 32;

    for (uint32_t i = warp_id; i < num_updates; i += total_warps) {
        EdgeUpdate e = updates[i];

        uint8_t owner_u = get_owner(e.u, local_partition->d_partition_map);
        uint8_t owner_v = get_owner(e.v, local_partition->d_partition_map);

        // Determine which endpoint we own
        bool we_own_u = (owner_u == this_gpu_id);
        uint32_t owned_vtx  = we_own_u ? e.u : e.v;
        uint32_t ghost_vtx  = we_own_u ? e.v : e.u;
        uint8_t  remote_gpu = we_own_u ? owner_v : owner_u;

        uint32_t owned_idx = get_owned_index(owned_vtx, local_partition);
        if (owned_idx == HASH_EMPTY) continue;

        GPUPartition* remote_partition = all_partitions[remote_gpu];

        if (e.type == UpdateType::ADD) {
            if (lane_id == 0) {
                // Update local adjacency to include ghost neighbor
                add_ghost_neighbor(owned_idx, ghost_vtx, local_partition);

                // Versioned pull: check if ghost is stale
                versioned_pull(ghost_vtx, local_partition, remote_partition);

                uint32_t ghost_idx = get_ghost_index(ghost_vtx, local_partition);
                uint16_t ghost_color = (ghost_idx != HASH_EMPTY) ?
                    local_partition->ghost[ghost_idx].color : COLOR_UNCOLORED;

                // Update bitmap/histogram
                update_bitmap_for_new_neighbor(owned_idx, ghost_color, local_partition);

                // If owned vertex is uncolored, assign it a color now
                if (local_partition->owned[owned_idx].color == COLOR_UNCOLORED) {
                    uint16_t c = FindAvailableColor(owned_idx, local_partition, histogram_threshold);
                    local_partition->owned[owned_idx].color = c;
                    local_partition->owned[owned_idx].version = 1;
                }

                // Check conflict
                uint16_t owned_color = local_partition->owned[owned_idx].color;
                if (owned_color == ghost_color && owned_color != COLOR_UNCOLORED) {
                    // Priority resolution
                    uint64_t owned_priority = local_partition->init_priority[owned_vtx];
                    uint64_t ghost_priority = local_partition->init_priority[ghost_vtx];

                    if (owned_priority < ghost_priority) {
                        // We own the loser -- recolor
                        recolor_vertex(owned_idx, local_partition, histogram_threshold);
                    }
                    // else: remote GPU owns the loser and will recolor
                }
            }
        }

        else if (e.type == UpdateType::DEL) {
            if (lane_id == 0) {
                uint32_t ghost_idx = get_ghost_index(ghost_vtx, local_partition);
                uint16_t ghost_color = (ghost_idx != HASH_EMPTY) ?
                    local_partition->ghost[ghost_idx].color : COLOR_UNCOLORED;

                remove_ghost_neighbor(owned_idx, ghost_vtx, local_partition);

                if (ghost_color != COLOR_INVALID) {
                    update_bitmap_for_removed_neighbor(owned_idx, ghost_color, local_partition);
                    try_color_reduction(owned_idx, ghost_color, local_partition, histogram_threshold);
                }
            }
        }

        __syncwarp();
    }
}
