#include "streamgc.cuh"
#include "color_select.cuh"

// Snapshot owned vertex colors to epoch_snapshot for query serving
__global__ void epoch_snapshot_kernel(
    GPUPartition* partition,
    uint32_t      num_vertices_global
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < partition->num_owned) {
        uint32_t global_id = partition->owned_to_global[tid];
        if (global_id < num_vertices_global) {
            partition->epoch_snapshot[global_id] = partition->owned[tid].color;
        }
    }
}

// Compact dynamic CSR: remove INVALID entries and rebuild row_ptr
// This runs on GPU as a kernel for each vertex
__global__ void compact_csr_kernel(
    GPUPartition* partition
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= partition->num_owned) return;

    uint32_t start = partition->row_ptr[tid];
    uint32_t end   = partition->row_ptr[tid + 1];

    // Compact: move valid entries to the front
    uint32_t write_pos = start;
    for (uint32_t j = start; j < end; j++) {
        if (partition->col_idx[j] != CSR_INVALID) {
            if (write_pos != j) {
                partition->col_idx[write_pos] = partition->col_idx[j];
            }
            write_pos++;
        }
    }

    // Fill remaining with INVALID
    for (uint32_t j = write_pos; j < end; j++) {
        partition->col_idx[j] = CSR_INVALID;
    }

    // Similarly compact ghost adjacency
    uint32_t gstart = partition->ghost_row_ptr[tid];
    uint32_t gend   = partition->ghost_row_ptr[tid + 1];

    write_pos = gstart;
    for (uint32_t j = gstart; j < gend; j++) {
        if (partition->ghost_col_idx[j] != CSR_INVALID) {
            if (write_pos != j) {
                partition->ghost_col_idx[write_pos] = partition->ghost_col_idx[j];
            }
            write_pos++;
        }
    }
    for (uint32_t j = write_pos; j < gend; j++) {
        partition->ghost_col_idx[j] = CSR_INVALID;
    }
}

// Drain the local overflow buffer: retry inserting edges that failed during streaming.
// Must run AFTER compact_csr_kernel so freed slots are available.
// Edges that still don't fit after compaction are counted for diagnostics.
__global__ void drain_overflow_local_kernel(
    GPUPartition* partition,
    uint32_t      overflow_count,
    uint32_t*     num_failed  // optional: count edges that still don't fit
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= overflow_count) return;

    uint32_t local_idx = partition->overflow_local[tid].local_vertex_idx;
    uint32_t neighbor  = partition->overflow_local[tid].neighbor_global;

    uint32_t start = partition->row_ptr[local_idx];
    uint32_t end   = partition->row_ptr[local_idx + 1];

    for (uint32_t j = start; j < end; j++) {
        uint32_t old = atomicCAS(&partition->col_idx[j], CSR_INVALID, neighbor);
        if (old == CSR_INVALID) return;     // inserted
        if (old == neighbor) return;         // already present (dedup)
    }
    // Still no room after compaction — count for diagnostics
    if (num_failed) atomicAdd(num_failed, 1);
}

// Drain the ghost overflow buffer
__global__ void drain_overflow_ghost_kernel(
    GPUPartition* partition,
    uint32_t      overflow_count,
    uint32_t*     num_failed
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= overflow_count) return;

    uint32_t local_idx = partition->overflow_ghost[tid].local_vertex_idx;
    uint32_t neighbor  = partition->overflow_ghost[tid].neighbor_global;

    uint32_t start = partition->ghost_row_ptr[local_idx];
    uint32_t end   = partition->ghost_row_ptr[local_idx + 1];

    for (uint32_t j = start; j < end; j++) {
        uint32_t old = atomicCAS(&partition->ghost_col_idx[j], CSR_INVALID, neighbor);
        if (old == CSR_INVALID) return;
        if (old == neighbor) return;
    }
    if (num_failed) atomicAdd(num_failed, 1);
}

// Rebuild bitmaps for all owned vertices (restores accuracy after deletions)
__global__ void rebuild_bitmaps_kernel(
    GPUPartition* partition
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= partition->num_owned) return;

    // Only rebuild for vertices using bitmap mode
    if (partition->high_degree_map != nullptr &&
        partition->high_degree_map[tid] != HASH_EMPTY) {
        return;  // histogram vertices don't need bitmap rebuild
    }

    uint64_t new_bitmap;
    rebuild_bitmap(tid, partition, &new_bitmap);
    partition->neighbor_bitmap[tid] = new_bitmap;
}
