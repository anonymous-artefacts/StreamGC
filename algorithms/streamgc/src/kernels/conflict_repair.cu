#include "streamgc.cuh"
#include "color_select.cuh"

// Forward declaration
__device__ void recolor_vertex(uint32_t local_vertex_idx, GPUPartition* partition, uint32_t histogram_threshold);

// Scan all owned vertices: for each, check if any neighbor has the same color.
// If conflict found, recolor the lower-priority vertex.
// Also assigns colors to any uncolored vertices that have neighbors.
__global__ void conflict_repair_kernel(
    GPUPartition* partition,
    uint32_t      histogram_threshold,
    uint32_t*     num_conflicts  // output: count of conflicts found
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= partition->num_owned) return;

    uint16_t my_color = partition->owned[tid].color;

    // Check if this vertex has any neighbors at all
    uint32_t start = partition->row_ptr[tid];
    uint32_t end   = partition->row_ptr[tid + 1];
    uint32_t gstart = partition->ghost_row_ptr[tid];
    uint32_t gend   = partition->ghost_row_ptr[tid + 1];

    bool has_neighbors = false;
    for (uint32_t j = start; j < end && !has_neighbors; j++) {
        if (partition->col_idx[j] != CSR_INVALID) has_neighbors = true;
    }
    for (uint32_t j = gstart; j < gend && !has_neighbors; j++) {
        if (partition->ghost_col_idx[j] != CSR_INVALID) has_neighbors = true;
    }

    // Handle uncolored vertices that have edges — assign them a color
    if (my_color == COLOR_UNCOLORED) {
        if (has_neighbors) {
            uint16_t new_color = FindAvailableColor(tid, partition, histogram_threshold);
            partition->owned[tid].color = new_color;
            partition->owned[tid].version += 1;
            atomicAdd(num_conflicts, 1);  // count as conflict so repair loop continues
        }
        return;
    }

    uint32_t my_global = partition->owned_to_global[tid];
#ifdef STREAMGC_ABLATION_LIVE_DEG_PRIORITY
    // ABLATION: recompute priority from live local degree (vs. frozen init_degree).
    uint32_t live_deg = 0;
    for (uint32_t j = start; j < end; j++) {
        if (partition->col_idx[j] != CSR_INVALID) live_deg++;
    }
    for (uint32_t j = gstart; j < gend; j++) {
        if (partition->ghost_col_idx[j] != CSR_INVALID) live_deg++;
    }
    uint64_t my_priority = compute_priority(live_deg, my_global);
#else
    uint64_t my_priority = partition->init_priority[my_global];
#endif

    bool need_recolor = false;

    // Check local neighbors
    for (uint32_t j = start; j < end; j++) {
        uint32_t neighbor_global = partition->col_idx[j];
        if (neighbor_global == CSR_INVALID) continue;

        uint32_t neighbor_local = get_owned_index(neighbor_global, partition);
        if (neighbor_local == HASH_EMPTY) continue;

        uint16_t neighbor_color = partition->owned[neighbor_local].color;
        if (neighbor_color == my_color && neighbor_color != COLOR_UNCOLORED) {
            uint64_t neighbor_priority = partition->init_priority[neighbor_global];
            if (my_priority < neighbor_priority) {
                need_recolor = true;
                break;
            }
        }
    }

    // Also check ghost neighbors
    if (!need_recolor) {
        for (uint32_t j = gstart; j < gend; j++) {
            uint32_t ghost_global = partition->ghost_col_idx[j];
            if (ghost_global == CSR_INVALID) continue;

            uint32_t ghost_local = get_ghost_index(ghost_global, partition);
            if (ghost_local == HASH_EMPTY) continue;

            uint16_t ghost_color = partition->ghost[ghost_local].color;
            if (ghost_color == my_color && ghost_color != COLOR_UNCOLORED) {
                uint64_t ghost_priority = partition->init_priority[ghost_global];
                if (my_priority < ghost_priority) {
                    need_recolor = true;
                    break;
                }
            }
        }
    }

    if (need_recolor) {
        atomicAdd(num_conflicts, 1);

        // Bitmap was already rebuilt by rebuild_bitmaps_kernel before this kernel.
        // Just ensure our own current color is marked as unavailable so we pick a NEW color.
        if (my_color != COLOR_UNCOLORED && my_color <= 64) {
            partition->neighbor_bitmap[tid] |= (1ULL << (my_color - 1));
        }

        // Recolor using the pre-rebuilt bitmap
        uint16_t new_color = FindAvailableColor(tid, partition, histogram_threshold);
        partition->owned[tid].color = new_color;
        partition->owned[tid].version += 1;
    }
}
