#include "streamgc.cuh"
#include "color_select.cuh"

// Forward declarations from recolor.cu
__device__ void recolor_vertex(uint32_t local_vertex_idx, GPUPartition* partition, uint32_t histogram_threshold);
__device__ void try_color_reduction(uint32_t local_vertex_idx, uint16_t freed_color, GPUPartition* partition, uint32_t histogram_threshold);

// ---- Dynamic CSR operations ----

// Add a neighbor to vertex's local adjacency list
// If no slot available, writes to overflow buffer (drained during epoch compaction)
__device__ void add_local_neighbor(
    uint32_t local_vertex_idx,
    uint32_t neighbor_global,
    GPUPartition* partition
) {
    uint32_t start = partition->row_ptr[local_vertex_idx];
    uint32_t end   = partition->row_ptr[local_vertex_idx + 1];

    // Atomically claim first empty (INVALID) slot using CAS
    // Also check for duplicates (edge already present from CSR pre-population)
    for (uint32_t j = start; j < end; j++) {
        uint32_t old = atomicCAS(&partition->col_idx[j], CSR_INVALID, neighbor_global);
        if (old == CSR_INVALID) return;      // successfully claimed empty slot
        if (old == neighbor_global) return;   // already present (idempotent)
    }
    // No empty slot — write to overflow buffer for retry after compaction
#ifndef STREAMGC_ABLATION_NO_OVERFLOW
    uint32_t pos = atomicAdd(partition->overflow_local_count, 1);
    if (pos < OVERFLOW_CAPACITY) {
        partition->overflow_local[pos].local_vertex_idx = local_vertex_idx;
        partition->overflow_local[pos].neighbor_global = neighbor_global;
    }
#else
    // ABLATION: drop the edge instead of buffering. Visible as edge-count drift
    // and eventually as coloring-ratio inflation under heavy insertion pressure.
    (void)local_vertex_idx; (void)neighbor_global; (void)partition;
#endif
}

// Mark a neighbor for removal (lazy deletion)
__device__ void remove_local_neighbor(
    uint32_t local_vertex_idx,
    uint32_t neighbor_global,
    GPUPartition* partition
) {
    uint32_t start = partition->row_ptr[local_vertex_idx];
    uint32_t end   = partition->row_ptr[local_vertex_idx + 1];

    for (uint32_t j = start; j < end; j++) {
        if (partition->col_idx[j] == neighbor_global) {
            partition->col_idx[j] = CSR_INVALID;  // lazy deletion
            return;
        }
    }
}

// ---- Local update kernel ----
// Process a batch of local edges (both endpoints owned by this GPU)
// Warp-cooperative: each warp processes one edge
__global__ void local_update_kernel(
    EdgeUpdate*   updates,
    uint32_t      num_updates,
    GPUPartition* partition,
    uint32_t      histogram_threshold
) {
    uint32_t warp_id    = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    uint32_t lane_id    = threadIdx.x % 32;
    uint32_t total_warps = (gridDim.x * blockDim.x) / 32;

    for (uint32_t i = warp_id; i < num_updates; i += total_warps) {
        EdgeUpdate e = updates[i];

        uint32_t u_idx = get_owned_index(e.u, partition);
        uint32_t v_idx = get_owned_index(e.v, partition);

        if (u_idx == HASH_EMPTY || v_idx == HASH_EMPTY) continue;

        if (e.type == UpdateType::ADD) {
            // Update CSR adjacency
            if (lane_id == 0) {
                add_local_neighbor(u_idx, e.v, partition);
                add_local_neighbor(v_idx, e.u, partition);
            }
            __syncwarp();

            if (lane_id == 0) {
                // First: if either vertex is uncolored, assign it a color
                // This handles the initialization case where all vertices start at COLOR_UNCOLORED
                if (partition->owned[u_idx].color == COLOR_UNCOLORED) {
                    uint16_t c = FindAvailableColor(u_idx, partition, histogram_threshold);
                    partition->owned[u_idx].color = c;
                    partition->owned[u_idx].version = 1;
                }
                if (partition->owned[v_idx].color == COLOR_UNCOLORED) {
                    // Update bitmap with u's (now possibly assigned) color before coloring v
                    update_bitmap_for_new_neighbor(v_idx, partition->owned[u_idx].color, partition);
                    uint16_t c = FindAvailableColor(v_idx, partition, histogram_threshold);
                    partition->owned[v_idx].color = c;
                    partition->owned[v_idx].version = 1;
                }

                // Update bitmaps/histograms for both endpoints
                update_bitmap_for_new_neighbor(u_idx, partition->owned[v_idx].color, partition);
                update_bitmap_for_new_neighbor(v_idx, partition->owned[u_idx].color, partition);
            }
            __syncwarp();

            // Check conflict using warp vote
            bool has_conflict = (partition->owned[u_idx].color == partition->owned[v_idx].color)
                             && (partition->owned[u_idx].color != COLOR_UNCOLORED);

            uint32_t conflict_mask = __ballot_sync(0xFFFFFFFF, has_conflict);

            if (conflict_mask && lane_id == 0) {
                // Lower priority vertex recolors
                uint32_t global_u = partition->owned_to_global[u_idx];
                uint32_t global_v = partition->owned_to_global[v_idx];
                uint64_t pu = partition->init_priority[global_u];
                uint64_t pv = partition->init_priority[global_v];
                uint32_t loser_idx = (pu < pv) ? u_idx : v_idx;
                recolor_vertex(loser_idx, partition, histogram_threshold);
            }
        }

        else if (e.type == UpdateType::DEL) {
            uint16_t freed_color_u = partition->owned[v_idx].color;
            uint16_t freed_color_v = partition->owned[u_idx].color;

            if (lane_id == 0) {
                // Update CSR adjacency (mark for lazy deletion)
                remove_local_neighbor(u_idx, e.v, partition);
                remove_local_neighbor(v_idx, e.u, partition);

                // Update bitmaps/histograms
                update_bitmap_for_removed_neighbor(u_idx, freed_color_u, partition);
                update_bitmap_for_removed_neighbor(v_idx, freed_color_v, partition);

                // Color reduction attempt
                try_color_reduction(u_idx, freed_color_u, partition, histogram_threshold);
                try_color_reduction(v_idx, freed_color_v, partition, histogram_threshold);
            }
        }

        __syncwarp();
    }
}
