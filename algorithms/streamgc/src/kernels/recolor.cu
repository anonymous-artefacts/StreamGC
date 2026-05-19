#include "streamgc.cuh"
#include "color_select.cuh"

// Φ instrumentation: mark this boundary vertex as "in-flight conflict" iff
// at the moment of advance it has a ghost neighbour currently holding the same
// colour. This is the direct conflict-pair semantic of Φ per §5 Thm 1 — only
// advances that produce a same-colour cross-cut edge count. Advances that land
// on a colour no ghost holds are not Φ-conflicts (they still advance, but they
// are immediately consistent with the owner-side view of peer state).

__device__ inline void phi_bump_on_advance(
    uint32_t local_vertex_idx,
    uint16_t new_color,
    GPUPartition* partition
) {
#ifdef STREAMGC_INSTRUMENT_PHI
    if (partition->is_boundary == nullptr ||
        !partition->is_boundary[local_vertex_idx] ||
        partition->phi_dirty == nullptr) return;

    // Scan ghost neighbours for a same-colour match.
    bool found_conflict = false;
    uint32_t gstart = partition->ghost_row_ptr[local_vertex_idx];
    uint32_t gend   = partition->ghost_row_ptr[local_vertex_idx + 1];
    for (uint32_t j = gstart; j < gend; j++) {
        uint32_t ghost_global = partition->ghost_col_idx[j];
        if (ghost_global == CSR_INVALID) continue;
        uint32_t ghost_local = get_ghost_index(ghost_global, partition);
        if (ghost_local == HASH_EMPTY) continue;
        uint16_t gcolor = partition->ghost[ghost_local].color;
        if (gcolor == new_color) { found_conflict = true; break; }
    }
    if (!found_conflict) return;

    uint32_t  word_idx = local_vertex_idx / 4;
    uint32_t  byte_off = local_vertex_idx % 4;
    uint32_t* word_ptr = reinterpret_cast<uint32_t*>(partition->phi_dirty) + word_idx;
    uint32_t  byte_mask = 1u << (byte_off * 8);
    uint32_t  old = atomicOr(word_ptr, byte_mask);
    if (((old >> (byte_off * 8)) & 0xFFu) == 0) {
        int32_t new_val = atomicAdd(partition->phi_inflight, 1) + 1;
        if (new_val > 0) atomicMax(partition->phi_max, static_cast<uint32_t>(new_val));
    }
#else
    (void)local_vertex_idx; (void)new_color; (void)partition;
#endif
}

// Update bitmap/histogram for all neighbors of vertex that just recolored
__device__ void update_incremental_state(
    uint32_t local_vertex_idx,
    uint16_t old_color,
    uint16_t new_color,
    GPUPartition* partition
) {
    if (old_color == new_color) return;

    // Update local neighbors
    uint32_t start = partition->row_ptr[local_vertex_idx];
    uint32_t end   = partition->row_ptr[local_vertex_idx + 1];

    for (uint32_t j = start; j < end; j++) {
        uint32_t neighbor_global = partition->col_idx[j];
        if (neighbor_global == CSR_INVALID) continue;

        uint32_t neighbor_local = get_owned_index(neighbor_global, partition);
        if (neighbor_local != HASH_EMPTY) {
            update_neighbor_color_change(neighbor_local, old_color, new_color, partition);
        }
    }

    // Update ghost neighbors
    uint32_t gstart = partition->ghost_row_ptr[local_vertex_idx];
    uint32_t gend   = partition->ghost_row_ptr[local_vertex_idx + 1];

    for (uint32_t j = gstart; j < gend; j++) {
        uint32_t ghost_global = partition->ghost_col_idx[j];
        if (ghost_global == CSR_INVALID) continue;
        // Ghost neighbors don't have bitmaps on this GPU -- their owning GPU
        // will see the updated color via versioned_pull
    }
}

// Attempt to recolor vertex v to a new valid color
// Fast path: optimistic write -- dominates on power-law graphs
// Slow path: atomicCAS loop -- entered only on detected contention
__device__ void recolor_vertex(
    uint32_t local_vertex_idx,
    GPUPartition* partition,
    uint32_t histogram_threshold
) {
    uint16_t old_color = partition->owned[local_vertex_idx].color;
    uint16_t new_color = FindAvailableColor(local_vertex_idx, partition, histogram_threshold);

    if (new_color == old_color) return;  // no change needed

    // --- Fast path: optimistic write ---
    uint16_t old_ver = partition->owned[local_vertex_idx].version;
    uint16_t new_ver = old_ver + 1;

    partition->owned[local_vertex_idx].color   = new_color;
    partition->owned[local_vertex_idx].version = new_ver;

    // Validate: did anyone race us?
    __threadfence();
    if (partition->owned[local_vertex_idx].color == new_color &&
        partition->owned[local_vertex_idx].version == new_ver) {
        // Fast path succeeded
        phi_bump_on_advance(local_vertex_idx, new_color, partition);
        update_incremental_state(local_vertex_idx, old_color, new_color, partition);
        return;
    }

    // --- Slow path: CAS loop ---
    while (true) {
        uint32_t current_packed = *reinterpret_cast<uint32_t*>(
            &partition->owned[local_vertex_idx]
        );
        uint16_t current_color = unpack_color(current_packed);
        uint16_t current_ver   = unpack_version(current_packed);

        // CRITICAL: recompute with fresh state on each retry
        new_color = FindAvailableColor(local_vertex_idx, partition, histogram_threshold);
        if (new_color == current_color) return;  // already valid

        uint32_t new_packed = pack_state(new_color, current_ver + 1);
        uint32_t result = atomicCAS(
            reinterpret_cast<uint32_t*>(&partition->owned[local_vertex_idx]),
            current_packed,
            new_packed
        );

        if (result == current_packed) {
            // CAS succeeded
            phi_bump_on_advance(local_vertex_idx, new_color, partition);
            update_incremental_state(local_vertex_idx, current_color, new_color, partition);
            return;
        }
        // CAS failed: someone else updated -- retry with their new value
    }
}

// Try to recolor vertex to a lower color after a neighbor deletion
// This bounds color count growth over time
__device__ void try_color_reduction(
    uint32_t local_vertex_idx,
    uint16_t freed_color,
    GPUPartition* partition,
    uint32_t histogram_threshold
) {
    uint16_t current_color = partition->owned[local_vertex_idx].color;
    if (current_color == COLOR_UNCOLORED) return;

    // Only attempt reduction if the freed color is lower than current
    if (freed_color >= current_color) return;

    // Check if freed_color is actually available now
    uint16_t best = FindAvailableColor(local_vertex_idx, partition, histogram_threshold);
    if (best < current_color) {
        // Recolor to the lower color
        recolor_vertex(local_vertex_idx, partition, histogram_threshold);
    }
}
