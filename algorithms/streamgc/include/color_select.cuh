#pragma once
#include "streamgc.cuh"
#include <cstdint>

// ---- Bitmap mode: for vertices with init_degree <= histogram_threshold ----
// Uses neighbor_bitmap[v] -- bit i set means color (i+1) is used by a neighbor
// __ffsll(~bitmap) gives position of first zero bit (1-indexed), so color = that value
// Supports up to 64 colors in bitmap mode

__device__ inline uint16_t find_available_color_bitmap(
    uint32_t local_owned_idx,
    const GPUPartition* partition
) {
    uint64_t bitmap = partition->neighbor_bitmap[local_owned_idx];
    int pos = __ffsll(~bitmap);
    if (pos != 0) {
        return static_cast<uint16_t>(pos);  // colors are 1-indexed (bit 0 = color 1)
    }

    // All 64 bitmap colors occupied — scan actual neighbors for first available color
    // This handles power-law graphs where even low-degree vertices may need color > 64
    bool used[256];
    for (int i = 0; i < 256; i++) used[i] = false;

    // Scan local neighbors
    uint32_t start = partition->row_ptr[local_owned_idx];
    uint32_t end   = partition->row_ptr[local_owned_idx + 1];
    for (uint32_t j = start; j < end; j++) {
        uint32_t ng = partition->col_idx[j];
        if (ng == CSR_INVALID) continue;
        uint32_t nl = get_owned_index(ng, partition);
        if (nl != HASH_EMPTY) {
            uint16_t c = partition->owned[nl].color;
            if (c > 0 && c < 256) used[c] = true;
        }
    }
    // Scan ghost neighbors
    uint32_t gs = partition->ghost_row_ptr[local_owned_idx];
    uint32_t ge = partition->ghost_row_ptr[local_owned_idx + 1];
    for (uint32_t j = gs; j < ge; j++) {
        uint32_t gg = partition->ghost_col_idx[j];
        if (gg == CSR_INVALID) continue;
        uint32_t gl = get_ghost_index(gg, partition);
        if (gl != HASH_EMPTY) {
            uint16_t c = partition->ghost[gl].color;
            if (c > 0 && c < 256) used[c] = true;
        }
    }
    for (uint16_t c = 1; c < 256; c++) {
        if (!used[c]) return c;
    }
    return 256;  // extremely rare: all 255 colors in use
}

// ---- Histogram mode: for vertices with init_degree > histogram_threshold ----
// freq_histogram[high_degree_idx * MAX_HISTOGRAM_COLORS + c] = count of neighbors using color (c+1)
// Find first c where count == 0; for very high degree (>10000): find minimum count

__device__ inline uint16_t find_available_color_histogram(
    uint32_t local_owned_idx,
    const GPUPartition* partition
) {
    uint32_t hd_idx = partition->high_degree_map[local_owned_idx];
    if (hd_idx == HASH_EMPTY) {
        // Fallback: shouldn't happen if dispatch is correct
        return find_available_color_bitmap(local_owned_idx, partition);
    }

    const uint32_t* hist = partition->freq_histogram + hd_idx * MAX_HISTOGRAM_COLORS;

    // First pass: find first color with count == 0
    for (uint32_t c = 0; c < MAX_HISTOGRAM_COLORS; c++) {
        if (hist[c] == 0) {
            return static_cast<uint16_t>(c + 1);  // colors are 1-indexed
        }
    }

    // All colors have at least one neighbor using them -- find minimum frequency
    // This reduces downstream conflict probability
    uint32_t min_count = hist[0];
    uint32_t min_color = 0;
    for (uint32_t c = 1; c < MAX_HISTOGRAM_COLORS; c++) {
        if (hist[c] < min_count) {
            min_count = hist[c];
            min_color = c;
        }
    }
    return static_cast<uint16_t>(min_color + 1);
}

// ---- Unified dispatch ----

__device__ inline uint16_t FindAvailableColor(
    uint32_t local_owned_idx,
    const GPUPartition* partition,
    uint32_t histogram_threshold
) {
#if defined(STREAMGC_ABLATION_FORCE_BITMAP)
    (void)histogram_threshold;
    return find_available_color_bitmap(local_owned_idx, partition);
#elif defined(STREAMGC_ABLATION_FORCE_HISTOGRAM)
    (void)histogram_threshold;
    if (partition->high_degree_map != nullptr &&
        partition->high_degree_map[local_owned_idx] != HASH_EMPTY) {
        return find_available_color_histogram(local_owned_idx, partition);
    }
    return find_available_color_bitmap(local_owned_idx, partition);
#else
    if (partition->high_degree_map != nullptr &&
        partition->high_degree_map[local_owned_idx] != HASH_EMPTY) {
        return find_available_color_histogram(local_owned_idx, partition);
    }
    return find_available_color_bitmap(local_owned_idx, partition);
#endif
}

// ---- Incremental bitmap/histogram maintenance ----

// Called when a neighbor of owned vertex changes color from old_color to new_color
__device__ inline void update_neighbor_color_change(
    uint32_t local_owned_idx,
    uint16_t old_color,
    uint16_t new_color,
    GPUPartition* partition
) {
    if (old_color == new_color) return;
    if (old_color == COLOR_UNCOLORED && new_color == COLOR_UNCOLORED) return;

    // Update bitmap
    if (old_color != COLOR_UNCOLORED && old_color <= 64) {
        // Note: we can't just clear the bit -- other neighbors might use this color
        // For bitmap mode, we need reference counting or rescan
        // SIMPLIFICATION: bitmap bit clearing requires checking if any other neighbor
        // has the same color. For correctness, we use atomicAnd only if no other
        // neighbor has old_color. In practice, the bitmap is an approximation for
        // the fast path -- recolor will find a valid color even if bitmap is slightly stale.
        // The histogram path is exact.
    }
    if (new_color != COLOR_UNCOLORED && new_color <= 64) {
        atomicOr(
            reinterpret_cast<unsigned long long*>(&partition->neighbor_bitmap[local_owned_idx]),
            1ULL << (new_color - 1)
        );
    }

    // Update histogram (exact)
    uint32_t hd_idx = partition->high_degree_map ? partition->high_degree_map[local_owned_idx] : HASH_EMPTY;
    if (hd_idx != HASH_EMPTY) {
        uint32_t* hist = partition->freq_histogram + hd_idx * MAX_HISTOGRAM_COLORS;
        if (old_color != COLOR_UNCOLORED && old_color <= MAX_HISTOGRAM_COLORS) {
            atomicSub(&hist[old_color - 1], 1);
        }
        if (new_color != COLOR_UNCOLORED && new_color <= MAX_HISTOGRAM_COLORS) {
            atomicAdd(&hist[new_color - 1], 1);
        }
    }
}

// Called when a new neighbor is added with the given color
__device__ inline void update_bitmap_for_new_neighbor(
    uint32_t local_owned_idx,
    uint16_t neighbor_color,
    GPUPartition* partition
) {
    if (neighbor_color == COLOR_UNCOLORED) return;

    // Update bitmap
    if (neighbor_color <= 64) {
        atomicOr(
            reinterpret_cast<unsigned long long*>(&partition->neighbor_bitmap[local_owned_idx]),
            1ULL << (neighbor_color - 1)
        );
    }

    // Update histogram
    uint32_t hd_idx = partition->high_degree_map ? partition->high_degree_map[local_owned_idx] : HASH_EMPTY;
    if (hd_idx != HASH_EMPTY) {
        uint32_t* hist = partition->freq_histogram + hd_idx * MAX_HISTOGRAM_COLORS;
        if (neighbor_color <= MAX_HISTOGRAM_COLORS) {
            atomicAdd(&hist[neighbor_color - 1], 1);
        }
    }
}

// Called when a neighbor is removed with the given color
// For bitmap: we must rescan neighbors to see if any other neighbor has this color
// For histogram: just decrement
__device__ inline void update_bitmap_for_removed_neighbor(
    uint32_t local_owned_idx,
    uint16_t neighbor_color,
    GPUPartition* partition
) {
    if (neighbor_color == COLOR_UNCOLORED) return;

    // Histogram: straightforward decrement
    uint32_t hd_idx = partition->high_degree_map ? partition->high_degree_map[local_owned_idx] : HASH_EMPTY;
    if (hd_idx != HASH_EMPTY) {
        uint32_t* hist = partition->freq_histogram + hd_idx * MAX_HISTOGRAM_COLORS;
        if (neighbor_color <= MAX_HISTOGRAM_COLORS) {
            atomicSub(&hist[neighbor_color - 1], 1);
        }
    }

    // Bitmap: need to check if any remaining neighbor uses this color
    // For now, we leave the bit set (conservative) -- this means bitmap may overcount
    // but will never undercount, so FindAvailableColor_Bitmap remains correct
    // (it may skip a valid color, but never assigns a conflicting one)
    // Periodic bitmap rebuild at epoch boundary restores accuracy
}

// Rebuild bitmap from scratch for a vertex by scanning all neighbors
// Called during epoch maintenance to restore bitmap accuracy after deletions
__device__ inline void rebuild_bitmap(
    uint32_t local_owned_idx,
    const GPUPartition* partition,
    uint64_t* out_bitmap
) {
    uint64_t bm = 0;
    uint32_t start = partition->row_ptr[local_owned_idx];
    uint32_t end   = partition->row_ptr[local_owned_idx + 1];

    // Scan local neighbors
    for (uint32_t j = start; j < end; j++) {
        uint32_t neighbor_global = partition->col_idx[j];
        if (neighbor_global == CSR_INVALID) continue;
        uint32_t neighbor_local = get_owned_index(neighbor_global, partition);
        if (neighbor_local != HASH_EMPTY) {
            uint16_t c = partition->owned[neighbor_local].color;
            if (c != COLOR_UNCOLORED && c <= 64) {
                bm |= (1ULL << (c - 1));
            }
        }
    }

    // Scan ghost neighbors
    uint32_t gstart = partition->ghost_row_ptr[local_owned_idx];
    uint32_t gend   = partition->ghost_row_ptr[local_owned_idx + 1];
    for (uint32_t j = gstart; j < gend; j++) {
        uint32_t ghost_global = partition->ghost_col_idx[j];
        if (ghost_global == CSR_INVALID) continue;
        uint32_t ghost_local = get_ghost_index(ghost_global, partition);
        if (ghost_local != HASH_EMPTY) {
            uint16_t c = partition->ghost[ghost_local].color;
            if (c != COLOR_UNCOLORED && c <= 64) {
                bm |= (1ULL << (c - 1));
            }
        }
    }

    *out_bitmap = bm;
}
