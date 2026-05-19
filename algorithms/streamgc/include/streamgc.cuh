#pragma once
#include "types.h"
#include "vertex_state.cuh"
#include "partition.cuh"
#include "priority.cuh"
#include "epoch.cuh"
#include <cstdint>
#include <cstdio>
#include <cstdlib>

static_assert(sizeof(EdgeUpdate) == 16, "EdgeUpdate must be exactly 16 bytes");

// Warp batch size for persistent kernel
constexpr uint32_t WARP_BATCH_SIZE = 4;

// Overflow buffer entry: edge that couldn't be inserted into CSR due to full adjacency
struct OverflowEdge {
    uint32_t local_vertex_idx;   // local index of the vertex whose adj list was full
    uint32_t neighbor_global;    // global ID of the neighbor to insert
};

// Overflow buffer capacity (number of entries per buffer)
constexpr uint32_t OVERFLOW_CAPACITY = 262144;  // 256K entries = 2MB per buffer

// Per-GPU partition state -- all arrays in HBM
struct GPUPartition {
    // Owned vertex state
    VertexState* owned;
    uint32_t*    owned_to_global;

    // Ghost vertex state
    VertexState* ghost;
    uint32_t*    ghost_to_global;

    // Reverse index maps (open-addressing hash maps)
    uint32_t* global_to_owned;
    uint32_t* global_to_ghost;
    uint32_t  index_map_size;  // power-of-2 for owned map
    uint32_t  ghost_map_size;  // power-of-2 for ghost map

    // Color occupancy bitmaps (64-bit for __ffsll)
    uint64_t* neighbor_bitmap;

    // Frequency histogram for high-degree vertices
    uint32_t* freq_histogram;
    uint32_t* high_degree_map;  // local_owned_idx -> histogram row idx (HASH_EMPTY = bitmap mode)
    uint32_t  num_high_degree;

    // Fixed priority
    uint64_t* init_priority;

    // Epoch snapshot for query serving
    uint16_t* epoch_snapshot;

    // Dynamic CSR adjacency for owned vertices (local neighbors)
    uint32_t* row_ptr;
    uint32_t* col_idx;
    uint32_t* col_idx_capacity;

    // Ghost adjacency (boundary neighbors)
    uint32_t* ghost_row_ptr;
    uint32_t* ghost_col_idx;
    uint32_t* ghost_col_idx_capacity;

    // Ring buffer for incoming edge stream
    EdgeUpdate* stream_buffer;
    uint32_t*   stream_head;
    uint32_t*   stream_tail;
    uint32_t    stream_mask;

    // Epoch and update tracking
    uint64_t* update_counter;
    uint32_t* epoch_ready_flag;
    uint32_t* epoch_counter;

    // Partition map pointer (unified memory, readable from device)
    PartitionEntry* d_partition_map;

    // Overflow buffers for edges that couldn't fit in CSR
    OverflowEdge* overflow_local;       // local CSR overflow
    OverflowEdge* overflow_ghost;       // ghost CSR overflow
    uint32_t*     overflow_local_count; // atomic counter
    uint32_t*     overflow_ghost_count; // atomic counter

    // Counters
    uint32_t num_owned;
    uint32_t num_ghost;
    uint32_t num_edges_local;
    uint32_t max_owned;
    uint32_t max_ghost;
    uint32_t gpu_id;

    // Φ instrumentation (present only when compiled with -DSTREAMGC_INSTRUMENT_PHI).
    //
    // Definition: Φ per §5 Thm 1 is the instantaneous count of *in-flight conflicts*
    // — distinct boundary vertices whose most recent local color advance has not
    // yet been consumed by every peer that shares a boundary edge. We implement
    // this with a per-vertex `phi_dirty` bit that is SET on a color advance (only
    // if it was previously clear — idempotent) and CLEARED when the last peer
    // consumes the advance. phi_inflight tracks the distinct-vertex count (bounded
    // by #boundary vertices rather than by cumulative advance count) and phi_max
    // is its sticky maximum.
    //
    // is_boundary[local_owned_idx]: precomputed 1/0 flag indicating the owned vertex
    // has at least one ghost-side neighbor (set at partition build time).
    int32_t*  phi_inflight;    // signed so underflow on spurious decrements is visible
    uint32_t* phi_max;         // sticky-max via atomicMax
    uint8_t*  is_boundary;     // max_owned bytes
    uint8_t*  phi_dirty;       // max_owned bytes; 1 = advance outstanding to some peer
};

// System-wide configuration
struct StreamGCConfig {
    uint32_t num_gpus;
    uint32_t num_vertices;
    uint64_t num_edges;
    uint32_t stream_buffer_size;  // default: 1 << 20 (1M edges)
    uint32_t epoch_size;
    uint32_t priority_refresh_interval;
    uint32_t histogram_threshold;
    float    rebalance_threshold;
};

inline StreamGCConfig default_config() {
    StreamGCConfig cfg;
    cfg.num_gpus = 1;
    cfg.num_vertices = 0;
    cfg.num_edges = 0;
    cfg.stream_buffer_size = 1 << 20;
    cfg.epoch_size = EPOCH_SIZE_DEFAULT;
    cfg.priority_refresh_interval = PRIORITY_REFRESH_INTERVAL_DEFAULT;
    cfg.histogram_threshold = HISTOGRAM_THRESHOLD_DEFAULT;
    cfg.rebalance_threshold = REBALANCE_THRESHOLD_DEFAULT;
    return cfg;
}

// CUDA error checking macro
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d -- %s\n",                 \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while(0)

// ---- Hash map device functions for index lookup ----

// Open-addressing hash map lookup
__device__ inline uint32_t hash_lookup(
    const uint32_t* map,
    uint32_t map_size,
    uint32_t key
) {
    uint32_t mask = map_size - 1;
    uint32_t slot = (key * 2654435761u) & mask;  // Knuth multiplicative hash
    for (uint32_t probe = 0; probe < map_size; probe++) {
        uint32_t idx = (slot + probe) & mask;
        // Map stores: even slots = key, odd slots = value
        if (map[idx * 2] == key) return map[idx * 2 + 1];
        if (map[idx * 2] == HASH_EMPTY) return HASH_EMPTY;
    }
    return HASH_EMPTY;
}

__device__ inline uint32_t get_owned_index(uint32_t global_id, const GPUPartition* partition) {
    return hash_lookup(partition->global_to_owned, partition->index_map_size, global_id);
}

__device__ inline uint32_t get_ghost_index(uint32_t global_id, const GPUPartition* partition) {
    return hash_lookup(partition->global_to_ghost, partition->ghost_map_size, global_id);
}

// Host-side hash map lookup (for initialization and testing)
inline uint32_t hash_lookup_host(
    const uint32_t* map,
    uint32_t map_size,
    uint32_t key
) {
    uint32_t mask = map_size - 1;
    uint32_t slot = (key * 2654435761u) & mask;
    for (uint32_t probe = 0; probe < map_size; probe++) {
        uint32_t idx = (slot + probe) & mask;
        if (map[idx * 2] == key) return map[idx * 2 + 1];
        if (map[idx * 2] == HASH_EMPTY) return HASH_EMPTY;
    }
    return HASH_EMPTY;
}

// Host-side hash map insertion
inline void hash_insert_host(
    uint32_t* map,
    uint32_t map_size,
    uint32_t key,
    uint32_t value
) {
    uint32_t mask = map_size - 1;
    uint32_t slot = (key * 2654435761u) & mask;
    for (uint32_t probe = 0; probe < map_size; probe++) {
        uint32_t idx = (slot + probe) & mask;
        if (map[idx * 2] == HASH_EMPTY || map[idx * 2] == key) {
            map[idx * 2] = key;
            map[idx * 2 + 1] = value;
            return;
        }
    }
    // Should never reach here if map is properly sized (2x entries)
    fprintf(stderr, "FATAL: hash map full during insert\n");
    exit(EXIT_FAILURE);
}
