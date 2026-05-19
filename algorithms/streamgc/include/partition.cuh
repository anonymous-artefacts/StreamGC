#pragma once
#include <cstdint>

// Color selection constants. The histogram size can be overridden at compile
// time for graphs whose per-partition high-degree count would exceed HBM
#ifndef STREAMGC_MAX_HISTOGRAM_COLORS
#define STREAMGC_MAX_HISTOGRAM_COLORS 1024
#endif
constexpr uint32_t MAX_HISTOGRAM_COLORS = STREAMGC_MAX_HISTOGRAM_COLORS;

// Histogram threshold: vertices with init_degree > this use histogram mode
constexpr uint32_t HISTOGRAM_THRESHOLD_DEFAULT = 64;

// Two-version partition entry for hazard-epoch migration
struct PartitionEntry {
    uint8_t  current_owner;    // authoritative owner GPU after migration
    uint8_t  prev_owner;       // source GPU, still processes updates during grace period
    uint16_t migration_epoch;  // epoch when migration started; 0 = not migrating
};

// Global partition map -- CUDA unified memory, coherent across all GPUs via NVLink
// DECLARATION in header (extern). DEFINITION and allocation in streamgc.cu only.
// Host-side only: device code accesses via GPUPartition.d_partition_map pointer
extern PartitionEntry* partition_map;
extern uint32_t        partition_map_size;

// Device-side: get_owner and is_migrating take partition_map as parameter
// The pointer is stored in GPUPartition and passed to kernels
__device__ inline uint8_t get_owner(uint32_t vertex_id, const PartitionEntry* pmap) {
    return pmap[vertex_id].current_owner;
}

__host__ inline uint8_t get_owner_host(uint32_t vertex_id) {
    return partition_map[vertex_id].current_owner;
}

__device__ inline bool is_migrating(uint32_t vertex_id, uint16_t current_epoch, const PartitionEntry* pmap) {
    return pmap[vertex_id].migration_epoch == current_epoch
        && pmap[vertex_id].prev_owner != pmap[vertex_id].current_owner;
}
