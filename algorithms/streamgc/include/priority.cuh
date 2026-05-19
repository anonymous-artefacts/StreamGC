#pragma once
#include <cstdint>

// Priority: (init_degree << 32) | vertex_id
// Higher value = higher priority = keeps color in conflict
// Deterministic: no two vertices can have the same priority (vertex_id is unique)

__host__ __device__ inline uint64_t compute_priority(uint32_t init_degree, uint32_t vertex_id) {
    return (static_cast<uint64_t>(init_degree) << 32) | static_cast<uint64_t>(vertex_id);
}

__host__ __device__ inline uint32_t priority_degree(uint64_t priority) {
    return static_cast<uint32_t>(priority >> 32);
}

__host__ __device__ inline uint32_t priority_vertex_id(uint64_t priority) {
    return static_cast<uint32_t>(priority & 0xFFFFFFFF);
}

// Returns true if priority_a wins over priority_b (higher priority keeps color)
__host__ __device__ inline bool priority_wins(uint64_t priority_a, uint64_t priority_b) {
    return priority_a > priority_b;
}

#ifdef STREAMGC_ABLATION_LIVE_DEG_PRIORITY
// Count live neighbors in owned CSR row (ABLATION: live-degree priority).
__device__ inline uint32_t live_degree_owned(uint32_t local_idx, const struct GPUPartition* partition);
#endif
