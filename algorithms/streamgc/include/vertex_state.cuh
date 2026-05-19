#pragma once
#include <cstdint>

// 32-bit packed state: color (16 bits) + version (16 bits)
// CRITICAL: alignas(4) ensures hardware atomic load/store on NVIDIA GPUs
// A naturally-aligned 32-bit load returns either complete old or complete new word
// Torn reads (half-old, half-new) are impossible by hardware memory model guarantee
struct alignas(4) VertexState {
    uint16_t color;    // current color assigned to this vertex (0 = uncolored)
    uint16_t version;  // increments on every recolor; wraps at 65535
};

static_assert(alignof(VertexState) == 4, "VertexState must be 4-byte aligned for atomic guarantees");
static_assert(sizeof(VertexState) == 4, "VertexState must be exactly 4 bytes");

// Pack color and version into a single uint32_t for atomicCAS operations
__host__ __device__ inline uint32_t pack_state(uint16_t color, uint16_t version) {
    return (static_cast<uint32_t>(version) << 16) | static_cast<uint32_t>(color);
}

__host__ __device__ inline uint16_t unpack_color(uint32_t packed) {
    return static_cast<uint16_t>(packed & 0xFFFF);
}

__host__ __device__ inline uint16_t unpack_version(uint32_t packed) {
    return static_cast<uint16_t>((packed >> 16) & 0xFFFF);
}

// Sentinel values
constexpr uint16_t COLOR_UNCOLORED   = 0;
constexpr uint16_t COLOR_MIGRATING   = 0xFFFE;  // vertex is being migrated
constexpr uint16_t COLOR_INVALID     = 0xFFFF;  // tombstone: vertex deleted
constexpr uint16_t VERSION_INIT      = 0;
