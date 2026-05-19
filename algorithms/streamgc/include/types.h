#pragma once
// Shared types for both CUDA (.cuh) and plain C++ (.cpp) code
// No CUDA-specific qualifiers in this file

#include <cstdint>

// Edge update types
enum class UpdateType : uint8_t {
    ADD   = 0,
    DEL   = 1,
    QUERY_POINT = 2,
    QUERY_NEIGHBORHOOD = 3
};

// A single streaming update -- 16 bytes total
struct alignas(16) EdgeUpdate {
    uint32_t   u;
    uint32_t   v;
    UpdateType type;
    uint8_t    _pad[7];
};

// Sentinel constants
constexpr uint32_t HASH_EMPTY  = 0xFFFFFFFF;
constexpr uint32_t CSR_INVALID = 0xFFFFFFFF;
