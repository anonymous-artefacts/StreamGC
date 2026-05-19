#include "streamgc.cuh"
#include <cstdio>
#include <cassert>
#include <cstring>
#include <cuda_runtime.h>

// Definition of partition_map (required by extern in partition.cuh)
PartitionEntry* partition_map = nullptr;
uint32_t        partition_map_size = 0;

static int passed = 0;
static int failed = 0;

#define TEST(name) \
    printf("  TEST: %-50s ", #name); \
    test_##name(); \
    printf("PASS\n"); passed++;

// Versioned pull device function -- implemented inline for testing
__device__ bool versioned_pull(
    uint32_t ghost_global_id,
    GPUPartition* local_partition,
    GPUPartition* remote_partition
) {
    uint32_t local_ghost_idx = get_ghost_index(ghost_global_id, local_partition);
    if (local_ghost_idx == HASH_EMPTY) return false;

    uint16_t local_ver = local_partition->ghost[local_ghost_idx].version;

    uint32_t remote_owned_idx = get_owned_index(ghost_global_id, remote_partition);
    if (remote_owned_idx == HASH_EMPTY) return false;

    // Single 32-bit atomic load
    uint32_t remote_word = *reinterpret_cast<const uint32_t*>(
        &remote_partition->owned[remote_owned_idx]
    );

    uint16_t remote_color   = unpack_color(remote_word);
    uint16_t remote_version = unpack_version(remote_word);

    if (remote_version == local_ver) {
        return false;  // ghost is fresh
    }

    // Update ghost
    local_partition->ghost[local_ghost_idx].color   = remote_color;
    local_partition->ghost[local_ghost_idx].version = remote_version;
    return true;
}

// Test kernel: single thread does a versioned pull
__global__ void kernel_versioned_pull(
    GPUPartition* local_part,
    GPUPartition* remote_part,
    uint32_t ghost_global_id,
    bool* result
) {
    *result = versioned_pull(ghost_global_id, local_part, remote_part);
}

// Helper: build a minimal hash map on device with one entry
void build_single_entry_hash_map(
    uint32_t** d_map,
    uint32_t map_size,
    uint32_t key,
    uint32_t value
) {
    // Allocate key-value pairs (2 * map_size entries)
    uint32_t* h_map = new uint32_t[map_size * 2];
    for (uint32_t i = 0; i < map_size * 2; i++) h_map[i] = HASH_EMPTY;

    // Insert using the same hash function as the device code
    uint32_t mask = map_size - 1;
    uint32_t slot = (key * 2654435761u) & mask;
    h_map[slot * 2] = key;
    h_map[slot * 2 + 1] = value;

    CUDA_CHECK(cudaMalloc(d_map, map_size * 2 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(*d_map, h_map, map_size * 2 * sizeof(uint32_t), cudaMemcpyHostToDevice));
    delete[] h_map;
}

void test_pull_no_update_when_fresh() {
    // Ghost has same version as remote owned -- should not pull
    const uint32_t MAP_SIZE = 4;  // power of 2
    const uint32_t GHOST_GLOBAL_ID = 42;

    // Set up remote partition: vertex 42 owned, color=5, version=3
    VertexState* d_remote_owned;
    uint32_t* d_remote_g2o;
    CUDA_CHECK(cudaMalloc(&d_remote_owned, sizeof(VertexState)));
    VertexState remote_state = {5, 3};
    CUDA_CHECK(cudaMemcpy(d_remote_owned, &remote_state, sizeof(VertexState), cudaMemcpyHostToDevice));
    build_single_entry_hash_map(&d_remote_g2o, MAP_SIZE, GHOST_GLOBAL_ID, 0);

    // Set up local partition: vertex 42 is ghost, color=5, version=3 (fresh)
    VertexState* d_local_ghost;
    uint32_t* d_local_g2ghost;
    CUDA_CHECK(cudaMalloc(&d_local_ghost, sizeof(VertexState)));
    VertexState local_ghost_state = {5, 3};
    CUDA_CHECK(cudaMemcpy(d_local_ghost, &local_ghost_state, sizeof(VertexState), cudaMemcpyHostToDevice));
    build_single_entry_hash_map(&d_local_g2ghost, MAP_SIZE, GHOST_GLOBAL_ID, 0);

    // Build GPUPartition structs on device
    GPUPartition h_local, h_remote;
    memset(&h_local, 0, sizeof(GPUPartition));
    memset(&h_remote, 0, sizeof(GPUPartition));

    h_local.ghost = d_local_ghost;
    h_local.global_to_ghost = d_local_g2ghost;
    h_local.ghost_map_size = MAP_SIZE;

    h_remote.owned = d_remote_owned;
    h_remote.global_to_owned = d_remote_g2o;
    h_remote.index_map_size = MAP_SIZE;

    GPUPartition *d_local, *d_remote;
    CUDA_CHECK(cudaMalloc(&d_local, sizeof(GPUPartition)));
    CUDA_CHECK(cudaMalloc(&d_remote, sizeof(GPUPartition)));
    CUDA_CHECK(cudaMemcpy(d_local, &h_local, sizeof(GPUPartition), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_remote, &h_remote, sizeof(GPUPartition), cudaMemcpyHostToDevice));

    bool* d_result;
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(bool)));

    kernel_versioned_pull<<<1, 1>>>(d_local, d_remote, GHOST_GLOBAL_ID, d_result);
    CUDA_CHECK(cudaDeviceSynchronize());

    bool h_result;
    CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(bool), cudaMemcpyDeviceToHost));
    assert(h_result == false);  // No update needed

    // Verify ghost was not modified
    VertexState h_ghost;
    CUDA_CHECK(cudaMemcpy(&h_ghost, d_local_ghost, sizeof(VertexState), cudaMemcpyDeviceToHost));
    assert(h_ghost.color == 5);
    assert(h_ghost.version == 3);

    cudaFree(d_remote_owned); cudaFree(d_remote_g2o);
    cudaFree(d_local_ghost); cudaFree(d_local_g2ghost);
    cudaFree(d_local); cudaFree(d_remote); cudaFree(d_result);
}

void test_pull_updates_when_stale() {
    // Ghost has old version -- should pull new color and version
    const uint32_t MAP_SIZE = 4;
    const uint32_t GHOST_GLOBAL_ID = 42;

    // Remote: color=8, version=5
    VertexState* d_remote_owned;
    uint32_t* d_remote_g2o;
    CUDA_CHECK(cudaMalloc(&d_remote_owned, sizeof(VertexState)));
    VertexState remote_state = {8, 5};
    CUDA_CHECK(cudaMemcpy(d_remote_owned, &remote_state, sizeof(VertexState), cudaMemcpyHostToDevice));
    build_single_entry_hash_map(&d_remote_g2o, MAP_SIZE, GHOST_GLOBAL_ID, 0);

    // Local ghost: color=3, version=2 (stale)
    VertexState* d_local_ghost;
    uint32_t* d_local_g2ghost;
    CUDA_CHECK(cudaMalloc(&d_local_ghost, sizeof(VertexState)));
    VertexState local_ghost_state = {3, 2};
    CUDA_CHECK(cudaMemcpy(d_local_ghost, &local_ghost_state, sizeof(VertexState), cudaMemcpyHostToDevice));
    build_single_entry_hash_map(&d_local_g2ghost, MAP_SIZE, GHOST_GLOBAL_ID, 0);

    GPUPartition h_local, h_remote;
    memset(&h_local, 0, sizeof(GPUPartition));
    memset(&h_remote, 0, sizeof(GPUPartition));

    h_local.ghost = d_local_ghost;
    h_local.global_to_ghost = d_local_g2ghost;
    h_local.ghost_map_size = MAP_SIZE;

    h_remote.owned = d_remote_owned;
    h_remote.global_to_owned = d_remote_g2o;
    h_remote.index_map_size = MAP_SIZE;

    GPUPartition *d_local, *d_remote;
    CUDA_CHECK(cudaMalloc(&d_local, sizeof(GPUPartition)));
    CUDA_CHECK(cudaMalloc(&d_remote, sizeof(GPUPartition)));
    CUDA_CHECK(cudaMemcpy(d_local, &h_local, sizeof(GPUPartition), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_remote, &h_remote, sizeof(GPUPartition), cudaMemcpyHostToDevice));

    bool* d_result;
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(bool)));

    kernel_versioned_pull<<<1, 1>>>(d_local, d_remote, GHOST_GLOBAL_ID, d_result);
    CUDA_CHECK(cudaDeviceSynchronize());

    bool h_result;
    CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(bool), cudaMemcpyDeviceToHost));
    assert(h_result == true);  // Pull happened

    // Verify ghost was updated
    VertexState h_ghost;
    CUDA_CHECK(cudaMemcpy(&h_ghost, d_local_ghost, sizeof(VertexState), cudaMemcpyDeviceToHost));
    assert(h_ghost.color == 8);
    assert(h_ghost.version == 5);

    cudaFree(d_remote_owned); cudaFree(d_remote_g2o);
    cudaFree(d_local_ghost); cudaFree(d_local_g2ghost);
    cudaFree(d_local); cudaFree(d_remote); cudaFree(d_result);
}

void test_pull_atomic_consistency() {
    // Verify that the 32-bit read is atomic: both color and version from same write
    const uint32_t MAP_SIZE = 4;
    const uint32_t GHOST_GLOBAL_ID = 99;

    // Remote: color=0xABCD, version=0x1234
    VertexState* d_remote_owned;
    uint32_t* d_remote_g2o;
    CUDA_CHECK(cudaMalloc(&d_remote_owned, sizeof(VertexState)));
    VertexState remote_state = {0xABCD, 0x1234};
    CUDA_CHECK(cudaMemcpy(d_remote_owned, &remote_state, sizeof(VertexState), cudaMemcpyHostToDevice));
    build_single_entry_hash_map(&d_remote_g2o, MAP_SIZE, GHOST_GLOBAL_ID, 0);

    // Local ghost: stale
    VertexState* d_local_ghost;
    uint32_t* d_local_g2ghost;
    CUDA_CHECK(cudaMalloc(&d_local_ghost, sizeof(VertexState)));
    VertexState local_ghost_state = {1, 0};
    CUDA_CHECK(cudaMemcpy(d_local_ghost, &local_ghost_state, sizeof(VertexState), cudaMemcpyHostToDevice));
    build_single_entry_hash_map(&d_local_g2ghost, MAP_SIZE, GHOST_GLOBAL_ID, 0);

    GPUPartition h_local, h_remote;
    memset(&h_local, 0, sizeof(GPUPartition));
    memset(&h_remote, 0, sizeof(GPUPartition));
    h_local.ghost = d_local_ghost;
    h_local.global_to_ghost = d_local_g2ghost;
    h_local.ghost_map_size = MAP_SIZE;
    h_remote.owned = d_remote_owned;
    h_remote.global_to_owned = d_remote_g2o;
    h_remote.index_map_size = MAP_SIZE;

    GPUPartition *d_local, *d_remote;
    CUDA_CHECK(cudaMalloc(&d_local, sizeof(GPUPartition)));
    CUDA_CHECK(cudaMalloc(&d_remote, sizeof(GPUPartition)));
    CUDA_CHECK(cudaMemcpy(d_local, &h_local, sizeof(GPUPartition), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_remote, &h_remote, sizeof(GPUPartition), cudaMemcpyHostToDevice));

    bool* d_result;
    CUDA_CHECK(cudaMalloc(&d_result, sizeof(bool)));

    kernel_versioned_pull<<<1, 1>>>(d_local, d_remote, GHOST_GLOBAL_ID, d_result);
    CUDA_CHECK(cudaDeviceSynchronize());

    VertexState h_ghost;
    CUDA_CHECK(cudaMemcpy(&h_ghost, d_local_ghost, sizeof(VertexState), cudaMemcpyDeviceToHost));
    assert(h_ghost.color == 0xABCD);
    assert(h_ghost.version == 0x1234);

    cudaFree(d_remote_owned); cudaFree(d_remote_g2o);
    cudaFree(d_local_ghost); cudaFree(d_local_g2ghost);
    cudaFree(d_local); cudaFree(d_remote); cudaFree(d_result);
}

int main() {
    printf("=== Versioned Pull Unit Tests ===\n");

    TEST(pull_no_update_when_fresh);
    TEST(pull_updates_when_stale);
    TEST(pull_atomic_consistency);

    printf("\n=== Results: %d passed, %d failed ===\n", passed, failed);
    return failed > 0 ? 1 : 0;
}
