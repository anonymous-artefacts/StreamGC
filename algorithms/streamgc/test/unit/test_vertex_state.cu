#include "vertex_state.cuh"
#include <cstdio>
#include <cassert>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d -- %s\n",                 \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while(0)

static int passed = 0;
static int failed = 0;

#define TEST(name) \
    printf("  TEST: %-50s ", #name); \
    test_##name(); \
    printf("PASS\n"); passed++;

// Helper: simple kernel to test device-side functions
__global__ void kernel_pack_unpack(uint32_t* result, uint16_t color, uint16_t version) {
    uint32_t packed = pack_state(color, version);
    result[0] = packed;
    result[1] = unpack_color(packed);
    result[2] = unpack_version(packed);
}

__global__ void kernel_atomic_cas(VertexState* state, uint16_t new_color, uint16_t new_version, uint32_t* success) {
    uint32_t old_packed = *reinterpret_cast<uint32_t*>(state);
    uint16_t old_color = unpack_color(old_packed);
    uint16_t old_ver = unpack_version(old_packed);

    uint32_t expected = pack_state(old_color, old_ver);
    uint32_t desired  = pack_state(new_color, new_version);

    uint32_t result = atomicCAS(reinterpret_cast<uint32_t*>(state), expected, desired);
    *success = (result == expected) ? 1 : 0;
}

__global__ void kernel_concurrent_cas(VertexState* state, uint32_t* win_count) {
    // Each thread tries to CAS from current to its own color
    uint16_t my_color = static_cast<uint16_t>(threadIdx.x + 1);

    for (int attempt = 0; attempt < 10; attempt++) {
        uint32_t old_packed = *reinterpret_cast<uint32_t*>(state);
        uint16_t old_ver = unpack_version(old_packed);
        uint32_t desired = pack_state(my_color, old_ver + 1);
        uint32_t result = atomicCAS(reinterpret_cast<uint32_t*>(state), old_packed, desired);
        if (result == old_packed) {
            atomicAdd(win_count, 1);
            break;
        }
    }
}

void test_pack_unpack_basic() {
    uint32_t packed = pack_state(42, 7);
    assert(unpack_color(packed) == 42);
    assert(unpack_version(packed) == 7);
}

void test_pack_unpack_zero() {
    uint32_t packed = pack_state(0, 0);
    assert(unpack_color(packed) == 0);
    assert(unpack_version(packed) == 0);
}

void test_pack_unpack_max() {
    uint32_t packed = pack_state(0xFFFF, 0xFFFF);
    assert(unpack_color(packed) == 0xFFFF);
    assert(unpack_version(packed) == 0xFFFF);
}

void test_pack_unpack_sentinels() {
    uint32_t p1 = pack_state(COLOR_UNCOLORED, VERSION_INIT);
    assert(unpack_color(p1) == COLOR_UNCOLORED);
    assert(unpack_version(p1) == VERSION_INIT);

    uint32_t p2 = pack_state(COLOR_MIGRATING, 100);
    assert(unpack_color(p2) == COLOR_MIGRATING);
    assert(unpack_version(p2) == 100);

    uint32_t p3 = pack_state(COLOR_INVALID, 200);
    assert(unpack_color(p3) == COLOR_INVALID);
    assert(unpack_version(p3) == 200);
}

void test_struct_alignment() {
    assert(alignof(VertexState) == 4);
    assert(sizeof(VertexState) == 4);
}

void test_struct_fields() {
    VertexState vs;
    vs.color = 5;
    vs.version = 3;
    uint32_t* as_u32 = reinterpret_cast<uint32_t*>(&vs);
    assert(unpack_color(*as_u32) == 5);
    assert(unpack_version(*as_u32) == 3);
}

void test_device_pack_unpack() {
    uint32_t* d_result;
    CUDA_CHECK(cudaMalloc(&d_result, 3 * sizeof(uint32_t)));

    kernel_pack_unpack<<<1, 1>>>(d_result, 123, 456);
    CUDA_CHECK(cudaDeviceSynchronize());

    uint32_t h_result[3];
    CUDA_CHECK(cudaMemcpy(h_result, d_result, 3 * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    assert(h_result[0] == pack_state(123, 456));
    assert(h_result[1] == 123);
    assert(h_result[2] == 456);

    cudaFree(d_result);
}

void test_atomic_cas_single() {
    VertexState* d_state;
    uint32_t* d_success;
    CUDA_CHECK(cudaMalloc(&d_state, sizeof(VertexState)));
    CUDA_CHECK(cudaMalloc(&d_success, sizeof(uint32_t)));

    VertexState init = {5, 2};
    CUDA_CHECK(cudaMemcpy(d_state, &init, sizeof(VertexState), cudaMemcpyHostToDevice));

    kernel_atomic_cas<<<1, 1>>>(d_state, 10, 3, d_success);
    CUDA_CHECK(cudaDeviceSynchronize());

    uint32_t h_success;
    CUDA_CHECK(cudaMemcpy(&h_success, d_success, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    assert(h_success == 1);

    VertexState h_state;
    CUDA_CHECK(cudaMemcpy(&h_state, d_state, sizeof(VertexState), cudaMemcpyDeviceToHost));
    assert(h_state.color == 10);
    assert(h_state.version == 3);

    cudaFree(d_state);
    cudaFree(d_success);
}

void test_atomic_cas_concurrent() {
    // 32 threads all try to CAS the same VertexState
    // Exactly one should win per round
    VertexState* d_state;
    uint32_t* d_win_count;
    CUDA_CHECK(cudaMalloc(&d_state, sizeof(VertexState)));
    CUDA_CHECK(cudaMalloc(&d_win_count, sizeof(uint32_t)));

    VertexState init = {0, 0};
    uint32_t zero = 0;
    CUDA_CHECK(cudaMemcpy(d_state, &init, sizeof(VertexState), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_win_count, &zero, sizeof(uint32_t), cudaMemcpyHostToDevice));

    kernel_concurrent_cas<<<1, 32>>>(d_state, d_win_count);
    CUDA_CHECK(cudaDeviceSynchronize());

    uint32_t h_win_count;
    CUDA_CHECK(cudaMemcpy(&h_win_count, d_win_count, sizeof(uint32_t), cudaMemcpyDeviceToHost));

    // At least one thread should have won (likely many with retries)
    assert(h_win_count >= 1);

    // Final state should be valid
    VertexState h_state;
    CUDA_CHECK(cudaMemcpy(&h_state, d_state, sizeof(VertexState), cudaMemcpyDeviceToHost));
    assert(h_state.color >= 1 && h_state.color <= 32);
    assert(h_state.version >= 1);

    cudaFree(d_state);
    cudaFree(d_win_count);
}

void test_version_wrap() {
    uint32_t packed = pack_state(5, 0xFFFF);
    uint16_t ver = unpack_version(packed);
    uint16_t new_ver = ver + 1;  // should wrap to 0
    assert(new_ver == 0);

    uint32_t new_packed = pack_state(5, new_ver);
    assert(unpack_color(new_packed) == 5);
    assert(unpack_version(new_packed) == 0);
}

void test_pack_roundtrip_exhaustive() {
    // Test a selection of color/version pairs
    uint16_t test_vals[] = {0, 1, 2, 63, 64, 65, 127, 128, 255, 256, 1000, 0xFFFE, 0xFFFF};
    int n = sizeof(test_vals) / sizeof(test_vals[0]);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            uint32_t packed = pack_state(test_vals[i], test_vals[j]);
            assert(unpack_color(packed) == test_vals[i]);
            assert(unpack_version(packed) == test_vals[j]);
        }
    }
}

int main() {
    printf("=== VertexState Unit Tests ===\n");

    TEST(pack_unpack_basic);
    TEST(pack_unpack_zero);
    TEST(pack_unpack_max);
    TEST(pack_unpack_sentinels);
    TEST(struct_alignment);
    TEST(struct_fields);
    TEST(device_pack_unpack);
    TEST(atomic_cas_single);
    TEST(atomic_cas_concurrent);
    TEST(version_wrap);
    TEST(pack_roundtrip_exhaustive);

    printf("\n=== Results: %d passed, %d failed ===\n", passed, failed);
    return failed > 0 ? 1 : 0;
}
