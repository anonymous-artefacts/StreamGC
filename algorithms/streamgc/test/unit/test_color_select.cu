#include "color_select.cuh"
#include <cstdio>
#include <cassert>
#include <cstring>
#include <cuda_runtime.h>

#define TEST_CUDA_CHECK(call)                                               \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d -- %s\n",                 \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while(0)

// Provide definitions for partition_map (required by partition.cuh extern)
PartitionEntry* partition_map = nullptr;
uint32_t        partition_map_size = 0;

static int passed = 0;
static int failed = 0;

#define TEST(name) \
    printf("  TEST: %-50s ", #name); \
    test_##name(); \
    printf("PASS\n"); passed++;

// Kernel to test bitmap color selection on device
__global__ void kernel_find_color_bitmap(
    uint64_t* neighbor_bitmap,
    uint32_t* high_degree_map,
    uint16_t* result,
    uint32_t vertex_idx
) {
    // Build a minimal GPUPartition on the stack for testing
    GPUPartition part;
    memset(&part, 0, sizeof(GPUPartition));
    part.neighbor_bitmap = neighbor_bitmap;
    part.high_degree_map = high_degree_map;
    part.freq_histogram = nullptr;

    *result = FindAvailableColor(vertex_idx, &part, HISTOGRAM_THRESHOLD_DEFAULT);
}

__global__ void kernel_find_color_histogram(
    uint64_t* neighbor_bitmap,
    uint32_t* high_degree_map,
    uint32_t* freq_histogram,
    uint16_t* result,
    uint32_t vertex_idx
) {
    GPUPartition part;
    memset(&part, 0, sizeof(GPUPartition));
    part.neighbor_bitmap = neighbor_bitmap;
    part.high_degree_map = high_degree_map;
    part.freq_histogram = freq_histogram;

    *result = FindAvailableColor(vertex_idx, &part, HISTOGRAM_THRESHOLD_DEFAULT);
}

void test_bitmap_empty() {
    // No neighbors -- bitmap is 0, should return color 1
    uint64_t* d_bitmap;
    uint32_t* d_hd_map;
    uint16_t* d_result;

    TEST_CUDA_CHECK(cudaMalloc(&d_bitmap, sizeof(uint64_t)));
    TEST_CUDA_CHECK(cudaMalloc(&d_hd_map, sizeof(uint32_t)));
    TEST_CUDA_CHECK(cudaMalloc(&d_result, sizeof(uint16_t)));

    uint64_t bm = 0;
    uint32_t hd = HASH_EMPTY;
    TEST_CUDA_CHECK(cudaMemcpy(d_bitmap, &bm, sizeof(uint64_t), cudaMemcpyHostToDevice));
    TEST_CUDA_CHECK(cudaMemcpy(d_hd_map, &hd, sizeof(uint32_t), cudaMemcpyHostToDevice));

    kernel_find_color_bitmap<<<1, 1>>>(d_bitmap, d_hd_map, d_result, 0);
    TEST_CUDA_CHECK(cudaDeviceSynchronize());

    uint16_t h_result;
    TEST_CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(uint16_t), cudaMemcpyDeviceToHost));
    assert(h_result == 1);

    cudaFree(d_bitmap); cudaFree(d_hd_map); cudaFree(d_result);
}

void test_bitmap_first_color_taken() {
    // Color 1 taken (bit 0 set), should return color 2
    uint64_t* d_bitmap;
    uint32_t* d_hd_map;
    uint16_t* d_result;

    TEST_CUDA_CHECK(cudaMalloc(&d_bitmap, sizeof(uint64_t)));
    TEST_CUDA_CHECK(cudaMalloc(&d_hd_map, sizeof(uint32_t)));
    TEST_CUDA_CHECK(cudaMalloc(&d_result, sizeof(uint16_t)));

    uint64_t bm = 0x1;  // bit 0 set = color 1 taken
    uint32_t hd = HASH_EMPTY;
    TEST_CUDA_CHECK(cudaMemcpy(d_bitmap, &bm, sizeof(uint64_t), cudaMemcpyHostToDevice));
    TEST_CUDA_CHECK(cudaMemcpy(d_hd_map, &hd, sizeof(uint32_t), cudaMemcpyHostToDevice));

    kernel_find_color_bitmap<<<1, 1>>>(d_bitmap, d_hd_map, d_result, 0);
    TEST_CUDA_CHECK(cudaDeviceSynchronize());

    uint16_t h_result;
    TEST_CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(uint16_t), cudaMemcpyDeviceToHost));
    assert(h_result == 2);

    cudaFree(d_bitmap); cudaFree(d_hd_map); cudaFree(d_result);
}

void test_bitmap_scattered_colors() {
    // Colors 1,2,3,5 taken (bits 0,1,2,4), should return color 4
    uint64_t* d_bitmap;
    uint32_t* d_hd_map;
    uint16_t* d_result;

    TEST_CUDA_CHECK(cudaMalloc(&d_bitmap, sizeof(uint64_t)));
    TEST_CUDA_CHECK(cudaMalloc(&d_hd_map, sizeof(uint32_t)));
    TEST_CUDA_CHECK(cudaMalloc(&d_result, sizeof(uint16_t)));

    uint64_t bm = 0b10111;  // bits 0,1,2,4 set
    uint32_t hd = HASH_EMPTY;
    TEST_CUDA_CHECK(cudaMemcpy(d_bitmap, &bm, sizeof(uint64_t), cudaMemcpyHostToDevice));
    TEST_CUDA_CHECK(cudaMemcpy(d_hd_map, &hd, sizeof(uint32_t), cudaMemcpyHostToDevice));

    kernel_find_color_bitmap<<<1, 1>>>(d_bitmap, d_hd_map, d_result, 0);
    TEST_CUDA_CHECK(cudaDeviceSynchronize());

    uint16_t h_result;
    TEST_CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(uint16_t), cudaMemcpyDeviceToHost));
    assert(h_result == 4);

    cudaFree(d_bitmap); cudaFree(d_hd_map); cudaFree(d_result);
}

void test_bitmap_all_64_taken() {
    // All 64 bits set -- should return 65
    uint64_t* d_bitmap;
    uint32_t* d_hd_map;
    uint16_t* d_result;

    TEST_CUDA_CHECK(cudaMalloc(&d_bitmap, sizeof(uint64_t)));
    TEST_CUDA_CHECK(cudaMalloc(&d_hd_map, sizeof(uint32_t)));
    TEST_CUDA_CHECK(cudaMalloc(&d_result, sizeof(uint16_t)));

    uint64_t bm = 0xFFFFFFFFFFFFFFFFULL;
    uint32_t hd = HASH_EMPTY;
    TEST_CUDA_CHECK(cudaMemcpy(d_bitmap, &bm, sizeof(uint64_t), cudaMemcpyHostToDevice));
    TEST_CUDA_CHECK(cudaMemcpy(d_hd_map, &hd, sizeof(uint32_t), cudaMemcpyHostToDevice));

    kernel_find_color_bitmap<<<1, 1>>>(d_bitmap, d_hd_map, d_result, 0);
    TEST_CUDA_CHECK(cudaDeviceSynchronize());

    uint16_t h_result;
    TEST_CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(uint16_t), cudaMemcpyDeviceToHost));
    assert(h_result == 65);

    cudaFree(d_bitmap); cudaFree(d_hd_map); cudaFree(d_result);
}

void test_histogram_empty() {
    // All histogram counts 0 -- should return color 1
    uint64_t* d_bitmap;
    uint32_t* d_hd_map;
    uint32_t* d_hist;
    uint16_t* d_result;

    TEST_CUDA_CHECK(cudaMalloc(&d_bitmap, sizeof(uint64_t)));
    TEST_CUDA_CHECK(cudaMalloc(&d_hd_map, sizeof(uint32_t)));
    TEST_CUDA_CHECK(cudaMalloc(&d_hist, MAX_HISTOGRAM_COLORS * sizeof(uint32_t)));
    TEST_CUDA_CHECK(cudaMalloc(&d_result, sizeof(uint16_t)));

    uint64_t bm = 0;
    uint32_t hd = 0;  // vertex 0 maps to histogram row 0
    TEST_CUDA_CHECK(cudaMemcpy(d_bitmap, &bm, sizeof(uint64_t), cudaMemcpyHostToDevice));
    TEST_CUDA_CHECK(cudaMemcpy(d_hd_map, &hd, sizeof(uint32_t), cudaMemcpyHostToDevice));
    TEST_CUDA_CHECK(cudaMemset(d_hist, 0, MAX_HISTOGRAM_COLORS * sizeof(uint32_t)));

    kernel_find_color_histogram<<<1, 1>>>(d_bitmap, d_hd_map, d_hist, d_result, 0);
    TEST_CUDA_CHECK(cudaDeviceSynchronize());

    uint16_t h_result;
    TEST_CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(uint16_t), cudaMemcpyDeviceToHost));
    assert(h_result == 1);

    cudaFree(d_bitmap); cudaFree(d_hd_map); cudaFree(d_hist); cudaFree(d_result);
}

void test_histogram_first_colors_taken() {
    // Colors 1-5 have neighbors using them, should return color 6
    uint64_t* d_bitmap;
    uint32_t* d_hd_map;
    uint32_t* d_hist;
    uint16_t* d_result;

    TEST_CUDA_CHECK(cudaMalloc(&d_bitmap, sizeof(uint64_t)));
    TEST_CUDA_CHECK(cudaMalloc(&d_hd_map, sizeof(uint32_t)));
    TEST_CUDA_CHECK(cudaMalloc(&d_hist, MAX_HISTOGRAM_COLORS * sizeof(uint32_t)));
    TEST_CUDA_CHECK(cudaMalloc(&d_result, sizeof(uint16_t)));

    uint64_t bm = 0;
    uint32_t hd = 0;
    TEST_CUDA_CHECK(cudaMemcpy(d_bitmap, &bm, sizeof(uint64_t), cudaMemcpyHostToDevice));
    TEST_CUDA_CHECK(cudaMemcpy(d_hd_map, &hd, sizeof(uint32_t), cudaMemcpyHostToDevice));
    TEST_CUDA_CHECK(cudaMemset(d_hist, 0, MAX_HISTOGRAM_COLORS * sizeof(uint32_t)));

    // Set counts for colors 1-5 (indices 0-4)
    uint32_t counts[5] = {3, 2, 1, 5, 4};
    TEST_CUDA_CHECK(cudaMemcpy(d_hist, counts, 5 * sizeof(uint32_t), cudaMemcpyHostToDevice));

    kernel_find_color_histogram<<<1, 1>>>(d_bitmap, d_hd_map, d_hist, d_result, 0);
    TEST_CUDA_CHECK(cudaDeviceSynchronize());

    uint16_t h_result;
    TEST_CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(uint16_t), cudaMemcpyDeviceToHost));
    assert(h_result == 6);

    cudaFree(d_bitmap); cudaFree(d_hd_map); cudaFree(d_hist); cudaFree(d_result);
}

void test_histogram_gap() {
    // Colors 1,2 taken, 3 free, 4 taken -- should return 3
    uint64_t* d_bitmap;
    uint32_t* d_hd_map;
    uint32_t* d_hist;
    uint16_t* d_result;

    TEST_CUDA_CHECK(cudaMalloc(&d_bitmap, sizeof(uint64_t)));
    TEST_CUDA_CHECK(cudaMalloc(&d_hd_map, sizeof(uint32_t)));
    TEST_CUDA_CHECK(cudaMalloc(&d_hist, MAX_HISTOGRAM_COLORS * sizeof(uint32_t)));
    TEST_CUDA_CHECK(cudaMalloc(&d_result, sizeof(uint16_t)));

    uint64_t bm = 0;
    uint32_t hd = 0;
    TEST_CUDA_CHECK(cudaMemcpy(d_bitmap, &bm, sizeof(uint64_t), cudaMemcpyHostToDevice));
    TEST_CUDA_CHECK(cudaMemcpy(d_hd_map, &hd, sizeof(uint32_t), cudaMemcpyHostToDevice));
    TEST_CUDA_CHECK(cudaMemset(d_hist, 0, MAX_HISTOGRAM_COLORS * sizeof(uint32_t)));

    uint32_t counts[4] = {2, 1, 0, 3};  // indices 0,1,2,3 = colors 1,2,3,4
    TEST_CUDA_CHECK(cudaMemcpy(d_hist, counts, 4 * sizeof(uint32_t), cudaMemcpyHostToDevice));

    kernel_find_color_histogram<<<1, 1>>>(d_bitmap, d_hd_map, d_hist, d_result, 0);
    TEST_CUDA_CHECK(cudaDeviceSynchronize());

    uint16_t h_result;
    TEST_CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(uint16_t), cudaMemcpyDeviceToHost));
    assert(h_result == 3);

    cudaFree(d_bitmap); cudaFree(d_hd_map); cudaFree(d_hist); cudaFree(d_result);
}

void test_dispatch_uses_bitmap_for_low_degree() {
    // Vertex with HASH_EMPTY in high_degree_map should use bitmap mode
    uint64_t* d_bitmap;
    uint32_t* d_hd_map;
    uint16_t* d_result;

    TEST_CUDA_CHECK(cudaMalloc(&d_bitmap, sizeof(uint64_t)));
    TEST_CUDA_CHECK(cudaMalloc(&d_hd_map, sizeof(uint32_t)));
    TEST_CUDA_CHECK(cudaMalloc(&d_result, sizeof(uint16_t)));

    uint64_t bm = 0b11;  // colors 1,2 taken
    uint32_t hd = HASH_EMPTY;  // low degree -> bitmap mode
    TEST_CUDA_CHECK(cudaMemcpy(d_bitmap, &bm, sizeof(uint64_t), cudaMemcpyHostToDevice));
    TEST_CUDA_CHECK(cudaMemcpy(d_hd_map, &hd, sizeof(uint32_t), cudaMemcpyHostToDevice));

    kernel_find_color_bitmap<<<1, 1>>>(d_bitmap, d_hd_map, d_result, 0);
    TEST_CUDA_CHECK(cudaDeviceSynchronize());

    uint16_t h_result;
    TEST_CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(uint16_t), cudaMemcpyDeviceToHost));
    assert(h_result == 3);  // should use bitmap, find color 3

    cudaFree(d_bitmap); cudaFree(d_hd_map); cudaFree(d_result);
}

int main() {
    printf("=== Color Select Unit Tests ===\n");

    TEST(bitmap_empty);
    TEST(bitmap_first_color_taken);
    TEST(bitmap_scattered_colors);
    TEST(bitmap_all_64_taken);
    TEST(histogram_empty);
    TEST(histogram_first_colors_taken);
    TEST(histogram_gap);
    TEST(dispatch_uses_bitmap_for_low_degree);

    printf("\n=== Results: %d passed, %d failed ===\n", passed, failed);
    return failed > 0 ? 1 : 0;
}
