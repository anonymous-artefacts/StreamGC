// t_rc_atomic.cu — measure T_rc: time for a single atomic 32-bit
//
// Build:  nvcc -O2 -arch=sm_90 -o t_rc_atomic t_rc_atomic.cu
// Run:    ./t_rc_atomic
// Output: median of 11 runs, in nanoseconds per atomic write.

#include <cstdio>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
    return 1; } } while (0)

// One warp, one word, many CAS-with-version-increment operations.
__global__ void trc_loop(uint32_t* word, int iters) {
    if (threadIdx.x != 0) return;
    uint32_t cur = *word;
    for (int i = 0; i < iters; ++i) {
        uint32_t version = (cur >> 16) + 1;
        uint32_t color = (cur & 0xFFFF) ^ 1;    // toggle low bit
        uint32_t nxt = (version << 16) | color;
        uint32_t got = atomicCAS(word, cur, nxt);
        cur = (got == cur) ? nxt : got;
    }
}

int main() {
    CK(cudaSetDevice(0));

    uint32_t* word = nullptr;
    CK(cudaMalloc(&word, sizeof(uint32_t)));
    uint32_t init = 0;
    CK(cudaMemcpy(word, &init, sizeof(uint32_t), cudaMemcpyHostToDevice));

    const int iters = 1000000;
    const int runs = 11;
    std::vector<double> medians;

    cudaEvent_t start, stop;
    CK(cudaEventCreate(&start));
    CK(cudaEventCreate(&stop));

    trc_loop<<<1, 32>>>(word, 1024);
    CK(cudaDeviceSynchronize());

    for (int r = 0; r < runs; ++r) {
        CK(cudaEventRecord(start));
        trc_loop<<<1, 32>>>(word, iters);
        CK(cudaEventRecord(stop));
        CK(cudaEventSynchronize(stop));
        float ms = 0.f;
        CK(cudaEventElapsedTime(&ms, start, stop));
        double ns_per = (ms * 1e6) / iters;
        medians.push_back(ns_per);
    }
    std::sort(medians.begin(), medians.end());
    printf("T_rc_median_ns = %.1f\n", medians[runs / 2]);
    return 0;
}
