// pingpong_nvlink.cu — measure L_nvlink (single-word peer-load RT) between
// two H100 GPUs on the same NVSwitch / NVLink 4 fabric.
//
// Build:  nvcc -O2 -arch=sm_90 -o pingpong_nvlink pingpong_nvlink.cu
// Run:    ./pingpong_nvlink          (uses device 0 and device 1)
// Output: median RT latency in nanoseconds over 10 000 iterations.

#include <cstdio>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
    return 1; } } while (0)

// Strided peer-load over a buffer larger than the reader's L2
// so every read misses L2 and must cross NVLink to GPU 1's HBM3.
__global__ void peer_read_loop(const uint32_t* __restrict__ remote_buf,
                               uint32_t buf_words, uint32_t stride_words,
                               uint32_t* sink, int iters) {
    uint32_t x = 0;
    uint32_t idx = 0;
    for (int i = 0; i < iters; ++i) {
        idx += stride_words;
        if (idx >= buf_words) idx -= buf_words;
        // volatile load: bypass L1, L2 line kicked by stride crossing cache-line.
        x ^= __ldcv(remote_buf + idx);
    }
    if (threadIdx.x == 0) *sink = x;
}

int main() {
    int ndev = 0;
    CK(cudaGetDeviceCount(&ndev));
    if (ndev < 2) {
        fprintf(stderr, "need >= 2 GPUs; found %d\n", ndev);
        return 1;
    }

    int can_access = 0;
    CK(cudaDeviceCanAccessPeer(&can_access, 0, 1));
    if (!can_access) {
        fprintf(stderr, "peer access 0 -> 1 not available\n");
        return 1;
    }
    CK(cudaSetDevice(0));
    CK(cudaDeviceEnablePeerAccess(1, 0));

    // 256 MiB > H100 L2 (60 MiB) forces misses regardless of reader-side caching.
    const uint32_t buf_words = 64u * 1024u * 1024u;
    const uint32_t stride_words = 32u;  // 128 B = one cache line, 2 MiB in 16 K iters
    uint32_t *remote = nullptr, *sink = nullptr;
    CK(cudaSetDevice(1));
    CK(cudaMalloc(&remote, buf_words * sizeof(uint32_t)));
    CK(cudaMemset(remote, 0xA5, buf_words * sizeof(uint32_t)));

    CK(cudaSetDevice(0));
    CK(cudaMalloc(&sink, sizeof(uint32_t)));

    const int iters = 10000;
    const int runs = 11;
    std::vector<double> medians;

    cudaEvent_t start, stop;
    CK(cudaEventCreate(&start));
    CK(cudaEventCreate(&stop));

    // warm up
    peer_read_loop<<<1, 1>>>(remote, buf_words, stride_words, sink, 1024);
    CK(cudaDeviceSynchronize());

    for (int r = 0; r < runs; ++r) {
        CK(cudaEventRecord(start));
        peer_read_loop<<<1, 1>>>(remote, buf_words, stride_words, sink, iters);
        CK(cudaEventRecord(stop));
        CK(cudaEventSynchronize(stop));
        float ms = 0.f;
        CK(cudaEventElapsedTime(&ms, start, stop));
        double ns_per = (ms * 1e6) / iters;
        medians.push_back(ns_per);
    }
    std::sort(medians.begin(), medians.end());
    printf("L_nvlink_median_ns = %.1f\n", medians[runs / 2]);
    return 0;
}
