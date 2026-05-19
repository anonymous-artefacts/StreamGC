#include "types.h"
#include <cstdio>
#include <cstdint>
#include <atomic>
#include <chrono>
#include <vector>
#include <thread>

struct FeederState {
    EdgeUpdate*     stream_buffer;   // ring buffer in pinned host memory
    uint32_t*       stream_tail;     // atomic write pointer (host side)
    uint32_t*       stream_head;     // read pointer (GPU side)
    uint32_t        stream_mask;     // power-of-2 mask
    volatile bool*  shutdown_flag;

    // Timing
    double init_time_ms;
    double bench_time_ms;

    // Status
    bool init_complete;
    bool bench_complete;
};

// Check if ring buffer is full
// Full when: (tail - head) >= buffer_size
// This uses the ring buffer invariant: tail always >= head
static inline bool is_buffer_full(
    const uint32_t* tail,
    const uint32_t* head,
    uint32_t buffer_size
) {
    uint32_t t = *tail;
    uint32_t h = *head;
    return (t - h) >= buffer_size;
}

// Wait for buffer space with backoff
static inline void wait_for_space(
    const uint32_t* tail,
    const uint32_t* head,
    uint32_t buffer_size
) {
    uint32_t spins = 0;
    while (is_buffer_full(tail, head, buffer_size)) {
        spins++;
        if (spins > 1000) {
            std::this_thread::yield();
            spins = 0;
        }
    }
}

// Feed a batch of edge updates into the ring buffer
// Called for both init and bench phases
static void feed_edges(
    FeederState& state,
    const std::vector<std::pair<uint32_t, uint32_t>>& edges,
    const std::vector<uint8_t>& types,  // 0=ADD, 1=DEL (empty = all ADD)
    uint32_t buffer_size
) {
    for (size_t i = 0; i < edges.size(); i++) {
        wait_for_space(state.stream_tail, state.stream_head, buffer_size);

        uint32_t slot = (*state.stream_tail) & state.stream_mask;

        // Write the update
        EdgeUpdate& update = state.stream_buffer[slot];
        update.u = edges[i].first;
        update.v = edges[i].second;

        if (types.empty()) {
            // Init phase: all ADDs
            update.type = static_cast<UpdateType>(0);  // ADD
        } else {
            update.type = static_cast<UpdateType>(types[i]);
        }

        // Memory fence to ensure update is visible before advancing tail
        __sync_synchronize();

        // Advance tail
        (*state.stream_tail)++;
    }
}

// Wait for GPU to drain all pending updates
static void wait_for_drain(const FeederState& state) {
    while (*state.stream_head != *state.stream_tail) {
        std::this_thread::yield();
    }
}

// Main feeder thread function
void feeder_thread_func(
    FeederState& state,
    const std::vector<std::pair<uint32_t, uint32_t>>& init_edges,
    const std::vector<std::pair<uint32_t, uint32_t>>& bench_edges,
    const std::vector<uint8_t>& bench_types,
    uint32_t buffer_size
) {
    state.init_complete = false;
    state.bench_complete = false;

    // Phase 1: Feed init stream
    printf("[StreamGC] Feeder: starting init stream (%zu edges)\n", init_edges.size());
    auto t0 = std::chrono::high_resolution_clock::now();

    std::vector<uint8_t> empty_types;  // all ADD
    feed_edges(state, init_edges, empty_types, buffer_size);

    // Wait for GPU to process all init edges
    wait_for_drain(state);

    auto t1 = std::chrono::high_resolution_clock::now();
    state.init_time_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    state.init_complete = true;
    printf("[StreamGC] Feeder: init complete in %.2f ms\n", state.init_time_ms);

    // Phase 2: Feed benchmark stream
    printf("[StreamGC] Feeder: starting bench stream (%zu events)\n", bench_edges.size());
    auto t2 = std::chrono::high_resolution_clock::now();

    feed_edges(state, bench_edges, bench_types, buffer_size);

    // Wait for drain
    wait_for_drain(state);

    auto t3 = std::chrono::high_resolution_clock::now();
    state.bench_time_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
    state.bench_complete = true;
    printf("[StreamGC] Feeder: bench complete in %.2f ms (%.2f M updates/sec)\n",
           state.bench_time_ms,
           bench_edges.size() / state.bench_time_ms / 1000.0);

    // Signal shutdown
    *state.shutdown_flag = true;
}
