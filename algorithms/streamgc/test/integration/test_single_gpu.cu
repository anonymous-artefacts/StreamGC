#include "streamgc.cuh"
#include "color_select.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <cassert>

// Definition of partition_map (required by partition.cuh extern declarations)
PartitionEntry* partition_map = nullptr;
uint32_t        partition_map_size = 0;

// Kernel declarations (defined in separate .cu translation units, linked via -rdc=true)
__global__ void local_update_kernel(EdgeUpdate* updates, uint32_t num_updates, GPUPartition* partition, uint32_t histogram_threshold);
__global__ void conflict_repair_kernel(GPUPartition* partition, uint32_t histogram_threshold, uint32_t* num_conflicts);
__global__ void rebuild_bitmaps_kernel(GPUPartition* partition);
__global__ void compact_csr_kernel(GPUPartition* partition);

// Test helper: build a simple CSR from edge list
struct SimpleCSR {
    uint32_t num_vertices;
    std::vector<uint32_t> row_ptr;
    std::vector<uint32_t> col_idx;
    std::vector<uint32_t> degree;
};

SimpleCSR build_csr(uint32_t nv, const std::vector<std::pair<uint32_t,uint32_t>>& edges) {
    SimpleCSR csr;
    csr.num_vertices = nv;
    csr.degree.resize(nv, 0);
    // Count degrees (symmetric)
    for (auto& e : edges) {
        csr.degree[e.first]++;
        csr.degree[e.second]++;
    }
    // Build row_ptr with 2x capacity
    csr.row_ptr.resize(nv + 1, 0);
    for (uint32_t i = 0; i < nv; i++) {
        csr.row_ptr[i + 1] = csr.row_ptr[i] + csr.degree[i] * 2;
    }
    // Fill col_idx
    uint32_t total = csr.row_ptr[nv];
    csr.col_idx.resize(total, CSR_INVALID);
    std::vector<uint32_t> pos(nv);
    for (uint32_t i = 0; i < nv; i++) pos[i] = csr.row_ptr[i];
    for (auto& e : edges) {
        csr.col_idx[pos[e.first]++] = e.second;
        csr.col_idx[pos[e.second]++] = e.first;
    }
    return csr;
}

// Allocate a minimal GPU partition for testing
GPUPartition* alloc_test_partition(const SimpleCSR& csr) {
    uint32_t nv = csr.num_vertices;

    GPUPartition h_part;
    memset(&h_part, 0, sizeof(GPUPartition));
    h_part.num_owned = nv;
    h_part.num_ghost = 0;
    h_part.max_owned = nv;
    h_part.max_ghost = 1;
    h_part.gpu_id = 0;
    h_part.num_high_degree = 0;

    // Hash map size
    uint32_t map_size = 1;
    while (map_size < nv * 2) map_size <<= 1;
    h_part.index_map_size = map_size;
    h_part.ghost_map_size = 4;

    // Allocate arrays
    CUDA_CHECK(cudaMalloc(&h_part.owned, nv * sizeof(VertexState)));
    CUDA_CHECK(cudaMemset(h_part.owned, 0, nv * sizeof(VertexState)));
    CUDA_CHECK(cudaMalloc(&h_part.owned_to_global, nv * sizeof(uint32_t)));

    CUDA_CHECK(cudaMalloc(&h_part.ghost, sizeof(VertexState)));
    CUDA_CHECK(cudaMalloc(&h_part.ghost_to_global, sizeof(uint32_t)));

    // Hash maps
    CUDA_CHECK(cudaMalloc(&h_part.global_to_owned, map_size * 2 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(h_part.global_to_owned, 0xFF, map_size * 2 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&h_part.global_to_ghost, 4 * 2 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(h_part.global_to_ghost, 0xFF, 4 * 2 * sizeof(uint32_t)));

    // Bitmaps
    CUDA_CHECK(cudaMalloc(&h_part.neighbor_bitmap, nv * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemset(h_part.neighbor_bitmap, 0, nv * sizeof(uint64_t)));

    // High degree (none)
    CUDA_CHECK(cudaMalloc(&h_part.high_degree_map, nv * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(h_part.high_degree_map, 0xFF, nv * sizeof(uint32_t)));
    h_part.freq_histogram = nullptr;

    // Priority
    CUDA_CHECK(cudaMallocManaged(&h_part.init_priority, nv * sizeof(uint64_t)));
    for (uint32_t i = 0; i < nv; i++) {
        h_part.init_priority[i] = ((uint64_t)csr.degree[i] << 32) | i;
    }

    // Epoch snapshot
    CUDA_CHECK(cudaMallocManaged(&h_part.epoch_snapshot, nv * sizeof(uint16_t)));
    CUDA_CHECK(cudaMemset(h_part.epoch_snapshot, 0, nv * sizeof(uint16_t)));

    // CSR
    uint32_t total_slots = csr.row_ptr[nv];
    CUDA_CHECK(cudaMalloc(&h_part.row_ptr, (nv + 1) * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(h_part.row_ptr, csr.row_ptr.data(), (nv + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&h_part.col_idx, total_slots * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(h_part.col_idx, csr.col_idx.data(), total_slots * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&h_part.col_idx_capacity, nv * sizeof(uint32_t)));
    // capacity per vertex
    {
        std::vector<uint32_t> cap(nv);
        for (uint32_t i = 0; i < nv; i++) cap[i] = csr.row_ptr[i+1] - csr.row_ptr[i];
        CUDA_CHECK(cudaMemcpy(h_part.col_idx_capacity, cap.data(), nv * sizeof(uint32_t), cudaMemcpyHostToDevice));
    }

    // Ghost CSR (empty)
    CUDA_CHECK(cudaMalloc(&h_part.ghost_row_ptr, (nv + 1) * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(h_part.ghost_row_ptr, 0, (nv + 1) * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&h_part.ghost_col_idx, sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(h_part.ghost_col_idx, 0xFF, sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&h_part.ghost_col_idx_capacity, nv * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(h_part.ghost_col_idx_capacity, 0, nv * sizeof(uint32_t)));

    // Ring buffer (unused but must be allocated)
    CUDA_CHECK(cudaHostAlloc(&h_part.stream_buffer, sizeof(EdgeUpdate), cudaHostAllocMapped));
    CUDA_CHECK(cudaHostAlloc(&h_part.stream_head, sizeof(uint32_t), cudaHostAllocMapped));
    CUDA_CHECK(cudaHostAlloc(&h_part.stream_tail, sizeof(uint32_t), cudaHostAllocMapped));
    *h_part.stream_head = 0; *h_part.stream_tail = 0; h_part.stream_mask = 0;

    // Counters
    CUDA_CHECK(cudaMallocManaged(&h_part.update_counter, sizeof(uint64_t)));
    *h_part.update_counter = 0;
    CUDA_CHECK(cudaMallocManaged(&h_part.epoch_ready_flag, sizeof(uint32_t)));
    *h_part.epoch_ready_flag = 0;
    CUDA_CHECK(cudaMallocManaged(&h_part.epoch_counter, sizeof(uint32_t)));
    *h_part.epoch_counter = 0;

    // owned_to_global and hash map (identity mapping for single-GPU)
    {
        std::vector<uint32_t> o2g(nv);
        std::vector<uint32_t> hash(map_size * 2, HASH_EMPTY);
        for (uint32_t i = 0; i < nv; i++) {
            o2g[i] = i;
            hash_insert_host(hash.data(), map_size, i, i);
        }
        CUDA_CHECK(cudaMemcpy(h_part.owned_to_global, o2g.data(), nv * sizeof(uint32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(h_part.global_to_owned, hash.data(), map_size * 2 * sizeof(uint32_t), cudaMemcpyHostToDevice));
    }

    h_part.d_partition_map = nullptr;

    GPUPartition* d_part;
    CUDA_CHECK(cudaMalloc(&d_part, sizeof(GPUPartition)));
    CUDA_CHECK(cudaMemcpy(d_part, &h_part, sizeof(GPUPartition), cudaMemcpyHostToDevice));
    return d_part;
}

int passed = 0, failed = 0;
#define TEST(name) { printf("  TEST: %s ... ", #name); fflush(stdout); }
#define PASS() { printf("PASSED\n"); passed++; }
#define FAIL(msg) { printf("FAILED: %s\n", msg); failed++; }

// ---- Test 1: Small graph coloring ----
void test_small_graph_coloring() {
    TEST(small_graph_coloring);

    // K4 complete graph (4 vertices, 6 edges) - needs 4 colors
    std::vector<std::pair<uint32_t,uint32_t>> edges = {
        {0,1}, {0,2}, {0,3}, {1,2}, {1,3}, {2,3}
    };
    auto csr = build_csr(4, edges);
    GPUPartition* d_part = alloc_test_partition(csr);

    // Stream all edges through local_update_kernel
    std::vector<EdgeUpdate> updates(edges.size());
    for (size_t i = 0; i < edges.size(); i++) {
        updates[i].u = edges[i].first;
        updates[i].v = edges[i].second;
        updates[i].type = UpdateType::ADD;
        memset(updates[i]._pad, 0, sizeof(updates[i]._pad));
    }

    EdgeUpdate* d_updates;
    CUDA_CHECK(cudaMalloc(&d_updates, updates.size() * sizeof(EdgeUpdate)));
    CUDA_CHECK(cudaMemcpy(d_updates, updates.data(), updates.size() * sizeof(EdgeUpdate), cudaMemcpyHostToDevice));

    int blocks = ((int)updates.size() * 32 + 255) / 256;
    local_update_kernel<<<blocks, 256>>>(d_updates, updates.size(), d_part, 64);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Repair
    uint32_t* d_nc;
    CUDA_CHECK(cudaMallocManaged(&d_nc, sizeof(uint32_t)));
    for (int iter = 0; iter < 20; iter++) {
        rebuild_bitmaps_kernel<<<1, 256>>>(d_part);
        CUDA_CHECK(cudaDeviceSynchronize());
        *d_nc = 0;
        conflict_repair_kernel<<<1, 256>>>(d_part, 64, d_nc);
        CUDA_CHECK(cudaDeviceSynchronize());
        if (*d_nc == 0) break;
    }

    // Pull colors and validate
    GPUPartition hp;
    CUDA_CHECK(cudaMemcpy(&hp, d_part, sizeof(GPUPartition), cudaMemcpyDeviceToHost));
    std::vector<VertexState> states(4);
    CUDA_CHECK(cudaMemcpy(states.data(), hp.owned, 4 * sizeof(VertexState), cudaMemcpyDeviceToHost));

    // Check: all colored
    bool all_colored = true;
    for (int i = 0; i < 4; i++) {
        if (states[i].color == COLOR_UNCOLORED) { all_colored = false; break; }
    }

    // Check: no conflicts
    bool no_conflicts = true;
    for (auto& e : edges) {
        if (states[e.first].color == states[e.second].color) { no_conflicts = false; break; }
    }

    // K4 needs exactly 4 colors
    uint16_t max_c = 0;
    for (int i = 0; i < 4; i++) if (states[i].color > max_c) max_c = states[i].color;

    if (all_colored && no_conflicts && max_c >= 4) PASS()
    else {
        char msg[256];
        snprintf(msg, sizeof(msg), "colored=%d noconflict=%d maxcolor=%u", all_colored, no_conflicts, max_c);
        FAIL(msg);
    }

    cudaFree(d_updates);
    cudaFree(d_nc);
}

// ---- Test 2: Edge deletion ----
void test_edge_deletion() {
    TEST(edge_deletion);

    // Triangle: 3 vertices, 3 edges
    std::vector<std::pair<uint32_t,uint32_t>> edges = {{0,1}, {1,2}, {0,2}};
    auto csr = build_csr(3, edges);
    GPUPartition* d_part = alloc_test_partition(csr);

    // Add all edges
    std::vector<EdgeUpdate> adds(3);
    for (int i = 0; i < 3; i++) {
        adds[i].u = edges[i].first; adds[i].v = edges[i].second;
        adds[i].type = UpdateType::ADD;
        memset(adds[i]._pad, 0, sizeof(adds[i]._pad));
    }
    EdgeUpdate* d_updates;
    CUDA_CHECK(cudaMalloc(&d_updates, 3 * sizeof(EdgeUpdate)));
    CUDA_CHECK(cudaMemcpy(d_updates, adds.data(), 3 * sizeof(EdgeUpdate), cudaMemcpyHostToDevice));
    local_update_kernel<<<1, 96>>>(d_updates, 3, d_part, 64);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Repair
    uint32_t* d_nc;
    CUDA_CHECK(cudaMallocManaged(&d_nc, sizeof(uint32_t)));
    for (int iter = 0; iter < 10; iter++) {
        rebuild_bitmaps_kernel<<<1, 256>>>(d_part);
        CUDA_CHECK(cudaDeviceSynchronize());
        *d_nc = 0;
        conflict_repair_kernel<<<1, 256>>>(d_part, 64, d_nc);
        CUDA_CHECK(cudaDeviceSynchronize());
        if (*d_nc == 0) break;
    }

    // Now delete edge (0,2)
    EdgeUpdate del;
    del.u = 0; del.v = 2; del.type = UpdateType::DEL;
    memset(del._pad, 0, sizeof(del._pad));
    CUDA_CHECK(cudaMemcpy(d_updates, &del, sizeof(EdgeUpdate), cudaMemcpyHostToDevice));
    local_update_kernel<<<1, 32>>>(d_updates, 1, d_part, 64);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Validate: graph is now a path 0-1-2, needs only 2 colors
    GPUPartition hp;
    CUDA_CHECK(cudaMemcpy(&hp, d_part, sizeof(GPUPartition), cudaMemcpyDeviceToHost));
    std::vector<VertexState> states(3);
    CUDA_CHECK(cudaMemcpy(states.data(), hp.owned, 3 * sizeof(VertexState), cudaMemcpyDeviceToHost));

    // Check no conflict on remaining edges (0,1) and (1,2)
    bool ok = (states[0].color != states[1].color) && (states[1].color != states[2].color);
    bool all_colored = states[0].color != COLOR_UNCOLORED && states[1].color != COLOR_UNCOLORED && states[2].color != COLOR_UNCOLORED;

    if (ok && all_colored) PASS()
    else FAIL("conflict or uncolored after deletion");

    cudaFree(d_updates);
    cudaFree(d_nc);
}

// ---- Test 3: Larger graph (cycle) ----
void test_cycle_graph() {
    TEST(cycle_graph_100);

    const uint32_t N = 100;
    std::vector<std::pair<uint32_t,uint32_t>> edges;
    for (uint32_t i = 0; i < N; i++) {
        edges.push_back({i, (i + 1) % N});
    }
    auto csr = build_csr(N, edges);
    GPUPartition* d_part = alloc_test_partition(csr);

    // Stream edges
    std::vector<EdgeUpdate> updates(edges.size());
    for (size_t i = 0; i < edges.size(); i++) {
        updates[i].u = edges[i].first; updates[i].v = edges[i].second;
        updates[i].type = UpdateType::ADD;
        memset(updates[i]._pad, 0, sizeof(updates[i]._pad));
    }
    EdgeUpdate* d_updates;
    CUDA_CHECK(cudaMalloc(&d_updates, updates.size() * sizeof(EdgeUpdate)));
    CUDA_CHECK(cudaMemcpy(d_updates, updates.data(), updates.size() * sizeof(EdgeUpdate), cudaMemcpyHostToDevice));

    int blocks = ((int)updates.size() * 32 + 255) / 256;
    local_update_kernel<<<blocks, 256>>>(d_updates, updates.size(), d_part, 64);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Repair
    uint32_t* d_nc;
    CUDA_CHECK(cudaMallocManaged(&d_nc, sizeof(uint32_t)));
    for (int iter = 0; iter < 20; iter++) {
        rebuild_bitmaps_kernel<<<1, 256>>>(d_part);
        CUDA_CHECK(cudaDeviceSynchronize());
        *d_nc = 0;
        conflict_repair_kernel<<<1, 256>>>(d_part, 64, d_nc);
        CUDA_CHECK(cudaDeviceSynchronize());
        if (*d_nc == 0) break;
    }

    // Validate
    GPUPartition hp;
    CUDA_CHECK(cudaMemcpy(&hp, d_part, sizeof(GPUPartition), cudaMemcpyDeviceToHost));
    std::vector<VertexState> states(N);
    CUDA_CHECK(cudaMemcpy(states.data(), hp.owned, N * sizeof(VertexState), cudaMemcpyDeviceToHost));

    // Even cycle needs 2 colors, odd cycle needs 3
    bool all_colored = true;
    uint16_t max_c = 0;
    for (uint32_t i = 0; i < N; i++) {
        if (states[i].color == COLOR_UNCOLORED) all_colored = false;
        if (states[i].color > max_c) max_c = states[i].color;
    }

    bool no_conflicts = true;
    for (auto& e : edges) {
        if (states[e.first].color == states[e.second].color &&
            states[e.first].color != COLOR_UNCOLORED) {
            no_conflicts = false; break;
        }
    }

    // Even 100-cycle needs 2 colors, but streaming may use up to 3
    if (all_colored && no_conflicts && max_c <= 3) PASS()
    else {
        char msg[256];
        snprintf(msg, sizeof(msg), "colored=%d noconflict=%d maxcolor=%u", all_colored, no_conflicts, max_c);
        FAIL(msg);
    }

    cudaFree(d_updates);
    cudaFree(d_nc);
}

// ---- Test 4: Mixed add/delete stream ----
void test_mixed_stream() {
    TEST(mixed_add_delete_stream);

    // Star graph: center=0 connected to 1,2,3,4,5
    std::vector<std::pair<uint32_t,uint32_t>> edges = {
        {0,1}, {0,2}, {0,3}, {0,4}, {0,5}
    };
    auto csr = build_csr(6, edges);
    GPUPartition* d_part = alloc_test_partition(csr);

    EdgeUpdate* d_updates;
    CUDA_CHECK(cudaMalloc(&d_updates, 10 * sizeof(EdgeUpdate)));

    // Add all star edges
    std::vector<EdgeUpdate> updates(5);
    for (int i = 0; i < 5; i++) {
        updates[i].u = edges[i].first; updates[i].v = edges[i].second;
        updates[i].type = UpdateType::ADD;
        memset(updates[i]._pad, 0, sizeof(updates[i]._pad));
    }
    CUDA_CHECK(cudaMemcpy(d_updates, updates.data(), 5 * sizeof(EdgeUpdate), cudaMemcpyHostToDevice));
    local_update_kernel<<<1, 160>>>(d_updates, 5, d_part, 64);
    CUDA_CHECK(cudaDeviceSynchronize());

    uint32_t* d_nc;
    CUDA_CHECK(cudaMallocManaged(&d_nc, sizeof(uint32_t)));
    for (int iter = 0; iter < 10; iter++) {
        rebuild_bitmaps_kernel<<<1, 256>>>(d_part);
        CUDA_CHECK(cudaDeviceSynchronize());
        *d_nc = 0;
        conflict_repair_kernel<<<1, 256>>>(d_part, 64, d_nc);
        CUDA_CHECK(cudaDeviceSynchronize());
        if (*d_nc == 0) break;
    }

    // Delete edges to 4 and 5, add edge 1-2
    std::vector<EdgeUpdate> mixed(3);
    mixed[0].u = 0; mixed[0].v = 4; mixed[0].type = UpdateType::DEL;
    memset(mixed[0]._pad, 0, sizeof(mixed[0]._pad));
    mixed[1].u = 0; mixed[1].v = 5; mixed[1].type = UpdateType::DEL;
    memset(mixed[1]._pad, 0, sizeof(mixed[1]._pad));
    mixed[2].u = 1; mixed[2].v = 2; mixed[2].type = UpdateType::ADD;
    memset(mixed[2]._pad, 0, sizeof(mixed[2]._pad));
    CUDA_CHECK(cudaMemcpy(d_updates, mixed.data(), 3 * sizeof(EdgeUpdate), cudaMemcpyHostToDevice));
    local_update_kernel<<<1, 96>>>(d_updates, 3, d_part, 64);
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int iter = 0; iter < 10; iter++) {
        rebuild_bitmaps_kernel<<<1, 256>>>(d_part);
        CUDA_CHECK(cudaDeviceSynchronize());
        *d_nc = 0;
        conflict_repair_kernel<<<1, 256>>>(d_part, 64, d_nc);
        CUDA_CHECK(cudaDeviceSynchronize());
        if (*d_nc == 0) break;
    }

    // Validate: remaining edges are (0,1),(0,2),(0,3),(1,2)
    GPUPartition hp;
    CUDA_CHECK(cudaMemcpy(&hp, d_part, sizeof(GPUPartition), cudaMemcpyDeviceToHost));
    std::vector<VertexState> states(6);
    CUDA_CHECK(cudaMemcpy(states.data(), hp.owned, 6 * sizeof(VertexState), cudaMemcpyDeviceToHost));

    // Check active edges
    std::vector<std::pair<uint32_t,uint32_t>> active = {{0,1},{0,2},{0,3},{1,2}};
    bool no_conflicts = true;
    for (auto& e : active) {
        if (states[e.first].color == states[e.second].color &&
            states[e.first].color != COLOR_UNCOLORED) {
            no_conflicts = false; break;
        }
    }

    if (no_conflicts) PASS()
    else FAIL("conflict on active edges after mixed stream");

    cudaFree(d_updates);
    cudaFree(d_nc);
}

int main() {
    printf("=== StreamGC Single-GPU Integration Tests ===\n\n");

    test_small_graph_coloring();
    test_edge_deletion();
    test_cycle_graph();
    test_mixed_stream();

    printf("\n=== Results: %d passed, %d failed ===\n", passed, failed);
    return failed > 0 ? 1 : 0;
}
