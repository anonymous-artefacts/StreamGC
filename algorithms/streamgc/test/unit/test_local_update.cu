#include "streamgc.cuh"
#include "color_select.cuh"
#include <cstdio>
#include <cassert>
#include <cstring>
#include <cuda_runtime.h>

// Definition of partition_map
PartitionEntry* partition_map = nullptr;
uint32_t        partition_map_size = 0;

static int passed = 0;
static int failed = 0;

#define TEST(name) \
    printf("  TEST: %-50s ", #name); \
    test_##name(); \
    printf("PASS\n"); passed++;

// ---- Simplified recolor for testing ----
// Uses FindAvailableColor from color_select.cuh

__device__ void test_recolor_vertex(
    uint32_t local_vertex_idx,
    GPUPartition* partition,
    uint32_t histogram_threshold
) {
    uint16_t old_color = partition->owned[local_vertex_idx].color;
    uint16_t new_color = FindAvailableColor(local_vertex_idx, partition, histogram_threshold);

    if (new_color == old_color) return;

    // Simple write for single-threaded test (no CAS needed)
    partition->owned[local_vertex_idx].color = new_color;
    partition->owned[local_vertex_idx].version += 1;
}

// ---- Test: ADD edge between different-colored vertices -> no recolor ----

__global__ void kernel_test_no_conflict(
    GPUPartition* partition,
    uint32_t u_idx,
    uint32_t v_idx,
    uint32_t* recolor_happened
) {
    uint16_t color_u = partition->owned[u_idx].color;
    uint16_t color_v = partition->owned[v_idx].color;

    // Update bitmaps for new neighbors
    update_bitmap_for_new_neighbor(u_idx, color_v, partition);
    update_bitmap_for_new_neighbor(v_idx, color_u, partition);

    // Check conflict
    bool has_conflict = (color_u == color_v) && (color_u != COLOR_UNCOLORED);

    if (has_conflict) {
        *recolor_happened = 1;
        // Lower priority recolors
        uint64_t pu = partition->init_priority[partition->owned_to_global[u_idx]];
        uint64_t pv = partition->init_priority[partition->owned_to_global[v_idx]];
        uint32_t loser_idx = (pu < pv) ? u_idx : v_idx;
        test_recolor_vertex(loser_idx, partition, HISTOGRAM_THRESHOLD_DEFAULT);
    } else {
        *recolor_happened = 0;
    }
}

// Helper to set up a minimal 2-vertex partition on device
struct TestPartitionSetup {
    GPUPartition* d_partition;
    VertexState* d_owned;
    uint64_t* d_bitmap;
    uint64_t* d_priority;
    uint32_t* d_owned_to_global;
    uint32_t* d_high_degree_map;
    uint32_t* d_global_to_owned;

    void init(uint16_t color_u, uint16_t color_v, uint32_t degree_u, uint32_t degree_v) {
        const uint32_t NUM_VERTS = 2;
        const uint32_t MAP_SIZE = 4;

        // Allocate arrays
        CUDA_CHECK(cudaMalloc(&d_owned, NUM_VERTS * sizeof(VertexState)));
        CUDA_CHECK(cudaMalloc(&d_bitmap, NUM_VERTS * sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc(&d_priority, 2 * sizeof(uint64_t)));  // for global IDs 0,1
        CUDA_CHECK(cudaMalloc(&d_owned_to_global, NUM_VERTS * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_high_degree_map, NUM_VERTS * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_global_to_owned, MAP_SIZE * 2 * sizeof(uint32_t)));

        // Initialize owned states
        VertexState states[2] = {{color_u, 0}, {color_v, 0}};
        CUDA_CHECK(cudaMemcpy(d_owned, states, 2 * sizeof(VertexState), cudaMemcpyHostToDevice));

        // Initialize bitmaps (empty)
        uint64_t bitmaps[2] = {0, 0};
        CUDA_CHECK(cudaMemcpy(d_bitmap, bitmaps, 2 * sizeof(uint64_t), cudaMemcpyHostToDevice));

        // Initialize priorities
        uint64_t priorities[2] = {
            compute_priority(degree_u, 0),
            compute_priority(degree_v, 1)
        };
        CUDA_CHECK(cudaMemcpy(d_priority, priorities, 2 * sizeof(uint64_t), cudaMemcpyHostToDevice));

        // owned_to_global: [0] -> global 0, [1] -> global 1
        uint32_t o2g[2] = {0, 1};
        CUDA_CHECK(cudaMemcpy(d_owned_to_global, o2g, 2 * sizeof(uint32_t), cudaMemcpyHostToDevice));

        // high_degree_map: all low degree (bitmap mode)
        uint32_t hd[2] = {HASH_EMPTY, HASH_EMPTY};
        CUDA_CHECK(cudaMemcpy(d_high_degree_map, hd, 2 * sizeof(uint32_t), cudaMemcpyHostToDevice));

        // global_to_owned hash map
        uint32_t h_map[MAP_SIZE * 2];
        for (uint32_t i = 0; i < MAP_SIZE * 2; i++) h_map[i] = HASH_EMPTY;
        // Insert key=0 -> value=0 and key=1 -> value=1
        uint32_t mask = MAP_SIZE - 1;
        uint32_t slot0 = (0 * 2654435761u) & mask;
        h_map[slot0 * 2] = 0;
        h_map[slot0 * 2 + 1] = 0;
        uint32_t slot1 = (1 * 2654435761u) & mask;
        // Handle collision
        if (h_map[slot1 * 2] != HASH_EMPTY) {
            slot1 = (slot1 + 1) & mask;
        }
        h_map[slot1 * 2] = 1;
        h_map[slot1 * 2 + 1] = 1;
        CUDA_CHECK(cudaMemcpy(d_global_to_owned, h_map, MAP_SIZE * 2 * sizeof(uint32_t), cudaMemcpyHostToDevice));

        // Build GPUPartition
        GPUPartition h_part;
        memset(&h_part, 0, sizeof(GPUPartition));
        h_part.owned = d_owned;
        h_part.neighbor_bitmap = d_bitmap;
        h_part.init_priority = d_priority;
        h_part.owned_to_global = d_owned_to_global;
        h_part.high_degree_map = d_high_degree_map;
        h_part.global_to_owned = d_global_to_owned;
        h_part.index_map_size = MAP_SIZE;
        h_part.num_owned = NUM_VERTS;
        h_part.freq_histogram = nullptr;

        CUDA_CHECK(cudaMalloc(&d_partition, sizeof(GPUPartition)));
        CUDA_CHECK(cudaMemcpy(d_partition, &h_part, sizeof(GPUPartition), cudaMemcpyHostToDevice));
    }

    void cleanup() {
        cudaFree(d_owned);
        cudaFree(d_bitmap);
        cudaFree(d_priority);
        cudaFree(d_owned_to_global);
        cudaFree(d_high_degree_map);
        cudaFree(d_global_to_owned);
        cudaFree(d_partition);
    }

    void read_state(VertexState* out, int idx) {
        CUDA_CHECK(cudaMemcpy(out, d_owned + idx, sizeof(VertexState), cudaMemcpyDeviceToHost));
    }
};

void test_add_no_conflict() {
    // Vertex 0: color=1, Vertex 1: color=2 -- no conflict
    TestPartitionSetup setup;
    setup.init(1, 2, 10, 5);

    uint32_t* d_recolor;
    CUDA_CHECK(cudaMalloc(&d_recolor, sizeof(uint32_t)));

    kernel_test_no_conflict<<<1, 1>>>(setup.d_partition, 0, 1, d_recolor);
    CUDA_CHECK(cudaDeviceSynchronize());

    uint32_t h_recolor;
    CUDA_CHECK(cudaMemcpy(&h_recolor, d_recolor, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    assert(h_recolor == 0);  // No recolor

    // Verify colors unchanged
    VertexState s0, s1;
    setup.read_state(&s0, 0);
    setup.read_state(&s1, 1);
    assert(s0.color == 1);
    assert(s1.color == 2);

    cudaFree(d_recolor);
    setup.cleanup();
}

void test_add_with_conflict_lower_priority_recolors() {
    // Vertex 0: color=3, degree=10 -> higher priority
    // Vertex 1: color=3, degree=5  -> lower priority, should recolor
    TestPartitionSetup setup;
    setup.init(3, 3, 10, 5);

    uint32_t* d_recolor;
    CUDA_CHECK(cudaMalloc(&d_recolor, sizeof(uint32_t)));

    kernel_test_no_conflict<<<1, 1>>>(setup.d_partition, 0, 1, d_recolor);
    CUDA_CHECK(cudaDeviceSynchronize());

    uint32_t h_recolor;
    CUDA_CHECK(cudaMemcpy(&h_recolor, d_recolor, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    assert(h_recolor == 1);  // Conflict detected

    // Vertex 0 (higher priority) keeps color 3
    VertexState s0;
    setup.read_state(&s0, 0);
    assert(s0.color == 3);
    assert(s0.version == 0);  // No recolor

    // Vertex 1 (lower priority) should have recolored
    VertexState s1;
    setup.read_state(&s1, 1);
    assert(s1.color != 3);     // Must not be 3 anymore
    assert(s1.color != 0);     // Must be a valid color
    assert(s1.version == 1);   // Version incremented

    cudaFree(d_recolor);
    setup.cleanup();
}

void test_add_with_conflict_same_degree_higher_id_wins() {
    // Same degree, vertex 1 has higher id -> vertex 1 wins, vertex 0 recolors
    TestPartitionSetup setup;
    setup.init(5, 5, 10, 10);  // same degree

    uint32_t* d_recolor;
    CUDA_CHECK(cudaMalloc(&d_recolor, sizeof(uint32_t)));

    kernel_test_no_conflict<<<1, 1>>>(setup.d_partition, 0, 1, d_recolor);
    CUDA_CHECK(cudaDeviceSynchronize());

    uint32_t h_recolor;
    CUDA_CHECK(cudaMemcpy(&h_recolor, d_recolor, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    assert(h_recolor == 1);

    // Vertex 1 (higher id, same degree = higher priority) keeps color
    VertexState s1;
    setup.read_state(&s1, 1);
    assert(s1.color == 5);
    assert(s1.version == 0);

    // Vertex 0 (lower id = lower priority) recolored
    VertexState s0;
    setup.read_state(&s0, 0);
    assert(s0.color != 5);
    assert(s0.version == 1);

    cudaFree(d_recolor);
    setup.cleanup();
}

void test_uncolored_no_conflict() {
    // One vertex uncolored -- no conflict even if technically "same color" (both 0)
    TestPartitionSetup setup;
    setup.init(COLOR_UNCOLORED, COLOR_UNCOLORED, 10, 5);

    uint32_t* d_recolor;
    CUDA_CHECK(cudaMalloc(&d_recolor, sizeof(uint32_t)));

    kernel_test_no_conflict<<<1, 1>>>(setup.d_partition, 0, 1, d_recolor);
    CUDA_CHECK(cudaDeviceSynchronize());

    uint32_t h_recolor;
    CUDA_CHECK(cudaMemcpy(&h_recolor, d_recolor, sizeof(uint32_t), cudaMemcpyDeviceToHost));
    assert(h_recolor == 0);  // Uncolored is not a conflict

    cudaFree(d_recolor);
    setup.cleanup();
}

int main() {
    printf("=== Local Update Unit Tests ===\n");

    TEST(add_no_conflict);
    TEST(add_with_conflict_lower_priority_recolors);
    TEST(add_with_conflict_same_degree_higher_id_wins);
    TEST(uncolored_no_conflict);

    printf("\n=== Results: %d passed, %d failed ===\n", passed, failed);
    return failed > 0 ? 1 : 0;
}
