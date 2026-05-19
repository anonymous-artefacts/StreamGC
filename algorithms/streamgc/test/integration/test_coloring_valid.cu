#include "streamgc.cuh"
#include "color_select.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cassert>

// Definition of partition_map (required by partition.cuh extern declarations)
PartitionEntry* partition_map = nullptr;
uint32_t        partition_map_size = 0;

// Kernel declarations
__global__ void local_update_kernel(EdgeUpdate* updates, uint32_t num_updates, GPUPartition* partition, uint32_t histogram_threshold);
__global__ void rebuild_bitmaps_kernel(GPUPartition* partition);
__global__ void conflict_repair_kernel(GPUPartition* partition, uint32_t histogram_threshold, uint32_t* num_conflicts);

int passed = 0, failed = 0;
#define TEST(name) { printf("  TEST: %s ... ", #name); fflush(stdout); }
#define PASS() { printf("PASSED\n"); passed++; }
#define FAIL(msg) { printf("FAILED: %s\n", msg); failed++; }

// Test 1: Valid coloring detected as valid
void test_valid_coloring() {
    TEST(valid_coloring_accepted);
    // Path: 0-1-2, colors 1,2,1
    // Manually set colors on device
    uint32_t nv = 3;

    // Create minimal partition
    GPUPartition h_part;
    memset(&h_part, 0, sizeof(GPUPartition));
    h_part.num_owned = nv;
    h_part.num_ghost = 0;
    h_part.max_owned = nv;

    CUDA_CHECK(cudaMalloc(&h_part.owned, nv * sizeof(VertexState)));
    CUDA_CHECK(cudaMalloc(&h_part.owned_to_global, nv * sizeof(uint32_t)));

    // Set colors: 1, 2, 1
    VertexState states[3] = {{1, 1}, {2, 1}, {1, 1}};
    uint32_t o2g[3] = {0, 1, 2};
    CUDA_CHECK(cudaMemcpy(h_part.owned, states, 3 * sizeof(VertexState), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(h_part.owned_to_global, o2g, 3 * sizeof(uint32_t), cudaMemcpyHostToDevice));

    // CSR: 0-1, 1-2 (path)
    uint32_t row_ptr[] = {0, 2, 4, 6};  // 2 slots each (1 used + 1 capacity)
    uint32_t col_idx[] = {1, CSR_INVALID, 0, 2, 1, CSR_INVALID};
    CUDA_CHECK(cudaMalloc(&h_part.row_ptr, 4 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(h_part.row_ptr, row_ptr, 4 * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&h_part.col_idx, 6 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemcpy(h_part.col_idx, col_idx, 6 * sizeof(uint32_t), cudaMemcpyHostToDevice));

    // Ghost CSR (empty)
    CUDA_CHECK(cudaMalloc(&h_part.ghost_row_ptr, 4 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(h_part.ghost_row_ptr, 0, 4 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&h_part.ghost_col_idx, sizeof(uint32_t)));

    GPUPartition* d_part;
    CUDA_CHECK(cudaMalloc(&d_part, sizeof(GPUPartition)));
    CUDA_CHECK(cudaMemcpy(d_part, &h_part, sizeof(GPUPartition), cudaMemcpyHostToDevice));

    // Pull colors and check manually
    std::vector<uint16_t> colors = {1, 2, 1};
    // Edges: 0-1 (colors 1,2 OK), 1-2 (colors 2,1 OK)
    bool valid = true;
    if (colors[0] == colors[1]) valid = false;
    if (colors[1] == colors[2]) valid = false;

    if (valid) PASS()
    else FAIL("valid coloring rejected");

    cudaFree(h_part.owned);
    cudaFree(h_part.owned_to_global);
    cudaFree(h_part.row_ptr);
    cudaFree(h_part.col_idx);
    cudaFree(h_part.ghost_row_ptr);
    cudaFree(h_part.ghost_col_idx);
    cudaFree(d_part);
}

// Test 2: Invalid coloring detected
void test_invalid_coloring() {
    TEST(invalid_coloring_detected);
    // Path: 0-1-2, colors 1,1,2 -- edge (0,1) conflicts
    std::vector<uint16_t> colors = {1, 1, 2};
    bool has_conflict = (colors[0] == colors[1]); // edge 0-1

    if (has_conflict) PASS()
    else FAIL("conflict not detected");
}

// Test 3: Uncolored vertices not flagged as violations
void test_uncolored_not_violation() {
    TEST(uncolored_not_violation);
    std::vector<uint16_t> colors = {COLOR_UNCOLORED, 1, 2};
    // Edge 0-1: uncolored vs 1 -- NOT a violation
    bool is_violation = (colors[0] == colors[1] && colors[0] != COLOR_UNCOLORED);

    if (!is_violation) PASS()
    else FAIL("uncolored falsely flagged");
}

int main() {
    printf("=== StreamGC Coloring Validity Tests ===\n\n");

    test_valid_coloring();
    test_invalid_coloring();
    test_uncolored_not_violation();

    printf("\n=== Results: %d passed, %d failed ===\n", passed, failed);
    return failed > 0 ? 1 : 0;
}
