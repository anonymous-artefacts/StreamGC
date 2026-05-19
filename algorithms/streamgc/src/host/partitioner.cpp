#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#include <algorithm>
#include <numeric>

// Forward declarations (defined in graph_loader.cpp)
struct CSRGraph;

// Partition assignment result
struct PartitionAssignment {
    uint32_t  num_vertices;
    uint32_t  num_gpus;
    uint8_t*  owner;          // [num_vertices] -> owning GPU id
    uint32_t* vertex_count;   // [num_gpus] -> vertices per partition
    uint64_t* edge_count;     // [num_gpus] -> edges per partition
    std::vector<std::vector<uint32_t>> owned_vertices;  // per-GPU list of owned vertex ids
    std::vector<std::vector<uint32_t>> ghost_vertices;  // per-GPU list of ghost vertex ids
};

// Degree-aware vertex-cut partitioning
// Assigns each vertex to the GPU that minimizes:
//   sum_neighbors_already_assigned_to_gpu + lambda * vertex_count(gpu)
// lambda balances edge clustering vs vertex balance
PartitionAssignment degree_aware_vertex_cut(
    const uint32_t* row_ptr,
    const uint32_t* col_idx,
    const uint32_t* degree,
    uint32_t num_vertices,
    uint64_t num_edges,
    uint32_t num_gpus,
    float lambda
) {
    PartitionAssignment assign;
    assign.num_vertices = num_vertices;
    assign.num_gpus = num_gpus;
    assign.owner = (uint8_t*)calloc(num_vertices, sizeof(uint8_t));
    assign.vertex_count = (uint32_t*)calloc(num_gpus, sizeof(uint32_t));
    assign.edge_count = (uint64_t*)calloc(num_gpus, sizeof(uint64_t));
    assign.owned_vertices.resize(num_gpus);
    assign.ghost_vertices.resize(num_gpus);

    if (num_gpus == 1) {
        // Single GPU: all vertices owned by GPU 0
        for (uint32_t v = 0; v < num_vertices; v++) {
            assign.owner[v] = 0;
            assign.owned_vertices[0].push_back(v);
        }
        assign.vertex_count[0] = num_vertices;
        assign.edge_count[0] = num_edges;
        printf("[StreamGC] Single-GPU partition: %u vertices, %lu edges\n",
               num_vertices, num_edges);
        return assign;
    }

    // Sort vertices by degree (descending) for greedy assignment
    std::vector<uint32_t> sorted_vertices(num_vertices);
    std::iota(sorted_vertices.begin(), sorted_vertices.end(), 0);
    std::sort(sorted_vertices.begin(), sorted_vertices.end(),
              [&degree](uint32_t a, uint32_t b) { return degree[a] > degree[b]; });

    // Track how many neighbors each vertex has in each partition
    // Use a simple scoring approach
    std::vector<std::vector<uint32_t>> neighbor_count(num_vertices, std::vector<uint32_t>(num_gpus, 0));

    for (uint32_t i = 0; i < num_vertices; i++) {
        uint32_t v = sorted_vertices[i];

        // Score each GPU
        float best_score = 1e18f;
        uint8_t best_gpu = 0;

        for (uint32_t g = 0; g < num_gpus; g++) {
            // Count neighbors already assigned to this GPU
            float affinity = static_cast<float>(neighbor_count[v][g]);
            float balance_penalty = lambda * static_cast<float>(assign.vertex_count[g]);
            float score = -affinity + balance_penalty;  // minimize: fewer neighbors is worse, more vertices is worse

            if (score < best_score) {
                best_score = score;
                best_gpu = static_cast<uint8_t>(g);
            }
        }

        assign.owner[v] = best_gpu;
        assign.owned_vertices[best_gpu].push_back(v);
        assign.vertex_count[best_gpu]++;

        // Update neighbor counts for v's neighbors
        for (uint32_t j = row_ptr[v]; j < row_ptr[v + 1]; j++) {
            uint32_t w = col_idx[j];
            neighbor_count[w][best_gpu]++;
        }
    }

    // Count edges per partition and identify ghost vertices
    for (uint32_t g = 0; g < num_gpus; g++) {
        std::vector<bool> is_ghost(num_vertices, false);
        for (uint32_t v : assign.owned_vertices[g]) {
            for (uint32_t j = row_ptr[v]; j < row_ptr[v + 1]; j++) {
                uint32_t w = col_idx[j];
                if (assign.owner[w] == g) {
                    assign.edge_count[g]++;
                } else {
                    assign.edge_count[g]++;  // boundary edge counted on both sides
                    if (!is_ghost[w]) {
                        is_ghost[w] = true;
                        assign.ghost_vertices[g].push_back(w);
                    }
                }
            }
        }
    }

    printf("[StreamGC] Degree-aware vertex-cut partitioning (lambda=%.2f):\n", lambda);
    for (uint32_t g = 0; g < num_gpus; g++) {
        printf("[StreamGC]   GPU %u: %u vertices, %lu edges, %zu ghosts\n",
               g, assign.vertex_count[g], assign.edge_count[g],
               assign.ghost_vertices[g].size());
    }

    return assign;
}

void free_partition_assignment(PartitionAssignment& assign) {
    free(assign.owner);
    free(assign.vertex_count);
    free(assign.edge_count);
    assign.owner = nullptr;
    assign.vertex_count = nullptr;
    assign.edge_count = nullptr;
}
