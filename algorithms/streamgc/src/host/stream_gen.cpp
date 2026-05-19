#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <random>
#include <unordered_set>
#include <unordered_map>

// Forward declarations
struct CSRGraph;
struct EdgeUpdate;
enum class UpdateType : uint8_t;

// Stream configuration
struct StreamConfig {
    uint64_t stream_size;    // total number of update events
    uint32_t insert_pct;     // percentage of events that are insertions (0-100)
    uint64_t random_seed;    // for reproducibility
};

struct PairHash {
    size_t operator()(const std::pair<uint32_t, uint32_t>& p) const {
        return std::hash<uint64_t>()((uint64_t)p.first << 32 | p.second);
    }
};

// Generate Phase 1 init stream: all base graph edges in degree-descending order
// Order: by min(deg(u), deg(v)) descending — hubs colored first
std::vector<std::pair<uint32_t, uint32_t>> generate_init_stream(
    const uint32_t* row_ptr,
    const uint32_t* col_idx,
    const uint32_t* degree,
    uint32_t num_vertices
) {
    // Collect undirected edges (u < v only)
    std::vector<std::pair<uint32_t, uint32_t>> edges;
    for (uint32_t u = 0; u < num_vertices; u++) {
        for (uint32_t j = row_ptr[u]; j < row_ptr[u + 1]; j++) {
            uint32_t v = col_idx[j];
            if (u < v) {
                edges.push_back({u, v});
            }
        }
    }

    // Sort by min(deg(u), deg(v)) descending, then max(deg(u), deg(v)) descending
    std::sort(edges.begin(), edges.end(),
        [&degree](const auto& a, const auto& b) {
            uint32_t min_a = std::min(degree[a.first], degree[a.second]);
            uint32_t min_b = std::min(degree[b.first], degree[b.second]);
            if (min_a != min_b) return min_a > min_b;
            uint32_t max_a = std::max(degree[a.first], degree[a.second]);
            uint32_t max_b = std::max(degree[b.first], degree[b.second]);
            return max_a > max_b;
        }
    );

    printf("[StreamGC] Init stream: %zu edges (degree-descending order)\n", edges.size());
    return edges;
}

// Generate Phase 2 benchmark stream: mixed insert/delete perturbation
// Maintains live edge set and deleted pool for re-insertion
std::vector<std::pair<uint32_t, uint32_t>> generate_bench_stream_edges(
    const uint32_t* row_ptr,
    const uint32_t* col_idx,
    const uint32_t* degree,
    uint32_t num_vertices,
    const StreamConfig& config,
    std::vector<uint8_t>& out_types  // 0 = ADD, 1 = DEL
) {
    std::mt19937_64 rng(config.random_seed);

    // Initialize live edge set from base graph
    std::unordered_set<std::pair<uint32_t, uint32_t>, PairHash> live_edges;
    std::vector<std::pair<uint32_t, uint32_t>> live_edge_vec;

    for (uint32_t u = 0; u < num_vertices; u++) {
        for (uint32_t j = row_ptr[u]; j < row_ptr[u + 1]; j++) {
            uint32_t v = col_idx[j];
            if (u < v) {
                live_edges.insert({u, v});
                live_edge_vec.push_back({u, v});
            }
        }
    }

    // Deleted pool for re-insertion
    std::vector<std::pair<uint32_t, uint32_t>> deleted_pool;

    // Degree distribution for preferential attachment
    std::vector<uint32_t> cum_degree(num_vertices);
    uint64_t total_degree = 0;
    for (uint32_t v = 0; v < num_vertices; v++) {
        total_degree += degree[v];
        cum_degree[v] = static_cast<uint32_t>(std::min(total_degree, (uint64_t)UINT32_MAX));
    }

    std::vector<std::pair<uint32_t, uint32_t>> stream;
    out_types.clear();
    stream.reserve(config.stream_size);
    out_types.reserve(config.stream_size);

    std::uniform_int_distribution<uint32_t> pct_dist(0, 99);

    for (uint64_t i = 0; i < config.stream_size; i++) {
        bool do_insert = (pct_dist(rng) < config.insert_pct);

        if (do_insert) {
            // INSERT
            if (!deleted_pool.empty()) {
                // Re-insert from deleted pool
                std::uniform_int_distribution<size_t> pool_dist(0, deleted_pool.size() - 1);
                size_t idx = pool_dist(rng);
                auto edge = deleted_pool[idx];

                // Remove from pool (swap with last)
                deleted_pool[idx] = deleted_pool.back();
                deleted_pool.pop_back();

                live_edges.insert(edge);
                live_edge_vec.push_back(edge);
                stream.push_back(edge);
                out_types.push_back(0);  // ADD
            } else {
                // Generate synthetic edge via preferential attachment
                // Sample two vertices weighted by degree
                std::uniform_int_distribution<uint32_t> vert_dist(0, num_vertices - 1);
                uint32_t u = vert_dist(rng);
                uint32_t v = vert_dist(rng);
                if (u == v) { v = (v + 1) % num_vertices; }
                uint32_t a = std::min(u, v);
                uint32_t b = std::max(u, v);

                if (live_edges.find({a, b}) == live_edges.end()) {
                    live_edges.insert({a, b});
                    live_edge_vec.push_back({a, b});
                    stream.push_back({a, b});
                    out_types.push_back(0);  // ADD
                } else {
                    // Edge already exists, try again (count this attempt)
                    i--;
                    continue;
                }
            }
        } else {
            // DELETE
            if (live_edge_vec.empty()) {
                // No edges to delete, force insert instead
                i--;
                continue;
            }

            std::uniform_int_distribution<size_t> edge_dist(0, live_edge_vec.size() - 1);
            size_t idx = edge_dist(rng);
            auto edge = live_edge_vec[idx];

            // Remove from live set
            live_edges.erase(edge);
            // Swap-remove from vector
            live_edge_vec[idx] = live_edge_vec.back();
            live_edge_vec.pop_back();

            deleted_pool.push_back(edge);
            stream.push_back(edge);
            out_types.push_back(1);  // DEL
        }
    }

    printf("[StreamGC] Bench stream: %zu events (%u%% insert, seed=%lu)\n",
           stream.size(), config.insert_pct, config.random_seed);
    printf("[StreamGC]   Live edges after stream: %zu, Deleted pool: %zu\n",
           live_edges.size(), deleted_pool.size());

    return stream;
}
