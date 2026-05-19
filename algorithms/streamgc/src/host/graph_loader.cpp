#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <vector>
#include <algorithm>
#include <unordered_set>
#include <string>
#ifdef _OPENMP
#include <omp.h>
#include <parallel/algorithm>
#endif

// CSR graph representation
struct CSRGraph {
    uint32_t  num_vertices;
    uint64_t  num_edges;       // number of directed edges (undirected * 2)
    uint32_t* row_ptr;         // [num_vertices + 1]
    uint32_t* col_idx;         // [num_edges]
    uint32_t* degree;          // [num_vertices]
};

struct PairHash {
    size_t operator()(const std::pair<uint32_t, uint32_t>& p) const {
        return std::hash<uint64_t>()((uint64_t)p.first << 32 | p.second);
    }
};

CSRGraph load_graph_csr(const char* filename) {
    FILE* f = fopen(filename, "r");
    if (!f) {
        fprintf(stderr, "ERROR: Cannot open graph file: %s\n", filename);
        exit(EXIT_FAILURE);
    }

    printf("[StreamGC] Loading graph: %s\n", filename);
    fflush(stdout);

    char line[4096];
    bool is_pattern = false;

    // Read banner line
    if (fgets(line, sizeof(line), f)) {
        if (strstr(line, "pattern")) is_pattern = true;
    }

    // Skip comment lines (start with %)
    while (fgets(line, sizeof(line), f)) {
        if (line[0] != '%') break;
    }

    // Parse header: rows cols nnz
    uint32_t rows = 0, cols = 0;
    uint64_t nnz = 0;
    {
        // Use %llu for uint64_t portability
        unsigned long long nnz_tmp = 0;
        if (sscanf(line, "%u %u %llu", &rows, &cols, &nnz_tmp) != 3) {
            fprintf(stderr, "ERROR: Failed to parse MTX header: '%s'\n", line);
            fclose(f);
            exit(EXIT_FAILURE);
        }
        nnz = nnz_tmp;
    }

    uint32_t num_vertices = (rows > cols) ? rows : cols;
    printf("[StreamGC]   Header: %u x %u, nnz=%lu, pattern=%d\n",
           rows, cols, (unsigned long)nnz, is_pattern);
    fflush(stdout);

    // Read edges into vectors (faster than unordered_set for moderate sizes)
    std::vector<std::pair<uint32_t, uint32_t>> raw_edges;
    raw_edges.reserve(nnz);

    for (uint64_t i = 0; i < nnz; i++) {
        uint32_t u = 0, v = 0;
        if (is_pattern) {
            if (fscanf(f, "%u %u", &u, &v) != 2) {
                fprintf(stderr, "WARNING: Failed to read edge %lu\n", (unsigned long)i);
                break;
            }
        } else {
            // Read u, v, then skip any weight value on the rest of the line
            if (fscanf(f, "%u %u", &u, &v) != 2) {
                fprintf(stderr, "WARNING: Failed to read edge %lu\n", (unsigned long)i);
                break;
            }
            // Skip rest of line (weight)
            int c;
            while ((c = fgetc(f)) != '\n' && c != EOF);
        }

        // MTX is 1-indexed -> 0-indexed
        u--; v--;

        // Skip self-loops
        if (u == v) continue;

        // Normalize: always store (min, max)
        if (u > v) { uint32_t tmp = u; u = v; v = tmp; }
        raw_edges.push_back({u, v});
    }
    fclose(f);

    printf("[StreamGC]   Read %zu raw edges\n", raw_edges.size());
    fflush(stdout);

    // Sort and deduplicate
#ifdef _OPENMP
    __gnu_parallel::sort(raw_edges.begin(), raw_edges.end());
#else
    std::sort(raw_edges.begin(), raw_edges.end());
#endif
    raw_edges.erase(std::unique(raw_edges.begin(), raw_edges.end()), raw_edges.end());

    printf("[StreamGC]   After dedup: %zu undirected edges\n", raw_edges.size());
    fflush(stdout);

    // Build symmetric edge list
    uint64_t num_directed_edges = raw_edges.size() * 2;

    // Compute degree
    uint32_t* degree = (uint32_t*)calloc(num_vertices, sizeof(uint32_t));
    for (auto& [a, b] : raw_edges) {
        degree[a]++;
        degree[b]++;
    }
    printf("[StreamGC]   Computed degree\n"); fflush(stdout);

    // Build CSR
    uint32_t* row_ptr = (uint32_t*)malloc((num_vertices + 1) * sizeof(uint32_t));
    row_ptr[0] = 0;
    for (uint32_t v = 0; v < num_vertices; v++) {
        row_ptr[v + 1] = row_ptr[v] + degree[v];
    }

    uint32_t* col_idx = (uint32_t*)malloc(num_directed_edges * sizeof(uint32_t));
    uint32_t* offsets = (uint32_t*)calloc(num_vertices, sizeof(uint32_t));

    for (auto& [a, b] : raw_edges) {
        col_idx[row_ptr[a] + offsets[a]++] = b;
        col_idx[row_ptr[b] + offsets[b]++] = a;
    }
    free(offsets);
    printf("[StreamGC]   Scattered edges into CSR\n"); fflush(stdout);

    // Sort adjacency lists (parallel over rows; per-row sorts remain serial
    // but rows are independent so OpenMP parallelizes cleanly).
#ifdef _OPENMP
    #pragma omp parallel for schedule(dynamic, 1024)
#endif
    for (uint32_t v = 0; v < num_vertices; v++) {
        std::sort(col_idx + row_ptr[v], col_idx + row_ptr[v + 1]);
    }
    printf("[StreamGC]   Sorted adjacency lists\n"); fflush(stdout);

    CSRGraph graph;
    graph.num_vertices = num_vertices;
    graph.num_edges = num_directed_edges;
    graph.row_ptr = row_ptr;
    graph.col_idx = col_idx;
    graph.degree = degree;

    printf("[StreamGC] Loaded: %u vertices, %lu edges (directed), %zu undirected\n",
           num_vertices, (unsigned long)num_directed_edges, raw_edges.size());
    fflush(stdout);

    return graph;
}

void free_csr_graph(CSRGraph& graph) {
    free(graph.row_ptr);
    free(graph.col_idx);
    free(graph.degree);
    graph.row_ptr = nullptr;
    graph.col_idx = nullptr;
    graph.degree = nullptr;
}
