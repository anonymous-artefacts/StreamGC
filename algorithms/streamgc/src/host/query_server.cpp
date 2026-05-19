#include <cstdio>
#include <cstdint>
#include <vector>
#include <chrono>

// Query result for point query
struct PointQueryResult {
    uint32_t vertex_id;
    uint16_t color;
    double   latency_us;
    uint64_t snapshot_age_updates;  // updates since last epoch boundary
};

// Query result for neighborhood query
struct NeighborhoodQueryResult {
    uint32_t vertex_id;
    uint64_t color_bitmap;   // bitmap mode result
    double   latency_us;
    uint64_t snapshot_age_updates;
};

// Point query: O(1) lookup from epoch_snapshot
uint16_t point_query(
    const uint16_t* epoch_snapshot,
    uint32_t vertex_id
) {
    return epoch_snapshot[vertex_id];
}

// Neighborhood query from bitmap
uint64_t neighborhood_query_bitmap(
    const uint64_t* neighbor_bitmap,
    uint32_t local_owned_idx
) {
    return neighbor_bitmap[local_owned_idx];
}

// Write query results CSV
void write_query_csv(
    const char* filename,
    const std::vector<PointQueryResult>& point_results,
    const std::vector<NeighborhoodQueryResult>& neighborhood_results
) {
    FILE* f = fopen(filename, "w");
    if (!f) {
        fprintf(stderr, "ERROR: Cannot open query CSV: %s\n", filename);
        return;
    }

    fprintf(f, "query_id,vertex_id,query_type,color_returned,query_latency_us,snapshot_age_updates\n");

    uint32_t qid = 1;
    for (const auto& r : point_results) {
        fprintf(f, "%u,%u,point,%u,%.2f,%lu\n",
                qid++, r.vertex_id, r.color, r.latency_us, r.snapshot_age_updates);
    }
    for (const auto& r : neighborhood_results) {
        fprintf(f, "%u,%u,neighborhood,0x%lx,%.2f,%lu\n",
                qid++, r.vertex_id, r.color_bitmap, r.latency_us, r.snapshot_age_updates);
    }

    fclose(f);
    printf("[StreamGC] Query results written: %s (%u queries)\n", filename, qid - 1);
}
