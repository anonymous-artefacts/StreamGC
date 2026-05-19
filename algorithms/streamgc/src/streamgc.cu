#include "streamgc.cuh"
#include "color_select.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <vector>
#include <algorithm>
#include <numeric>
#include <thread>
#include <chrono>
#include <string>
#include <unordered_set>
#include <unordered_map>
#include <random>
#include <sys/stat.h>

// ---- Definition of partition_map (declared extern in partition.cuh) ----
PartitionEntry* partition_map = nullptr;
uint32_t        partition_map_size = 0;

// ---- CSR Graph (from graph_loader.cpp) ----
struct CSRGraph {
    uint32_t  num_vertices;
    uint64_t  num_edges;
    uint32_t* row_ptr;
    uint32_t* col_idx;
    uint32_t* degree;
};
extern CSRGraph load_graph_csr(const char* filename);
extern void free_csr_graph(CSRGraph& graph);

// ---- Partition Assignment (from partitioner.cpp) ----
struct PartitionAssignment {
    uint32_t  num_vertices;
    uint32_t  num_gpus;
    uint8_t*  owner;
    uint32_t* vertex_count;
    uint64_t* edge_count;
    std::vector<std::vector<uint32_t>> owned_vertices;
    std::vector<std::vector<uint32_t>> ghost_vertices;
};
extern PartitionAssignment degree_aware_vertex_cut(
    const uint32_t* row_ptr, const uint32_t* col_idx, const uint32_t* degree,
    uint32_t num_vertices, uint64_t num_edges, uint32_t num_gpus, float lambda);
extern void free_partition_assignment(PartitionAssignment& assign);

// ---- Stream Generation (from stream_gen.cpp) ----
struct StreamConfig {
    uint64_t stream_size;
    uint32_t insert_pct;
    uint64_t random_seed;
};
extern std::vector<std::pair<uint32_t, uint32_t>> generate_init_stream(
    const uint32_t* row_ptr, const uint32_t* col_idx, const uint32_t* degree, uint32_t num_vertices);
extern std::vector<std::pair<uint32_t, uint32_t>> generate_bench_stream_edges(
    const uint32_t* row_ptr, const uint32_t* col_idx, const uint32_t* degree,
    uint32_t num_vertices, const StreamConfig& config, std::vector<uint8_t>& out_types);

// ---- Feeder (from feeder_thread.cpp) ----
struct FeederState {
    EdgeUpdate*     stream_buffer;
    uint32_t*       stream_tail;
    uint32_t*       stream_head;
    uint32_t        stream_mask;
    volatile bool*  shutdown_flag;
    double init_time_ms;
    double bench_time_ms;
    bool init_complete;
    bool bench_complete;
};
extern void feeder_thread_func(
    FeederState& state,
    const std::vector<std::pair<uint32_t, uint32_t>>& init_edges,
    const std::vector<std::pair<uint32_t, uint32_t>>& bench_edges,
    const std::vector<uint8_t>& bench_types,
    uint32_t buffer_size);

// ---- Kernel declarations ----
__global__ void local_update_kernel(EdgeUpdate* updates, uint32_t num_updates, GPUPartition* partition, uint32_t histogram_threshold);
__global__ void boundary_update_kernel(EdgeUpdate* updates, uint32_t num_updates, GPUPartition* local_partition, GPUPartition** all_partitions, uint32_t this_gpu_id, uint32_t histogram_threshold);
__global__ void epoch_snapshot_kernel(GPUPartition* partition, uint32_t num_vertices_global);
__global__ void compact_csr_kernel(GPUPartition* partition);
__global__ void drain_overflow_local_kernel(GPUPartition* partition, uint32_t overflow_count, uint32_t* num_failed);
__global__ void drain_overflow_ghost_kernel(GPUPartition* partition, uint32_t overflow_count, uint32_t* num_failed);
__global__ void rebuild_bitmaps_kernel(GPUPartition* partition);
__global__ void conflict_repair_kernel(GPUPartition* partition, uint32_t histogram_threshold, uint32_t* num_conflicts);
__global__ void ghost_sync_kernel(GPUPartition* local_partition, GPUPartition** all_partitions, uint32_t num_gpus);

// ---- Profiling accumulators for process_stream ----
struct StreamProfile {
    double total_ms          = 0;  // wall-clock for entire stream
    double classify_ms       = 0;  // edge classification (multi-GPU only)
    double local_kernel_ms   = 0;  // local_update_kernel time
    double boundary_kernel_ms= 0;  // boundary_update_kernel time (multi-GPU only)
    double gpu_sync_ms       = 0;  // cudaDeviceSynchronize after kernels
    double ghost_sync_ms     = 0;  // ghost_sync_kernel time (multi-GPU only)
    double epoch_ms          = 0;  // epoch maintenance time
    double memcpy_ms         = 0;  // host-to-device memcpy time
    double query_latency_us  = 0;  // avg point query latency (sampled)
    uint64_t query_count     = 0;  // number of queries sampled
    double query_total_us    = 0;  // total query time for averaging
};

// ---- NVLink Peer Access ----
void enable_peer_access(uint32_t num_gpus) {
    for (uint32_t i = 0; i < num_gpus; i++) {
        CUDA_CHECK(cudaSetDevice(i));
        for (uint32_t j = 0; j < num_gpus; j++) {
            if (i == j) continue;
            int can_access = 0;
            CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access, i, j));
            if (can_access) {
                cudaError_t err = cudaDeviceEnablePeerAccess(j, 0);
                if (err == cudaErrorPeerAccessAlreadyEnabled) {
                    cudaGetLastError();  // clear the error
                } else if (err != cudaSuccess) {
                    fprintf(stderr, "WARNING: Failed to enable peer access GPU %u -> %u: %s\n",
                            i, j, cudaGetErrorString(err));
                } else {
                    printf("[StreamGC] Peer access: GPU %u -> GPU %u: ENABLED\n", i, j);
                }
            } else {
                fprintf(stderr, "WARNING: GPU %u cannot peer-access GPU %u. NVLink not available.\n", i, j);
            }
        }
    }
}

// ---- Partition Map Allocation ----
void allocate_partition_map(uint32_t num_vertices, uint32_t num_gpus) {
    partition_map_size = num_vertices;
    size_t bytes = num_vertices * sizeof(PartitionEntry);

    CUDA_CHECK(cudaMallocManaged(&partition_map, bytes));
    CUDA_CHECK(cudaMemAdvise(partition_map, bytes, cudaMemAdviseSetReadMostly, cudaCpuDeviceId));
    for (uint32_t gpu = 0; gpu < num_gpus; gpu++) {
        CUDA_CHECK(cudaMemAdvise(partition_map, bytes, cudaMemAdviseSetAccessedBy, gpu));
    }
    memset(partition_map, 0, bytes);
}

// ---- GPU Partition Allocation ----
GPUPartition* allocate_gpu_partition(
    const PartitionAssignment& assign,
    const CSRGraph& graph,
    uint32_t gpu_id,
    const StreamGCConfig& config
) {
    CUDA_CHECK(cudaSetDevice(gpu_id));

    uint32_t num_owned = assign.vertex_count[gpu_id];
    uint32_t num_ghost = assign.ghost_vertices[gpu_id].size();

    // Estimate edges
    uint64_t local_edges = 0;
    uint64_t ghost_edges = 0;
    for (uint32_t v : assign.owned_vertices[gpu_id]) {
        for (uint32_t j = graph.row_ptr[v]; j < graph.row_ptr[v + 1]; j++) {
            if (assign.owner[graph.col_idx[j]] == gpu_id) local_edges++;
            else ghost_edges++;
        }
    }

    // Over-allocate for dynamic growth. EXTRA_SLACK_PCT matches the per-row
    // capacity policy below (§4.1), so the initial CSR slab is sized exactly
    // equal to what the per-row sum will consume — no realloc at build time.
    // Default 300 = 4x total (count + 300% extra). 
#ifndef STREAMGC_CSR_EXTRA_SLACK_PCT
#define STREAMGC_CSR_EXTRA_SLACK_PCT 300
#endif
    uint32_t max_owned = num_owned + num_owned / 4;  // 25% headroom for migration
    uint32_t max_ghost = num_ghost + num_ghost / 4;
    // Add 8*num_owned safety margin for the min-8-per-row floor in the per-row
    // capacity loop below; this keeps the initial slab >= the eventual per-row sum
    // and avoids a realloc that would briefly hold 2x the CSR in HBM.
    uint64_t slab_safety = static_cast<uint64_t>(num_owned) * 8;
    uint64_t max_local_edges = local_edges + (local_edges * STREAMGC_CSR_EXTRA_SLACK_PCT) / 100 + slab_safety;
    uint64_t max_ghost_edges = ghost_edges + (ghost_edges * STREAMGC_CSR_EXTRA_SLACK_PCT) / 100 + slab_safety;

    // Hash map sizes (power of 2, at least 2x entries)
    uint32_t owned_map_size = 1;
    while (owned_map_size < num_owned * 2) owned_map_size <<= 1;
    uint32_t ghost_map_size = 1;
    while (ghost_map_size < (num_ghost > 0 ? num_ghost * 2 : 4)) ghost_map_size <<= 1;

    // Allocate GPUPartition on host, then copy to device
    GPUPartition h_part;
    memset(&h_part, 0, sizeof(GPUPartition));
    h_part.num_owned = num_owned;
    h_part.num_ghost = num_ghost;
    h_part.num_edges_local = static_cast<uint32_t>(local_edges);
    h_part.max_owned = max_owned;
    h_part.max_ghost = max_ghost;
    h_part.gpu_id = gpu_id;
    h_part.d_partition_map = partition_map;  // unified memory pointer, device-accessible
    h_part.index_map_size = owned_map_size;
    h_part.ghost_map_size = ghost_map_size;

    // Owned state
    CUDA_CHECK(cudaMalloc(&h_part.owned, max_owned * sizeof(VertexState)));
    CUDA_CHECK(cudaMemset(h_part.owned, 0, max_owned * sizeof(VertexState)));
    CUDA_CHECK(cudaMalloc(&h_part.owned_to_global, max_owned * sizeof(uint32_t)));

    // Ghost state
    CUDA_CHECK(cudaMalloc(&h_part.ghost, max_ghost * sizeof(VertexState)));
    CUDA_CHECK(cudaMemset(h_part.ghost, 0, max_ghost * sizeof(VertexState)));
    CUDA_CHECK(cudaMalloc(&h_part.ghost_to_global, max_ghost * sizeof(uint32_t)));

    // Hash maps (key-value interleaved: 2 * map_size entries)
    CUDA_CHECK(cudaMalloc(&h_part.global_to_owned, owned_map_size * 2 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(h_part.global_to_owned, 0xFF, owned_map_size * 2 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&h_part.global_to_ghost, ghost_map_size * 2 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(h_part.global_to_ghost, 0xFF, ghost_map_size * 2 * sizeof(uint32_t)));

    // Bitmaps
    CUDA_CHECK(cudaMalloc(&h_part.neighbor_bitmap, max_owned * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemset(h_part.neighbor_bitmap, 0, max_owned * sizeof(uint64_t)));

    // High-degree map (all HASH_EMPTY initially)
    CUDA_CHECK(cudaMalloc(&h_part.high_degree_map, max_owned * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(h_part.high_degree_map, 0xFF, max_owned * sizeof(uint32_t)));
    h_part.num_high_degree = 0;

    // Identify high-degree vertices and allocate histogram
    std::vector<uint32_t> hd_vertices;
    for (uint32_t i = 0; i < num_owned; i++) {
        uint32_t global_id = assign.owned_vertices[gpu_id][i];
        if (graph.degree[global_id] > config.histogram_threshold) {
            hd_vertices.push_back(i);  // local index
        }
    }
    h_part.num_high_degree = hd_vertices.size();

    if (!hd_vertices.empty()) {
        CUDA_CHECK(cudaMalloc(&h_part.freq_histogram,
                              hd_vertices.size() * MAX_HISTOGRAM_COLORS * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemset(h_part.freq_histogram, 0,
                              hd_vertices.size() * MAX_HISTOGRAM_COLORS * sizeof(uint32_t)));

        // Upload high_degree_map entries
        std::vector<uint32_t> hd_map_host(max_owned, HASH_EMPTY);
        for (uint32_t idx = 0; idx < hd_vertices.size(); idx++) {
            hd_map_host[hd_vertices[idx]] = idx;
        }
        CUDA_CHECK(cudaMemcpy(h_part.high_degree_map, hd_map_host.data(),
                              max_owned * sizeof(uint32_t), cudaMemcpyHostToDevice));
    } else {
        h_part.freq_histogram = nullptr;
    }

    // Priority (unified memory so all GPUs can read)
    CUDA_CHECK(cudaMallocManaged(&h_part.init_priority, config.num_vertices * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemAdvise(h_part.init_priority, config.num_vertices * sizeof(uint64_t),
                             cudaMemAdviseSetReadMostly, cudaCpuDeviceId));

    // Epoch snapshot (unified memory for query serving)
    CUDA_CHECK(cudaMallocManaged(&h_part.epoch_snapshot, config.num_vertices * sizeof(uint16_t)));
    CUDA_CHECK(cudaMemset(h_part.epoch_snapshot, 0, config.num_vertices * sizeof(uint16_t)));

    // Local CSR
    CUDA_CHECK(cudaMalloc(&h_part.row_ptr, (max_owned + 1) * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(h_part.row_ptr, 0, (max_owned + 1) * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&h_part.col_idx, max_local_edges * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(h_part.col_idx, 0xFF, max_local_edges * sizeof(uint32_t)));  // INVALID
    CUDA_CHECK(cudaMalloc(&h_part.col_idx_capacity, max_owned * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(h_part.col_idx_capacity, 0, max_owned * sizeof(uint32_t)));

    // Ghost CSR
    CUDA_CHECK(cudaMalloc(&h_part.ghost_row_ptr, (max_owned + 1) * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(h_part.ghost_row_ptr, 0, (max_owned + 1) * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&h_part.ghost_col_idx, (max_ghost_edges > 0 ? max_ghost_edges : 1) * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(h_part.ghost_col_idx, 0xFF, (max_ghost_edges > 0 ? max_ghost_edges : 1) * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&h_part.ghost_col_idx_capacity, max_owned * sizeof(uint32_t)));
    CUDA_CHECK(cudaMemset(h_part.ghost_col_idx_capacity, 0, max_owned * sizeof(uint32_t)));

    // Overflow buffers for edges that don't fit in CSR
    CUDA_CHECK(cudaMalloc(&h_part.overflow_local, OVERFLOW_CAPACITY * sizeof(OverflowEdge)));
    CUDA_CHECK(cudaMalloc(&h_part.overflow_ghost, OVERFLOW_CAPACITY * sizeof(OverflowEdge)));
    CUDA_CHECK(cudaMallocManaged(&h_part.overflow_local_count, sizeof(uint32_t)));
    CUDA_CHECK(cudaMallocManaged(&h_part.overflow_ghost_count, sizeof(uint32_t)));
    *h_part.overflow_local_count = 0;
    *h_part.overflow_ghost_count = 0;

    // Ring buffer (pinned host memory for CPU->GPU transfer)
    uint32_t buf_size = config.stream_buffer_size;
    CUDA_CHECK(cudaHostAlloc(&h_part.stream_buffer, buf_size * sizeof(EdgeUpdate), cudaHostAllocMapped));
    CUDA_CHECK(cudaHostAlloc(&h_part.stream_head, sizeof(uint32_t), cudaHostAllocMapped));
    CUDA_CHECK(cudaHostAlloc(&h_part.stream_tail, sizeof(uint32_t), cudaHostAllocMapped));
    *h_part.stream_head = 0;
    *h_part.stream_tail = 0;
    h_part.stream_mask = buf_size - 1;

    // Counters
    CUDA_CHECK(cudaMallocManaged(&h_part.update_counter, sizeof(uint64_t)));
    *h_part.update_counter = 0;
    CUDA_CHECK(cudaMallocManaged(&h_part.epoch_ready_flag, sizeof(uint32_t)));
    *h_part.epoch_ready_flag = 0;
    CUDA_CHECK(cudaMallocManaged(&h_part.epoch_counter, sizeof(uint32_t)));
    *h_part.epoch_counter = 0;

    // Φ instrumentation
#ifdef STREAMGC_INSTRUMENT_PHI
    CUDA_CHECK(cudaMallocManaged(&h_part.phi_inflight, sizeof(int32_t)));
    CUDA_CHECK(cudaMallocManaged(&h_part.phi_max, sizeof(uint32_t)));
    *h_part.phi_inflight = 0;
    *h_part.phi_max = 0;
    CUDA_CHECK(cudaMalloc(&h_part.is_boundary, max_owned * sizeof(uint8_t)));
    CUDA_CHECK(cudaMemset(h_part.is_boundary, 0, max_owned * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&h_part.phi_dirty, max_owned * sizeof(uint8_t)));
    CUDA_CHECK(cudaMemset(h_part.phi_dirty, 0, max_owned * sizeof(uint8_t)));
#else
    h_part.phi_inflight = nullptr;
    h_part.phi_max = nullptr;
    h_part.is_boundary = nullptr;
    h_part.phi_dirty = nullptr;
#endif

    // ---- Populate owned_to_global and hash maps ----
    {
        std::vector<uint32_t> o2g(num_owned);
        // Build host hash map for owned
        std::vector<uint32_t> owned_hash(owned_map_size * 2, HASH_EMPTY);

        for (uint32_t i = 0; i < num_owned; i++) {
            uint32_t global_id = assign.owned_vertices[gpu_id][i];
            o2g[i] = global_id;
            hash_insert_host(owned_hash.data(), owned_map_size, global_id, i);
        }
        CUDA_CHECK(cudaMemcpy(h_part.owned_to_global, o2g.data(),
                              num_owned * sizeof(uint32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(h_part.global_to_owned, owned_hash.data(),
                              owned_map_size * 2 * sizeof(uint32_t), cudaMemcpyHostToDevice));

        // Build ghost hash map
        std::vector<uint32_t> g2g(num_ghost);
        std::vector<uint32_t> ghost_hash(ghost_map_size * 2, HASH_EMPTY);

        for (uint32_t i = 0; i < num_ghost; i++) {
            uint32_t global_id = assign.ghost_vertices[gpu_id][i];
            g2g[i] = global_id;
            hash_insert_host(ghost_hash.data(), ghost_map_size, global_id, i);
        }
        CUDA_CHECK(cudaMemcpy(h_part.ghost_to_global, g2g.data(),
                              num_ghost * sizeof(uint32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(h_part.global_to_ghost, ghost_hash.data(),
                              ghost_map_size * 2 * sizeof(uint32_t), cudaMemcpyHostToDevice));
    }

    // ---- Build initial CSR from base graph ----
    {
        std::vector<uint32_t> local_row_ptr(max_owned + 1, 0);
        std::vector<uint32_t> ghost_row_ptr_h(max_owned + 1, 0);
#ifdef STREAMGC_INSTRUMENT_PHI
        std::vector<uint8_t> is_boundary_h(max_owned, 0);
#endif

        // Count edges per owned vertex
        for (uint32_t i = 0; i < num_owned; i++) {
            uint32_t global_id = assign.owned_vertices[gpu_id][i];
            uint32_t local_count = 0, ghost_count = 0;
            for (uint32_t j = graph.row_ptr[global_id]; j < graph.row_ptr[global_id + 1]; j++) {
                uint32_t neighbor = graph.col_idx[j];
                if (assign.owner[neighbor] == gpu_id) local_count++;
                else ghost_count++;
            }
#ifdef STREAMGC_INSTRUMENT_PHI
            if (ghost_count > 0) is_boundary_h[i] = 1;
#endif
            // Per-row capacity uses the same EXTRA_SLACK_PCT as the partition-level
            // slab sizing above. Default 300 -> 4x per row (count + 3x extra).
            // Friendster build uses 50 -> 1.5x per row.
            uint32_t local_extra = (local_count * STREAMGC_CSR_EXTRA_SLACK_PCT) / 100;
            uint32_t ghost_extra = (ghost_count * STREAMGC_CSR_EXTRA_SLACK_PCT) / 100;
            uint32_t local_cap = std::max(local_count + local_extra, (uint32_t)8);
            uint32_t ghost_cap = std::max(ghost_count + ghost_extra, (uint32_t)8);
            local_row_ptr[i + 1] = local_row_ptr[i] + local_cap;
            ghost_row_ptr_h[i + 1] = ghost_row_ptr_h[i] + ghost_cap;
        }
#ifdef STREAMGC_INSTRUMENT_PHI
        CUDA_CHECK(cudaMemcpy(h_part.is_boundary, is_boundary_h.data(),
                              max_owned * sizeof(uint8_t), cudaMemcpyHostToDevice));
#endif

        // Realloc col_idx arrays if needed
        uint64_t total_local_slots = local_row_ptr[num_owned];
        uint64_t total_ghost_slots = ghost_row_ptr_h[num_owned];

        if (total_local_slots > max_local_edges) {
            cudaFree(h_part.col_idx);
            max_local_edges = total_local_slots;
            CUDA_CHECK(cudaMalloc(&h_part.col_idx, max_local_edges * sizeof(uint32_t)));
        }
        if (total_ghost_slots > max_ghost_edges && total_ghost_slots > 0) {
            cudaFree(h_part.ghost_col_idx);
            max_ghost_edges = total_ghost_slots;
            CUDA_CHECK(cudaMalloc(&h_part.ghost_col_idx, max_ghost_edges * sizeof(uint32_t)));
        }

        // Pre-populate CSR with base graph edges (remaining slots = INVALID)
        std::vector<uint32_t> local_col(total_local_slots, CSR_INVALID);
        std::vector<uint32_t> ghost_col(total_ghost_slots > 0 ? total_ghost_slots : 1, CSR_INVALID);
        std::vector<uint32_t> cap_h(max_owned, 0);

        for (uint32_t i = 0; i < num_owned; i++) {
            uint32_t global_id = assign.owned_vertices[gpu_id][i];
            uint32_t lpos = local_row_ptr[i];
            uint32_t gpos = ghost_row_ptr_h[i];

            for (uint32_t j = graph.row_ptr[global_id]; j < graph.row_ptr[global_id + 1]; j++) {
                uint32_t neighbor = graph.col_idx[j];
                if (assign.owner[neighbor] == gpu_id) {
                    local_col[lpos++] = neighbor;
                } else {
                    ghost_col[gpos++] = neighbor;
                }
            }
            cap_h[i] = (local_row_ptr[i + 1] - local_row_ptr[i]);
        }

        CUDA_CHECK(cudaMemcpy(h_part.row_ptr, local_row_ptr.data(),
                              (num_owned + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(h_part.col_idx, local_col.data(),
                              total_local_slots * sizeof(uint32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(h_part.col_idx_capacity, cap_h.data(),
                              num_owned * sizeof(uint32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(h_part.ghost_row_ptr, ghost_row_ptr_h.data(),
                              (num_owned + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice));
        if (total_ghost_slots > 0) {
            CUDA_CHECK(cudaMemcpy(h_part.ghost_col_idx, ghost_col.data(),
                                  total_ghost_slots * sizeof(uint32_t), cudaMemcpyHostToDevice));
        }
    }

    // Copy partition struct to device
    GPUPartition* d_part;
    CUDA_CHECK(cudaMalloc(&d_part, sizeof(GPUPartition)));
    CUDA_CHECK(cudaMemcpy(d_part, &h_part, sizeof(GPUPartition), cudaMemcpyHostToDevice));

    printf("[StreamGC] GPU %u partition allocated: %u owned, %u ghost, %u local edges, %u high-degree\n",
           gpu_id, num_owned, num_ghost, h_part.num_edges_local, h_part.num_high_degree);

    return d_part;
}

// ---- Validity Checker ----
// Validates coloring against the GPU's ACTUAL CSR (not the stale base graph)
bool check_coloring_valid(
    GPUPartition** d_partitions,
    uint32_t num_gpus,
    uint32_t num_vertices
) {
    // Pull all colors and CSR from GPU partitions
    std::vector<uint16_t> color(num_vertices, COLOR_UNCOLORED);

    // Also build an edge set from the GPU's actual CSR
    std::unordered_set<uint64_t> edge_set;
    std::vector<uint32_t> effective_degree(num_vertices, 0);

    for (uint32_t gpu = 0; gpu < num_gpus; gpu++) {
        GPUPartition h_part;
        CUDA_CHECK(cudaSetDevice(gpu));
        CUDA_CHECK(cudaMemcpy(&h_part, d_partitions[gpu], sizeof(GPUPartition), cudaMemcpyDeviceToHost));

        std::vector<VertexState> owned_states(h_part.num_owned);
        std::vector<uint32_t> o2g(h_part.num_owned);
        CUDA_CHECK(cudaMemcpy(owned_states.data(), h_part.owned,
                              h_part.num_owned * sizeof(VertexState), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(o2g.data(), h_part.owned_to_global,
                              h_part.num_owned * sizeof(uint32_t), cudaMemcpyDeviceToHost));

        for (uint32_t i = 0; i < h_part.num_owned; i++) {
            color[o2g[i]] = owned_states[i].color;
        }

        // Pull actual CSR edges
        std::vector<uint32_t> row_ptr_h(h_part.num_owned + 1);
        CUDA_CHECK(cudaMemcpy(row_ptr_h.data(), h_part.row_ptr,
                              (h_part.num_owned + 1) * sizeof(uint32_t), cudaMemcpyDeviceToHost));

        uint32_t total_slots = row_ptr_h[h_part.num_owned];
        std::vector<uint32_t> col_idx_h(total_slots);
        CUDA_CHECK(cudaMemcpy(col_idx_h.data(), h_part.col_idx,
                              total_slots * sizeof(uint32_t), cudaMemcpyDeviceToHost));

        for (uint32_t i = 0; i < h_part.num_owned; i++) {
            uint32_t u = o2g[i];
            for (uint32_t j = row_ptr_h[i]; j < row_ptr_h[i + 1]; j++) {
                uint32_t v = col_idx_h[j];
                if (v == CSR_INVALID) continue;
                effective_degree[u]++;
                uint32_t lo = std::min(u, v), hi = std::max(u, v);
                edge_set.insert(((uint64_t)lo << 32) | hi);
            }
        }

        // Also pull ghost CSR edges
        std::vector<uint32_t> ghost_rp(h_part.num_owned + 1);
        CUDA_CHECK(cudaMemcpy(ghost_rp.data(), h_part.ghost_row_ptr,
                              (h_part.num_owned + 1) * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        uint32_t ghost_total = ghost_rp[h_part.num_owned];
        if (ghost_total > 0) {
            std::vector<uint32_t> ghost_col(ghost_total);
            CUDA_CHECK(cudaMemcpy(ghost_col.data(), h_part.ghost_col_idx,
                                  ghost_total * sizeof(uint32_t), cudaMemcpyDeviceToHost));
            for (uint32_t i = 0; i < h_part.num_owned; i++) {
                uint32_t u = o2g[i];
                for (uint32_t j = ghost_rp[i]; j < ghost_rp[i + 1]; j++) {
                    uint32_t v = ghost_col[j];
                    if (v == CSR_INVALID) continue;
                    effective_degree[u]++;
                    uint32_t lo = std::min(u, v), hi = std::max(u, v);
                    edge_set.insert(((uint64_t)lo << 32) | hi);
                }
            }
        }
    }

    // Check coloring validity on actual edges
    uint32_t violations = 0;
    for (uint64_t edge : edge_set) {
        uint32_t u = (uint32_t)(edge >> 32);
        uint32_t v = (uint32_t)(edge & 0xFFFFFFFF);
        if (color[u] == color[v] && color[u] != COLOR_UNCOLORED) {
            violations++;
            if (violations <= 10) {
                fprintf(stderr, "VIOLATION: edge (%u, %u) both have color %u\n", u, v, color[u]);
            }
        }
    }

    // Check that all vertices with edges are colored
    uint32_t uncolored_with_edges = 0;
    for (uint32_t u = 0; u < num_vertices; u++) {
        if (color[u] == COLOR_UNCOLORED && effective_degree[u] > 0) {
            uncolored_with_edges++;
        }
    }

    if (violations > 0) {
        fprintf(stderr, "[StreamGC] COLORING INVALID: %u violations, %u uncolored-with-edges (edges: %zu)\n",
                violations, uncolored_with_edges, edge_set.size());
    } else {
        printf("[StreamGC] Coloring valid: 0 violations, %u uncolored-with-edges (edges: %zu)\n",
               uncolored_with_edges, edge_set.size());
    }

    // Count colors used
    uint16_t max_color = 0;
    for (uint32_t u = 0; u < num_vertices; u++) {
        if (color[u] > max_color) max_color = color[u];
    }
    printf("[StreamGC] Colors used: %u\n", max_color);

    return violations == 0;
}

// ---- Static Greedy Coloring (CPU reference) ----
uint16_t static_greedy_coloring(const CSRGraph& graph) {
    std::vector<uint16_t> color(graph.num_vertices, 0);
    std::vector<bool> used(graph.num_vertices + 1, false);

    for (uint32_t v = 0; v < graph.num_vertices; v++) {
        // Mark colors used by neighbors
        for (uint32_t j = graph.row_ptr[v]; j < graph.row_ptr[v + 1]; j++) {
            uint32_t w = graph.col_idx[j];
            if (color[w] != 0) {
                used[color[w]] = true;
            }
        }

        // Find smallest available color
        uint16_t c = 1;
        while (used[c]) c++;
        color[v] = c;

        // Unmark
        for (uint32_t j = graph.row_ptr[v]; j < graph.row_ptr[v + 1]; j++) {
            uint32_t w = graph.col_idx[j];
            if (color[w] != 0) {
                used[color[w]] = false;
            }
        }
    }

    uint16_t max_color = 0;
    for (uint32_t v = 0; v < graph.num_vertices; v++) {
        if (color[v] > max_color) max_color = color[v];
    }

    return max_color;
}

// ---- Write Coloring Output ----
void write_coloring(
    const char* filename,
    GPUPartition** d_partitions,
    uint32_t num_gpus,
    uint32_t num_vertices
) {
    std::vector<uint16_t> colors(num_vertices, 0);

    for (uint32_t gpu = 0; gpu < num_gpus; gpu++) {
        GPUPartition h_part;
        CUDA_CHECK(cudaSetDevice(gpu));
        CUDA_CHECK(cudaMemcpy(&h_part, d_partitions[gpu], sizeof(GPUPartition), cudaMemcpyDeviceToHost));

        std::vector<VertexState> owned_states(h_part.num_owned);
        std::vector<uint32_t> o2g(h_part.num_owned);
        CUDA_CHECK(cudaMemcpy(owned_states.data(), h_part.owned,
                              h_part.num_owned * sizeof(VertexState), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(o2g.data(), h_part.owned_to_global,
                              h_part.num_owned * sizeof(uint32_t), cudaMemcpyDeviceToHost));

        for (uint32_t i = 0; i < h_part.num_owned; i++) {
            colors[o2g[i]] = owned_states[i].color;
        }
    }

    FILE* f = fopen(filename, "wb");
    if (!f) {
        fprintf(stderr, "ERROR: Cannot write coloring file: %s\n", filename);
        return;
    }
    fwrite(colors.data(), sizeof(uint16_t), num_vertices, f);
    fclose(f);
    printf("[StreamGC] Coloring written: %s (%u vertices)\n", filename, num_vertices);

    // Also write text coloring: vertex_id color (one per line)
    char txt_path[520];
    snprintf(txt_path, sizeof(txt_path), "%s.txt", filename);
    FILE* tf = fopen(txt_path, "w");
    if (tf) {
        uint16_t max_c = *std::max_element(colors.begin(), colors.end());
        fprintf(tf, "# vertices: %u\n# colors: %u\n# algorithm: streamgc\n",
                num_vertices, max_c);
        for (uint32_t i = 0; i < num_vertices; i++) {
            fprintf(tf, "%u %u\n", i, colors[i]);
        }
        fclose(tf);
    }
}

// ---- Helper: ensure directory exists (recursive) ----
static void ensure_dir(const std::string& path) {
    size_t pos = 0;
    while ((pos = path.find('/', pos + 1)) != std::string::npos) {
        mkdir(path.substr(0, pos).c_str(), 0755);
    }
    mkdir(path.c_str(), 0755);
}

// ---- Main ----
int main(int argc, char** argv) {
    // Parse command-line args
    const char* graph_file = nullptr;
    uint64_t stream_size = 1000000;
    uint32_t insert_pct = 50;
    uint64_t seed = 42;
    uint32_t num_gpus = 1;
    const char* hw_tag = nullptr;
    const char* output_dir = nullptr;
    const char* log_dir = nullptr;
    uint32_t epoch_size = EPOCH_SIZE_DEFAULT;
    uint32_t histogram_threshold = HISTOGRAM_THRESHOLD_DEFAULT;
    bool validate = false;
    bool profile_hw = false;
    bool init_only = false;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--graph") == 0 && i + 1 < argc) graph_file = argv[++i];
        else if (strcmp(argv[i], "--stream") == 0 && i + 1 < argc) stream_size = atol(argv[++i]);
        else if (strcmp(argv[i], "--insert-pct") == 0 && i + 1 < argc) insert_pct = atoi(argv[++i]);
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) seed = atol(argv[++i]);
        else if (strcmp(argv[i], "--num-gpus") == 0 && i + 1 < argc) num_gpus = atoi(argv[++i]);
        else if (strcmp(argv[i], "--hw-tag") == 0 && i + 1 < argc) hw_tag = argv[++i];
        else if (strcmp(argv[i], "--output-dir") == 0 && i + 1 < argc) output_dir = argv[++i];
        else if (strcmp(argv[i], "--log-dir") == 0 && i + 1 < argc) log_dir = argv[++i];
        else if (strcmp(argv[i], "--epoch-size") == 0 && i + 1 < argc) epoch_size = atoi(argv[++i]);
        else if (strcmp(argv[i], "--histogram-threshold") == 0 && i + 1 < argc) histogram_threshold = atoi(argv[++i]);
        else if (strcmp(argv[i], "--validate") == 0) validate = true;
        else if (strcmp(argv[i], "--profile-hw") == 0) profile_hw = true;
        else if (strcmp(argv[i], "--init-only") == 0) init_only = true;
    }

    if (!graph_file) {
        fprintf(stderr, "Usage: %s --graph <path.mtx> [options]\n", argv[0]);
        return 1;
    }

    // Auto-detect hw_tag
    std::string hw_tag_str;
    if (hw_tag) {
        hw_tag_str = hw_tag;
    } else {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        hw_tag_str = prop.name;
        std::replace(hw_tag_str.begin(), hw_tag_str.end(), ' ', '_');
    }

    printf("[StreamGC] Hardware: %s, GPUs: %u\n", hw_tag_str.c_str(), num_gpus);

    // ---- Configuration ----
    StreamGCConfig config = default_config();
    config.num_gpus = num_gpus;
    config.epoch_size = epoch_size;
    config.histogram_threshold = histogram_threshold;

    // ---- Warm up CUDA runtime & enable peer access BEFORE the graph load ----
    // On very large graphs ( ~30 GB host resident set after
    // dedup) the first CUDA API call after a huge host allocation can return
    // cudaErrorUnknown because the driver's virtual-address-space probe collides
    // with the process's sbrk/mmap footprint. Establishing the CUDA context
    // first avoids that race.
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaFree(0));
    if (num_gpus > 1) {
        enable_peer_access(num_gpus);
    }

    // ---- Load graph ----
    CSRGraph graph = load_graph_csr(graph_file);
    config.num_vertices = graph.num_vertices;
    config.num_edges = graph.num_edges;

    printf("[StreamGC] Config: epoch_size=%u, histogram_threshold=%u\n",
           config.epoch_size, config.histogram_threshold);

    // ---- Allocate partition map ----
    allocate_partition_map(graph.num_vertices, num_gpus);

    // ---- Partition graph ----
    auto t_part_start = std::chrono::high_resolution_clock::now();
    PartitionAssignment assign = degree_aware_vertex_cut(
        graph.row_ptr, graph.col_idx, graph.degree,
        graph.num_vertices, graph.num_edges, num_gpus, 0.1f);
    auto t_part_end = std::chrono::high_resolution_clock::now();
    double partition_time_ms = std::chrono::duration<double, std::milli>(t_part_end - t_part_start).count();

    // Populate partition_map
    for (uint32_t v = 0; v < graph.num_vertices; v++) {
        partition_map[v].current_owner = assign.owner[v];
        partition_map[v].prev_owner = assign.owner[v];
        partition_map[v].migration_epoch = 0;
    }

    // ---- Allocate GPU partitions ----
    std::vector<GPUPartition*> d_partitions(num_gpus);
    for (uint32_t gpu = 0; gpu < num_gpus; gpu++) {
        d_partitions[gpu] = allocate_gpu_partition(assign, graph, gpu, config);
    }

    // ---- Compute init_priority ----
    {
        std::vector<uint64_t> priorities(graph.num_vertices);
        for (uint32_t v = 0; v < graph.num_vertices; v++) {
            priorities[v] = compute_priority(graph.degree[v], v);
        }
        // Copy to all GPU partitions (unified memory)
        for (uint32_t gpu = 0; gpu < num_gpus; gpu++) {
            GPUPartition h_part;
            CUDA_CHECK(cudaSetDevice(gpu));
            CUDA_CHECK(cudaMemcpy(&h_part, d_partitions[gpu], sizeof(GPUPartition), cudaMemcpyDeviceToHost));
            memcpy(h_part.init_priority, priorities.data(), graph.num_vertices * sizeof(uint64_t));
        }
    }

    // ---- Generate streams ----
    auto init_edges = generate_init_stream(graph.row_ptr, graph.col_idx, graph.degree, graph.num_vertices);

    StreamConfig stream_config;
    stream_config.stream_size = stream_size;
    stream_config.insert_pct = insert_pct;
    stream_config.random_seed = seed;

    std::vector<uint8_t> bench_types;
    auto bench_edges = generate_bench_stream_edges(
        graph.row_ptr, graph.col_idx, graph.degree,
        graph.num_vertices, stream_config, bench_types);

    // ---- Host-driven batch processing ----
    CUDA_CHECK(cudaSetDevice(0));

    const uint32_t BATCH_SIZE = 32768;

    // Per-GPU update buffers
    std::vector<EdgeUpdate*> d_updates_per_gpu(num_gpus);
    for (uint32_t gpu = 0; gpu < num_gpus; gpu++) {
        CUDA_CHECK(cudaSetDevice(gpu));
        CUDA_CHECK(cudaMalloc(&d_updates_per_gpu[gpu], BATCH_SIZE * sizeof(EdgeUpdate)));
    }
    CUDA_CHECK(cudaSetDevice(0));
    EdgeUpdate* d_updates = d_updates_per_gpu[0];  // shorthand for single-GPU

    // Device-side array of all partition pointers (for multi-GPU NVLink access)
    GPUPartition** d_all_partitions = nullptr;
    if (num_gpus > 1) {
        CUDA_CHECK(cudaMalloc(&d_all_partitions, num_gpus * sizeof(GPUPartition*)));
        CUDA_CHECK(cudaMemcpy(d_all_partitions, d_partitions.data(),
                              num_gpus * sizeof(GPUPartition*), cudaMemcpyHostToDevice));
    }

    // Conflict counter for repair pass
    uint32_t* d_num_conflicts;
    CUDA_CHECK(cudaMallocManaged(&d_num_conflicts, sizeof(uint32_t)));

    // Helper to process a stream of edges with per-stage profiling
    auto process_stream = [&](
        const std::vector<std::pair<uint32_t, uint32_t>>& edges,
        const std::vector<uint8_t>& types,  // empty = all ADD
        const char* phase_name,
        bool sample_queries = false  // enable query sampling (bench phase only)
    ) -> StreamProfile {
        StreamProfile prof;
        auto t0 = std::chrono::high_resolution_clock::now();
        auto ts = t0;  // reusable timestamp

        // For query sampling: pick random vertices to query after each epoch
        std::mt19937 query_rng(seed + 999);
        const uint32_t QUERIES_PER_EPOCH = 100;

        std::vector<EdgeUpdate> batch(BATCH_SIZE);
        uint64_t total = edges.size();
        uint64_t processed = 0;
        uint64_t updates_since_epoch = 0;

        while (processed < total) {
            uint32_t batch_count = std::min((uint64_t)BATCH_SIZE, total - processed);

            // Fill batch
            for (uint32_t i = 0; i < batch_count; i++) {
                batch[i].u = edges[processed + i].first;
                batch[i].v = edges[processed + i].second;
                batch[i].type = types.empty() ? UpdateType::ADD : (UpdateType)types[processed + i];
                memset(batch[i]._pad, 0, sizeof(batch[i]._pad));
            }

            if (num_gpus == 1) {
                // Single-GPU fast path
                ts = std::chrono::high_resolution_clock::now();
                CUDA_CHECK(cudaMemcpy(d_updates, batch.data(),
                                      batch_count * sizeof(EdgeUpdate), cudaMemcpyHostToDevice));
                auto t_memcpy = std::chrono::high_resolution_clock::now();
                prof.memcpy_ms += std::chrono::duration<double, std::milli>(t_memcpy - ts).count();

                int warps_needed = batch_count;
                int threads = std::min(warps_needed * 32, 256);
                int blocks = (warps_needed * 32 + threads - 1) / threads;
                blocks = std::max(blocks, 1);

                local_update_kernel<<<blocks, threads>>>(
                    d_updates, batch_count, d_partitions[0], histogram_threshold);
                CUDA_CHECK(cudaDeviceSynchronize());
                auto t_kernel = std::chrono::high_resolution_clock::now();
                prof.local_kernel_ms += std::chrono::duration<double, std::milli>(t_kernel - t_memcpy).count();
            } else {
                // Multi-GPU: classify edges into local and boundary per GPU
                ts = std::chrono::high_resolution_clock::now();
                std::vector<std::vector<EdgeUpdate>> local_batches(num_gpus);
                std::vector<std::vector<EdgeUpdate>> boundary_batches(num_gpus);

                for (uint32_t i = 0; i < batch_count; i++) {
                    uint8_t owner_u = partition_map[batch[i].u].current_owner;
                    uint8_t owner_v = partition_map[batch[i].v].current_owner;

                    if (owner_u == owner_v) {
                        local_batches[owner_u].push_back(batch[i]);
                    } else {
                        boundary_batches[owner_u].push_back(batch[i]);
                        boundary_batches[owner_v].push_back(batch[i]);
                    }
                }
                auto t_classify = std::chrono::high_resolution_clock::now();
                prof.classify_ms += std::chrono::duration<double, std::milli>(t_classify - ts).count();

                // Memcpy + launch local kernels on each GPU
                auto t_local_start = std::chrono::high_resolution_clock::now();
                for (uint32_t gpu = 0; gpu < num_gpus; gpu++) {
                    CUDA_CHECK(cudaSetDevice(gpu));

                    if (!local_batches[gpu].empty()) {
                        uint32_t n = local_batches[gpu].size();
                        CUDA_CHECK(cudaMemcpy(d_updates_per_gpu[gpu], local_batches[gpu].data(),
                                              n * sizeof(EdgeUpdate), cudaMemcpyHostToDevice));
                        int warps = n, thr = std::min(warps * 32, 256);
                        int blk = std::max((warps * 32 + thr - 1) / thr, 1);
                        local_update_kernel<<<blk, thr>>>(
                            d_updates_per_gpu[gpu], n, d_partitions[gpu], histogram_threshold);
                    }
                }
                // Sync after local kernels
                for (uint32_t gpu = 0; gpu < num_gpus; gpu++) {
                    CUDA_CHECK(cudaSetDevice(gpu));
                    CUDA_CHECK(cudaDeviceSynchronize());
                }
                auto t_local_done = std::chrono::high_resolution_clock::now();
                prof.local_kernel_ms += std::chrono::duration<double, std::milli>(t_local_done - t_local_start).count();

                // Launch boundary kernels
                auto t_boundary_start = std::chrono::high_resolution_clock::now();
                for (uint32_t gpu = 0; gpu < num_gpus; gpu++) {
                    CUDA_CHECK(cudaSetDevice(gpu));

                    if (!boundary_batches[gpu].empty()) {
                        uint32_t n = boundary_batches[gpu].size();
                        CUDA_CHECK(cudaMemcpy(d_updates_per_gpu[gpu], boundary_batches[gpu].data(),
                                              n * sizeof(EdgeUpdate), cudaMemcpyHostToDevice));
                        int warps = n, thr = std::min(warps * 32, 256);
                        int blk = std::max((warps * 32 + thr - 1) / thr, 1);
                        boundary_update_kernel<<<blk, thr>>>(
                            d_updates_per_gpu[gpu], n, d_partitions[gpu],
                            d_all_partitions, gpu, histogram_threshold);
                    }
                }
                // Sync after boundary kernels
                for (uint32_t gpu = 0; gpu < num_gpus; gpu++) {
                    CUDA_CHECK(cudaSetDevice(gpu));
                    CUDA_CHECK(cudaDeviceSynchronize());
                }
                auto t_boundary_done = std::chrono::high_resolution_clock::now();
                prof.boundary_kernel_ms += std::chrono::duration<double, std::milli>(t_boundary_done - t_boundary_start).count();

                // Ghost synchronization across GPUs
                auto t_ghost_start = std::chrono::high_resolution_clock::now();
                for (uint32_t gpu = 0; gpu < num_gpus; gpu++) {
                    CUDA_CHECK(cudaSetDevice(gpu));
                    GPUPartition h_part;
                    CUDA_CHECK(cudaMemcpy(&h_part, d_partitions[gpu], sizeof(GPUPartition), cudaMemcpyDeviceToHost));
                    if (h_part.num_ghost > 0) {
                        int blk = (h_part.num_ghost + 255) / 256;
                        ghost_sync_kernel<<<blk, 256>>>(d_partitions[gpu], d_all_partitions, num_gpus);
                    }
                }
                for (uint32_t gpu = 0; gpu < num_gpus; gpu++) {
                    CUDA_CHECK(cudaSetDevice(gpu));
                    CUDA_CHECK(cudaDeviceSynchronize());
                }
                auto t_ghost_done = std::chrono::high_resolution_clock::now();
                prof.ghost_sync_ms += std::chrono::duration<double, std::milli>(t_ghost_done - t_ghost_start).count();

                CUDA_CHECK(cudaSetDevice(0));
            }

            processed += batch_count;
            updates_since_epoch += batch_count;

            // Epoch maintenance: snapshot, compact CSR, drain overflow, rebuild bitmaps
            if (updates_since_epoch >= config.epoch_size) {
                auto t_epoch_start = std::chrono::high_resolution_clock::now();

                // Step 1: snapshot + compact CSR (frees INVALID slots)
                for (uint32_t gpu = 0; gpu < num_gpus; gpu++) {
                    CUDA_CHECK(cudaSetDevice(gpu));
                    GPUPartition h_part;
                    CUDA_CHECK(cudaMemcpy(&h_part, d_partitions[gpu], sizeof(GPUPartition), cudaMemcpyDeviceToHost));
                    int epoch_blocks = (h_part.num_owned + 255) / 256;
                    epoch_snapshot_kernel<<<epoch_blocks, 256>>>(d_partitions[gpu], config.num_vertices);
                    compact_csr_kernel<<<epoch_blocks, 256>>>(d_partitions[gpu]);
                }
                for (uint32_t gpu = 0; gpu < num_gpus; gpu++) {
                    CUDA_CHECK(cudaSetDevice(gpu));
                    CUDA_CHECK(cudaDeviceSynchronize());
                }

                // Step 2: drain overflow buffers (retry edges that failed insertion)
                for (uint32_t gpu = 0; gpu < num_gpus; gpu++) {
                    CUDA_CHECK(cudaSetDevice(gpu));
                    GPUPartition h_part;
                    CUDA_CHECK(cudaMemcpy(&h_part, d_partitions[gpu], sizeof(GPUPartition), cudaMemcpyDeviceToHost));

                    uint32_t local_overflow = *h_part.overflow_local_count;
                    uint32_t ghost_overflow = *h_part.overflow_ghost_count;

                    if (local_overflow > 0 || ghost_overflow > 0) {
                        // Reset counters AFTER reading the count
                        *h_part.overflow_local_count = 0;
                        *h_part.overflow_ghost_count = 0;

                        if (local_overflow > 0) {
                            uint32_t count = std::min(local_overflow, OVERFLOW_CAPACITY);
                            int blk = (count + 255) / 256;
                            drain_overflow_local_kernel<<<blk, 256>>>(d_partitions[gpu], count, nullptr);
                        }
                        if (ghost_overflow > 0) {
                            uint32_t count = std::min(ghost_overflow, OVERFLOW_CAPACITY);
                            int blk = (count + 255) / 256;
                            drain_overflow_ghost_kernel<<<blk, 256>>>(d_partitions[gpu], count, nullptr);
                        }
                    }
                }
                for (uint32_t gpu = 0; gpu < num_gpus; gpu++) {
                    CUDA_CHECK(cudaSetDevice(gpu));
                    CUDA_CHECK(cudaDeviceSynchronize());
                }

                // Step 3: rebuild bitmaps (after all edges are inserted)
                for (uint32_t gpu = 0; gpu < num_gpus; gpu++) {
                    CUDA_CHECK(cudaSetDevice(gpu));
                    GPUPartition h_part;
                    CUDA_CHECK(cudaMemcpy(&h_part, d_partitions[gpu], sizeof(GPUPartition), cudaMemcpyDeviceToHost));
                    int epoch_blocks = (h_part.num_owned + 255) / 256;
                    rebuild_bitmaps_kernel<<<epoch_blocks, 256>>>(d_partitions[gpu]);
                }
                for (uint32_t gpu = 0; gpu < num_gpus; gpu++) {
                    CUDA_CHECK(cudaSetDevice(gpu));
                    CUDA_CHECK(cudaDeviceSynchronize());
                }
                CUDA_CHECK(cudaSetDevice(0));

                auto t_epoch_done = std::chrono::high_resolution_clock::now();
                prof.epoch_ms += std::chrono::duration<double, std::milli>(t_epoch_done - t_epoch_start).count();

                // Query sampling: measure point query latency from epoch snapshot
                if (sample_queries) {
                    GPUPartition h_part;
                    CUDA_CHECK(cudaMemcpy(&h_part, d_partitions[0], sizeof(GPUPartition), cudaMemcpyDeviceToHost));
                    // Read snapshot from GPU 0 (covers owned vertices)
                    std::vector<VertexState> snapshot(h_part.num_owned);
                    CUDA_CHECK(cudaMemcpy(snapshot.data(), h_part.owned,
                                          h_part.num_owned * sizeof(VertexState), cudaMemcpyDeviceToHost));

                    std::uniform_int_distribution<uint32_t> qdist(0, h_part.num_owned - 1);
                    for (uint32_t q = 0; q < QUERIES_PER_EPOCH; q++) {
                        uint32_t idx = qdist(query_rng);
                        auto qt0 = std::chrono::high_resolution_clock::now();
                        volatile uint16_t color = snapshot[idx].color;
                        (void)color;
                        auto qt1 = std::chrono::high_resolution_clock::now();
                        prof.query_total_us += std::chrono::duration<double, std::micro>(qt1 - qt0).count();
                        prof.query_count++;
                    }
                }

                updates_since_epoch = 0;
            }

            // Progress report every 10%
            if (processed % (total / 10 + 1) < BATCH_SIZE) {
                printf("[StreamGC] %s: %lu / %lu (%.0f%%)\n",
                       phase_name, (unsigned long)processed, (unsigned long)total,
                       100.0 * processed / total);
                fflush(stdout);
            }
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        prof.total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (prof.query_count > 0) {
            prof.query_latency_us = prof.query_total_us / prof.query_count;
        }
        return prof;
    };

    // CPU-side serial repair for remaining conflicts that GPU parallel repair can't resolve.
    // Builds full merged adjacency (both CSR directions) using flat vector<vector<uint32_t>>.
    auto cpu_serial_repair = [&](const char* phase_name) {
        auto t_repair_start = std::chrono::high_resolution_clock::now();
        uint32_t N = graph.num_vertices;

        // Pull all vertex colors and CSR data from all GPUs
        std::vector<VertexState> states(N);

        struct GPUData {
            GPUPartition h_part;
            std::vector<uint32_t> o2g;
            std::vector<uint32_t> row_ptr_h;
            std::vector<uint32_t> col_idx_h;
            std::vector<uint32_t> ghost_rp;
            std::vector<uint32_t> ghost_col;
        };
        std::vector<GPUData> gpu_data(num_gpus);

        // Owner map: global_id -> (gpu, local_idx) for O(1) lookup
        struct VertexLoc { uint16_t gpu; uint32_t local_idx; };
        std::vector<VertexLoc> owner(N, {0xFFFF, 0});

        for (uint32_t gpu = 0; gpu < num_gpus; gpu++) {
            CUDA_CHECK(cudaSetDevice(gpu));
            auto& gd = gpu_data[gpu];
            CUDA_CHECK(cudaMemcpy(&gd.h_part, d_partitions[gpu], sizeof(GPUPartition), cudaMemcpyDeviceToHost));

            std::vector<VertexState> owned_states(gd.h_part.num_owned);
            gd.o2g.resize(gd.h_part.num_owned);
            CUDA_CHECK(cudaMemcpy(owned_states.data(), gd.h_part.owned,
                                  gd.h_part.num_owned * sizeof(VertexState), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(gd.o2g.data(), gd.h_part.owned_to_global,
                                  gd.h_part.num_owned * sizeof(uint32_t), cudaMemcpyDeviceToHost));

            for (uint32_t i = 0; i < gd.h_part.num_owned; i++) {
                states[gd.o2g[i]] = owned_states[i];
                owner[gd.o2g[i]] = {(uint16_t)gpu, i};
            }

            // Pull CSR data
            gd.row_ptr_h.resize(gd.h_part.num_owned + 1);
            CUDA_CHECK(cudaMemcpy(gd.row_ptr_h.data(), gd.h_part.row_ptr,
                                  (gd.h_part.num_owned + 1) * sizeof(uint32_t), cudaMemcpyDeviceToHost));
            uint32_t total_slots = gd.row_ptr_h[gd.h_part.num_owned];
            gd.col_idx_h.resize(total_slots);
            CUDA_CHECK(cudaMemcpy(gd.col_idx_h.data(), gd.h_part.col_idx,
                                  total_slots * sizeof(uint32_t), cudaMemcpyDeviceToHost));

            gd.ghost_rp.resize(gd.h_part.num_owned + 1);
            CUDA_CHECK(cudaMemcpy(gd.ghost_rp.data(), gd.h_part.ghost_row_ptr,
                                  (gd.h_part.num_owned + 1) * sizeof(uint32_t), cudaMemcpyDeviceToHost));
            uint32_t ghost_total = gd.ghost_rp[gd.h_part.num_owned];
            if (ghost_total > 0) {
                gd.ghost_col.resize(ghost_total);
                CUDA_CHECK(cudaMemcpy(gd.ghost_col.data(), gd.h_part.ghost_col_idx,
                                      ghost_total * sizeof(uint32_t), cudaMemcpyDeviceToHost));
            }
        }
        CUDA_CHECK(cudaSetDevice(0));

        auto t_pull_done = std::chrono::high_resolution_clock::now();

        // Build full adjacency using vector<vector> (both directions for asymmetric CSR)
        // Key optimization: SKIP sort/dedup — duplicates are harmless for both
        // conflict detection (idempotent marking) and greedy recolor (idempotent used[c]=true)
        // This eliminates the O(E*log(degree)) sort cost that dominated on large graphs.

        // Count degrees for pre-allocation
        std::vector<uint32_t> degree(N, 0);
        for (uint32_t gpu = 0; gpu < num_gpus; gpu++) {
            auto& gd = gpu_data[gpu];
            for (uint32_t i = 0; i < gd.h_part.num_owned; i++) {
                uint32_t u = gd.o2g[i];
                for (uint32_t j = gd.row_ptr_h[i]; j < gd.row_ptr_h[i + 1]; j++) {
                    uint32_t v = gd.col_idx_h[j];
                    if (v != CSR_INVALID && v < N && v != u) { degree[u]++; degree[v]++; }
                }
                for (uint32_t j = gd.ghost_rp[i]; j < gd.ghost_rp[i + 1]; j++) {
                    uint32_t v = gd.ghost_col[j];
                    if (v != CSR_INVALID && v < N && v != u) { degree[u]++; degree[v]++; }
                }
            }
        }

        std::vector<std::vector<uint32_t>> adj(N);
        for (uint32_t u = 0; u < N; u++) {
            if (degree[u] > 0) adj[u].reserve(degree[u]);
        }

        for (uint32_t gpu = 0; gpu < num_gpus; gpu++) {
            auto& gd = gpu_data[gpu];
            for (uint32_t i = 0; i < gd.h_part.num_owned; i++) {
                uint32_t u = gd.o2g[i];
                for (uint32_t j = gd.row_ptr_h[i]; j < gd.row_ptr_h[i + 1]; j++) {
                    uint32_t v = gd.col_idx_h[j];
                    if (v == CSR_INVALID || v >= N || v == u) continue;
                    adj[u].push_back(v);
                    adj[v].push_back(u);
                }
                for (uint32_t j = gd.ghost_rp[i]; j < gd.ghost_rp[i + 1]; j++) {
                    uint32_t v = gd.ghost_col[j];
                    if (v == CSR_INVALID || v >= N || v == u) continue;
                    adj[u].push_back(v);
                    adj[v].push_back(u);
                }
            }
        }

        // No sort/dedup needed — duplicates are harmless

        auto t_detect_done = std::chrono::high_resolution_clock::now();

        // Helper: greedy recolor vertex v
        auto greedy_recolor = [&](uint32_t v) -> bool {
            bool used[1024];
            std::memset(used, 0, sizeof(used));
            for (uint32_t w : adj[v]) {
                uint16_t nc = states[w].color;
                if (nc != COLOR_UNCOLORED && nc < 1024) used[nc] = true;
            }
            for (uint16_t c = 1; c < 1024; c++) {
                if (!used[c]) {
                    if (states[v].color != c) {
                        states[v].color = c;
                        states[v].version++;
                        return true;
                    }
                    return false;
                }
            }
            return false;
        };

        // Fix uncolored vertices that have edges
        uint32_t total_fixed = 0;
        for (uint32_t u = 0; u < N; u++) {
            if (states[u].color != COLOR_UNCOLORED || adj[u].empty()) continue;
            if (greedy_recolor(u)) total_fixed++;
        }

        // Iterative conflict repair with separated detect/fix phases
        for (int pass = 0; pass < 30; pass++) {
            std::vector<uint32_t> to_fix;
            for (uint32_t u = 0; u < N; u++) {
                for (uint32_t v : adj[u]) {
                    if (v <= u) continue;
                    if (states[u].color == states[v].color &&
                        states[u].color != COLOR_UNCOLORED) {
                        to_fix.push_back(v);
                    }
                }
            }
            if (to_fix.empty()) break;
            std::sort(to_fix.begin(), to_fix.end());
            to_fix.erase(std::unique(to_fix.begin(), to_fix.end()), to_fix.end());

            uint32_t fixed = 0;
            for (uint32_t v : to_fix) {
                if (greedy_recolor(v)) fixed++;
            }
            total_fixed += fixed;
            if (fixed == 0) break;
        }

        if (total_fixed > 0) {
            for (uint32_t gpu = 0; gpu < num_gpus; gpu++) {
                auto& gd = gpu_data[gpu];
                std::vector<VertexState> owned_states(gd.h_part.num_owned);
                for (uint32_t i = 0; i < gd.h_part.num_owned; i++) {
                    owned_states[i] = states[gd.o2g[i]];
                }
                CUDA_CHECK(cudaSetDevice(gpu));
                CUDA_CHECK(cudaMemcpy(gd.h_part.owned, owned_states.data(),
                                      gd.h_part.num_owned * sizeof(VertexState), cudaMemcpyHostToDevice));
            }
            CUDA_CHECK(cudaSetDevice(0));
        }

        auto t_repair_end = std::chrono::high_resolution_clock::now();
        double pull_ms = std::chrono::duration<double, std::milli>(t_detect_done - t_pull_done).count();
        double repair_ms = std::chrono::duration<double, std::milli>(t_repair_end - t_repair_start).count();
        printf("[StreamGC] %s CPU repair: fixed %u vertices (%.0f ms total, detect %.0f ms)\n",
               phase_name, total_fixed, repair_ms, pull_ms);
        fflush(stdout);
    };

    // Helper: drain any remaining overflow edges and compact CSR
    auto drain_and_compact = [&](const char* phase_name) {
        // Step 1: Compact CSR on all GPUs (frees INVALID slots)
        for (uint32_t gpu = 0; gpu < num_gpus; gpu++) {
            CUDA_CHECK(cudaSetDevice(gpu));
            GPUPartition h_part;
            CUDA_CHECK(cudaMemcpy(&h_part, d_partitions[gpu], sizeof(GPUPartition), cudaMemcpyDeviceToHost));
            int epoch_blocks = (h_part.num_owned + 255) / 256;
            compact_csr_kernel<<<epoch_blocks, 256>>>(d_partitions[gpu]);
        }
        for (uint32_t gpu = 0; gpu < num_gpus; gpu++) {
            CUDA_CHECK(cudaSetDevice(gpu));
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // Step 2: Drain overflow buffers
        uint32_t total_local_overflow = 0, total_ghost_overflow = 0;
        uint32_t total_failed = 0;
        *d_num_conflicts = 0;  // reuse as failed counter

        for (uint32_t gpu = 0; gpu < num_gpus; gpu++) {
            CUDA_CHECK(cudaSetDevice(gpu));
            GPUPartition h_part;
            CUDA_CHECK(cudaMemcpy(&h_part, d_partitions[gpu], sizeof(GPUPartition), cudaMemcpyDeviceToHost));

            uint32_t local_overflow = *h_part.overflow_local_count;
            uint32_t ghost_overflow = *h_part.overflow_ghost_count;
            total_local_overflow += local_overflow;
            total_ghost_overflow += ghost_overflow;

            if (local_overflow > 0 || ghost_overflow > 0) {
                // Reset counters
                *h_part.overflow_local_count = 0;
                *h_part.overflow_ghost_count = 0;

                if (local_overflow > 0) {
                    uint32_t count = std::min(local_overflow, OVERFLOW_CAPACITY);
                    int blk = (count + 255) / 256;
                    drain_overflow_local_kernel<<<blk, 256>>>(d_partitions[gpu], count, d_num_conflicts);
                }
                if (ghost_overflow > 0) {
                    uint32_t count = std::min(ghost_overflow, OVERFLOW_CAPACITY);
                    int blk = (count + 255) / 256;
                    drain_overflow_ghost_kernel<<<blk, 256>>>(d_partitions[gpu], count, d_num_conflicts);
                }
            }
        }
        for (uint32_t gpu = 0; gpu < num_gpus; gpu++) {
            CUDA_CHECK(cudaSetDevice(gpu));
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        total_failed = *d_num_conflicts;

        // Step 3: Rebuild bitmaps after drain
        for (uint32_t gpu = 0; gpu < num_gpus; gpu++) {
            CUDA_CHECK(cudaSetDevice(gpu));
            GPUPartition h_part;
            CUDA_CHECK(cudaMemcpy(&h_part, d_partitions[gpu], sizeof(GPUPartition), cudaMemcpyDeviceToHost));
            int blk = (h_part.num_owned + 255) / 256;
            rebuild_bitmaps_kernel<<<blk, 256>>>(d_partitions[gpu]);
        }
        for (uint32_t gpu = 0; gpu < num_gpus; gpu++) {
            CUDA_CHECK(cudaSetDevice(gpu));
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        CUDA_CHECK(cudaSetDevice(0));

        if (total_local_overflow > 0 || total_ghost_overflow > 0) {
            printf("[StreamGC] %s overflow drain: %u local + %u ghost attempted, %u failed\n",
                   phase_name, total_local_overflow, total_ghost_overflow, total_failed);
        }
        fflush(stdout);
    };

    // Helper: run conflict repair until convergence (all GPUs)
    auto run_conflict_repair = [&](const char* phase_name) {
        uint32_t prev_conflicts = UINT32_MAX;
        int plateau_count = 0;

        for (int iter = 0; iter < 20; iter++) {
            // Phase 1: Rebuild bitmaps + ghost sync on all GPUs
            for (uint32_t gpu = 0; gpu < num_gpus; gpu++) {
                CUDA_CHECK(cudaSetDevice(gpu));
                GPUPartition h_part;
                CUDA_CHECK(cudaMemcpy(&h_part, d_partitions[gpu], sizeof(GPUPartition), cudaMemcpyDeviceToHost));
                int blk = (h_part.num_owned + 255) / 256;
                if (blk > 0) rebuild_bitmaps_kernel<<<blk, 256>>>(d_partitions[gpu]);
            }
            for (uint32_t gpu = 0; gpu < num_gpus; gpu++) {
                CUDA_CHECK(cudaSetDevice(gpu));
                CUDA_CHECK(cudaDeviceSynchronize());
            }

            // Ghost sync for multi-GPU
            if (num_gpus > 1) {
                for (uint32_t gpu = 0; gpu < num_gpus; gpu++) {
                    CUDA_CHECK(cudaSetDevice(gpu));
                    GPUPartition h_part;
                    CUDA_CHECK(cudaMemcpy(&h_part, d_partitions[gpu], sizeof(GPUPartition), cudaMemcpyDeviceToHost));
                    if (h_part.num_ghost > 0) {
                        int blk = (h_part.num_ghost + 255) / 256;
                        ghost_sync_kernel<<<blk, 256>>>(d_partitions[gpu], d_all_partitions, num_gpus);
                    }
                }
                for (uint32_t gpu = 0; gpu < num_gpus; gpu++) {
                    CUDA_CHECK(cudaSetDevice(gpu));
                    CUDA_CHECK(cudaDeviceSynchronize());
                }
            }

            // Phase 2: Detect and repair conflicts on all GPUs
            *d_num_conflicts = 0;
            for (uint32_t gpu = 0; gpu < num_gpus; gpu++) {
                CUDA_CHECK(cudaSetDevice(gpu));
                GPUPartition h_part;
                CUDA_CHECK(cudaMemcpy(&h_part, d_partitions[gpu], sizeof(GPUPartition), cudaMemcpyDeviceToHost));
                int blk = (h_part.num_owned + 255) / 256;
                if (blk > 0) {
                    conflict_repair_kernel<<<blk, 256>>>(
                        d_partitions[gpu], histogram_threshold, d_num_conflicts);
                }
            }
            for (uint32_t gpu = 0; gpu < num_gpus; gpu++) {
                CUDA_CHECK(cudaSetDevice(gpu));
                CUDA_CHECK(cudaDeviceSynchronize());
            }
            CUDA_CHECK(cudaSetDevice(0));

            if (*d_num_conflicts == 0) {
                printf("[StreamGC] %s repair: converged after %d iteration(s)\n", phase_name, iter + 1);
                fflush(stdout);
                return;
            }
            printf("[StreamGC] %s repair iter %d: %u conflicts\n", phase_name, iter + 1, *d_num_conflicts);
            fflush(stdout);

            if (*d_num_conflicts == prev_conflicts) {
                plateau_count++;
                if (plateau_count >= 3) {
                    printf("[StreamGC] %s GPU repair: plateau at %u conflicts, deferring to CPU\n",
                           phase_name, *d_num_conflicts);
                    fflush(stdout);
                    return;
                }
            } else {
                plateau_count = 0;
            }
            prev_conflicts = *d_num_conflicts;
        }
        printf("[StreamGC] %s GPU repair: did not converge after 20 iterations\n", phase_name);
        fflush(stdout);
    };

    // Phase 1: Init stream
    printf("[StreamGC] Starting Phase 1: Init stream (%zu edges)...\n", init_edges.size());
    fflush(stdout);
    std::vector<uint8_t> empty_types;
    StreamProfile init_prof = process_stream(init_edges, empty_types, "Init", false);
    double init_time_ms = init_prof.total_ms;
    printf("[StreamGC] Init complete: %.2f ms\n", init_time_ms);
    fflush(stdout);

    // Drain any remaining overflow edges, then repair conflicts
    drain_and_compact("Init");
    run_conflict_repair("Init");
    // CPU serial repair with full adjacency — catches all remaining conflicts
    cpu_serial_repair("Init");

    // Phase 2: Bench stream (includes repair time in total measurement)
    printf("[StreamGC] Starting Phase 2: Bench stream (%zu events)...\n", bench_edges.size());
    fflush(stdout);
    auto bench_t0 = std::chrono::high_resolution_clock::now();
    StreamProfile bench_prof = process_stream(bench_edges, bench_types, "Bench", true);
    double bench_stream_ms = bench_prof.total_ms;
    printf("[StreamGC] Bench stream complete: %.2f ms\n", bench_stream_ms);
    fflush(stdout);

    // Drain any remaining overflow edges, then repair conflicts
    drain_and_compact("Bench");
    run_conflict_repair("Bench");
    // CPU serial repair with full adjacency — catches all remaining conflicts
    cpu_serial_repair("Bench");
    auto bench_t1 = std::chrono::high_resolution_clock::now();
    double bench_total_ms = std::chrono::duration<double, std::milli>(bench_t1 - bench_t0).count();
    double repair_ms = bench_total_ms - bench_stream_ms;

    cudaFree(d_updates);

    // ---- Report results ----
    printf("[StreamGC] Init time: %.2f ms\n", init_time_ms);
    printf("[StreamGC] Bench time: %.2f ms (stream: %.2f ms, repair: %.2f ms)\n",
           bench_total_ms, bench_stream_ms, repair_ms);
    if (bench_total_ms > 0) {
        double raw_tp = bench_edges.size() / bench_stream_ms / 1000.0;
        double eff_tp = bench_edges.size() / bench_total_ms / 1000.0;
        printf("[StreamGC] Throughput: %.2f M updates/sec (effective: %.2f M updates/sec)\n",
               raw_tp, eff_tp);
    }
#ifdef STREAMGC_INSTRUMENT_PHI
    {
        uint32_t phi_max_across_gpus = 0;
        int64_t  phi_inflight_total = 0;
        for (uint32_t gpu = 0; gpu < num_gpus; gpu++) {
            GPUPartition hp;
            CUDA_CHECK(cudaSetDevice(gpu));
            CUDA_CHECK(cudaMemcpy(&hp, d_partitions[gpu], sizeof(GPUPartition), cudaMemcpyDeviceToHost));
            if (hp.phi_max != nullptr) {
                phi_max_across_gpus = std::max(phi_max_across_gpus, *hp.phi_max);
                phi_inflight_total += *hp.phi_inflight;
            }
        }
        printf("[StreamGC] Phi instrumentation: max_inflight=%u (peak across all GPUs), "
               "residual_inflight=%ld (sum)\n",
               phi_max_across_gpus, (long)phi_inflight_total);
    }
#endif
    fflush(stdout);

    // ---- Validity check ----
    if (validate) {
        bool valid = check_coloring_valid(d_partitions.data(), num_gpus, graph.num_vertices);
        if (!valid) {
            fprintf(stderr, "[StreamGC] VALIDATION FAILED\n");
        }
    }

    // ---- Write output ----
    if (output_dir) {
        ensure_dir(output_dir);

        // Extract graph name from path
        std::string graph_path(graph_file);
        size_t slash = graph_path.rfind('/');
        size_t dot = graph_path.rfind('.');
        std::string graph_name = graph_path.substr(slash + 1, dot - slash - 1);

        char coloring_path[512];
        snprintf(coloring_path, sizeof(coloring_path), "%s/streamgc_%ugpu_stream%lu_%upct.coloring",
                 output_dir, num_gpus, stream_size, insert_pct);
        write_coloring(coloring_path, d_partitions.data(), num_gpus, graph.num_vertices);

        // Static greedy reference
        printf("[StreamGC] Running static greedy coloring (CPU reference)...\n");
        uint16_t greedy_colors = static_greedy_coloring(graph);
        printf("[StreamGC] Static greedy: %u colors\n", greedy_colors);

        // Write .coloring.meta
        char meta_path[512];
        snprintf(meta_path, sizeof(meta_path), "%s.meta", coloring_path);
        FILE* mf = fopen(meta_path, "w");
        if (mf) {
            // Count colors used by StreamGC
            std::vector<uint16_t> colors(graph.num_vertices, 0);
            for (uint32_t gpu = 0; gpu < num_gpus; gpu++) {
                GPUPartition hp;
                CUDA_CHECK(cudaSetDevice(gpu));
                CUDA_CHECK(cudaMemcpy(&hp, d_partitions[gpu], sizeof(GPUPartition), cudaMemcpyDeviceToHost));
                std::vector<VertexState> os(hp.num_owned);
                std::vector<uint32_t> og(hp.num_owned);
                CUDA_CHECK(cudaMemcpy(os.data(), hp.owned, hp.num_owned * sizeof(VertexState), cudaMemcpyDeviceToHost));
                CUDA_CHECK(cudaMemcpy(og.data(), hp.owned_to_global, hp.num_owned * sizeof(uint32_t), cudaMemcpyDeviceToHost));
                for (uint32_t i = 0; i < hp.num_owned; i++) colors[og[i]] = os[i].color;
            }
            uint16_t max_c = *std::max_element(colors.begin(), colors.end());
            float ratio = (greedy_colors > 0) ? (float)max_c / greedy_colors : 0.0f;

            double raw_tp = bench_stream_ms > 0 ? bench_edges.size() / bench_stream_ms / 1000.0 : 0;
            double eff_tp = bench_total_ms > 0 ? bench_edges.size() / bench_total_ms / 1000.0 : 0;

            fprintf(mf, "{\n");
            fprintf(mf, "  \"algorithm\": \"streamgc\",\n");
            fprintf(mf, "  \"graph\": \"%s\",\n", graph_name.c_str());
            fprintf(mf, "  \"num_vertices\": %u,\n", graph.num_vertices);
            fprintf(mf, "  \"num_edges\": %lu,\n", (unsigned long)graph.num_edges);
            fprintf(mf, "  \"num_colors_used\": %u,\n", max_c);
            fprintf(mf, "  \"static_greedy_colors\": %u,\n", greedy_colors);
            fprintf(mf, "  \"color_ratio\": %.2f,\n", ratio);
            fprintf(mf, "  \"init_time_ms\": %.2f,\n", init_time_ms);
            fprintf(mf, "  \"bench_time_ms\": %.2f,\n", bench_total_ms);
            fprintf(mf, "  \"bench_stream_ms\": %.2f,\n", bench_stream_ms);
            fprintf(mf, "  \"repair_time_ms\": %.2f,\n", repair_ms);
            fprintf(mf, "  \"throughput_mups\": %.2f,\n", raw_tp);
            fprintf(mf, "  \"effective_throughput_mups\": %.2f,\n", eff_tp);
#ifdef STREAMGC_INSTRUMENT_PHI
            {
                uint32_t phi_max_across = 0;
                int64_t  phi_residual = 0;
                for (uint32_t g = 0; g < num_gpus; g++) {
                    GPUPartition hp2;
                    CUDA_CHECK(cudaSetDevice(g));
                    CUDA_CHECK(cudaMemcpy(&hp2, d_partitions[g], sizeof(GPUPartition), cudaMemcpyDeviceToHost));
                    if (hp2.phi_max != nullptr) {
                        phi_max_across = std::max(phi_max_across, *hp2.phi_max);
                        phi_residual += *hp2.phi_inflight;
                    }
                }
                fprintf(mf, "  \"phi_max_inflight\": %u,\n", phi_max_across);
                fprintf(mf, "  \"phi_residual_inflight\": %ld,\n", (long)phi_residual);
            }
#endif
            // Per-stage profiling (bench phase)
            fprintf(mf, "  \"classify_ms\": %.2f,\n", bench_prof.classify_ms);
            fprintf(mf, "  \"local_kernel_ms\": %.2f,\n", bench_prof.local_kernel_ms);
            fprintf(mf, "  \"boundary_kernel_ms\": %.2f,\n", bench_prof.boundary_kernel_ms);
            fprintf(mf, "  \"gpu_sync_ms\": %.2f,\n", bench_prof.gpu_sync_ms);
            fprintf(mf, "  \"ghost_sync_ms\": %.2f,\n", bench_prof.ghost_sync_ms);
            fprintf(mf, "  \"epoch_maintenance_ms\": %.2f,\n", bench_prof.epoch_ms);
            fprintf(mf, "  \"memcpy_ms\": %.2f,\n", bench_prof.memcpy_ms);
            // Query latency
            fprintf(mf, "  \"query_latency_us\": %.2f,\n", bench_prof.query_latency_us);
            fprintf(mf, "  \"query_count\": %lu,\n", (unsigned long)bench_prof.query_count);
            // Partition info
            fprintf(mf, "  \"partition_time_ms\": %.2f,\n", partition_time_ms);
            fprintf(mf, "  \"partition_info\": [\n");
            for (uint32_t gpu = 0; gpu < num_gpus; gpu++) {
                fprintf(mf, "    {\"gpu\": %u, \"owned\": %u, \"ghost\": %u, \"edges\": %lu}%s\n",
                        gpu, assign.vertex_count[gpu],
                        (uint32_t)assign.ghost_vertices[gpu].size(),
                        (unsigned long)assign.edge_count[gpu],
                        (gpu < num_gpus - 1) ? "," : "");
            }
            fprintf(mf, "  ],\n");
            // Experiment config
            fprintf(mf, "  \"stream_size\": %lu,\n", stream_size);
            fprintf(mf, "  \"insert_pct\": %u,\n", insert_pct);
            fprintf(mf, "  \"seed\": %lu,\n", seed);
            fprintf(mf, "  \"num_gpus\": %u,\n", num_gpus);
            fprintf(mf, "  \"hw_tag\": \"%s\"\n", hw_tag_str.c_str());
            fprintf(mf, "}\n");
            fclose(mf);
            printf("[StreamGC] Meta written: %s (ratio: %.2f)\n", meta_path, ratio);
        }
    }

    // ---- Cleanup ----
    cudaFree(partition_map);
    free_csr_graph(graph);
    free_partition_assignment(assign);

    printf("[StreamGC] Done.\n");
    return 0;
}
