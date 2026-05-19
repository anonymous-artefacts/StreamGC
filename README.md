# StreamGC

This is the artefact accompanying the paper.
## How to run

### Prerequisites

- CUDA 13.0+, gcc 9+ (C++17), GNU Make 4+
- Graph files in Matrix-Market (`.mtx`) format under `data/`

### Build

```bash
make streamgc                          # multi-arch (sm_80 + sm_90)
cd algorithms/streamgc && make fast    # native arch only, faster compile
make test                              # unit + integration tests
```

Binary: `algorithms/streamgc/streaming_gc`.

### Single run

```bash
./algorithms/streamgc/streaming_gc \
    --graph data/com-Amazon.mtx \
    --stream 100000 \
    --insert-pct 50 \
    --seed 42 \
    --num-gpus 1 \
    --hw-tag H100 \
    --output-dir results/H100/com-Amazon \
    --validate
```

### Common flags

| Flag | Meaning |
|---|---|
| `--graph <path>` | Path to `.mtx` graph file |
| `--stream <n>` | Stream length (number of edge updates) |
| `--insert-pct <n>` | Insert percentage (0–100) |
| `--seed <n>` | Random seed (default 42) |
| `--num-gpus <n>` | GPUs to use (1, 2, 4, or 8) |
| `--hw-tag <tag>` | Hardware label, used in the output dir |
| `--output-dir <dir>` | Where to write `.coloring` and `.coloring.meta` |
| `--validate` | Run correctness check at the end |

---

## Graphs

The ten graphs used in the paper, all SuiteSparse `.mtx` downloads. `E/V` is the average per-vertex edge count.

| Graph | V | E | E/V |
|---|---:|---:|---:|
| delaunay_n17     |     131,072 |       786,352 |   6.0 |
| com-Amazon       |     334,863 |     1,851,744 |   5.5 |
| com-Youtube      |   1,134,890 |     5,975,248 |   5.3 |
| as-Skitter       |   1,696,415 |    22,190,596 |  13.1 |
| com-LiveJournal  |   3,997,962 |    69,362,378 |  17.4 |
| kron_g500-logn20 |   1,048,576 |    89,238,804 |  85.1 |
| com-Orkut        |   3,072,441 |   234,370,166 |  76.3 |
| com-Friendster   |  65,608,366 | 3,612,134,270 |  55.1 |
| agatha-2015      | 107,000,000 | 3,000,000,000 |  28.0 |
| moliere-2016     |  23,300,000 |   180,300,000 |   7.7 |

---

## Code structure

```
StreamingGC/
├── README.md                  This file
├── Makefile                   Top-level build dispatcher
├── algorithms/streamgc/
│   ├── include/               Public headers (.cuh, .h)
│   ├── src/
│   │   ├── streamgc.cu        Top-level orchestration, profiling, CSV output
│   │   ├── kernels/           Device-side kernels (one .cu per role)
│   │   └── host/              Host-side helpers (loader, partitioner, dispatch)
│   ├── test/                  Unit + integration tests
│   └── Makefile               Per-algorithm build
└── benchmarks/microbench/     Φ-bound microbenchmarks (T_rc, L_nvlink)
```

### Headers (`algorithms/streamgc/include/`)

| File | Role |
|---|---|
| `vertex_state.cuh` | 32-bit packed `(color, version)` word and pack/unpack helpers |
| `partition.cuh` | Vertex → GPU mapping, ghost set lookups |
| `epoch.cuh` | Epoch interval and refresh constants |
| `color_select.cuh` | Bitmap / histogram colour-selection dispatch |
| `priority.cuh` | Frozen initial-degree priority and tie-break |
| `streamgc.cuh` | Top-level structs (`GPUPartition`, `StreamGCConfig`, `EdgeUpdate`) |
| `types.h` | Shared types between `.cu` and `.cpp` |

### Kernels (`algorithms/streamgc/src/kernels/`)

| File | Role |
|---|---|
| `local_update.cu` | Warp-cooperative handler for owned-edge updates |
| `boundary_update.cu` | Cross-partition handler reading through ghost mirrors |
| `versioned_pull.cu` | Single-atomic peer load with version check |
| `recolor.cu` | Fast-path colour update + slow-path CAS retry |
| `conflict_repair.cu` | Tier-1 GPU repair (≤ 20 iterations, bounded) |
| `epoch_maintenance.cu` | Snapshot publish, CSR compact, bitmap rebuild |

### Host pipeline (`algorithms/streamgc/src/host/`)

| File | Role |
|---|---|
| `graph_loader.cpp` | `.mtx` reader, degree counter, symmetrisation |
| `partitioner.cpp`  | Degree-aware static partition, ghost set computation |
| `stream_gen.cpp`   | Synthetic init + bench stream (uniform random, seed-deterministic) |
| `feeder_thread.cpp`| Host → GPU batch dispatcher (4 K updates per `cudaMemcpy`) |
| `query_server.cpp` | Snapshot-backed point and neighborhood query handler |

---

## Code map

A single edge update travels through the system in this order. Read the code in this order if you're trying to understand the hot path.

1. **Stream generation — `host/stream_gen.cpp`.** Edge updates `(u, v, op)` are produced from a fixed seed; `op` is `INSERT` or `DELETE` according to `--insert-pct`. The bench stream is materialised before the run starts, so the GPU is never waiting on the host RNG.

2. **Host dispatch — `host/feeder_thread.cpp`.** Updates are batched into chunks of 4 096 and pushed to the GPU via `cudaMemcpy`. The earlier persistent-kernel + ring-buffer design hung on UVA coherence and was replaced with a plain memcpy loop.

3. **Local kernel — `kernels/local_update.cu`.** One warp per edge. The warp identifies whether the edge is fully-owned or boundary, votes via `__ballot_sync` on whether a recolour is needed, and dispatches to `recolor.cu` if so. Conflicts among warps on the same vertex are detected inline.

4. **Boundary kernel — `kernels/boundary_update.cu` + `versioned_pull.cu`.** If the edge crosses a partition boundary, the ghost-side endpoint is pulled with a single 32-bit atomic load through NVLink. The version field guards against a torn read; a stale pull retries.

5. **Colour selection — `priority.cuh` → `color_select.cuh` → `recolor.cu`.** The frozen priority `π(v) = (init_deg(v) << 32) | id(v)` decides which endpoint must move. The mover queries the per-vertex used-colour set (a 64-bit bitmap when Δ < 64, a histogram otherwise) and CAS-writes the new packed `(color, version+1)` word.

6. **In-flight repair — `kernels/conflict_repair.cu`.** Once per epoch (default `EPOCH_SIZE = 10 000` events), a bounded GPU sweep visits any boundary vertex that may still hold a conflict and re-runs the recolour. The sweep is bounded to ≤ 20 iterations to keep the persistent kernel's tail latency tight.

7. **Epoch maintenance — `kernels/epoch_maintenance.cu`.** At the same cadence, `epoch_snapshot[v] = owned[v].color` is published into unified memory, the dynamic CSR is compacted (free slots reclaimed, overflow buffer drained), and bitmaps are rebuilt from the live state. This is the only kernel that touches the snapshot — readers and the streaming kernel never share a cacheline.

8. **Tier-2 host repair — `host/feeder_thread.cpp` (background).** A CPU thread walks the bidirectional adjacency once per epoch and re-colours any vertex Tier-1 left unfixed. This is the correctness safety net — it guarantees a proper colouring at every epoch boundary even when Tier-1's bound is too tight.

9. **Query path — `host/query_server.cpp`.** Queries read directly from `epoch_snapshot[v]` (point) or the per-partition `neighbor_bitmap` (neighbourhood). Both are unified-memory reads. The query never enters the streaming kernel's update path, never blocks an update, and runs in O(1).

10. **Output — `streamgc.cu`.** The final coloring is written to `<output-dir>/streamgc_stream<N>_<pct>pct.coloring` (binary vertex→colour array) and `.coloring.meta` (JSON with `init_ms`, `stream_ms`, `repair_ms`, `bench_ms`, `throughput_mups`, `effective_throughput_mups`, `query_latency_us`, plus per-kernel breakdowns). `scripts/collect_results.sh` parses these into a single CSV.
