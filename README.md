# StreamGC

This is the artifact for the IPDPS '27 paper *StreamGC: Scalable Coloring for Streaming Graphs*. StreamGC is a GPU-native graph coloring system that services updates per event rather than in batches. One warp handles each edge update and detects conflicts with a single warp vote. A frozen priority decides which endpoint recolors, and a two-tier repair at every epoch boundary publishes a valid snapshot for queries. Queries never block updates and see a coloring that is at most one epoch old.

## How to run

### Prerequisites

- CUDA 13.1 (the paper's toolchain), gcc 9+ (C++17), GNU Make 4+, OpenMP
- NVIDIA A100, H100, or B100 GPUs. Multi-GPU runs need NVLink with peer access.
- Graph files in Matrix-Market (`.mtx`) format under `data/`

### Build

```bash
make streamgc                          # multi-arch build
cd algorithms/streamgc && make fast    # native arch only, faster compile
make test                              # unit + integration tests
```

Binary: `algorithms/streamgc/streaming_gc`.

Two compile-time options are available:

| Define | Default | Meaning |
|---|---|---|
| `STREAMGC_CSR_EXTRA_SLACK_PCT` | `300` | Extra row capacity in percent. `300` gives each row 4× its initial degree (α = 4 in the paper). |
| `STREAMGC_INSTRUMENT_PHI` | off | Enables the Φ counters (conflicts in flight and maximum per epoch) used for the repair results in §IV-C. |

### Single run

```bash
./algorithms/streamgc/stream_gc \
    --graph data/com-LiveJournal.mtx \
    --stream 1000000 \
    --insert-pct 50 \
    --epoch-size 32768 \
    --seed 42 \
    --num-gpus 1 \
    --hw-tag H100 \
    --output-dir results/H100/com-LiveJournal \
    --validate
```

This is the paper's default configuration: one million updates, half insertions and half deletions, and an epoch size of E = 32,768, so the stream spans 31 epochs.

### Flags

| Flag | Default | Meaning |
|---|---|---|
| `--graph <path>` | required | Path to the `.mtx` graph file |
| `--stream <n>` | `1000000` | Stream length (number of edge updates) |
| `--insert-pct <n>` | `50` | Percentage of insertions (0–100) |
| `--epoch-size <n>` | `32768` | Updates per epoch (E). Larger epochs raise throughput but increase staleness and repair work (§IV-D). |
| `--histogram-threshold <n>` | `64` | Initial degree above which a vertex selects colors from a histogram instead of a 64-bit bitmap |
| `--seed <n>` | `42` | Random seed. The same seed gives a byte-identical stream. |
| `--num-gpus <n>` | `1` | GPUs to use (1, 2, 4, or 8) |
| `--hw-tag <tag>` | GPU name | Hardware label used in the output directory |
| `--output-dir <dir>` | none | Where to write `.coloring` and `.coloring.meta` |
| `--validate` | off | Check that the final coloring is proper |

---

## Experimental setup

### Graphs

The ten graphs from the paper (Table III), all from the SuiteSparse Matrix Collection. |E| counts undirected edges, and the average degree is 2|E|/|V|. G1–G8 run on one GPU, while G9 and G10 run across multiple GPUs.

| | Graph | \|V\| | \|E\| | Avg. deg. |
|---|---|---:|---:|---:|
| G1  | delaunay_n17     |     131,072 |       393,176 |   6.00 |
| G2  | com-Amazon       |     334,863 |       925,872 |   5.53 |
| G3  | com-Youtube      |   1,134,890 |     2,987,624 |   5.27 |
| G4  | as-Skitter       |   1,696,415 |    11,095,298 |  13.08 |
| G5  | com-LiveJournal  |   3,997,962 |    34,681,189 |  17.35 |
| G6  | kron_g500-logn20 |   1,048,576 |    44,619,402 |  85.11 |
| G7  | com-Orkut        |   3,072,441 |   117,185,083 |  76.28 |
| G8  | com-Friendster   |  65,608,366 | 1,806,067,135 |  55.06 |
| G9  | Moliere_2016     |     30.24 M |        3.33 B | 220.55 |
| G10 | Agatha-2015      |    183.96 M |        5.79 B |  63.00 |

### Update streams

| Stream | What it does |
|---|---|
| Churn (default) | Deletes a live edge or restores a previously deleted one. Keeps the degree distribution and produces few conflicts. |
| Uniform | Inserts random non-adjacent vertex pairs |
| Hold-out | Inserts a held-out 10% of the graph's edges |
| Dense | Draws updates from the k-core at the 90th percentile of core numbers, around high-degree vertices |
| Dense growth | The dense stream with two insertions per deletion |

### Baselines

- **ECL-GC** (Alabandi et al., PPoPP '20): static GPU coloring, rerun on every update. It is also the color-quality reference.
- **LVCU** (Khanda et al., HiPC '22): batch-dynamic GPU coloring. We swept its batch size from 1K to 10M updates and use the fastest setting, 100K updates per batch.

Every result is the average of ten runs (variance below 2%), and all systems process the same seeded stream.

---

## Code structure

```
StreamGC/
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
| `vertex_state.cuh` | Packed 32-bit `(color, version)` word and pack/unpack helpers |
| `partition.cuh` | Vertex-to-GPU mapping and ghost set lookups |
| `epoch.cuh` | Epoch size and refresh constants |
| `color_select.cuh` | Bitmap or histogram color selection |
| `priority.cuh` | Frozen priority π(v) = (init_deg(v) << 32) \| id(v) |
| `streamgc.cuh` | Top-level structs (`GPUPartition`, `StreamGCConfig`, `EdgeUpdate`) |
| `types.h` | Types shared between `.cu` and `.cpp` |

### Kernels (`algorithms/streamgc/src/kernels/`)

| File | Role |
|---|---|
| `local_update.cu` | Warp-cooperative handler for updates whose endpoints are both owned |
| `boundary_update.cu` | Handler for updates that cross partitions, reading through ghost copies |
| `versioned_pull.cu` | Single 32-bit system-scoped load of a remote vertex with version check |
| `recolor.cu` | Optimistic recolor (store, fence, re-read) with a version-checked CAS retry |
| `conflict_repair.cu` | Tier-1 GPU repair (at most 20 iterations, early exit after 3 without progress) |
| `epoch_maintenance.cu` | Row compaction, overflow drain, bitmap rebuild, and snapshot publication |

### Host pipeline (`algorithms/streamgc/src/host/`)

| File | Role |
|---|---|
| `graph_loader.cpp` | `.mtx` reader, degree counter, symmetrization |
| `partitioner.cpp`  | Degree-aware vertex cut and ghost set computation |
| `stream_gen.cpp`   | Seeded generation of the initial graph and the update stream |
| `feeder_thread.cpp`| Host-to-GPU dispatch in batches of 4,096 updates, and the Tier-2 CPU repair |
| `query_server.cpp` | Point queries served from the published snapshot |

---

## Code map

A single edge update travels through the system in this order. Read the code in this order to follow the hot path.

1. **Stream generation (`host/stream_gen.cpp`).** Edge updates `(u, v, op)` are produced from a fixed seed, with `op` set to `INSERT` or `DELETE` according to `--insert-pct`. The stream is built before the run starts, so the GPU never waits on the host random number generator.

2. **Dispatch (`host/feeder_thread.cpp`).** Updates are copied to the GPU in batches of 4,096. On the GPU, the streaming kernel assigns one warp to each update, so no kernel is launched per event. An update becomes visible to queries after its batch and the next epoch boundary.

3. **Adjacency update (`kernels/local_update.cu`).** Each vertex row has max(4 · init_deg(v), 8) slots, allocated at load time. An insertion claims a free slot in each endpoint's row with one `atomicCAS`, and a deletion marks its slot invalid until the next compaction. If a row is full, the edge goes to a per-partition overflow buffer of 2¹⁸ entries and is applied at the next epoch boundary. If that buffer is also full, the edge is discarded and stays invisible to repair. The initial graph never overflows.

4. **Conflict detection (`kernels/local_update.cu`).** The warp reads both endpoints' packed states with one aligned 32-bit load each, and all 32 lanes evaluate the same-color predicate in one `__ballot_sync`. Detecting a conflict costs one vote inside the update, not a separate pass.

5. **Boundary updates (`kernels/boundary_update.cu`, `versioned_pull.cu`).** If an edge crosses partitions, the remote endpoint is read from its ghost copy. Ghosts are refreshed after each streaming pass with a single `ld.acquire.sys.u32` across NVLink. The load cannot tear, and the version field identifies stale ghosts without a CAS loop.

6. **Recolor (`priority.cuh`, `color_select.cuh`, `recolor.cu`).** The frozen priority π(v) = (init_deg(v) << 32) | id(v) picks the lower-priority endpoint, and every warp computes it identically, so concurrent warps agree without communicating. The loser picks a new color. A vertex with init_deg(v) ≤ 64 uses a 64-bit bitmap of neighbor colors and takes the lowest free color with `__ffsll`. Higher-degree vertices keep exact counts over 1,024 colors. The recolor stores `pack(c', version + 1)`, fences, and re-reads. If another warp intervened, it recomputes and retries with a version-checked `atomicCAS`.

7. **Epoch maintenance (`kernels/epoch_maintenance.cu`).** Every E updates (default 32,768), the epoch manager pauses the stream. It compacts every row, drains the overflow buffer into the compacted rows, and rebuilds the bitmaps, so the adjacency is consistent before repair begins.

8. **Tier-1 repair (`kernels/conflict_repair.cu`).** A parallel GPU sweep recolors the lower-priority endpoint of every conflicting edge, using the same frozen-priority rule. It stops when no conflicts remain, after 20 iterations, or after 3 iterations without progress.

9. **Tier-2 repair (`host/feeder_thread.cpp`).** If Tier 1 leaves any conflicts, the adjacency is copied to the host and a sequential greedy pass recolors in priority order. This always terminates with a proper coloring (Proposition 1). It copies 16 bytes per stored edge, so it runs only when Tier 1 fails.

10. **Snapshot publication (`kernels/epoch_maintenance.cu`).** After repair, the coloring is copied into the unified-memory snapshot (2 bytes per vertex) and published behind a memory fence, and the stream resumes. Queries keep reading the previous snapshot until the new one is published.

11. **Query path (`host/query_server.cpp`).** A point query is one read of the published snapshot. It never enters the update path, never blocks an update, and runs in O(1).

12. **Output (`streamgc.cu`).** The final coloring is written to `<output-dir>/streamgc_stream<N>_<pct>pct.coloring` (a binary vertex-to-color array) and `.coloring.meta`, a JSON file with `init_ms`, `stream_ms`, `repair_ms`, `bench_ms`, `throughput_mups`, `effective_throughput_mups`, `query_latency_us`, and per-kernel breakdowns. `scripts/collect_results.sh` collects these into one CSV.

---

## Multi-GPU

Graphs that exceed one GPU's memory are partitioned across NVLink-connected GPUs with a degree-aware vertex cut (`partitioner.cpp`). Vertices are placed in descending order of degree, each on the GPU holding the most of its already-placed neighbors, minus λ = 0.1 times that GPU's vertex count to keep partitions balanced. Every remote neighbor is mirrored as a read-only ghost copy. Cross-partition conflicts are fixed by the normal epoch repair. Ghosts refresh once per epoch, so boundary decisions lag by at most one epoch, the same bound as queries.

## Known limitations

- Rows are sized at load time. Streams that add many edges to low-degree vertices, such as dense growth, can fill the overflow buffer, and those edges are discarded.
- Priorities are frozen at load time. A vertex that grows into a hub keeps its low priority.
- Slot scans run on one lane, so warps that draw high-degree vertices do more work.
- Partitions are not rebalanced while the stream runs.
