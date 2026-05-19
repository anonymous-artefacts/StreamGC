#!/bin/bash
# Run all benchmarks on available graphs
# Usage: ./run_all_benchmarks.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_DIR"

HW_TAG=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | tr ' ' '_' || echo "unknown")
RESULTS_DIR="results/${HW_TAG}/benchmarks"
mkdir -p "$RESULTS_DIR"

echo "=== StreamGC Benchmark Suite ==="
echo "Hardware: $HW_TAG"
echo "Results: $RESULTS_DIR"
echo ""

GRAPHS=(delaunay_n17 com-Amazon)

for graph in "${GRAPHS[@]}"; do
    MTX="../../data/${graph}.mtx"
    if [ ! -f "$MTX" ]; then
        echo "SKIP: $MTX not found"
        continue
    fi

    echo "--- Throughput: $graph ---"
    bash test/benchmark/bench_throughput.sh "$MTX" "$HW_TAG" | tee "$RESULTS_DIR/${graph}_throughput.csv"
    echo ""

    echo "--- Color Quality: $graph ---"
    bash test/benchmark/bench_color_count.sh "$MTX" "$HW_TAG" | tee "$RESULTS_DIR/${graph}_colors.csv"
    echo ""
done

echo "=== Benchmarks complete ==="
echo "Results in: $RESULTS_DIR"
