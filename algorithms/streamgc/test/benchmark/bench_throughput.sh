#!/bin/bash
# Throughput benchmark - runs StreamGC with various stream sizes
# Usage: ./bench_throughput.sh <graph_path> [hw_tag]

GRAPH=${1:?Usage: $0 <graph.mtx> [hw_tag]}
HW_TAG=${2:-$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' '_')}
BINARY=./streaming_gc

echo "# StreamGC Throughput Benchmark"
echo "# graph: $GRAPH"
echo "# hw_tag: $HW_TAG"
echo "stream_size,insert_pct,raw_throughput_mups,effective_throughput_mups,init_ms,bench_ms,repair_ms,colors,ratio"

for STREAM in 1000 10000 100000 1000000; do
    for PCT in 20 50 80; do
        OUTPUT=$($BINARY --graph "$GRAPH" --stream $STREAM --insert-pct $PCT --seed 42 \
            --num-gpus 1 --hw-tag "$HW_TAG" --validate 2>&1)

        # Parse output
        RAW_TP=$(echo "$OUTPUT" | grep -oP 'Throughput: \K[0-9.]+')
        EFF_TP=$(echo "$OUTPUT" | grep -oP 'effective: \K[0-9.]+')
        INIT_MS=$(echo "$OUTPUT" | grep -oP 'Init time: \K[0-9.]+')
        BENCH_MS=$(echo "$OUTPUT" | grep -oP 'Bench time: \K[0-9.]+')
        REPAIR_MS=$(echo "$OUTPUT" | grep -oP 'repair: \K[0-9.]+')
        COLORS=$(echo "$OUTPUT" | grep -oP 'Colors used: \K[0-9]+')
        RATIO=$(echo "$OUTPUT" | grep -oP 'ratio: \K[0-9.]+')

        echo "${STREAM},${PCT},${RAW_TP:-0},${EFF_TP:-0},${INIT_MS:-0},${BENCH_MS:-0},${REPAIR_MS:-0},${COLORS:-0},${RATIO:-0}"
    done
done
