#!/bin/bash
# Color quality benchmark
# Usage: ./bench_color_count.sh <graph_path> [hw_tag]

GRAPH=${1:?Usage: $0 <graph.mtx> [hw_tag]}
HW_TAG=${2:-$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' '_')}
BINARY=./streaming_gc

echo "# StreamGC Color Quality Benchmark"
echo "# graph: $GRAPH"
echo "stream_size,insert_pct,streamgc_colors,greedy_colors,ratio,valid"

for STREAM in 1000 10000 100000 1000000; do
    for PCT in 20 50 80; do
        OUTPUT=$($BINARY --graph "$GRAPH" --stream $STREAM --insert-pct $PCT --seed 42 \
            --num-gpus 1 --hw-tag "$HW_TAG" \
            --output-dir /tmp/streamgc_bench --validate 2>&1)

        COLORS=$(echo "$OUTPUT" | grep -oP 'Colors used: \K[0-9]+')
        GREEDY=$(echo "$OUTPUT" | grep -oP 'Static greedy: \K[0-9]+')
        RATIO=$(echo "$OUTPUT" | grep -oP 'ratio: \K[0-9.]+')
        VALID=$(echo "$OUTPUT" | grep -c 'Coloring valid')

        echo "${STREAM},${PCT},${COLORS:-0},${GREEDY:-0},${RATIO:-0},${VALID}"
    done
done
