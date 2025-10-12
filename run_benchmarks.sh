#!/bin/bash

# Script to run ΨQRH benchmarks
echo "🚀 Running ΨQRH Benchmarks"
echo "=========================="

# Set default values
MODEL=${MODEL:-both}
DEVICE=${DEVICE:-cuda}
EPOCHS=${EPOCHS:-3}
SEQ_LEN=${SEQ_LEN:-512}
OUTPUT_DIR=${OUTPUT_DIR:-benchmark_results}

echo "Configuration:"
echo "  Model: $MODEL"
echo "  Device: $DEVICE"
echo "  Epochs: $EPOCHS"
echo "  Sequence Length: $SEQ_LEN"
echo "  Output Directory: $OUTPUT_DIR"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run benchmark
echo "Starting benchmark execution..."
python generate_benchmark_data.py \
    --model "$MODEL" \
    --device "$DEVICE" \
    --epochs "$EPOCHS" \
    --seq-len "$SEQ_LEN" \
    --output-dir "$OUTPUT_DIR" \
    --generate-tables

echo ""
echo "✅ Benchmark completed!"
echo "📁 Results saved to: $OUTPUT_DIR"
echo ""
echo "Generated files:"
ls -la "$OUTPUT_DIR"/*.json "$OUTPUT_DIR"/*.tex 2>/dev/null || echo "No result files found"

echo ""
echo "To view results:"
echo "  cat $OUTPUT_DIR/benchmark_summary.json | jq ."
echo ""
echo "To use tables in LaTeX paper:"
echo "  cp $OUTPUT_DIR/*.tex paper/"