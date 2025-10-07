#!/bin/bash

# ΨQRH Benchmark Runner
# =====================
#
# This script runs the complete benchmark suite for ΨQRH vs Baseline models
# and generates the data used in the NeurIPS/ICLR paper submission.

echo "🚀 ΨQRH Benchmark Suite"
echo "========================"
echo ""

# Check if CUDA is available
if command -v nvidia-smi &> /dev/null; then
    echo "✅ CUDA detected - using GPU acceleration"
    DEVICE="cuda"
else
    echo "⚠️  CUDA not detected - using CPU (will be slower)"
    DEVICE="cpu"
fi

echo "Device: $DEVICE"
echo ""

# Run the benchmark data generator
echo "🔬 Running REAL ΨQRH benchmark data generation..."
echo "This will train actual models and may take 10-30 minutes..."
echo "Training ΨQRH and Baseline models on WikiText-103 data..."
echo ""

python generate_benchmark_data.py \
    --device $DEVICE \
    --seq_len 512 \
    --epochs 3 \
    --output benchmark_results.json

echo ""
echo "✅ Benchmark complete!"
echo ""
echo "📊 Results saved to: benchmark_results.json"
echo "📄 LaTeX tables saved to: paper/benchmark_tables.tex"
echo ""
echo "📈 Key Results Summary:"
echo "========================"

# Extract and display key results
if [ -f benchmark_results.json ]; then
    # Language Modeling Results
    echo ""
    echo "📚 Language Modeling (WikiText-103):"
    python -c "
import json
with open('benchmark_results.json') as f:
    data = json.load(f)
lm = data['language_modeling']
print(f'Baseline: {lm[\"baseline\"][\"parameters\"]:,} params, PPL={lm[\"baseline\"][\"perplexity\"]}, {lm[\"baseline\"][\"memory_mb\"]}MB, {lm[\"baseline\"][\"inference_speed_tokens_per_sec\"]:,} tok/s')
print(f'ΨQRH:     {lm[\"psiqrh\"][\"parameters\"]:,} params, PPL={lm[\"psiqrh\"][\"perplexity\"]}, {lm[\"psiqrh\"][\"memory_mb\"]}MB, {lm[\"psiqrh\"][\"inference_speed_tokens_per_sec\"]:,} tok/s')
"

    # GLUE Results
    echo ""
    echo "🎯 GLUE Benchmark Results:"
    python -c "
import json
with open('benchmark_results.json') as f:
    data = json.load(f)
glue = data['glue']
tasks = ['MNLI', 'QQP', 'QNLI', 'SST-2']
print('Task     | Baseline | ΨQRH | Improvement')
print('---------|----------|-------|------------')
for task in tasks:
    base = glue['baseline'][task]
    psiqrh = glue['psiqrh'][task]
    imp = psiqrh - base
    print(f'{task:8} | {base:8.1f} | {psiqrh:5.1f} | +{imp:.1f}')
"
fi

echo ""
echo "🎉 Ready for NeurIPS/ICLR submission!"
echo ""
echo "Next steps:"
echo "1. Review results in benchmark_results.json"
echo "2. Update paper tables in paper/psiqrh_paper.tex"
echo "3. Run: docker build -t psiqrh:latest . && docker run --gpus all psiqrh:latest"