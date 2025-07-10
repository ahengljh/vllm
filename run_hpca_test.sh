#!/bin/bash
set -e

echo "🚀 HPCA Test Setup and Execution"
echo "================================="

# Check if vLLM is properly installed
echo "📦 Checking vLLM installation..."
python -c "import vllm; print(f'vLLM version: {vllm.__version__}')" || {
    echo "❌ vLLM not found. Please install vLLM first."
    exit 1
}

# Install additional dependencies
echo "📦 Installing additional dependencies..."
pip install -r requirements_hpca.txt

# Check if model exists
MODEL_PATH="../Qwen3-8B"
if [ ! -d "$MODEL_PATH" ]; then
    echo "⚠️  Model not found at $MODEL_PATH"
    echo "Please update the model path in hpca.py or ensure the model is available"
fi

# Set environment variables for optimal performance
export CUDA_VISIBLE_DEVICES=0,1,2,3
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

echo "🔧 Environment setup complete"
echo "Starting HPCA compression performance test..."
echo "This may take 30-60 minutes depending on your hardware"
echo ""

# Run the test
python hpca.py

echo ""
echo "✅ HPCA test completed!"
echo "Check the generated files:"
echo "  - hpca_compression_results.json (detailed metrics)"
echo "  - hpca_compression_analysis.png (visualizations)" 