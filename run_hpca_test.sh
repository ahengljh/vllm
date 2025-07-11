#!/bin/bash

# HPCA Test Runner Script
# This script sets up the environment and runs comprehensive KV cache compression tests

set -e  # Exit on any error

echo "🚀 HPCA: High-Performance Cache Analysis Test Suite"
echo "=================================================="

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check Python version
echo "📋 Checking Python environment..."
if ! command_exists python3; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "✅ Python version: $PYTHON_VERSION"

# Check if pip is available
if ! command_exists pip3; then
    echo "❌ pip3 is required but not installed."
    exit 1
fi

# Install required dependencies
echo ""
echo "📦 Installing required dependencies..."

# Core dependencies
CORE_DEPS=(
    "torch"
    "numpy"
    "pandas"
    "matplotlib"
    "seaborn" 
    "psutil"
    "argparse"
)

# Optional dependencies (for full functionality)
OPTIONAL_DEPS=(
    "vllm"
    "transformers"
    "accelerate"
)

echo "Installing core dependencies..."
for dep in "${CORE_DEPS[@]}"; do
    echo "  Installing $dep..."
    pip3 install "$dep" --quiet || echo "⚠️  Warning: Failed to install $dep"
done

echo ""
echo "Installing optional dependencies (may take longer)..."
for dep in "${OPTIONAL_DEPS[@]}"; do
    echo "  Installing $dep..."
    pip3 install "$dep" --quiet || echo "⚠️  Warning: Failed to install $dep (tests will run with mock data)"
done

# Check GPU availability
echo ""
echo "🔍 Checking GPU availability..."
if command_exists nvidia-smi; then
    echo "✅ NVIDIA GPU detected:"
    nvidia-smi --query-gpu=gpu_name,memory.total,memory.free --format=csv,noheader,nounits | head -1
    GPU_AVAILABLE=true
else
    echo "⚠️  No NVIDIA GPU detected. Tests will run on CPU (slower)."
    GPU_AVAILABLE=false
fi

# Set environment variables
echo ""
echo "⚙️  Setting up environment..."
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export CUDA_VISIBLE_DEVICES=0  # Use first GPU if available

# Parse command line arguments
MODELS=("microsoft/DialoGPT-medium")  # Default lightweight model
OUTPUT_DIR="hpca_results"
QUICK_MODE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --models)
            shift
            MODELS=()
            while [[ $# -gt 0 && ! $1 =~ ^-- ]]; do
                MODELS+=("$1")
                shift
            done
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --quick)
            QUICK_MODE=true
            shift
            ;;
        --large-model)
            MODELS=("meta-llama/Llama-2-7b-hf")
            echo "🔥 Using large model: ${MODELS[0]}"
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --models MODEL1 MODEL2 ...    Specify models to test"
            echo "  --output-dir DIR              Output directory (default: hpca_results)"
            echo "  --quick                       Run quick test with fewer configurations"
            echo "  --large-model                 Test with Llama-2-7b instead of DialoGPT"
            echo "  --help                        Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --quick                                    # Quick test"
            echo "  $0 --models gpt2 microsoft/DialoGPT-medium   # Test multiple models"
            echo "  $0 --large-model                             # Test with Llama-2-7b"
            echo "  $0 --output-dir my_results                   # Custom output directory"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Display test configuration
echo ""
echo "🧪 Test Configuration:"
echo "   Models: ${MODELS[*]}"
echo "   Output: $OUTPUT_DIR"
echo "   Quick mode: $QUICK_MODE"
echo "   GPU available: $GPU_AVAILABLE"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Build command arguments
PYTHON_ARGS=("--output-dir" "$OUTPUT_DIR")

if [ ${#MODELS[@]} -gt 0 ]; then
    PYTHON_ARGS+=("--models")
    PYTHON_ARGS+=("${MODELS[@]}")
fi

if [ "$QUICK_MODE" = true ]; then
    PYTHON_ARGS+=("--quick")
fi

# Run the test
echo ""
echo "🚀 Starting HPCA compression tests..."
echo "=================================================="

# Add timestamp
START_TIME=$(date)
echo "Start time: $START_TIME"

# Run Python test script
if python3 hpca.py "${PYTHON_ARGS[@]}"; then
    END_TIME=$(date)
    echo ""
    echo "=================================================="
    echo "✅ HPCA tests completed successfully!"
    echo "Start time: $START_TIME"
    echo "End time: $END_TIME"
    echo ""
    echo "📊 Results available in:"
    echo "   📁 Directory: $OUTPUT_DIR/"
    echo "   📄 Summary: $OUTPUT_DIR/summary_report.md"
    echo "   📈 Plots: $OUTPUT_DIR/plots/"
    echo "   📋 Raw data: $OUTPUT_DIR/comprehensive_results.json"
    echo ""
    echo "🔑 Quick view of results:"
    if [ -f "$OUTPUT_DIR/summary_report.md" ]; then
        echo "   $(head -20 "$OUTPUT_DIR/summary_report.md" | tail -10)"
    fi
    echo "=================================================="
else
    echo ""
    echo "❌ HPCA tests failed. Check the error messages above."
    exit 1
fi

# Optional: Open results if on macOS
if [[ "$OSTYPE" == "darwin"* ]] && command_exists open; then
    echo ""
    read -p "📖 Open results directory? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        open "$OUTPUT_DIR"
    fi
fi