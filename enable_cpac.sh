#!/bin/bash
# Script to enable CPAC in vLLM build

echo "Enabling CPAC (Cross-Page Adaptive Compression) in vLLM..."

# Check if we're in the vLLM directory
if [ ! -f "CMakeLists.txt" ] || [ ! -d "csrc" ]; then
    echo "Error: This script must be run from the vLLM root directory"
    exit 1
fi

# Apply the patch to integrate CPAC into CMakeLists.txt and torch_bindings.cpp
echo "Applying CPAC integration patch..."
if [ -f "cpac_integration.patch" ]; then
    git apply cpac_integration.patch
    echo "✓ Patch applied successfully"
else
    echo "⚠️  Warning: cpac_integration.patch not found"
    echo "   You'll need to manually add the CPAC sources to CMakeLists.txt"
fi

# Build with CPAC enabled
echo "Building vLLM with CPAC support..."
export VLLM_USE_PRECOMPILED=1

# Clean previous builds
rm -rf build dist

# Install with CPAC
uv pip install -U -e . --torch-backend=auto

# Verify CPAC is available
echo "Verifying CPAC installation..."
python -c "
try:
    import vllm._C
    if hasattr(vllm._C, 'cpac_ops'):
        print('✓ CPAC CUDA kernels loaded successfully!')
        print('  You can now use enable_cpac=True in vLLM')
    else:
        print('⚠️  Warning: CPAC ops not found in compiled extension')
        print('  Make sure CMakeLists.txt includes the CPAC sources')
except ImportError as e:
    print(f'✗ Error: Could not import vllm._C: {e}')
"

echo "Done! To use CPAC:"
echo "  llm = LLM(model='your-model', enable_cpac=True)"