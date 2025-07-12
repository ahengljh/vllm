# Building and Using CPAC with vLLM

## Overview
CPAC (Cross-Page Adaptive Compression) is a novel KV cache compression method that exploits cross-page similarities in vLLM's PagedAttention architecture.

## Building CPAC

### Method 1: Using the Integration Script (Recommended)

```bash
# From vLLM root directory
./enable_cpac.sh
```

This script will:
1. Apply the necessary patches to CMakeLists.txt
2. Build vLLM with CPAC support
3. Verify the installation

### Method 2: Manual Integration

1. **Add CPAC sources to CMakeLists.txt**:
   
   Edit `CMakeLists.txt` and add the CPAC source files:
   ```cmake
   set(VLLM_EXT_SRC
     # ... existing sources ...
     "csrc/cpac_kernels.cu"
     "csrc/cpac_ops.cpp"
     # ... rest of sources ...
   )
   ```

2. **Update torch_bindings.cpp**:
   
   Add the CPAC registration:
   ```cpp
   // At the top, add:
   void register_cpac_ops(py::module& m);
   
   // In PYBIND11_MODULE, add:
   #ifdef VLLM_CPAC_ENABLED
     register_cpac_ops(m);
   #endif
   ```

3. **Build vLLM**:
   ```bash
   VLLM_USE_PRECOMPILED=1 uv pip install -U -e . --torch-backend=auto
   ```

### Method 3: Using Environment Variables

Set CPAC to be included during build:
```bash
export VLLM_CPAC_ENABLED=1
VLLM_USE_PRECOMPILED=1 uv pip install -U -e . --torch-backend=auto
```

## Using CPAC

### Basic Usage

```python
from vllm import LLM, SamplingParams

# Enable CPAC via constructor
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    enable_cpac=True,
    cpac_config={
        "similarity_threshold": 0.85,
        "compression_level": 2,
        "adaptive_compression": True,
    }
)
```

### Environment Variable Configuration

```bash
# Enable CPAC
export VLLM_CPAC_ENABLED=true

# Configure CPAC parameters
export VLLM_CPAC_SIMILARITY_THRESHOLD=0.85
export VLLM_CPAC_COMPRESSION_LEVEL=2
export VLLM_CPAC_ADAPTIVE=true
export VLLM_CPAC_DELTA_BITS=8

# Run vLLM - CPAC will be automatically enabled
python your_script.py
```

### Advanced Configuration

```python
from vllm.attention.ops.cpac_compression import CPACConfig

# Create custom configuration
cpac_config = CPACConfig(
    similarity_threshold=0.85,      # Pages must be 85% similar
    max_cluster_size=8,            # Max pages per cluster
    min_compression_ratio=2.0,      # Minimum compression benefit
    top_k_similar=4,               # Check top 4 similar pages
    adaptive_compression=True,      # Adapt to memory pressure
    compression_level=2,           # 1: light, 2: moderate, 3: aggressive
    use_importance_weighting=True,  # Weight by attention importance
    delta_bits=8,                  # Quantization bits
)

llm = LLM(
    model="your-model",
    enable_cpac=True,
    cpac_config=cpac_config
)
```

## Verifying Installation

```python
# Check if CPAC is available
import vllm._C

if hasattr(vllm._C, 'cpac_ops'):
    print("✓ CPAC is available!")
    
    # Test basic functionality
    import torch
    from vllm.attention.ops.cpac_ops import cpac_kernel
    
    # Create test data
    test_page = torch.randn(32, 128, 16, device='cuda', dtype=torch.float16)
    
    # Test feature extraction
    features = cpac_kernel.compute_features(test_page)
    print(f"Feature shape: {features.shape}")
else:
    print("✗ CPAC not found in build")
```

## Monitoring CPAC Performance

```python
# Get compression statistics
if hasattr(llm.llm_engine.scheduler, 'block_manager'):
    stats = llm.llm_engine.scheduler.block_manager.get_compression_stats()
    print(f"Compressed blocks: {stats['compressed_blocks']}")
    print(f"Compression ratio: {stats['compression_ratio']:.2f}x")
    print(f"Memory saved: {stats['memory_saved_mb']:.1f} MB")
```

## Troubleshooting

### CPAC not available after build
1. Check that the CUDA files are in `csrc/`:
   ```bash
   ls csrc/cpac_*
   ```

2. Verify CMakeLists.txt includes CPAC sources:
   ```bash
   grep -n "cpac" CMakeLists.txt
   ```

3. Check build logs for CPAC compilation:
   ```bash
   pip install -v -e . 2>&1 | grep -i cpac
   ```

### Performance issues
- Reduce `top_k_similar` to check fewer pages
- Increase `min_compression_ratio` threshold
- Disable `use_importance_weighting` for faster decisions
- Use environment variable `VLLM_CPAC_COMPRESSION_LEVEL=1` for lighter compression

### Memory not being saved
- Check memory pressure with `nvidia-smi`
- CPAC may not compress if memory is abundant
- Force compression with `VLLM_CPAC_COMPRESSION_LEVEL=3`

## Benchmarking CPAC

Run the benchmark suite:
```bash
cd tests/attention
python test_cpac_compression.py

cd ../../benchmarks
python cpac_benchmark.py
```

This will generate:
- `cpac_compression_report.png` - Detailed performance metrics
- `cpac_comparison.png` - Comparison with other methods
- `cpac_benchmark_results.csv` - Raw benchmark data

## Development

To modify CPAC:

1. Edit source files:
   - Core logic: `vllm/attention/ops/cpac_compression.py`
   - CUDA kernels: `csrc/cpac_kernels.cu`
   - Python bindings: `csrc/cpac_ops.cpp`

2. Rebuild:
   ```bash
   VLLM_USE_PRECOMPILED=1 uv pip install -U -e . --torch-backend=auto
   ```

3. Test changes:
   ```bash
   pytest tests/attention/test_cpac_compression.py::TestCPACCompression::test_delta_compression_accuracy -v
   ```

## Support

For issues or questions:
1. Check the technical paper: `docs/cpac_technical_paper.md`
2. See the user guide: `vllm/attention/ops/cpac_README.md`
3. Open an issue with the `cpac` label