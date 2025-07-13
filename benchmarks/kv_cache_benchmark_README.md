# KV-Cache Redundancy Analysis for vLLM

This directory contains benchmarking scripts to analyze page-level redundancy in vLLM's KV-cache implementation. The goal is to identify compression opportunities that could reduce memory usage while maintaining model performance.

## Scripts Overview

### 1. `kv_cache_analysis.py`
Main analysis script that runs experiments and generates similarity metrics between KV-cache pages.

**Features:**
- Computes multiple similarity metrics (cosine, L2, Hamming, Jaccard)
- Identifies compression opportunities based on similarity thresholds
- Generates visualizations and detailed reports
- Supports multiple models and custom prompts

### 2. `kv_cache_page_monitor.py`
Real-time monitoring module that hooks into vLLM's cache operations to capture page-level data.

**Features:**
- Monkey-patches vLLM's cache operations
- Captures cache snapshots during inference
- Performs real-time similarity analysis
- Runs periodic background analysis

### 3. `run_kv_experiments.py`
Comprehensive experiment runner that coordinates multiple experiments with different configurations.

**Features:**
- Predefined experiment configurations (general, repetitive, code generation)
- Tracks memory usage and performance metrics
- Generates aggregate reports across experiments
- Supports custom experiment configurations

## Installation Requirements

```bash
# Install vLLM (if not already installed)
pip install vllm

# Install additional dependencies
pip install torch numpy matplotlib seaborn transformers

# For better performance monitoring
pip install psutil gputil
```

## Usage

### Quick Test
Run a minimal test to verify everything works:
```bash
python run_kv_experiments.py --quick-test --model "facebook/opt-125m"
```

### Full Experiments
Run comprehensive experiments with default configurations:
```bash
python run_kv_experiments.py --output-dir results/full_run --model "meta-llama/Llama-2-7b-hf"
```

### Custom Analysis
Run analysis with custom prompts:
```bash
python kv_cache_analysis.py \
  --model "meta-llama/Llama-2-7b-hf" \
  --prompts-file prompts.txt \
  --max-tokens 512 \
  --output-dir results/custom_analysis
```

### Experiment Configuration
Create a custom experiment configuration file (`experiments.json`):
```json
[
  {
    "name": "custom_experiment",
    "model": "meta-llama/Llama-2-7b-hf",
    "prompts": ["Your custom prompt 1", "Your custom prompt 2"],
    "max_tokens": 256,
    "temperature": 0.7,
    "max_model_len": 2048
  }
]
```

Then run:
```bash
python run_kv_experiments.py --experiments experiments.json
```

## Output Files

The scripts generate several output files:

1. **Analysis Reports** (`analysis_report.json`):
   - Similarity statistics per layer
   - Compression opportunities
   - Memory saving estimates

2. **Visualizations**:
   - `similarity_distribution.png`: Distribution of page similarities
   - `layer_similarity.png`: Layer-wise similarity patterns
   - `compression_opportunities.png`: Potential memory savings by layer

3. **Monitoring Data** (`monitoring_data/`):
   - Real-time cache snapshots
   - Access pattern statistics
   - Temporal analysis results

4. **Experiment Results** (`experiment_results/`):
   - Individual experiment outputs
   - Aggregate statistics
   - Final comprehensive report

## Key Findings to Look For

1. **High Similarity Regions**: Pages with >0.9 cosine similarity are candidates for compression
2. **Layer Patterns**: Some layers may show more redundancy than others
3. **Content-Dependent Redundancy**: Repetitive prompts should show higher redundancy
4. **Temporal Patterns**: Similar pages accessed at different times can be deduplicated

## Memory Estimation

The scripts estimate potential memory savings based on:
- Redundancy ratio: Percentage of highly similar pages
- Compression potential: Assumes 50% size reduction for similar pages
- Total cache size: Based on model configuration and sequence length

## Limitations and Notes

1. **Monitoring Hooks**: The page monitor uses monkey-patching which may not capture all cache operations in newer vLLM versions
2. **Synthetic Analysis**: Without actual cache data access, some metrics are estimated
3. **GPU Memory**: Experiments require significant GPU memory for larger models
4. **Performance Impact**: Monitoring adds overhead to inference time

## Next Steps

Based on the analysis results, you can:
1. Design compression algorithms targeting high-redundancy patterns
2. Implement page deduplication mechanisms
3. Develop adaptive compression based on layer-specific patterns
4. Create memory-efficient cache management strategies

## Troubleshooting

- **Import Errors**: Ensure vLLM is properly installed with CUDA support
- **Memory Errors**: Reduce `max_model_len` or use smaller models
- **Hook Failures**: Check vLLM version compatibility with monitoring hooks
- **No Results**: Verify model name and authentication for gated models