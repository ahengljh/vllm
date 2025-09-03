# Triton 3.4.0 Specific Configurations

This directory contains MoE kernel configurations optimized specifically for Triton version 3.4.0.

## Usage

When running vLLM with Triton 3.4.0, configurations in this directory will be automatically selected over default configurations if they exist.

## Adding Configurations

To add optimized configurations for Triton 3.4.0:

1. Ensure you have Triton 3.4.0 installed
2. Run the benchmark tool:
   ```bash
   python benchmarks/kernels/benchmark_moe.py \
       --model <your_model> \
       --output-dir vllm/model_executor/layers/fused_moe/configs/triton_3_4_0/
   ```
3. The generated JSON files will be automatically used when running with Triton 3.4.0

## File Naming Convention

Files should follow the same naming convention as the main configs directory:
`E=<num_experts>,N=<intermediate_size>,device_name=<device>[,dtype=<dtype>][,block_shape=[<n>,<k>]].json`