#!/bin/bash
# Script to run the sampling optimization benchmark on GPU server

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate hpca25_ljh

# Navigate to vllm directory
cd ~/ljh/vllm

# Run the benchmark
echo "Running sampling optimization benchmark..."
python benchmark_sampling_optimization.py