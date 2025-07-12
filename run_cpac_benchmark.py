#!/usr/bin/env python
"""
Run CPAC compression benchmarks
"""

import sys
import os
import torch
import time
import numpy as np
import pandas as pd
from typing import Dict, List

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Check if CUDA is available
if not torch.cuda.is_available():
    print("Error: CUDA is not available. CPAC benchmarks require GPU.")
    sys.exit(1)

print("=== CPAC Compression Benchmark ===")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print()

# Import CPAC components
try:
    from vllm.attention.ops.cpac_compression import CPACManager, CPACConfig
    from vllm.attention.ops.cpac_ops import (
        compute_page_features, compute_similarity, 
        compress_delta, decompress_delta, HAS_CPAC_OPS
    )
    print(f"CPAC CUDA ops available: {HAS_CPAC_OPS}")
except ImportError as e:
    print(f"Error importing CPAC components: {e}")
    print("Make sure CPAC is properly built and installed.")
    sys.exit(1)

def benchmark_compression_operations():
    """Benchmark individual CPAC operations"""
    print("\n--- Benchmarking Individual Operations ---")
    
    # Test parameters
    block_size = 16
    num_heads = 32
    head_size = 128
    num_blocks = 100
    
    # Generate test data
    torch.manual_seed(42)
    key_cache = torch.randn(
        num_blocks, num_heads, head_size // 16, block_size, 16,
        dtype=torch.float16, device='cuda'
    )
    
    # Reshape for operations
    test_block = key_cache[0].view(num_heads, head_size, block_size)
    
    # Warmup
    for _ in range(10):
        _ = compute_page_features(test_block)
    
    torch.cuda.synchronize()
    
    # 1. Feature extraction benchmark
    num_iterations = 100
    torch.cuda.synchronize()
    start = time.time()
    
    for i in range(num_iterations):
        features = compute_page_features(key_cache[i % num_blocks].view(num_heads, head_size, block_size))
    
    torch.cuda.synchronize()
    feature_time = (time.time() - start) / num_iterations * 1000
    print(f"Feature extraction: {feature_time:.3f} ms/block")
    
    # 2. Similarity computation benchmark
    feat1 = compute_page_features(key_cache[0].view(num_heads, head_size, block_size))
    feat2 = compute_page_features(key_cache[1].view(num_heads, head_size, block_size))
    
    torch.cuda.synchronize()
    start = time.time()
    
    for _ in range(num_iterations):
        sim = compute_similarity(feat1, feat2)
    
    torch.cuda.synchronize()
    similarity_time = (time.time() - start) / num_iterations * 1000
    print(f"Similarity computation: {similarity_time:.3f} ms/pair")
    
    # 3. Delta compression benchmark
    base_block = key_cache[0]
    target_block = key_cache[1]
    
    torch.cuda.synchronize()
    start = time.time()
    
    for i in range(num_iterations):
        quantized, scale = compress_delta(
            key_cache[i % num_blocks], 
            key_cache[(i+1) % num_blocks]
        )
    
    torch.cuda.synchronize()
    compress_time = (time.time() - start) / num_iterations * 1000
    print(f"Delta compression: {compress_time:.3f} ms/block")
    
    # 4. Decompression benchmark
    quantized, scale = compress_delta(target_block, base_block)
    
    torch.cuda.synchronize()
    start = time.time()
    
    for _ in range(num_iterations):
        decompressed = decompress_delta(quantized, base_block, scale.item())
    
    torch.cuda.synchronize()
    decompress_time = (time.time() - start) / num_iterations * 1000
    print(f"Delta decompression: {decompress_time:.3f} ms/block")
    
    return {
        'feature_extraction_ms': feature_time,
        'similarity_computation_ms': similarity_time,
        'compression_ms': compress_time,
        'decompression_ms': decompress_time
    }

def benchmark_compression_ratios():
    """Benchmark compression ratios for different similarity levels"""
    print("\n--- Benchmarking Compression Ratios ---")
    
    block_size = 16
    num_heads = 32
    head_size = 128
    
    # Generate base block
    torch.manual_seed(42)
    base_block = torch.randn(num_heads, head_size // 16, block_size, 16,
                            dtype=torch.float16, device='cuda')
    
    results = []
    
    # Test different noise levels (similarity levels)
    noise_levels = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    
    for noise_level in noise_levels:
        # Create similar block with controlled noise
        similar_block = base_block + torch.randn_like(base_block) * noise_level
        
        # Compute similarity
        feat1 = compute_page_features(base_block.view(num_heads, head_size, block_size))
        feat2 = compute_page_features(similar_block.view(num_heads, head_size, block_size))
        similarity = compute_similarity(feat1, feat2).item()
        
        # Compress
        quantized, scale = compress_delta(similar_block, base_block, delta_bits=8)
        
        # Calculate compression ratio
        original_size = similar_block.numel() * similar_block.element_size()
        compressed_size = quantized.numel() + 4  # +4 bytes for scale
        compression_ratio = original_size / compressed_size
        
        # Measure reconstruction error
        decompressed = decompress_delta(quantized, base_block, scale.item())
        mse = torch.mean((similar_block - decompressed) ** 2).item()
        relative_error = torch.norm(similar_block - decompressed) / torch.norm(similar_block)
        
        results.append({
            'noise_level': noise_level,
            'similarity': similarity,
            'compression_ratio': compression_ratio,
            'mse': mse,
            'relative_error': relative_error.item()
        })
        
        print(f"Noise: {noise_level:.2f}, Similarity: {similarity:.3f}, "
              f"Ratio: {compression_ratio:.2f}x, Error: {relative_error:.4f}")
    
    return pd.DataFrame(results)

def benchmark_cpac_manager():
    """Benchmark full CPAC manager with realistic workload"""
    print("\n--- Benchmarking CPAC Manager ---")
    
    # Configuration
    block_size = 16
    num_heads = 32
    head_size = 128
    num_blocks = 1000  # Larger scale test
    
    config = CPACConfig(
        similarity_threshold=0.85,
        compression_level=2,
        adaptive_compression=True,
    )
    
    manager = CPACManager(
        block_size=block_size,
        num_gpu_blocks=num_blocks,
        num_kv_heads=num_heads,
        head_size=head_size,
        config=config,
    )
    
    # Generate cache with patterns
    torch.manual_seed(42)
    key_cache = torch.randn(
        num_blocks, num_heads, head_size // 16, block_size, 16,
        dtype=torch.float16, device='cuda'
    )
    value_cache = torch.randn(
        num_blocks, num_heads, head_size, block_size,
        dtype=torch.float16, device='cuda'
    )
    
    # Create groups of similar blocks (simulate real patterns)
    print("Creating similar block groups...")
    for group in range(num_blocks // 10):
        base_idx = group * 10
        base_key = key_cache[base_idx]
        base_value = value_cache[base_idx]
        
        # Create variations
        for offset in range(1, 5):
            idx = base_idx + offset
            if idx < num_blocks:
                noise_scale = 0.1 * offset
                key_cache[idx] = base_key + torch.randn_like(base_key) * noise_scale
                value_cache[idx] = base_value + torch.randn_like(base_value) * noise_scale
    
    # Benchmark compression
    print("Running compression benchmark...")
    compressed_count = 0
    total_time = 0
    
    torch.cuda.synchronize()
    start_time = time.time()
    
    for block_idx in range(min(500, num_blocks)):
        if block_idx % 100 == 0:
            print(f"  Processing block {block_idx}/{min(500, num_blocks)}...")
        
        block_start = time.time()
        success = manager.compress_block(
            block_idx,
            key_cache[block_idx],
            value_cache[block_idx]
        )
        block_time = time.time() - block_start
        total_time += block_time
        
        if success:
            compressed_count += 1
    
    torch.cuda.synchronize()
    total_time = time.time() - start_time
    
    # Get statistics
    stats = manager.get_compression_stats()
    
    print(f"\nResults:")
    print(f"  Blocks processed: {min(500, num_blocks)}")
    print(f"  Blocks compressed: {compressed_count}")
    print(f"  Compression success rate: {compressed_count/min(500, num_blocks)*100:.1f}%")
    print(f"  Average compression ratio: {stats['average_ratio']:.2f}x")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Average time per block: {total_time/min(500, num_blocks)*1000:.2f}ms")
    
    # Benchmark decompression
    print("\nBenchmarking decompression...")
    decompress_times = []
    
    for block_idx in list(manager.compressed_blocks)[:100]:
        temp_key = torch.zeros_like(key_cache[block_idx])
        temp_value = torch.zeros_like(value_cache[block_idx])
        
        torch.cuda.synchronize()
        start = time.time()
        manager.decompress_block(block_idx, temp_key, temp_value)
        torch.cuda.synchronize()
        decompress_time = (time.time() - start) * 1000
        decompress_times.append(decompress_time)
    
    if decompress_times:
        print(f"  Average decompression time: {np.mean(decompress_times):.2f}ms")
        print(f"  Decompression throughput: {1000/np.mean(decompress_times):.1f} blocks/s")
    
    return stats

def compare_with_baseline():
    """Compare CPAC with simple quantization baseline"""
    print("\n--- Comparing with Baseline Methods ---")
    
    block_size = 16
    num_heads = 32
    head_size = 128
    num_blocks = 100
    
    # Generate test data
    torch.manual_seed(42)
    key_cache = torch.randn(
        num_blocks, num_heads, head_size // 16, block_size, 16,
        dtype=torch.float16, device='cuda'
    )
    
    results = []
    
    # 1. CPAC compression
    print("Testing CPAC...")
    manager = CPACManager(
        block_size=block_size,
        num_gpu_blocks=num_blocks,
        num_kv_heads=num_heads,
        head_size=head_size,
    )
    
    cpac_compressed = 0
    torch.cuda.synchronize()
    cpac_start = time.time()
    
    for i in range(num_blocks):
        if manager.compress_block(i, key_cache[i], key_cache[i]):
            cpac_compressed += 1
    
    torch.cuda.synchronize()
    cpac_time = time.time() - cpac_start
    
    cpac_stats = manager.get_compression_stats()
    results.append({
        'method': 'CPAC',
        'compression_ratio': cpac_stats['average_ratio'],
        'blocks_compressed': cpac_compressed,
        'time_ms': cpac_time * 1000,
        'throughput_blocks_per_sec': num_blocks / cpac_time
    })
    
    # 2. Simple INT8 quantization baseline
    print("Testing INT8 quantization...")
    torch.cuda.synchronize()
    int8_start = time.time()
    
    for i in range(num_blocks):
        scale = key_cache[i].abs().max() / 127
        quantized = (key_cache[i] / scale).to(torch.int8)
    
    torch.cuda.synchronize()
    int8_time = time.time() - int8_start
    
    results.append({
        'method': 'INT8 Quantization',
        'compression_ratio': 2.0,  # FP16 to INT8
        'blocks_compressed': num_blocks,
        'time_ms': int8_time * 1000,
        'throughput_blocks_per_sec': num_blocks / int8_time
    })
    
    # Print comparison
    df = pd.DataFrame(results)
    print("\nComparison Results:")
    print(df.to_string(index=False))
    
    return df

def main():
    """Run all benchmarks"""
    
    # 1. Individual operations
    op_results = benchmark_compression_operations()
    
    # 2. Compression ratios
    ratio_results = benchmark_compression_ratios()
    
    # 3. Full CPAC manager
    manager_results = benchmark_cpac_manager()
    
    # 4. Comparison with baselines
    comparison_results = compare_with_baseline()
    
    # Summary
    print("\n=== BENCHMARK SUMMARY ===")
    print(f"Feature extraction: {op_results['feature_extraction_ms']:.2f} ms/block")
    print(f"Compression: {op_results['compression_ms']:.2f} ms/block")
    print(f"Decompression: {op_results['decompression_ms']:.2f} ms/block")
    print(f"Best compression ratio: {ratio_results['compression_ratio'].max():.2f}x")
    print(f"CPAC vs INT8 speedup: {comparison_results.iloc[1]['time_ms'] / comparison_results.iloc[0]['time_ms']:.2f}x")
    
    # Save results
    print("\nSaving detailed results to cpac_benchmark_results.csv...")
    ratio_results.to_csv('cpac_benchmark_results.csv', index=False)
    print("Done!")

if __name__ == "__main__":
    main()