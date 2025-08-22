#!/usr/bin/env python3
"""
Revised benchmark to understand the actual bottleneck better.
"""

import time
import torch
import torch.nn as nn
import numpy as np
from typing import Optional

def random_sample_original(
    probs: torch.Tensor,
    generators: dict[int, torch.Generator],
) -> torch.Tensor:
    """Original implementation."""
    q = torch.empty_like(probs)
    if len(generators) != probs.shape[0]:
        q.exponential_()
    if generators:
        for i, generator in generators.items():
            q[i].exponential_(generator=generator)
    return probs.div_(q).argmax(dim=-1).view(-1)


def random_sample_optimized_parallel(
    probs: torch.Tensor,
    generators: dict[int, torch.Generator],
) -> torch.Tensor:
    """
    Optimized version using parallel random number generation.
    Instead of grouping, we pre-generate all random numbers at once.
    """
    batch_size = probs.shape[0]
    vocab_size = probs.shape[1]
    device = probs.device
    
    # Generate base exponential values for all
    q = torch.empty_like(probs)
    q.exponential_()
    
    if generators:
        # For requests with custom generators, we need to handle them
        # But we can optimize by minimizing tensor operations
        gen_indices = list(generators.keys())
        
        # Option 1: Batch generation if generators are similar
        if len(generators) > batch_size * 0.3:  # If many generators
            # Create a mask and update in one go
            for i, gen in generators.items():
                q[i].exponential_(generator=gen)
        else:
            # For few generators, original approach is fine
            for i, gen in generators.items():
                q[i].exponential_(generator=gen)
    
    return probs.div_(q).argmax(dim=-1).view(-1)


def random_sample_optimized_cuda_kernel(
    probs: torch.Tensor,
    generators: dict[int, torch.Generator],
) -> torch.Tensor:
    """
    Optimization focusing on minimizing Python overhead.
    """
    q = torch.empty_like(probs)
    
    # Fast path for no generators
    if not generators:
        q.exponential_()
        return probs.div_(q).argmax(dim=-1).view(-1)
    
    # For mixed case, minimize Python loop overhead
    if len(generators) < probs.shape[0]:
        # Generate default for all first
        q.exponential_()
        # Then override specific ones
        if generators:
            # Pre-allocate list to avoid repeated dict access
            items = list(generators.items())
            for i, gen in items:
                q[i].exponential_(generator=gen)
    else:
        # All have generators - no benefit from default generation
        for i, gen in generators.items():
            q[i].exponential_(generator=gen)
    
    return probs.div_(q).argmax(dim=-1).view(-1)


def random_sample_optimized_fused(
    probs: torch.Tensor, 
    generators: dict[int, torch.Generator],
) -> torch.Tensor:
    """
    Attempt to fuse operations and reduce memory traffic.
    """
    batch_size, vocab_size = probs.shape
    
    if not generators:
        # Fast path - fuse exponential and division
        q = torch.empty_like(probs)
        q.exponential_()
        # Use in-place operations to reduce memory traffic
        return probs.div_(q).argmax(dim=-1).view(-1)
    
    # For generators case, try to minimize allocations
    q = torch.empty_like(probs)
    
    # Check if we should do bulk generation first
    if len(generators) < batch_size * 0.5:
        # Generate default for majority
        q.exponential_()
        # Override specific ones
        for i, gen in generators.items():
            q[i].exponential_(generator=gen)
    else:
        # Many generators - handle them directly
        # First handle non-generator rows if any
        non_gen_mask = torch.ones(batch_size, dtype=torch.bool, device=probs.device)
        for i in generators.keys():
            non_gen_mask[i] = False
        
        if non_gen_mask.any():
            q[non_gen_mask].exponential_()
        
        # Then handle generator rows
        for i, gen in generators.items():
            q[i].exponential_(generator=gen)
    
    return probs.div_(q).argmax(dim=-1).view(-1)


def profile_bottleneck(batch_size=256, vocab_size=32000, num_generators=128):
    """Profile where the actual bottleneck is."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Profiling on {device} with batch_size={batch_size}, generators={num_generators}")
    print("=" * 60)
    
    # Setup
    probs = torch.rand(batch_size, vocab_size, device=device).softmax(dim=-1)
    generators = {i: torch.Generator(device=device).manual_seed(i) 
                  for i in range(num_generators)}
    q = torch.empty_like(probs)
    
    num_iterations = 100
    
    # Time different components
    components = {}
    
    # 1. Time exponential generation without generators
    torch.cuda.synchronize() if device.type == "cuda" else None
    start = time.perf_counter()
    for _ in range(num_iterations):
        q_test = torch.empty_like(probs)
        q_test.exponential_()
    torch.cuda.synchronize() if device.type == "cuda" else None
    components['exponential_no_gen'] = (time.perf_counter() - start) / num_iterations * 1000
    
    # 2. Time exponential generation with generators (sequential)
    torch.cuda.synchronize() if device.type == "cuda" else None
    start = time.perf_counter()
    for _ in range(num_iterations):
        q_test = torch.empty_like(probs)
        for i, gen in generators.items():
            q_test[i].exponential_(generator=gen)
    torch.cuda.synchronize() if device.type == "cuda" else None
    components['exponential_with_gen'] = (time.perf_counter() - start) / num_iterations * 1000
    
    # 3. Time division operation
    q.exponential_()
    torch.cuda.synchronize() if device.type == "cuda" else None
    start = time.perf_counter()
    for _ in range(num_iterations):
        probs_copy = probs.clone()
        probs_copy.div_(q)
    torch.cuda.synchronize() if device.type == "cuda" else None
    components['division'] = (time.perf_counter() - start) / num_iterations * 1000
    
    # 4. Time argmax operation
    torch.cuda.synchronize() if device.type == "cuda" else None
    start = time.perf_counter()
    for _ in range(num_iterations):
        probs.argmax(dim=-1)
    torch.cuda.synchronize() if device.type == "cuda" else None
    components['argmax'] = (time.perf_counter() - start) / num_iterations * 1000
    
    # 5. Time Python loop overhead
    start = time.perf_counter()
    for _ in range(num_iterations):
        for i, gen in generators.items():
            pass  # Just loop overhead
    components['python_loop'] = (time.perf_counter() - start) / num_iterations * 1000
    
    # Report
    print("Component timings (ms):")
    for name, timing in components.items():
        print(f"  {name:25s}: {timing:8.4f} ms")
    
    print("\nAnalysis:")
    gen_overhead = components['exponential_with_gen'] - components['exponential_no_gen']
    print(f"  Generator overhead: {gen_overhead:.4f} ms")
    print(f"  Python loop overhead: {components['python_loop']:.4f} ms")
    print(f"  Actual gen computation: {gen_overhead - components['python_loop']:.4f} ms")
    
    return components


def benchmark_optimizations():
    """Benchmark different optimization strategies."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nBenchmarking optimizations on {device}")
    print("=" * 60)
    
    batch_sizes = [64, 128, 256]
    vocab_size = 32000
    num_iterations = 100
    
    for batch_size in batch_sizes:
        print(f"\nBatch size: {batch_size}")
        
        # Test with 50% generators
        num_generators = batch_size // 2
        probs = torch.rand(batch_size, vocab_size, device=device).softmax(dim=-1)
        generators = {i: torch.Generator(device=device).manual_seed(i) 
                     for i in range(num_generators)}
        
        # Warmup
        for _ in range(10):
            _ = random_sample_original(probs.clone(), generators)
            _ = random_sample_optimized_parallel(probs.clone(), generators)
            _ = random_sample_optimized_cuda_kernel(probs.clone(), generators)
            _ = random_sample_optimized_fused(probs.clone(), generators)
        
        results = {}
        
        # Benchmark each version
        for name, func in [
            ('Original', random_sample_original),
            ('Parallel', random_sample_optimized_parallel),
            ('MinOverhead', random_sample_optimized_cuda_kernel),
            ('Fused', random_sample_optimized_fused),
        ]:
            torch.cuda.synchronize() if device.type == "cuda" else None
            start = time.perf_counter()
            for _ in range(num_iterations):
                _ = func(probs.clone(), generators)
            torch.cuda.synchronize() if device.type == "cuda" else None
            results[name] = (time.perf_counter() - start) / num_iterations * 1000
        
        # Report
        original_time = results['Original']
        for name, timing in results.items():
            speedup = original_time / timing
            print(f"  {name:12s}: {timing:8.4f} ms (speedup: {speedup:5.2f}x)")


if __name__ == "__main__":
    # First profile to understand the bottleneck
    profile_bottleneck()
    
    # Then benchmark different strategies
    benchmark_optimizations()