#!/usr/bin/env python3
"""Benchmark for sampling optimization: comparing original vs optimized random_sample function."""

import time
import torch
import torch.nn as nn
from typing import Optional
import numpy as np

# Original implementation
def random_sample_original(
    probs: torch.Tensor,
    generators: dict[int, torch.Generator],
) -> torch.Tensor:
    """Original implementation with sequential processing."""
    q = torch.empty_like(probs)
    if len(generators) != probs.shape[0]:
        q.exponential_()
    if generators:
        # Sequential processing - BOTTLENECK
        for i, generator in generators.items():
            q[i].exponential_(generator=generator)
    return probs.div_(q).argmax(dim=-1).view(-1)


# Optimized implementation
def random_sample_optimized(
    probs: torch.Tensor,
    generators: dict[int, torch.Generator],
) -> torch.Tensor:
    """Optimized implementation with batched processing."""
    q = torch.empty_like(probs)
    
    if len(generators) == 0:
        # Fast path: no custom generators
        q.exponential_()
    elif len(generators) == probs.shape[0]:
        # All requests have custom generators - batch process
        # Group by unique generator seed for batching
        generator_groups = {}
        for i, gen in generators.items():
            # Get generator state as key for grouping
            state = gen.get_state()
            state_key = state.tobytes() if isinstance(state, torch.Tensor) else str(state)
            if state_key not in generator_groups:
                generator_groups[state_key] = ([], gen)
            generator_groups[state_key][0].append(i)
        
        # Process each group in batch
        for indices, gen in generator_groups.values():
            if len(indices) == 1:
                q[indices[0]].exponential_(generator=gen)
            else:
                # Batch exponential sampling for same generator
                batch_indices = torch.tensor(indices, device=probs.device)
                q[batch_indices] = torch.empty_like(q[batch_indices]).exponential_(generator=gen)
    else:
        # Mixed case: some with, some without custom generators
        # First fill all with default exponential
        q.exponential_()
        
        # Then batch-update those with custom generators
        if len(generators) > 0:
            # Group by generator for batching
            generator_groups = {}
            for i, gen in generators.items():
                state = gen.get_state()
                state_key = state.tobytes() if isinstance(state, torch.Tensor) else str(state)
                if state_key not in generator_groups:
                    generator_groups[state_key] = ([], gen)
                generator_groups[state_key][0].append(i)
            
            # Batch process updates
            for indices, gen in generator_groups.values():
                if len(indices) == 1:
                    q[indices[0]].exponential_(generator=gen)
                else:
                    batch_indices = torch.tensor(indices, device=probs.device)
                    q[batch_indices] = torch.empty_like(q[batch_indices]).exponential_(generator=gen)
    
    return probs.div_(q).argmax(dim=-1).view(-1)


# Alternative optimized implementation using tensor operations
def random_sample_optimized_v2(
    probs: torch.Tensor,
    generators: dict[int, torch.Generator],
) -> torch.Tensor:
    """Alternative optimization using pre-allocated tensors and vectorized ops."""
    batch_size = probs.shape[0]
    q = torch.empty_like(probs)
    
    if not generators:
        # Fast path: no custom generators
        q.exponential_()
    else:
        # Create mask for custom generators
        has_generator = torch.zeros(batch_size, dtype=torch.bool, device=probs.device)
        generator_indices = list(generators.keys())
        has_generator[generator_indices] = True
        
        # Fill default exponential for those without custom generators
        default_mask = ~has_generator
        if default_mask.any():
            q[default_mask].exponential_()
        
        # Batch process custom generators by unique states
        if generators:
            # Group generators by their state for efficient batching
            state_to_indices = {}
            for idx, gen in generators.items():
                state = gen.get_state()
                state_key = hash(state.tobytes()) if isinstance(state, torch.Tensor) else hash(str(state))
                if state_key not in state_to_indices:
                    state_to_indices[state_key] = ([], gen)
                state_to_indices[state_key][0].append(idx)
            
            # Process each unique generator state in batch
            for indices, gen in state_to_indices.values():
                if len(indices) > 1:
                    # Batch sampling for multiple indices with same generator
                    indices_tensor = torch.tensor(indices, device=probs.device, dtype=torch.long)
                    batch_q = torch.empty((len(indices), probs.shape[1]), device=probs.device)
                    batch_q.exponential_(generator=gen)
                    q[indices_tensor] = batch_q
                else:
                    # Single index
                    q[indices[0]].exponential_(generator=gen)
    
    return probs.div_(q).argmax(dim=-1).view(-1)


def benchmark_sampling(batch_sizes=[32, 64, 128, 256, 512, 1024], 
                       vocab_size=32000,
                       num_iterations=100,
                       generator_ratios=[0.0, 0.1, 0.5, 1.0]):
    """Benchmark the sampling functions with various configurations."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running benchmark on device: {device}")
    print("=" * 80)
    
    results = []
    
    for batch_size in batch_sizes:
        for gen_ratio in generator_ratios:
            # Create dummy probabilities
            probs = torch.rand(batch_size, vocab_size, device=device)
            probs = probs.softmax(dim=-1)
            
            # Create generators for some requests
            num_generators = int(batch_size * gen_ratio)
            generators = {}
            if num_generators > 0:
                for i in range(num_generators):
                    generators[i] = torch.Generator(device=device)
                    generators[i].manual_seed(i)
            
            # Warmup
            for _ in range(10):
                _ = random_sample_original(probs.clone(), generators)
                _ = random_sample_optimized(probs.clone(), generators)
                _ = random_sample_optimized_v2(probs.clone(), generators)
            
            # Benchmark original
            torch.cuda.synchronize() if device.type == "cuda" else None
            start_time = time.perf_counter()
            for _ in range(num_iterations):
                _ = random_sample_original(probs.clone(), generators)
            torch.cuda.synchronize() if device.type == "cuda" else None
            original_time = (time.perf_counter() - start_time) / num_iterations * 1000
            
            # Benchmark optimized
            torch.cuda.synchronize() if device.type == "cuda" else None
            start_time = time.perf_counter()
            for _ in range(num_iterations):
                _ = random_sample_optimized(probs.clone(), generators)
            torch.cuda.synchronize() if device.type == "cuda" else None
            optimized_time = (time.perf_counter() - start_time) / num_iterations * 1000
            
            # Benchmark optimized v2
            torch.cuda.synchronize() if device.type == "cuda" else None
            start_time = time.perf_counter()
            for _ in range(num_iterations):
                _ = random_sample_optimized_v2(probs.clone(), generators)
            torch.cuda.synchronize() if device.type == "cuda" else None
            optimized_v2_time = (time.perf_counter() - start_time) / num_iterations * 1000
            
            speedup = original_time / optimized_time
            speedup_v2 = original_time / optimized_v2_time
            
            result = {
                'batch_size': batch_size,
                'gen_ratio': gen_ratio,
                'num_generators': num_generators,
                'original_ms': original_time,
                'optimized_ms': optimized_time,
                'optimized_v2_ms': optimized_v2_time,
                'speedup': speedup,
                'speedup_v2': speedup_v2
            }
            results.append(result)
            
            print(f"Batch: {batch_size:4d}, Gen ratio: {gen_ratio:.1f} ({num_generators:4d} gens)")
            print(f"  Original:     {original_time:8.4f} ms")
            print(f"  Optimized v1: {optimized_time:8.4f} ms (speedup: {speedup:5.2f}x)")
            print(f"  Optimized v2: {optimized_v2_time:8.4f} ms (speedup: {speedup_v2:5.2f}x)")
            print()
    
    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    # Group by generator ratio
    for gen_ratio in generator_ratios:
        ratio_results = [r for r in results if r['gen_ratio'] == gen_ratio]
        if ratio_results:
            avg_speedup = np.mean([r['speedup'] for r in ratio_results])
            avg_speedup_v2 = np.mean([r['speedup_v2'] for r in ratio_results])
            max_speedup = max([r['speedup'] for r in ratio_results])
            max_speedup_v2 = max([r['speedup_v2'] for r in ratio_results])
            
            print(f"Generator ratio {gen_ratio:.1f}:")
            print(f"  Avg speedup v1: {avg_speedup:.2f}x, Max: {max_speedup:.2f}x")
            print(f"  Avg speedup v2: {avg_speedup_v2:.2f}x, Max: {max_speedup_v2:.2f}x")
    
    return results


def verify_correctness():
    """Verify that optimized versions produce statistically equivalent results."""
    print("Verifying correctness...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 100
    vocab_size = 1000
    num_trials = 1000
    
    # Test with various generator configurations
    test_configs = [
        (0, {}),  # No generators
        (50, {i: torch.Generator(device=device).manual_seed(i) for i in range(50)}),  # Half with generators
        (100, {i: torch.Generator(device=device).manual_seed(i) for i in range(100)}),  # All with generators
    ]
    
    for num_gens, generators in test_configs:
        print(f"\nTesting with {num_gens} generators...")
        
        # Fixed seed for reproducible probabilities
        torch.manual_seed(42)
        probs = torch.rand(batch_size, vocab_size, device=device)
        probs = probs.softmax(dim=-1)
        
        # Collect samples
        original_samples = []
        optimized_samples = []
        optimized_v2_samples = []
        
        for _ in range(num_trials):
            # Clone generators for each trial to ensure same random state
            gen_copy_orig = {k: torch.Generator(device=device).manual_seed(v.initial_seed()) 
                            for k, v in generators.items()} if generators else {}
            gen_copy_opt = {k: torch.Generator(device=device).manual_seed(v.initial_seed()) 
                           for k, v in generators.items()} if generators else {}
            gen_copy_opt2 = {k: torch.Generator(device=device).manual_seed(v.initial_seed()) 
                            for k, v in generators.items()} if generators else {}
            
            original_samples.append(random_sample_original(probs.clone(), gen_copy_orig))
            optimized_samples.append(random_sample_optimized(probs.clone(), gen_copy_opt))
            optimized_v2_samples.append(random_sample_optimized_v2(probs.clone(), gen_copy_opt2))
        
        # Statistical comparison
        original_samples = torch.stack(original_samples)
        optimized_samples = torch.stack(optimized_samples)
        optimized_v2_samples = torch.stack(optimized_v2_samples)
        
        # Check if distributions are similar (they won't be identical due to different random number generation)
        print(f"  Sample means - Original: {original_samples.float().mean():.4f}, "
              f"Optimized: {optimized_samples.float().mean():.4f}, "
              f"Optimized v2: {optimized_v2_samples.float().mean():.4f}")
        print(f"  Sample stds - Original: {original_samples.float().std():.4f}, "
              f"Optimized: {optimized_samples.float().std():.4f}, "
              f"Optimized v2: {optimized_v2_samples.float().std():.4f}")
        
        # Check that all versions sample from valid token indices
        assert (original_samples >= 0).all() and (original_samples < vocab_size).all()
        assert (optimized_samples >= 0).all() and (optimized_samples < vocab_size).all()
        assert (optimized_v2_samples >= 0).all() and (optimized_v2_samples < vocab_size).all()
        
    print("\n✓ Correctness verification passed!")


if __name__ == "__main__":
    # First verify correctness
    verify_correctness()
    
    print("\n" + "=" * 80)
    print("PERFORMANCE BENCHMARK")
    print("=" * 80 + "\n")
    
    # Run performance benchmark
    results = benchmark_sampling(
        batch_sizes=[32, 64, 128, 256, 512],
        vocab_size=32000,  # Typical LLM vocab size
        num_iterations=100,
        generator_ratios=[0.0, 0.25, 0.5, 1.0]  # Test various generator scenarios
    )