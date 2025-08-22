#!/usr/bin/env python3
"""Test script for the optimized random_sample function."""

import torch
import time
from vllm.v1.sample.ops.topk_topp_sampler_optimized import random_sample_optimized, random_sample

def test_basic_functionality():
    """Test that the optimized version produces valid outputs."""
    print("Testing basic functionality...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 32
    vocab_size = 32000
    
    # Create test probabilities
    probs = torch.rand(batch_size, vocab_size, device=device)
    probs = probs.softmax(dim=-1)
    
    # Test 1: No generators
    print("  Test 1: No generators...")
    result = random_sample_optimized(probs.clone(), {})
    assert result.shape == (batch_size,)
    assert (result >= 0).all() and (result < vocab_size).all()
    print("    ✓ Passed")
    
    # Test 2: Some generators
    print("  Test 2: With some generators...")
    generators = {}
    for i in range(10):
        generators[i] = torch.Generator(device=device)
        generators[i].manual_seed(i)
    
    result = random_sample_optimized(probs.clone(), generators)
    assert result.shape == (batch_size,)
    assert (result >= 0).all() and (result < vocab_size).all()
    print("    ✓ Passed")
    
    # Test 3: All generators
    print("  Test 3: All requests with generators...")
    generators = {}
    for i in range(batch_size):
        generators[i] = torch.Generator(device=device)
        generators[i].manual_seed(i)
    
    result = random_sample_optimized(probs.clone(), generators)
    assert result.shape == (batch_size,)
    assert (result >= 0).all() and (result < vocab_size).all()
    print("    ✓ Passed")
    
    print("✓ All functionality tests passed!\n")


def performance_comparison():
    """Compare performance between original and optimized versions."""
    print("Performance Comparison")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")
    
    batch_sizes = [32, 64, 128, 256]
    vocab_size = 32000
    num_iterations = 100
    
    for batch_size in batch_sizes:
        print(f"Batch size: {batch_size}")
        
        # Create test data
        probs = torch.rand(batch_size, vocab_size, device=device)
        probs = probs.softmax(dim=-1)
        
        # Test with 50% of requests having generators
        num_generators = batch_size // 2
        generators = {}
        for i in range(num_generators):
            generators[i] = torch.Generator(device=device)
            generators[i].manual_seed(i)
        
        # Warmup
        for _ in range(10):
            _ = random_sample(probs.clone(), generators)
            _ = random_sample_optimized(probs.clone(), generators)
        
        # Benchmark original
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_iterations):
            _ = random_sample(probs.clone(), generators)
        if device.type == "cuda":
            torch.cuda.synchronize()
        original_time = (time.perf_counter() - start) / num_iterations * 1000
        
        # Benchmark optimized
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(num_iterations):
            _ = random_sample_optimized(probs.clone(), generators)
        if device.type == "cuda":
            torch.cuda.synchronize()
        optimized_time = (time.perf_counter() - start) / num_iterations * 1000
        
        speedup = original_time / optimized_time
        print(f"  Original:  {original_time:.4f} ms")
        print(f"  Optimized: {optimized_time:.4f} ms")
        print(f"  Speedup:   {speedup:.2f}x")
        print()


if __name__ == "__main__":
    test_basic_functionality()
    performance_comparison()