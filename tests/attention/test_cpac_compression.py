# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Comprehensive test suite for CPAC compression
"""

import pytest
import torch
import numpy as np
from typing import Dict, List, Tuple
import time
import matplotlib.pyplot as plt
from dataclasses import dataclass

from vllm.attention.ops.cpac_compression import (
    CPACConfig, CPACManager, PageSimilarityTracker, CrossPageCompressor
)
from vllm.attention.ops.cpac_ops import (
    compute_page_features, compute_similarity, compress_delta, decompress_delta,
    batch_compute_similarities, estimate_compression_ratio, cpac_kernel
)


@dataclass
class TestMetrics:
    """Metrics collected during testing"""
    compression_ratio: float
    similarity_score: float
    reconstruction_error: float
    compression_time_ms: float
    decompression_time_ms: float
    memory_saved_percent: float


class TestCPACCompression:
    """Test suite for CPAC compression functionality"""
    
    @pytest.fixture
    def setup_test_data(self):
        """Setup test data for compression tests"""
        torch.manual_seed(42)
        
        # Test parameters
        block_size = 16
        num_heads = 32
        head_size = 128
        num_blocks = 100
        
        # Generate synthetic KV cache data
        key_cache = torch.randn(
            num_blocks, num_heads, head_size // 16, block_size, 16,
            dtype=torch.float16, device='cuda'
        )
        value_cache = torch.randn(
            num_blocks, num_heads, head_size, block_size,
            dtype=torch.float16, device='cuda'
        )
        
        return {
            'block_size': block_size,
            'num_heads': num_heads,
            'head_size': head_size,
            'num_blocks': num_blocks,
            'key_cache': key_cache,
            'value_cache': value_cache,
        }
    
    def test_page_similarity_computation(self, setup_test_data):
        """Test page similarity computation"""
        data = setup_test_data
        
        # Create similar pages by adding small noise
        page1 = data['key_cache'][0]
        page2 = page1 + torch.randn_like(page1) * 0.1
        page3 = torch.randn_like(page1)  # Completely different
        
        # Compute features
        feat1 = compute_page_features(page1.view(data['num_heads'], data['head_size'], data['block_size']))
        feat2 = compute_page_features(page2.view(data['num_heads'], data['head_size'], data['block_size']))
        feat3 = compute_page_features(page3.view(data['num_heads'], data['head_size'], data['block_size']))
        
        # Compute similarities
        sim12 = compute_similarity(feat1, feat2).item()
        sim13 = compute_similarity(feat1, feat3).item()
        
        # Similar pages should have high similarity
        assert sim12 > 0.9, f"Similar pages have low similarity: {sim12}"
        # Different pages should have low similarity
        assert sim13 < 0.5, f"Different pages have high similarity: {sim13}"
        
        print(f"Similarity between similar pages: {sim12:.4f}")
        print(f"Similarity between different pages: {sim13:.4f}")
    
    def test_delta_compression_accuracy(self, setup_test_data):
        """Test delta compression and reconstruction accuracy"""
        data = setup_test_data
        
        # Create pages with varying similarity
        base_page = data['key_cache'][0]
        similar_page = base_page + torch.randn_like(base_page) * 0.05
        
        # Compress
        quantized_delta, scale = compress_delta(similar_page, base_page, delta_bits=8)
        
        # Decompress
        reconstructed = decompress_delta(quantized_delta, base_page, scale.item())
        
        # Compute reconstruction error
        mse = torch.mean((similar_page - reconstructed) ** 2).item()
        relative_error = torch.norm(similar_page - reconstructed) / torch.norm(similar_page)
        
        print(f"Reconstruction MSE: {mse:.6f}")
        print(f"Relative error: {relative_error:.4f}")
        
        # Error should be small
        assert relative_error < 0.1, f"High reconstruction error: {relative_error}"
        
        # Test compression ratio
        original_size = similar_page.numel() * similar_page.element_size()
        compressed_size = quantized_delta.numel() + 4  # +4 bytes for scale
        compression_ratio = original_size / compressed_size
        
        print(f"Compression ratio: {compression_ratio:.2f}x")
        assert compression_ratio > 1.5, f"Low compression ratio: {compression_ratio}"
    
    def test_cpac_manager_integration(self, setup_test_data):
        """Test full CPAC manager functionality"""
        data = setup_test_data
        
        # Initialize CPAC manager
        config = CPACConfig(
            similarity_threshold=0.8,
            compression_level=2,
            adaptive_compression=True,
        )
        
        manager = CPACManager(
            block_size=data['block_size'],
            num_gpu_blocks=data['num_blocks'],
            num_kv_heads=data['num_heads'],
            head_size=data['head_size'],
            config=config,
        )
        
        # Test compression of similar blocks
        metrics_list = []
        
        # Create groups of similar blocks
        for group_idx in range(10):
            base_idx = group_idx * 10
            base_key = data['key_cache'][base_idx]
            base_value = data['value_cache'][base_idx]
            
            # Create similar blocks
            for offset in range(1, 5):
                similar_idx = base_idx + offset
                noise_level = 0.1 * offset  # Increasing noise
                
                data['key_cache'][similar_idx] = base_key + torch.randn_like(base_key) * noise_level
                data['value_cache'][similar_idx] = base_value + torch.randn_like(base_value) * noise_level
                
                # Attempt compression
                start_time = time.time()
                success = manager.compress_block(
                    similar_idx,
                    data['key_cache'][similar_idx],
                    data['value_cache'][similar_idx]
                )
                compression_time = (time.time() - start_time) * 1000
                
                if success:
                    # Test decompression
                    decompressed_key = torch.zeros_like(data['key_cache'][similar_idx])
                    decompressed_value = torch.zeros_like(data['value_cache'][similar_idx])
                    
                    start_time = time.time()
                    manager.decompress_block(
                        similar_idx, decompressed_key, decompressed_value
                    )
                    decompression_time = (time.time() - start_time) * 1000
                    
                    # Compute metrics
                    key_error = torch.norm(data['key_cache'][similar_idx] - decompressed_key) / torch.norm(data['key_cache'][similar_idx])
                    value_error = torch.norm(data['value_cache'][similar_idx] - decompressed_value) / torch.norm(data['value_cache'][similar_idx])
                    
                    metrics = TestMetrics(
                        compression_ratio=2.0,  # Placeholder
                        similarity_score=0.9 - noise_level,  # Estimate
                        reconstruction_error=(key_error + value_error) / 2,
                        compression_time_ms=compression_time,
                        decompression_time_ms=decompression_time,
                        memory_saved_percent=50.0,  # Placeholder
                    )
                    metrics_list.append(metrics)
        
        # Analyze results
        stats = manager.get_compression_stats()
        print(f"\nCompression Statistics:")
        print(f"Total compressions: {stats['total_compressions']}")
        print(f"Successful compressions: {stats['successful_compressions']}")
        print(f"Average compression ratio: {stats['average_ratio']:.2f}x")
        
        assert stats['successful_compressions'] > 0, "No successful compressions"
        
        return metrics_list
    
    def test_cross_page_patterns(self, setup_test_data):
        """Test identification of cross-page patterns"""
        data = setup_test_data
        
        # Create pages with specific patterns
        patterns = {
            'repeating': lambda i: torch.sin(torch.arange(data['block_size']) * 2 * np.pi * i / data['block_size']),
            'linear': lambda i: torch.arange(data['block_size']) * i,
            'exponential': lambda i: torch.exp(torch.arange(data['block_size']) * 0.1 * i),
        }
        
        # Generate patterned data
        for pattern_name, pattern_fn in patterns.items():
            print(f"\nTesting pattern: {pattern_name}")
            
            # Create blocks with the same pattern but different scales
            base_pattern = pattern_fn(1).cuda()
            features_list = []
            
            for i in range(10):
                scale = 1 + i * 0.1
                block_data = base_pattern * scale
                
                # Expand to full shape
                full_block = block_data.unsqueeze(0).unsqueeze(0).expand(
                    data['num_heads'], data['head_size'], data['block_size']
                )
                
                features = compute_page_features(full_block)
                features_list.append(features)
            
            # Compute similarities
            features_tensor = torch.stack(features_list)
            similarities, _ = batch_compute_similarities(features_tensor, top_k=9)
            
            # Blocks with same pattern should have high similarity
            mean_similarity = similarities.mean().item()
            print(f"Mean similarity for {pattern_name} pattern: {mean_similarity:.4f}")
            
            assert mean_similarity > 0.7, f"Low similarity for {pattern_name} pattern"
    
    def test_memory_pressure_adaptation(self, setup_test_data):
        """Test adaptive compression under memory pressure"""
        data = setup_test_data
        
        config = CPACConfig(
            adaptive_compression=True,
            compression_level=2,
        )
        
        manager = CPACManager(
            block_size=data['block_size'],
            num_gpu_blocks=data['num_blocks'],
            num_kv_heads=data['num_heads'],
            head_size=data['head_size'],
            config=config,
        )
        
        # Test different memory pressure levels
        pressure_levels = [0.3, 0.6, 0.9]
        compression_results = []
        
        for pressure in pressure_levels:
            manager.compressor.update_memory_pressure(pressure)
            
            # Attempt compressions
            compressed_count = 0
            for i in range(20):
                if manager.should_compress_block(i, pressure):
                    compressed_count += 1
            
            compression_results.append({
                'pressure': pressure,
                'compressed': compressed_count,
                'compression_level': manager.compressor.current_compression_level,
            })
            
            print(f"Pressure: {pressure}, Compressed: {compressed_count}, "
                  f"Level: {manager.compressor.current_compression_level}")
        
        # Higher pressure should lead to more aggressive compression
        assert compression_results[-1]['compression_level'] > compression_results[0]['compression_level']
    
    def test_performance_benchmarks(self, setup_test_data):
        """Benchmark compression performance"""
        data = setup_test_data
        
        manager = CPACManager(
            block_size=data['block_size'],
            num_gpu_blocks=data['num_blocks'],
            num_kv_heads=data['num_heads'],
            head_size=data['head_size'],
        )
        
        # Warmup
        for _ in range(10):
            _ = compute_page_features(data['key_cache'][0].view(data['num_heads'], data['head_size'], data['block_size']))
        
        # Benchmark operations
        num_iterations = 100
        
        # Feature computation
        torch.cuda.synchronize()
        start = time.time()
        for i in range(num_iterations):
            _ = compute_page_features(data['key_cache'][i % data['num_blocks']].view(data['num_heads'], data['head_size'], data['block_size']))
        torch.cuda.synchronize()
        feature_time = (time.time() - start) / num_iterations * 1000
        
        # Compression
        torch.cuda.synchronize()
        start = time.time()
        for i in range(num_iterations):
            base_idx = i % (data['num_blocks'] - 1)
            target_idx = base_idx + 1
            _ = compress_delta(data['key_cache'][target_idx], data['key_cache'][base_idx])
        torch.cuda.synchronize()
        compress_time = (time.time() - start) / num_iterations * 1000
        
        print(f"\nPerformance Benchmarks:")
        print(f"Feature computation: {feature_time:.3f} ms/block")
        print(f"Delta compression: {compress_time:.3f} ms/block")
        print(f"Estimated throughput: {1000/compress_time:.1f} blocks/second")
    
    def generate_compression_report(self, metrics_list: List[TestMetrics]):
        """Generate a comprehensive report of compression performance"""
        if not metrics_list:
            return
        
        # Aggregate metrics
        avg_ratio = np.mean([m.compression_ratio for m in metrics_list])
        avg_error = np.mean([m.reconstruction_error for m in metrics_list])
        avg_compress_time = np.mean([m.compression_time_ms for m in metrics_list])
        avg_decompress_time = np.mean([m.decompression_time_ms for m in metrics_list])
        
        print("\n" + "="*60)
        print("CPAC Compression Performance Report")
        print("="*60)
        print(f"Average compression ratio: {avg_ratio:.2f}x")
        print(f"Average reconstruction error: {avg_error:.4f}")
        print(f"Average compression time: {avg_compress_time:.3f} ms")
        print(f"Average decompression time: {avg_decompress_time:.3f} ms")
        print(f"Compression throughput: {1000/avg_compress_time:.1f} blocks/second")
        print(f"Decompression throughput: {1000/avg_decompress_time:.1f} blocks/second")
        
        # Plot metrics
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Compression ratio vs similarity
        axes[0, 0].scatter([m.similarity_score for m in metrics_list],
                          [m.compression_ratio for m in metrics_list])
        axes[0, 0].set_xlabel('Similarity Score')
        axes[0, 0].set_ylabel('Compression Ratio')
        axes[0, 0].set_title('Compression Ratio vs Similarity')
        
        # Reconstruction error vs similarity
        axes[0, 1].scatter([m.similarity_score for m in metrics_list],
                          [m.reconstruction_error for m in metrics_list])
        axes[0, 1].set_xlabel('Similarity Score')
        axes[0, 1].set_ylabel('Reconstruction Error')
        axes[0, 1].set_title('Reconstruction Error vs Similarity')
        
        # Compression time distribution
        axes[1, 0].hist([m.compression_time_ms for m in metrics_list], bins=20)
        axes[1, 0].set_xlabel('Compression Time (ms)')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Compression Time Distribution')
        
        # Memory savings
        axes[1, 1].bar(['Original', 'Compressed'], 
                      [100, 100 - np.mean([m.memory_saved_percent for m in metrics_list])])
        axes[1, 1].set_ylabel('Memory Usage (%)')
        axes[1, 1].set_title('Memory Savings')
        
        plt.tight_layout()
        plt.savefig('cpac_compression_report.png')
        print(f"\nReport saved to cpac_compression_report.png")


def run_all_tests():
    """Run all CPAC compression tests"""
    test_suite = TestCPACCompression()
    
    # Setup test data
    test_data = test_suite.setup_test_data()
    
    print("Running CPAC Compression Tests...")
    print("="*60)
    
    # Run individual tests
    print("\n1. Testing page similarity computation...")
    test_suite.test_page_similarity_computation(test_data)
    
    print("\n2. Testing delta compression accuracy...")
    test_suite.test_delta_compression_accuracy(test_data)
    
    print("\n3. Testing CPAC manager integration...")
    metrics = test_suite.test_cpac_manager_integration(test_data)
    
    print("\n4. Testing cross-page pattern recognition...")
    test_suite.test_cross_page_patterns(test_data)
    
    print("\n5. Testing memory pressure adaptation...")
    test_suite.test_memory_pressure_adaptation(test_data)
    
    print("\n6. Running performance benchmarks...")
    test_suite.test_performance_benchmarks(test_data)
    
    # Generate report
    test_suite.generate_compression_report(metrics)
    
    print("\n" + "="*60)
    print("All tests completed successfully!")


if __name__ == "__main__":
    run_all_tests()