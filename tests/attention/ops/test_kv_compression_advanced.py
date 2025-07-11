"""Comprehensive tests for advanced KV cache compression.

This module tests:
1. Correctness of compression/decompression
2. Accuracy preservation
3. Memory efficiency
4. Performance benchmarks
"""

import pytest
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional
import time
import math
from dataclasses import dataclass

from vllm.attention.ops.kv_compression_advanced import (
    MultiScaleDecomposer, AttentionAwareCompressor, HierarchicalKVStorage,
    AdvancedKVCacheCompressor, AdvancedCompressionConfig
)
from vllm.attention.ops.kv_compression import (
    KVCacheCompressor, MagnitudeDirectionDecomposer
)


@dataclass
class CompressionMetrics:
    """Metrics for evaluating compression quality."""
    reconstruction_error: float
    cosine_similarity: float
    attention_preservation: float
    compression_ratio: float
    memory_saved_mb: float
    compression_time_ms: float
    decompression_time_ms: float


class TestMultiScaleDecomposition:
    """Test multi-scale decomposition correctness."""
    
    @pytest.mark.parametrize("batch_size", [1, 4])
    @pytest.mark.parametrize("num_heads", [8, 32])
    @pytest.mark.parametrize("seq_len", [128, 512])
    @pytest.mark.parametrize("head_dim", [64, 128])
    @pytest.mark.parametrize("learnable", [True, False])
    def test_decomposition_reconstruction(self, batch_size, num_heads, seq_len, head_dim, learnable):
        """Test that decomposition and recomposition preserves information."""
        torch.manual_seed(42)
        
        # Create test data
        kv_vectors = torch.randn(batch_size, num_heads, seq_len, head_dim)
        
        # Initialize decomposer
        decomposer = MultiScaleDecomposer(
            head_dim=head_dim,
            num_scales=3,
            learnable=learnable
        )
        
        # Decompose
        components = decomposer.decompose(kv_vectors)
        
        # Recompose
        reconstructed = decomposer.recompose(components)
        
        # Check reconstruction error
        reconstruction_error = F.mse_loss(reconstructed, kv_vectors).item()
        
        # For learnable decomposer, error should be very small initially
        if learnable:
            assert reconstruction_error < 0.1, f"Reconstruction error too high: {reconstruction_error}"
        else:
            # Fixed decomposition might have higher error
            assert reconstruction_error < 1.0, f"Reconstruction error too high: {reconstruction_error}"
        
        # Check shapes
        assert components['magnitude'].shape == (batch_size, num_heads, seq_len, 1)
        assert components['direction'].shape == kv_vectors.shape
        assert len(components['scale_components']) == 3
    
    def test_scale_component_properties(self):
        """Test properties of multi-scale components."""
        torch.manual_seed(42)
        
        # Create test data with known frequency components
        batch_size, num_heads, seq_len, head_dim = 2, 4, 256, 64
        
        # Create low-frequency component
        t = torch.linspace(0, 2 * np.pi, seq_len).unsqueeze(-1)
        low_freq = torch.sin(t).expand(batch_size, num_heads, seq_len, head_dim)
        
        # Create high-frequency component
        high_freq = torch.sin(10 * t).expand(batch_size, num_heads, seq_len, head_dim)
        
        # Combine
        kv_vectors = low_freq + 0.1 * high_freq
        
        # Decompose
        decomposer = MultiScaleDecomposer(head_dim=head_dim, learnable=False)
        components = decomposer.decompose(kv_vectors)
        
        # Check that different scales capture different frequency components
        # This is a simplified check - in practice we'd use FFT
        for i, scale_comp in enumerate(components['scale_components']):
            if isinstance(scale_comp, tuple):  # Wavelet decomposition
                low, high = scale_comp
                # Higher scales should have more high-frequency content
                assert low.shape[0] == batch_size
                assert high.shape[0] == batch_size


class TestAttentionAwareCompression:
    """Test attention-aware compression functionality."""
    
    @pytest.mark.parametrize("compression_ratio", [0.25, 0.5, 0.75])
    def test_importance_based_compression(self, compression_ratio):
        """Test that important tokens are preserved during compression."""
        torch.manual_seed(42)
        
        batch_size, num_heads, seq_len, head_dim = 2, 8, 128, 64
        
        # Create KV states with known important tokens
        key_states = torch.randn(batch_size, num_heads, seq_len, head_dim)
        value_states = torch.randn(batch_size, num_heads, seq_len, head_dim)
        
        # Make some tokens more important (higher magnitude)
        important_indices = [10, 20, 30, 40, 50]
        for idx in important_indices:
            key_states[:, :, idx, :] *= 5.0
            value_states[:, :, idx, :] *= 5.0
        
        # Initialize compressor
        compressor = AttentionAwareCompressor(
            num_heads=num_heads,
            head_dim=head_dim,
            compression_ratio=compression_ratio,
            use_importance_weighting=True
        )
        
        # Compute importance and compress
        importance_scores = compressor.compute_token_importance(key_states, value_states)
        compressed_keys, compressed_values, mask = compressor.compress_with_importance(
            key_states, value_states, importance_scores
        )
        
        # Check compression ratio
        actual_ratio = compressed_keys.shape[2] / seq_len
        assert abs(actual_ratio - compression_ratio) < 0.01
        
        # Check that important tokens are preserved
        # Get indices of preserved tokens
        preserved_indices = mask[0, 0].nonzero().squeeze(-1)
        
        # Most important tokens should be preserved
        preserved_important = sum(idx in preserved_indices for idx in important_indices)
        assert preserved_important >= min(len(important_indices) * 0.8, 
                                         compressed_keys.shape[2] * 0.3)
    
    def test_attention_pattern_preservation(self):
        """Test that compression preserves attention patterns."""
        torch.manual_seed(42)
        
        batch_size, num_heads, seq_len, head_dim = 1, 8, 64, 64
        
        # Create query, key, value
        query = torch.randn(batch_size, num_heads, seq_len, head_dim)
        key = torch.randn(batch_size, num_heads, seq_len, head_dim)
        value = torch.randn(batch_size, num_heads, seq_len, head_dim)
        
        # Compute original attention
        original_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(head_dim)
        original_attn = F.softmax(original_scores, dim=-1)
        original_output = torch.matmul(original_attn, value)
        
        # Compress KV cache
        compressor = AttentionAwareCompressor(
            num_heads=num_heads,
            head_dim=head_dim,
            compression_ratio=0.5
        )
        
        importance_scores = compressor.compute_token_importance(key, value, original_scores)
        compressed_keys, compressed_values, mask = compressor.compress_with_importance(
            key, value, importance_scores
        )
        
        # Compute attention with compressed KV
        compressed_scores = torch.matmul(query, compressed_keys.transpose(-2, -1)) / math.sqrt(head_dim)
        compressed_attn = F.softmax(compressed_scores, dim=-1)
        compressed_output = torch.matmul(compressed_attn, compressed_values)
        
        # Compare outputs (relaxed tolerance due to compression)
        cosine_sim = F.cosine_similarity(
            original_output.flatten(),
            compressed_output.flatten(),
            dim=0
        ).item()
        
        assert cosine_sim > 0.85, f"Attention pattern not well preserved: {cosine_sim}"


class TestHierarchicalStorage:
    """Test hierarchical storage system."""
    
    def test_storage_and_retrieval(self):
        """Test basic storage and retrieval functionality."""
        torch.manual_seed(42)
        
        storage = HierarchicalKVStorage(
            num_levels=3,
            compression_ratios=[1.0, 0.5, 0.25],
            quantization_bits=[16, 8, 4]
        )
        
        # Create test data
        batch_size, num_heads, seq_len, head_dim = 1, 8, 128, 64
        keys = torch.randn(batch_size, num_heads, seq_len, head_dim)
        values = torch.randn(batch_size, num_heads, seq_len, head_dim)
        importance = torch.rand(batch_size, num_heads, seq_len)
        
        # Store at different levels
        for level in range(3):
            page_id = level * 100
            storage.store(page_id, keys.clone(), values.clone(), importance, level=level)
        
        # Retrieve and check
        for level in range(3):
            page_id = level * 100
            retrieved_keys, retrieved_values = storage.retrieve(page_id)
            
            # Check shapes based on compression level
            expected_seq_len = int(seq_len * storage.compression_ratios[level])
            assert retrieved_keys.shape[2] == expected_seq_len
            assert retrieved_values.shape[2] == expected_seq_len
    
    def test_adaptive_promotion(self):
        """Test that frequently accessed pages get promoted."""
        storage = HierarchicalKVStorage()
        
        # Store page at lowest level (most compressed)
        page_id = 1
        keys = torch.randn(1, 8, 128, 64)
        values = torch.randn(1, 8, 128, 64)
        importance = torch.rand(1, 8, 128)
        
        storage.store(page_id, keys, values, importance, level=2)
        
        # Access multiple times
        for _ in range(15):
            storage.retrieve(page_id)
        
        # Check that page was promoted (should now be at higher level)
        # This is implementation-specific behavior
        found_level = None
        for level in range(storage.num_levels):
            if page_id in storage.storage_levels[level]:
                found_level = level
                break
        
        assert found_level is not None and found_level < 2, \
            "Frequently accessed page should be promoted to less compressed level"


class TestEndToEndCompression:
    """End-to-end tests for complete compression pipeline."""
    
    def test_accuracy_preservation(self):
        """Test that compression preserves model accuracy within acceptable bounds."""
        torch.manual_seed(42)
        
        # Configuration
        config = AdvancedCompressionConfig(
            num_scales=3,
            learnable_decomposition=True,
            compression_ratio=0.5,
            storage_levels=3
        )
        
        # Initialize compressor
        compressor = AdvancedKVCacheCompressor(config)
        
        # Simulate multiple KV cache pages
        num_pages = 10
        batch_size, num_heads, seq_len, head_dim = 1, 32, 256, 64
        
        all_metrics = []
        
        for page_id in range(num_pages):
            # Generate realistic KV cache data
            keys = torch.randn(batch_size, num_heads, seq_len, head_dim) * 0.1
            values = torch.randn(batch_size, num_heads, seq_len, head_dim) * 0.1
            
            # Add some structure (simulate real attention patterns)
            for i in range(0, seq_len, 16):
                keys[:, :, i:i+4, :] *= 2.0  # Emphasize certain positions
                values[:, :, i:i+4, :] *= 2.0
            
            # Measure compression
            start_time = time.time()
            compression_info = compressor.compress_page(page_id, keys, values)
            compression_time = (time.time() - start_time) * 1000
            
            # Measure decompression
            start_time = time.time()
            decompressed = compressor.decompress_page(page_id)
            decompression_time = (time.time() - start_time) * 1000
            
            if decompressed is not None:
                decompressed_keys, decompressed_values = decompressed
                
                # Calculate metrics
                metrics = self._calculate_compression_metrics(
                    keys, values,
                    decompressed_keys, decompressed_values,
                    compression_info,
                    compression_time,
                    decompression_time
                )
                all_metrics.append(metrics)
        
        # Aggregate metrics
        avg_reconstruction_error = np.mean([m.reconstruction_error for m in all_metrics])
        avg_cosine_similarity = np.mean([m.cosine_similarity for m in all_metrics])
        avg_compression_ratio = np.mean([m.compression_ratio for m in all_metrics])
        
        # Assert quality thresholds
        assert avg_reconstruction_error < 0.1, \
            f"Average reconstruction error too high: {avg_reconstruction_error}"
        assert avg_cosine_similarity > 0.95, \
            f"Average cosine similarity too low: {avg_cosine_similarity}"
        assert avg_compression_ratio < 0.6, \
            f"Compression not effective enough: {avg_compression_ratio}"
        
        print(f"\nCompression Quality Metrics:")
        print(f"  Avg Reconstruction Error: {avg_reconstruction_error:.4f}")
        print(f"  Avg Cosine Similarity: {avg_cosine_similarity:.4f}")
        print(f"  Avg Compression Ratio: {avg_compression_ratio:.4f}")
        print(f"  Avg Compression Time: {np.mean([m.compression_time_ms for m in all_metrics]):.2f}ms")
        print(f"  Avg Decompression Time: {np.mean([m.decompression_time_ms for m in all_metrics]):.2f}ms")
    
    def _calculate_compression_metrics(self,
                                     original_keys: torch.Tensor,
                                     original_values: torch.Tensor,
                                     decompressed_keys: torch.Tensor,
                                     decompressed_values: torch.Tensor,
                                     compression_info: dict,
                                     compression_time: float,
                                     decompression_time: float) -> CompressionMetrics:
        """Calculate comprehensive compression metrics."""
        # Handle size mismatch due to compression
        min_seq_len = min(original_keys.shape[2], decompressed_keys.shape[2])
        
        # Truncate or pad as needed
        if original_keys.shape[2] > min_seq_len:
            original_keys = original_keys[:, :, :min_seq_len, :]
            original_values = original_values[:, :, :min_seq_len, :]
        
        # Reconstruction error
        key_error = F.mse_loss(decompressed_keys, original_keys).item()
        value_error = F.mse_loss(decompressed_values, original_values).item()
        reconstruction_error = (key_error + value_error) / 2
        
        # Cosine similarity
        key_sim = F.cosine_similarity(
            original_keys.flatten(),
            decompressed_keys.flatten(),
            dim=0
        ).item()
        value_sim = F.cosine_similarity(
            original_values.flatten(),
            decompressed_values.flatten(),
            dim=0
        ).item()
        cosine_similarity = (key_sim + value_sim) / 2
        
        # Attention preservation (simplified)
        # In practice, we'd compute actual attention outputs
        attention_preservation = cosine_similarity  # Simplified metric
        
        # Memory calculations
        original_size = original_keys.numel() + original_values.numel()
        compressed_size = decompressed_keys.numel() + decompressed_values.numel()
        memory_saved_mb = (original_size - compressed_size) * 4 / (1024 * 1024)  # Assuming float32
        
        return CompressionMetrics(
            reconstruction_error=reconstruction_error,
            cosine_similarity=cosine_similarity,
            attention_preservation=attention_preservation,
            compression_ratio=compression_info.get('compression_ratio', 1.0),
            memory_saved_mb=memory_saved_mb,
            compression_time_ms=compression_time,
            decompression_time_ms=decompression_time
        )
    
    @pytest.mark.parametrize("model_size", ["small", "medium", "large"])
    def test_scalability(self, model_size):
        """Test compression scalability with different model sizes."""
        torch.manual_seed(42)
        
        # Define model configurations
        configs = {
            "small": (12, 64, 512),    # num_heads, head_dim, seq_len
            "medium": (32, 128, 1024),
            "large": (64, 128, 2048)
        }
        
        num_heads, head_dim, seq_len = configs[model_size]
        
        # Test both basic and advanced compression
        basic_compressor = KVCacheCompressor(enable_compression=True)
        
        advanced_config = AdvancedCompressionConfig(
            compression_ratio=0.5,
            learnable_decomposition=False  # Faster for testing
        )
        advanced_compressor = AdvancedKVCacheCompressor(advanced_config)
        
        # Generate test data
        keys = torch.randn(1, num_heads, seq_len, head_dim) * 0.1
        values = torch.randn(1, num_heads, seq_len, head_dim) * 0.1
        
        # Time basic compression
        start = time.time()
        basic_page = basic_compressor.compress_page(0, keys, values)
        basic_time = time.time() - start
        
        # Time advanced compression
        start = time.time()
        advanced_info = advanced_compressor.compress_page(0, keys, values)
        advanced_time = time.time() - start
        
        print(f"\n{model_size.upper()} Model Compression Times:")
        print(f"  Basic: {basic_time*1000:.2f}ms")
        print(f"  Advanced: {advanced_time*1000:.2f}ms")
        print(f"  Overhead: {(advanced_time/basic_time - 1)*100:.1f}%")
        
        # Advanced should not be prohibitively slower
        assert advanced_time < basic_time * 5, \
            f"Advanced compression too slow: {advanced_time/basic_time:.2f}x slower"


class TestCompressionComparison:
    """Compare different compression methods."""
    
    def test_compression_effectiveness(self):
        """Compare compression ratios and quality across methods."""
        torch.manual_seed(42)
        
        # Test configuration
        batch_size, num_heads, seq_len, head_dim = 1, 16, 512, 64
        keys = torch.randn(batch_size, num_heads, seq_len, head_dim) * 0.1
        values = torch.randn(batch_size, num_heads, seq_len, head_dim) * 0.1
        
        # Add realistic patterns
        for i in range(0, seq_len, 32):
            keys[:, :, i:i+8, :] *= 3.0  # Important tokens
            values[:, :, i:i+8, :] *= 3.0
        
        results = {}
        
        # Method 1: Simple magnitude-direction (baseline)
        basic_decomposer = MagnitudeDirectionDecomposer()
        magnitude, direction = basic_decomposer.decompose(keys[0])
        basic_reconstructed = basic_decomposer.recompose(magnitude, direction)
        results['basic'] = {
            'reconstruction_error': F.mse_loss(basic_reconstructed, keys[0]).item(),
            'storage_size': magnitude.numel() + direction.numel()
        }
        
        # Method 2: Multi-scale decomposition
        ms_decomposer = MultiScaleDecomposer(head_dim=head_dim, learnable=False)
        ms_components = ms_decomposer.decompose(keys)
        ms_reconstructed = ms_decomposer.recompose(ms_components)
        results['multi_scale'] = {
            'reconstruction_error': F.mse_loss(ms_reconstructed, keys).item(),
            'storage_size': sum(c.numel() for c in ms_components['scale_components'])
        }
        
        # Method 3: Attention-aware compression
        aa_compressor = AttentionAwareCompressor(
            num_heads=num_heads,
            head_dim=head_dim,
            compression_ratio=0.5
        )
        importance = aa_compressor.compute_token_importance(keys, values)
        compressed_k, compressed_v, _ = aa_compressor.compress_with_importance(
            keys, values, importance
        )
        results['attention_aware'] = {
            'reconstruction_error': 0.0,  # Not directly comparable
            'storage_size': compressed_k.numel() + compressed_v.numel(),
            'compression_ratio': compressed_k.shape[2] / seq_len
        }
        
        # Print comparison
        print("\nCompression Method Comparison:")
        for method, metrics in results.items():
            print(f"\n{method.upper()}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")
        
        # Assert multi-scale is better than basic
        assert results['multi_scale']['reconstruction_error'] <= \
               results['basic']['reconstruction_error'] * 1.5


if __name__ == "__main__":
    # Run specific tests for development
    test = TestEndToEndCompression()
    test.test_accuracy_preservation()
    test.test_scalability("medium") 