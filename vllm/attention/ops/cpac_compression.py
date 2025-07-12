# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
CPAC (Cross-Page Adaptive Compression) for KV Cache

This module implements a novel cross-page compression method that exploits
similarities between different pages in the KV cache to achieve better
compression ratios while maintaining model performance.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

from vllm import _custom_ops as ops
from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class CPACConfig:
    """Configuration for CPAC compression"""
    # Similarity threshold for page clustering (0.0 to 1.0)
    similarity_threshold: float = 0.85
    # Maximum number of pages in a cluster
    max_cluster_size: int = 8
    # Minimum compression ratio to apply compression
    min_compression_ratio: float = 2.0
    # Number of top similar pages to consider
    top_k_similar: int = 4
    # Enable adaptive compression based on memory pressure
    adaptive_compression: bool = True
    # Compression levels (1: light, 2: moderate, 3: aggressive)
    compression_level: int = 2
    # Use importance weighting in similarity calculation
    use_importance_weighting: bool = True
    # Delta encoding bits
    delta_bits: int = 8


class PageSimilarityTracker:
    """Tracks similarity between pages for efficient clustering"""
    
    def __init__(self, block_size: int, head_size: int, num_heads: int):
        self.block_size = block_size
        self.head_size = head_size
        self.num_heads = num_heads
        
        # Page feature cache for fast similarity computation
        self.page_features: Dict[int, torch.Tensor] = {}
        # Similarity matrix cache
        self.similarity_cache: Dict[Tuple[int, int], float] = {}
        # Page importance scores
        self.page_importance: Dict[int, float] = {}
        
    def compute_page_features(self, page_data: torch.Tensor) -> torch.Tensor:
        """Extract compact features from a page for similarity computation"""
        # Reshape to (num_heads, head_size, block_size)
        reshaped = page_data.view(self.num_heads, self.head_size, self.block_size)
        
        # Compute multiple feature types
        features = []
        
        # 1. Mean and variance per head
        mean_per_head = reshaped.mean(dim=[1, 2])
        var_per_head = reshaped.var(dim=[1, 2])
        features.extend([mean_per_head, var_per_head])
        
        # 2. Attention pattern signature (simplified)
        attention_signature = reshaped.abs().mean(dim=1)
        features.append(attention_signature.flatten())
        
        # 3. Frequency domain features (DCT coefficients)
        dct_features = self._compute_dct_features(reshaped)
        features.append(dct_features)
        
        return torch.cat(features, dim=-1)
    
    def _compute_dct_features(self, data: torch.Tensor, num_coeffs: int = 8) -> torch.Tensor:
        """Compute DCT coefficients as features"""
        # Simplified DCT using FFT
        fft_result = torch.fft.rfft(data, dim=-1)
        magnitudes = torch.abs(fft_result)[..., :num_coeffs]
        return magnitudes.flatten()
    
    def compute_similarity(self, page_idx1: int, page_idx2: int) -> float:
        """Compute similarity between two pages"""
        cache_key = (min(page_idx1, page_idx2), max(page_idx1, page_idx2))
        if cache_key in self.similarity_cache:
            return self.similarity_cache[cache_key]
        
        feat1 = self.page_features.get(page_idx1)
        feat2 = self.page_features.get(page_idx2)
        
        if feat1 is None or feat2 is None:
            return 0.0
        
        # Cosine similarity
        similarity = F.cosine_similarity(feat1.unsqueeze(0), feat2.unsqueeze(0)).item()
        
        # Apply importance weighting if enabled
        if page_idx1 in self.page_importance and page_idx2 in self.page_importance:
            imp1 = self.page_importance[page_idx1]
            imp2 = self.page_importance[page_idx2]
            # Reduce similarity if importance differs significantly
            importance_factor = 1.0 - abs(imp1 - imp2)
            similarity *= importance_factor
        
        self.similarity_cache[cache_key] = similarity
        return similarity
    
    def update_page_importance(self, page_idx: int, attention_scores: torch.Tensor):
        """Update importance score for a page based on attention patterns"""
        # Compute importance as the average maximum attention score
        max_attention = attention_scores.max(dim=-1).values.mean().item()
        self.page_importance[page_idx] = max_attention


class CrossPageCompressor:
    """Main compression engine for CPAC"""
    
    def __init__(self, config: CPACConfig, block_size: int, 
                 num_kv_heads: int, head_size: int):
        self.config = config
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        
        # Initialize similarity tracker
        self.similarity_tracker = PageSimilarityTracker(
            block_size, head_size, num_kv_heads
        )
        
        # Page clusters: maps page_idx to cluster_id
        self.page_clusters: Dict[int, int] = {}
        # Cluster representatives: maps cluster_id to base page_idx
        self.cluster_bases: Dict[int, int] = {}
        # Delta storage: maps page_idx to compressed delta
        self.compressed_deltas: Dict[int, torch.Tensor] = {}
        # Compression metadata
        self.compression_metadata: Dict[int, Dict] = {}
        
        # Adaptive compression state
        self.current_compression_level = config.compression_level
        self.memory_pressure = 0.0
        
    def compress_page(self, page_idx: int, page_data: torch.Tensor,
                     key_cache: torch.Tensor, value_cache: torch.Tensor) -> Optional[Dict]:
        """Compress a page using cross-page similarity"""
        # Extract features for similarity computation
        key_features = self.similarity_tracker.compute_page_features(key_cache)
        value_features = self.similarity_tracker.compute_page_features(value_cache)
        combined_features = torch.cat([key_features, value_features])
        self.similarity_tracker.page_features[page_idx] = combined_features
        
        # Find similar pages
        similar_pages = self._find_similar_pages(page_idx)
        
        if not similar_pages:
            return None
        
        # Try to compress using the most similar page as base
        best_compression = None
        best_ratio = 0.0
        
        for similar_idx, similarity in similar_pages:
            if similar_idx in self.cluster_bases.values():
                # This is a cluster base, try delta encoding
                compression_result = self._try_delta_compression(
                    page_idx, similar_idx, key_cache, value_cache
                )
                if compression_result and compression_result['ratio'] > best_ratio:
                    best_compression = compression_result
                    best_ratio = compression_result['ratio']
        
        if best_compression and best_ratio >= self.config.min_compression_ratio:
            # Store compression metadata
            self.compression_metadata[page_idx] = best_compression
            return best_compression
        
        return None
    
    def _find_similar_pages(self, page_idx: int) -> List[Tuple[int, float]]:
        """Find pages similar to the given page"""
        similarities = []
        
        for other_idx in self.similarity_tracker.page_features:
            if other_idx != page_idx:
                sim = self.similarity_tracker.compute_similarity(page_idx, other_idx)
                if sim >= self.config.similarity_threshold:
                    similarities.append((other_idx, sim))
        
        # Sort by similarity and return top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:self.config.top_k_similar]
    
    def _try_delta_compression(self, page_idx: int, base_idx: int,
                              key_cache: torch.Tensor, value_cache: torch.Tensor) -> Optional[Dict]:
        """Try to compress page using delta encoding from base"""
        # Get base page data
        base_key = self._get_page_data(base_idx, key_cache, is_key=True)
        base_value = self._get_page_data(base_idx, value_cache, is_key=False)
        
        # Compute deltas
        key_delta = key_cache - base_key
        value_delta = value_cache - base_value
        
        # Quantize deltas based on compression level
        quantized_key_delta, key_scale = self._quantize_delta(key_delta)
        quantized_value_delta, value_scale = self._quantize_delta(value_delta)
        
        # Calculate compression ratio
        original_size = key_cache.numel() + value_cache.numel()
        compressed_size = (quantized_key_delta.numel() * self.config.delta_bits // 8 +
                          quantized_value_delta.numel() * self.config.delta_bits // 8 +
                          4 * 4)  # Scales and metadata
        
        compression_ratio = original_size * key_cache.element_size() / compressed_size
        
        if compression_ratio >= self.config.min_compression_ratio:
            return {
                'type': 'delta',
                'base_idx': base_idx,
                'key_delta': quantized_key_delta,
                'value_delta': quantized_value_delta,
                'key_scale': key_scale,
                'value_scale': value_scale,
                'ratio': compression_ratio
            }
        
        return None
    
    def _quantize_delta(self, delta: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """Quantize delta values to reduce storage"""
        # Find scale
        max_val = delta.abs().max().item()
        if max_val == 0:
            return torch.zeros_like(delta, dtype=torch.int8), 1.0
        
        # Scale to fit in delta_bits
        max_int = (1 << (self.config.delta_bits - 1)) - 1
        scale = max_val / max_int
        
        # Quantize
        quantized = torch.round(delta / scale).clamp(-max_int, max_int).to(torch.int8)
        
        return quantized, scale
    
    def decompress_page(self, page_idx: int, key_cache_out: torch.Tensor,
                       value_cache_out: torch.Tensor) -> None:
        """Decompress a page back to original cache format"""
        metadata = self.compression_metadata.get(page_idx)
        if not metadata:
            return
        
        if metadata['type'] == 'delta':
            # Get base page
            base_idx = metadata['base_idx']
            base_key = self._get_page_data(base_idx, key_cache_out, is_key=True)
            base_value = self._get_page_data(base_idx, value_cache_out, is_key=False)
            
            # Dequantize deltas
            key_delta = metadata['key_delta'].float() * metadata['key_scale']
            value_delta = metadata['value_delta'].float() * metadata['value_scale']
            
            # Reconstruct
            key_cache_out.copy_(base_key + key_delta)
            value_cache_out.copy_(base_value + value_delta)
    
    def _get_page_data(self, page_idx: int, cache: torch.Tensor, is_key: bool) -> torch.Tensor:
        """Extract page data from cache tensor"""
        # This is a placeholder - actual implementation depends on cache layout
        return cache[page_idx]
    
    def update_memory_pressure(self, pressure: float):
        """Update memory pressure for adaptive compression"""
        self.memory_pressure = pressure
        if self.config.adaptive_compression:
            # Adjust compression level based on memory pressure
            if pressure > 0.8:
                self.current_compression_level = 3  # Aggressive
            elif pressure > 0.6:
                self.current_compression_level = 2  # Moderate
            else:
                self.current_compression_level = 1  # Light


class CPACManager:
    """High-level manager for CPAC compression in vLLM"""
    
    def __init__(self, block_size: int, num_gpu_blocks: int,
                 num_kv_heads: int, head_size: int, config: Optional[CPACConfig] = None):
        self.block_size = block_size
        self.num_gpu_blocks = num_gpu_blocks
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.config = config or CPACConfig()
        
        # Initialize compressor
        self.compressor = CrossPageCompressor(
            self.config, block_size, num_kv_heads, head_size
        )
        
        # Track compressed blocks
        self.compressed_blocks: Set[int] = set()
        # Compression statistics
        self.stats = {
            'total_compressions': 0,
            'successful_compressions': 0,
            'average_ratio': 0.0,
            'memory_saved': 0
        }
    
    def should_compress_block(self, block_idx: int, memory_pressure: float) -> bool:
        """Determine if a block should be compressed"""
        if block_idx in self.compressed_blocks:
            return False
        
        # Update memory pressure
        self.compressor.update_memory_pressure(memory_pressure)
        
        # Compress if memory pressure is high or block hasn't been accessed recently
        return memory_pressure > 0.5 or self._is_cold_block(block_idx)
    
    def _is_cold_block(self, block_idx: int) -> bool:
        """Check if block is cold (not recently accessed)"""
        # Placeholder - would integrate with vLLM's access tracking
        return False
    
    def compress_block(self, block_idx: int, key_cache: torch.Tensor,
                      value_cache: torch.Tensor) -> bool:
        """Compress a block and return success status"""
        compression_result = self.compressor.compress_page(
            block_idx, None, key_cache, value_cache
        )
        
        if compression_result:
            self.compressed_blocks.add(block_idx)
            self.stats['total_compressions'] += 1
            self.stats['successful_compressions'] += 1
            self.stats['average_ratio'] = (
                (self.stats['average_ratio'] * (self.stats['successful_compressions'] - 1) +
                 compression_result['ratio']) / self.stats['successful_compressions']
            )
            return True
        
        self.stats['total_compressions'] += 1
        return False
    
    def decompress_block(self, block_idx: int, key_cache_out: torch.Tensor,
                        value_cache_out: torch.Tensor) -> None:
        """Decompress a block for use"""
        if block_idx in self.compressed_blocks:
            self.compressor.decompress_page(block_idx, key_cache_out, value_cache_out)
            # Note: We keep it in compressed_blocks for tracking
    
    def get_compression_stats(self) -> Dict:
        """Get compression statistics"""
        return self.stats.copy()