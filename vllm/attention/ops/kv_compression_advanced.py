"""Advanced KV Cache Compression for vLLM

This module implements novel compression techniques including:
1. Multi-scale magnitude-direction decomposition
2. Attention-pattern aware compression
3. Learned compression parameters
4. Hierarchical storage with dynamic quantization
"""

import math
from typing import Dict, List, Optional, Tuple, Set, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from vllm.logger import init_logger

logger = init_logger(__name__)


class MultiScaleDecomposer(nn.Module):
    """
    Novel multi-scale decomposition that captures both local and global patterns
    in KV cache, going beyond simple magnitude-direction split.
    """
    
    def __init__(self, 
                 head_dim: int,
                 num_scales: int = 3,
                 learnable: bool = True,
                 epsilon: float = 1e-8):
        super().__init__()
        self.head_dim = head_dim
        self.num_scales = num_scales
        self.epsilon = epsilon
        self.learnable = learnable
        
        # Learnable scale-specific projection matrices
        if learnable:
            self.scale_projections = nn.ModuleList([
                nn.Linear(head_dim, head_dim // (2**i), bias=False)
                for i in range(num_scales)
            ])
            self.scale_reconstructions = nn.ModuleList([
                nn.Linear(head_dim // (2**i), head_dim, bias=False)
                for i in range(num_scales)
            ])
            
            # Initialize with orthogonal matrices for better decomposition
            for proj, recon in zip(self.scale_projections, self.scale_reconstructions):
                nn.init.orthogonal_(proj.weight)
                nn.init.orthogonal_(recon.weight)
        else:
            # Use fixed wavelet-like decomposition
            self.register_buffer('wavelet_filters', self._create_wavelet_filters())
    
    def _create_wavelet_filters(self) -> Tensor:
        """Create Haar wavelet filters for multi-scale decomposition."""
        filters = []
        for scale in range(self.num_scales):
            size = self.head_dim // (2**scale)
            # Simple Haar wavelet
            low_pass = torch.ones(size) / math.sqrt(size)
            high_pass = torch.ones(size)
            high_pass[size//2:] = -1
            high_pass = high_pass / math.sqrt(size)
            filters.append(torch.stack([low_pass, high_pass]))
        return torch.stack(filters)
    
    def decompose(self, kv_vectors: Tensor) -> Dict[str, Tensor]:
        """
        Decompose KV vectors into multi-scale components.
        
        Args:
            kv_vectors: [batch_size, num_heads, seq_len, head_dim]
            
        Returns:
            Dictionary containing:
            - 'magnitude': Global magnitude information
            - 'direction': Normalized direction vectors
            - 'scale_components': List of scale-specific components
            - 'residual': High-frequency residual information
        """
        batch_size, num_heads, seq_len, _ = kv_vectors.shape
        
        # Traditional magnitude-direction decomposition
        magnitude = torch.norm(kv_vectors, p=2, dim=-1, keepdim=True)
        safe_magnitude = torch.clamp(magnitude, min=self.epsilon)
        direction = kv_vectors / safe_magnitude
        
        # Multi-scale decomposition
        scale_components = []
        current_vectors = kv_vectors.view(-1, self.head_dim)  # Flatten for processing
        
        if self.learnable:
            for proj, recon in zip(self.scale_projections, self.scale_reconstructions):
                # Project to lower dimension
                component = proj(current_vectors)
                scale_components.append(component.view(batch_size, num_heads, seq_len, -1))
                
                # Subtract reconstructed component from current vectors
                reconstructed = recon(component)
                current_vectors = current_vectors - reconstructed
        else:
            # Fixed wavelet decomposition
            for scale_filter in self.wavelet_filters:
                # Apply low-pass and high-pass filters
                low_freq = F.conv1d(current_vectors.unsqueeze(1), 
                                   scale_filter[0].unsqueeze(0).unsqueeze(0))
                high_freq = F.conv1d(current_vectors.unsqueeze(1), 
                                    scale_filter[1].unsqueeze(0).unsqueeze(0))
                scale_components.append((low_freq, high_freq))
        
        # Residual contains highest frequency information
        residual = current_vectors.view(batch_size, num_heads, seq_len, -1)
        
        return {
            'magnitude': magnitude,
            'direction': direction,
            'scale_components': scale_components,
            'residual': residual
        }
    
    def recompose(self, components: Dict[str, Tensor]) -> Tensor:
        """Recompose from multi-scale components."""
        # Start with magnitude * direction
        base_reconstruction = components['magnitude'] * components['direction']
        
        if self.learnable and hasattr(self, 'scale_reconstructions'):
            # Add scale components
            for i, (scale_comp, recon) in enumerate(zip(components['scale_components'], 
                                                       self.scale_reconstructions)):
                batch_size, num_heads, seq_len, comp_dim = scale_comp.shape
                flat_comp = scale_comp.view(-1, comp_dim)
                reconstructed = recon(flat_comp)
                base_reconstruction = base_reconstruction + reconstructed.view(
                    batch_size, num_heads, seq_len, -1)
        
        # Add residual
        if 'residual' in components:
            base_reconstruction = base_reconstruction + components['residual']
        
        return base_reconstruction


class AttentionAwareCompressor(nn.Module):
    """
    Novel compression that considers attention patterns and importance scores
    for more intelligent KV cache compression.
    """
    
    def __init__(self,
                 num_heads: int,
                 head_dim: int,
                 compression_ratio: float = 0.5,
                 use_importance_weighting: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.compression_ratio = compression_ratio
        self.use_importance_weighting = use_importance_weighting
        
        # Learnable importance estimator
        if use_importance_weighting:
            self.importance_estimator = nn.Sequential(
                nn.Linear(head_dim, head_dim // 2),
                nn.ReLU(),
                nn.Linear(head_dim // 2, 1),
                nn.Sigmoid()
            )
        
        # Attention pattern analyzer
        self.attention_analyzer = nn.MultiheadAttention(
            embed_dim=head_dim,
            num_heads=1,  # Single head for efficiency
            batch_first=True
        )
    
    def compute_token_importance(self, 
                               key_states: Tensor, 
                               value_states: Tensor,
                               attention_scores: Optional[Tensor] = None) -> Tensor:
        """
        Compute importance scores for each token based on multiple factors.
        
        Args:
            key_states: [batch_size, num_heads, seq_len, head_dim]
            value_states: [batch_size, num_heads, seq_len, head_dim]
            attention_scores: Optional pre-computed attention scores
            
        Returns:
            importance_scores: [batch_size, num_heads, seq_len]
        """
        batch_size, num_heads, seq_len, _ = key_states.shape
        
        # Factor 1: Magnitude-based importance
        key_magnitude = torch.norm(key_states, p=2, dim=-1)
        value_magnitude = torch.norm(value_states, p=2, dim=-1)
        magnitude_importance = (key_magnitude + value_magnitude) / 2
        
        # Factor 2: Learned importance
        if self.use_importance_weighting:
            # Reshape for importance estimator
            combined_states = key_states + value_states  # Simple combination
            flat_states = combined_states.view(-1, self.head_dim)
            learned_importance = self.importance_estimator(flat_states)
            learned_importance = learned_importance.view(batch_size, num_heads, seq_len)
        else:
            learned_importance = torch.ones_like(magnitude_importance)
        
        # Factor 3: Attention-based importance (if available)
        if attention_scores is not None:
            # Average attention received by each token
            attention_importance = attention_scores.mean(dim=-2)  # Average over queries
        else:
            attention_importance = torch.ones_like(magnitude_importance)
        
        # Combine all factors
        importance_scores = (
            0.4 * F.normalize(magnitude_importance, dim=-1) +
            0.4 * learned_importance.squeeze(-1) +
            0.2 * F.normalize(attention_importance, dim=-1)
        )
        
        return importance_scores
    
    def compress_with_importance(self,
                                key_states: Tensor,
                                value_states: Tensor,
                                importance_scores: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Compress KV cache based on importance scores.
        
        Returns:
            compressed_keys, compressed_values, compression_mask
        """
        batch_size, num_heads, seq_len, head_dim = key_states.shape
        compressed_len = int(seq_len * self.compression_ratio)
        
        # Select top-k important tokens
        topk_importance, topk_indices = torch.topk(
            importance_scores, k=compressed_len, dim=-1, sorted=True
        )
        
        # Gather compressed states
        compressed_keys = torch.gather(
            key_states, 
            dim=2, 
            index=topk_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
        )
        compressed_values = torch.gather(
            value_states,
            dim=2,
            index=topk_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
        )
        
        # Create compression mask for reconstruction
        compression_mask = torch.zeros(batch_size, num_heads, seq_len, device=key_states.device)
        compression_mask.scatter_(2, topk_indices, 1.0)
        
        return compressed_keys, compressed_values, compression_mask


class HierarchicalKVStorage:
    """
    Novel hierarchical storage system with multiple compression levels
    for efficient memory usage.
    """
    
    def __init__(self,
                 num_levels: int = 3,
                 compression_ratios: List[float] = [1.0, 0.5, 0.25],
                 quantization_bits: List[int] = [16, 8, 4]):
        self.num_levels = num_levels
        self.compression_ratios = compression_ratios
        self.quantization_bits = quantization_bits
        
        # Storage for each level
        self.storage_levels = [
            {} for _ in range(num_levels)
        ]
        
        # Access statistics for adaptive promotion/demotion
        self.access_counts = {}
        self.last_access_time = {}
        self.current_time = 0
    
    def store(self, 
              page_id: int,
              key_states: Tensor,
              value_states: Tensor,
              importance_scores: Tensor,
              level: int = 0) -> None:
        """Store KV states at specified compression level."""
        if level >= self.num_levels:
            level = self.num_levels - 1
        
        # Apply level-specific compression
        compressed_data = self._compress_for_level(
            key_states, value_states, importance_scores, level
        )
        
        self.storage_levels[level][page_id] = compressed_data
        self.access_counts[page_id] = 0
        self.last_access_time[page_id] = self.current_time
    
    def _compress_for_level(self,
                           key_states: Tensor,
                           value_states: Tensor,
                           importance_scores: Tensor,
                           level: int) -> Dict[str, Tensor]:
        """Apply level-specific compression and quantization."""
        compression_ratio = self.compression_ratios[level]
        quant_bits = self.quantization_bits[level]
        
        # Compression
        if compression_ratio < 1.0:
            compressed_len = int(key_states.shape[2] * compression_ratio)
            topk_indices = torch.topk(importance_scores, k=compressed_len, dim=-1).indices
            
            key_states = torch.gather(
                key_states, dim=2,
                index=topk_indices.unsqueeze(-1).expand(-1, -1, -1, key_states.shape[-1])
            )
            value_states = torch.gather(
                value_states, dim=2,
                index=topk_indices.unsqueeze(-1).expand(-1, -1, -1, value_states.shape[-1])
            )
        else:
            topk_indices = None
        
        # Quantization
        if quant_bits < 16:
            key_states, key_scale = self._quantize(key_states, quant_bits)
            value_states, value_scale = self._quantize(value_states, quant_bits)
        else:
            key_scale = value_scale = None
        
        return {
            'keys': key_states,
            'values': value_states,
            'indices': topk_indices,
            'key_scale': key_scale,
            'value_scale': value_scale,
            'importance': importance_scores,
            'level': level
        }
    
    def _quantize(self, tensor: Tensor, bits: int) -> Tuple[Tensor, Tensor]:
        """Simple symmetric quantization."""
        if bits >= 16:
            return tensor, None
        
        # Compute scale
        max_val = tensor.abs().max()
        scale = max_val / (2**(bits-1) - 1)
        
        # Quantize
        quantized = torch.round(tensor / scale).clamp(-(2**(bits-1)), 2**(bits-1) - 1)
        
        # Convert to appropriate dtype
        if bits == 8:
            quantized = quantized.to(torch.int8)
        elif bits == 4:
            # Pack two 4-bit values into one byte (simplified)
            quantized = quantized.to(torch.int8)
        
        return quantized, scale
    
    def retrieve(self, page_id: int) -> Optional[Tuple[Tensor, Tensor]]:
        """Retrieve and decompress KV states."""
        # Update access statistics
        self.current_time += 1
        if page_id in self.access_counts:
            self.access_counts[page_id] += 1
            self.last_access_time[page_id] = self.current_time
        
        # Find which level contains the page
        for level in range(self.num_levels):
            if page_id in self.storage_levels[level]:
                compressed_data = self.storage_levels[level][page_id]
                
                # Consider promotion to higher level if frequently accessed
                if level > 0 and self.access_counts[page_id] > 10:
                    self._promote_page(page_id, level)
                
                return self._decompress_from_level(compressed_data)
        
        return None
    
    def _decompress_from_level(self, compressed_data: Dict[str, Tensor]) -> Tuple[Tensor, Tensor]:
        """Decompress data from storage format."""
        keys = compressed_data['keys']
        values = compressed_data['values']
        
        # Dequantize if necessary
        if compressed_data['key_scale'] is not None:
            keys = keys.float() * compressed_data['key_scale']
            values = values.float() * compressed_data['value_scale']
        
        return keys, values
    
    def _promote_page(self, page_id: int, current_level: int) -> None:
        """Promote frequently accessed page to less compressed level."""
        if current_level == 0:
            return
        
        # Retrieve and decompress
        compressed_data = self.storage_levels[current_level][page_id]
        keys, values = self._decompress_from_level(compressed_data)
        
        # Remove from current level
        del self.storage_levels[current_level][page_id]
        
        # Store at higher level (less compression)
        self.store(page_id, keys, values, 
                  compressed_data['importance'], level=current_level - 1)


@dataclass
class AdvancedCompressionConfig:
    """Configuration for advanced compression techniques."""
    # Multi-scale decomposition
    num_scales: int = 3
    learnable_decomposition: bool = True
    
    # Attention-aware compression
    use_attention_patterns: bool = True
    importance_weighting: bool = True
    compression_ratio: float = 0.5
    
    # Hierarchical storage
    storage_levels: int = 3
    level_compression_ratios: List[float] = None
    level_quantization_bits: List[int] = None
    
    # Adaptive parameters
    enable_online_learning: bool = True
    update_frequency: int = 100
    
    def __post_init__(self):
        if self.level_compression_ratios is None:
            self.level_compression_ratios = [1.0, 0.5, 0.25]
        if self.level_quantization_bits is None:
            self.level_quantization_bits = [16, 8, 4]


class AdvancedKVCacheCompressor(nn.Module):
    """
    Main class for advanced KV cache compression with novel techniques.
    """
    
    def __init__(self, config: AdvancedCompressionConfig):
        super().__init__()
        self.config = config
        
        # Initialize components
        self.multi_scale_decomposer = MultiScaleDecomposer(
            head_dim=64,  # This should be passed from model config
            num_scales=config.num_scales,
            learnable=config.learnable_decomposition
        )
        
        self.attention_compressor = AttentionAwareCompressor(
            num_heads=32,  # This should be passed from model config
            head_dim=64,
            compression_ratio=config.compression_ratio,
            use_importance_weighting=config.importance_weighting
        )
        
        self.hierarchical_storage = HierarchicalKVStorage(
            num_levels=config.storage_levels,
            compression_ratios=config.level_compression_ratios,
            quantization_bits=config.level_quantization_bits
        )
        
        # Online learning components
        if config.enable_online_learning:
            self.optimizer = torch.optim.AdamW(
                list(self.multi_scale_decomposer.parameters()) +
                list(self.attention_compressor.parameters()),
                lr=1e-4
            )
            self.update_counter = 0
    
    def compress_page(self,
                     page_id: int,
                     key_states: Tensor,
                     value_states: Tensor,
                     attention_scores: Optional[Tensor] = None) -> Dict[str, Any]:
        """
        Compress a KV cache page using advanced techniques.
        """
        # Multi-scale decomposition
        key_components = self.multi_scale_decomposer.decompose(key_states)
        value_components = self.multi_scale_decomposer.decompose(value_states)
        
        # Compute importance scores
        importance_scores = self.attention_compressor.compute_token_importance(
            key_states, value_states, attention_scores
        )
        
        # Compress based on importance
        compressed_keys, compressed_values, compression_mask = \
            self.attention_compressor.compress_with_importance(
                key_states, value_states, importance_scores
            )
        
        # Store in hierarchical storage
        compression_level = self._determine_compression_level(importance_scores)
        self.hierarchical_storage.store(
            page_id, compressed_keys, compressed_values, 
            importance_scores, level=compression_level
        )
        
        return {
            'page_id': page_id,
            'compression_level': compression_level,
            'compression_ratio': compressed_keys.shape[2] / key_states.shape[2],
            'importance_stats': {
                'mean': importance_scores.mean().item(),
                'std': importance_scores.std().item(),
                'max': importance_scores.max().item()
            }
        }
    
    def _determine_compression_level(self, importance_scores: Tensor) -> int:
        """Determine appropriate compression level based on importance."""
        mean_importance = importance_scores.mean().item()
        
        if mean_importance > 0.7:
            return 0  # Minimal compression for important pages
        elif mean_importance > 0.4:
            return 1  # Medium compression
        else:
            return 2  # High compression for less important pages
    
    def decompress_page(self, page_id: int) -> Optional[Tuple[Tensor, Tensor]]:
        """Retrieve and decompress a page."""
        return self.hierarchical_storage.retrieve(page_id)
    
    def update_compression_model(self, 
                               reconstruction_loss: Tensor,
                               importance_loss: Optional[Tensor] = None) -> None:
        """Update compression model parameters online."""
        if not self.config.enable_online_learning:
            return
        
        self.update_counter += 1
        if self.update_counter % self.config.update_frequency == 0:
            total_loss = reconstruction_loss
            if importance_loss is not None:
                total_loss = total_loss + 0.1 * importance_loss
            
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step() 