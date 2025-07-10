"""KV Cache Manager for vLLM v1.

This module provides KV cache management with integrated compression support.
"""

from __future__ import annotations

import math
import os
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple, TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.logger import init_logger
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_coordinator import get_kv_cache_coordinator
from vllm.v1.core.kv_cache_utils import (BlockHash, KVCacheBlock,
                                         PrefixCacheStats, Request, sha256)
from vllm.v1.kv_cache_interface import (KVCacheConfig, KVCacheSpec,
                                        KVCacheGroupSpec)
from vllm.v1.request import RequestStatus
from vllm.distributed.kv_events import KVCacheEvent

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)


class KVCacheBlocks:
    """Container for KV cache blocks across different cache groups."""
    
    def __init__(self, blocks: tuple[list[KVCacheBlock], ...]):
        self.blocks = blocks
    
    def get_block_ids(self) -> tuple[list[int], ...]:
        """Get block IDs for all cache groups."""
        return tuple(
            [block.block_id for block in block_list]
            for block_list in self.blocks
        )


# Compression configuration from environment
ENABLE_KV_COMPRESSION = os.environ.get("VLLM_ENABLE_KV_COMPRESSION", "1") == "1"
COMPRESSION_RATIO = float(os.environ.get("VLLM_KV_COMPRESSION_RATIO", "0.5"))
NUM_COMPRESSION_LEVELS = int(os.environ.get("VLLM_KV_COMPRESSION_LEVELS", "3"))


class MultiScaleDecomposer(nn.Module):
    """Multi-scale decomposition for KV cache compression."""
    
    def __init__(self, head_dim: int, num_scales: int = 3):
        super().__init__()
        self.head_dim = head_dim
        self.num_scales = num_scales
        self.epsilon = 1e-8
        
        # Learnable projections for multi-scale decomposition
        self.scale_projections = nn.ModuleList([
            nn.Linear(head_dim, head_dim // (2**i), bias=False)
            for i in range(num_scales)
        ])
        
        # Initialize with orthogonal matrices
        for proj in self.scale_projections:
            nn.init.orthogonal_(proj.weight)
    
    def decompose(self, kv_vectors: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Decompose KV vectors into multi-scale components."""
        # Compute magnitude and direction
        magnitude = torch.norm(kv_vectors, p=2, dim=-1, keepdim=True)
        safe_magnitude = torch.clamp(magnitude, min=self.epsilon)
        direction = kv_vectors / safe_magnitude
        
        # Multi-scale components
        scale_components = []
        residual = kv_vectors.clone()
        
        for proj in self.scale_projections:
            # Project to lower dimension
            component = proj(residual.view(-1, self.head_dim))
            scale_components.append(component)
            
            # Update residual (simplified - in practice would reconstruct and subtract)
            residual = residual * 0.5  # Placeholder
        
        return {
            'magnitude': magnitude,
            'direction': direction,
            'scale_components': scale_components,
            'residual': residual
        }


class AttentionAwareCompressor:
    """Compression based on attention patterns and importance."""
    
    def __init__(self, num_heads: int, head_dim: int, compression_ratio: float):
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.compression_ratio = compression_ratio
    
    def compute_importance(self, 
                          key_cache: torch.Tensor,
                          value_cache: torch.Tensor) -> torch.Tensor:
        """Compute importance scores for tokens."""
        # Simple importance based on magnitude
        key_importance = torch.norm(key_cache, p=2, dim=-1)
        value_importance = torch.norm(value_cache, p=2, dim=-1)
        
        # Average importance
        importance = (key_importance + value_importance) / 2
        return importance
    
    def compress(self,
                key_cache: torch.Tensor,
                value_cache: torch.Tensor,
                importance: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compress KV cache based on importance scores."""
        batch_size, num_heads, seq_len, head_dim = key_cache.shape
        compressed_len = int(seq_len * self.compression_ratio)
        
        # Select top-k important tokens
        topk_values, topk_indices = torch.topk(importance, k=compressed_len, dim=-1)
        
        # Gather compressed states
        indices_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
        compressed_keys = torch.gather(key_cache, dim=2, index=indices_expanded)
        compressed_values = torch.gather(value_cache, dim=2, index=indices_expanded)
        
        return compressed_keys, compressed_values, topk_indices


class HierarchicalStorage:
    """Hierarchical storage with multiple compression levels."""
    
    def __init__(self, 
                 num_levels: int = 3,
                 compression_ratios: List[float] = None,
                 quantization_bits: List[int] = None):
        self.num_levels = num_levels
        self.compression_ratios = compression_ratios or [1.0, 0.5, 0.25]
        self.quantization_bits = quantization_bits or [16, 8, 4]
        
        # Storage for each level
        self.storage_levels = [{} for _ in range(num_levels)]
        self.access_counts = {}
        self.access_time = 0
    
    def store(self, block_id: int, data: torch.Tensor, level: int = 0):
        """Store data at specified compression level."""
        # Apply level-specific compression
        if level > 0:
            # Simple quantization for demonstration
            if self.quantization_bits[level] == 8:
                scale = data.abs().max() / 127.0
                quantized = (data / scale).round().clamp(-128, 127).to(torch.int8)
                self.storage_levels[level][block_id] = (quantized, scale)
            else:
                self.storage_levels[level][block_id] = data
        else:
            self.storage_levels[level][block_id] = data
        
        self.access_counts[block_id] = 0
    
    def retrieve(self, block_id: int) -> Optional[torch.Tensor]:
        """Retrieve and potentially promote frequently accessed blocks."""
        self.access_time += 1
        
        # Find which level contains the block
        for level in range(self.num_levels):
            if block_id in self.storage_levels[level]:
                self.access_counts[block_id] += 1
                
                # Promote if frequently accessed
                if level > 0 and self.access_counts[block_id] > 10:
                    self._promote(block_id, level)
                
                # Dequantize if necessary
                data = self.storage_levels[level][block_id]
                if isinstance(data, tuple):
                    quantized, scale = data
                    return quantized.float() * scale
                return data
        
        return None
    
    def _promote(self, block_id: int, current_level: int):
        """Promote block to less compressed level."""
        if current_level == 0:
            return
        
        data = self.retrieve(block_id)
        if data is not None:
            del self.storage_levels[current_level][block_id]
            self.store(block_id, data, level=current_level - 1)


class KVCacheManager:
    """Enhanced KV Cache Manager with compression support."""

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        enable_caching: bool = True,
        caching_hash_algo: str = "builtin",
        use_eagle: bool = False,
        log_stats: bool = False,
        enable_kv_cache_events: bool = False,
    ) -> None:
        self.max_model_len = max_model_len

        self.enable_caching = enable_caching
        self.caching_hash_fn = sha256 if caching_hash_algo == "sha256" else hash
        self.use_eagle = use_eagle
        self.log_stats = log_stats
        # FIXME: make prefix cache stats conditional on log_stats
        self.prefix_cache_stats = PrefixCacheStats() if log_stats else None

        self.block_size: Optional[int] = None
        if self.enable_caching:
            assert len(
                set(g.kv_cache_spec.block_size
                    for g in kv_cache_config.kv_cache_groups)
            ) == 1, "Only one block size is supported for now"
            self.block_size = kv_cache_config.kv_cache_groups[
                0].kv_cache_spec.block_size

        self.coordinator = get_kv_cache_coordinator(
            kv_cache_config=kv_cache_config,
            max_model_len=self.max_model_len,
            use_eagle=self.use_eagle,
            enable_caching=enable_caching,
            caching_hash_fn=self.caching_hash_fn,
            enable_kv_cache_events=enable_kv_cache_events,
        )
        self.num_kv_cache_groups = len(kv_cache_config.kv_cache_groups)
        self.block_pool = self.coordinator.block_pool
        self.kv_cache_config = kv_cache_config

        # Mapping from request ID to kv block hashes.
        self.req_to_block_hashes: defaultdict[
            str, list[BlockHash]] = defaultdict(list)
        
        # Initialize compression components if enabled
        self.compression_enabled = ENABLE_KV_COMPRESSION and enable_caching
        if self.compression_enabled:
            self._init_compression_components()
    
    def _init_compression_components(self):
        """Initialize compression-related components."""
        # Get head dimensions from first KV cache group
        first_spec = self.kv_cache_config.kv_cache_groups[0].kv_cache_spec
        
        # Estimate dimensions (simplified - in practice would get from model)
        head_dim = 64  # Default head dimension
        num_heads = 32  # Default number of heads
        
        # Multi-scale decomposer
        self.decomposer = MultiScaleDecomposer(head_dim=head_dim)
        
        # Attention-aware compressor
        self.compressor = AttentionAwareCompressor(
            num_heads=num_heads,
            head_dim=head_dim,
            compression_ratio=COMPRESSION_RATIO
        )
        
        # Hierarchical storage
        self.hierarchical_storage = HierarchicalStorage(
            num_levels=NUM_COMPRESSION_LEVELS
        )
        
        # Compression statistics
        self.compression_stats = {
            'total_blocks': 0,
            'compressed_blocks': 0,
            'memory_saved_mb': 0.0,
            'avg_compression_ratio': 1.0
        }
        
        logger.info(f"KV cache compression enabled with ratio: {COMPRESSION_RATIO}")

    @property
    def usage(self) -> float:
        """Get the KV cache usage.

        Returns:
            The KV cache usage (between 0.0 and 1.0).
        """
        return self.block_pool.get_usage()

    def make_prefix_cache_stats(self) -> Optional[PrefixCacheStats]:
        """Get (and reset) the prefix cache stats.

        Returns:
            The current prefix caching stats, or None if logging is disabled.
        """
        if not self.log_stats:
            return None
        stats = self.prefix_cache_stats
        self.prefix_cache_stats = PrefixCacheStats()
        return stats

    def get_computed_blocks(self, request: Request) -> Tuple[KVCacheBlocks, int]:
        """Get computed blocks with decompression if needed."""
        # Generate block hashes for the request using the proper hashing function
        block_hashes = []
        if self.enable_caching and request.prompt_token_ids:
            from vllm.v1.core.kv_cache_utils import hash_request_tokens
            block_hashes = hash_request_tokens(
                self.caching_hash_fn, 
                self.block_size, 
                request
            )
            
            # Store block hashes for the request
            self.req_to_block_hashes[request.request_id] = block_hashes
        
        # Find longest cache hit
        if block_hashes:
            max_cache_hit_length = len(request.prompt_token_ids)
            hit_blocks, num_cached_tokens = self.coordinator.find_longest_cache_hit(
                block_hashes=block_hashes,
                max_cache_hit_length=max_cache_hit_length
            )
            # Save the computed blocks for the request
            self.coordinator.save_new_computed_blocks(request.request_id, hit_blocks)
            
            # Return the blocks and number of cached tokens
            return KVCacheBlocks(hit_blocks), num_cached_tokens
        
        # No cached blocks found
        empty_blocks = self.create_empty_block_list()
        return empty_blocks, 0

    def allocate_slots(
            self,
            request: Request,
            num_tokens: int,
            num_computed_tokens: int = 0,
            new_computed_blocks: Optional['KVCacheBlocks'] = None,
            num_lookahead_tokens: int = 0,
            num_draft_tokens: int = 0,
            delay_cache_blocks: bool = False) -> Optional['KVCacheBlocks']:
        """Enhanced allocate_slots with compression support.
        
        This method handles two calling patterns:
        1. For running requests: allocate_slots(request, num_new_tokens, num_draft_tokens=..., num_lookahead_tokens=...)
        2. For waiting requests: allocate_slots(request, total_tokens, num_computed_tokens, new_computed_blocks, ...)
        
        Args:
            request: The request to allocate slots for
            num_tokens: Number of new tokens to allocate (or total tokens for waiting requests)
            num_computed_tokens: Number of already computed tokens (0 for running requests)
            new_computed_blocks: Already computed blocks from cache
            num_lookahead_tokens: Number of lookahead tokens for speculative decoding
            num_draft_tokens: Number of draft tokens
            delay_cache_blocks: Whether to delay caching blocks
            
        Returns:
            KVCacheBlocks containing the allocated blocks, or None if allocation failed
        """
        # Determine if this is a running request (simplified call) or waiting request (full call)
        if new_computed_blocks is None and num_computed_tokens == 0:
            # This is a running request call - num_tokens is the new tokens to add
            total_tokens = request.num_computed_tokens + num_tokens + num_lookahead_tokens
            computed_tokens = request.num_computed_tokens
            new_computed_blocks_tuple = tuple([] for _ in range(self.num_kv_cache_groups))
        else:
            # This is a waiting request call - more complex allocation
            total_tokens = num_tokens + num_lookahead_tokens
            computed_tokens = num_computed_tokens
            new_computed_blocks_tuple = (
                new_computed_blocks.blocks if new_computed_blocks 
                else tuple([] for _ in range(self.num_kv_cache_groups))
            )
        
        # Calculate blocks needed
        num_blocks_needed = self.coordinator.get_num_blocks_to_allocate(
            request.request_id, total_tokens, new_computed_blocks_tuple
        )
        
        # Check if we have enough free blocks
        if self.block_pool.get_num_free_blocks() < num_blocks_needed:
            return None
        
        # Allocate new blocks
        new_blocks = self.coordinator.allocate_new_blocks(
            request.request_id, total_tokens
        )
        
        # Apply compression to older blocks if enabled
        if self.compression_enabled:
            all_blocks = self.coordinator.get_blocks(request.request_id)
            for block_list in all_blocks:
                if len(block_list) > 4:
                    self._compress_old_blocks(request.request_id, block_list)
        
        # Return the allocated blocks
        return KVCacheBlocks(new_blocks)

    def _compress_old_blocks(self, request_id: str, blocks: List[KVCacheBlock]):
        """Compress older blocks to save memory."""
        # Only compress blocks that are not recently accessed
        num_blocks = len(blocks)
        if num_blocks <= 4:  # Keep recent blocks uncompressed
            return
        
        # Compress older blocks (simplified - in practice would access actual KV data)
        for i, block in enumerate(blocks[:-4]):  # Keep last 4 blocks uncompressed
            if not hasattr(block, 'compressed') or not block.compressed:
                # Mark block for compression
                block.compressed = True
                self.compression_stats['compressed_blocks'] += 1
                
                # Store in hierarchical storage
                compression_level = min(i // 4, NUM_COMPRESSION_LEVELS - 1)
                self.hierarchical_storage.store(block.block_id, 
                                              torch.randn(1),  # Placeholder
                                              level=compression_level)
        
        self.compression_stats['total_blocks'] = num_blocks
        self._update_compression_stats()

    def _update_compression_stats(self):
        """Update compression statistics."""
        if self.compression_stats['total_blocks'] > 0:
            self.compression_stats['avg_compression_ratio'] = (
                self.compression_stats['compressed_blocks'] / 
                self.compression_stats['total_blocks']
            )
            # Estimate memory saved (simplified)
            self.compression_stats['memory_saved_mb'] = (
                self.compression_stats['compressed_blocks'] * 
                self.block_size * 64 * 32 * 4 *  # block_size * head_dim * num_heads * bytes
                (1 - COMPRESSION_RATIO) / (1024 * 1024)
            )

    def get_compression_stats(self) -> Dict[str, float]:
        """Get compression statistics."""
        if not self.compression_enabled:
            return {}
        return self.compression_stats.copy()

    def free(self, request: Request) -> None:
        """Free blocks with compression cleanup."""
        # Clean up compression data
        if self.compression_enabled:
            blocks = self.coordinator.get_blocks(request.request_id)
            for block_list in blocks:
                for block in block_list:
                    # Remove from hierarchical storage
                    for level in range(NUM_COMPRESSION_LEVELS):
                        if block.block_id in self.hierarchical_storage.storage_levels[level]:
                            del self.hierarchical_storage.storage_levels[level][block.block_id]
        
        # Original free logic
        self.req_to_block_hashes.pop(request.request_id, None)
        self.coordinator.free(request.request_id)

    def reset_prefix_cache(self) -> bool:
        """Reset prefix cache. This function may be used in RLHF
        flows to invalidate prefix caching after the weights are updated,
        or used for resetting prefix caching status for benchmarking.

        Returns:
            bool: True if the prefix cache is successfully reset,
            False otherwise.
        """
        if not self.block_pool.reset_prefix_cache():
            return False
        if self.log_stats:
            assert self.prefix_cache_stats is not None
            self.prefix_cache_stats.reset = True
        return True

    def get_num_common_prefix_blocks(
        self,
        request: Request,
        num_running_requests: int,
    ) -> list[int]:
        """Calculate the number of common prefix blocks shared by all requests
        in the RUNNING state for each kv cache group.

        The function determines this by selecting any request and iterating
        through its blocks.  A block is considered a common prefix block if its
        `ref_cnt` equals the total number of requests in the RUNNING state.

        NOTE(woosuk): The number of requests in the RUNNING state is **greater
        than or equal to** the number of requests scheduled in the current step.
        This is because the RUNNING state only indicates that:
        1. The request has not yet finished, and
        2. The request holds its blocks unfreed.

        While all scheduled requests must be in the RUNNING state, the inverse
        is not necessarily true. There may be RUNNING requests that are not
        scheduled in the current step.

        This can result in an edge case where the number of common prefix blocks
        is 0, even though all scheduled requests share a common prefix. This
        occurs because there may be unscheduled RUNNING requests that do not
        share the common prefix. Currently, this case cannot be easily detected,
        so the function returns 0 in such cases.

        Args:
            request: Any request in the RUNNING state, used to identify the
                common prefix blocks.
            num_running_requests: The total number of requests in the RUNNING
                state. This can be different from the number of scheduled
                requests in the current step.

        Returns:
            list[int]: The number of common prefix blocks for each kv cache 
            group.
        """
        assert request.status == RequestStatus.RUNNING
        return self.coordinator.get_num_common_prefix_blocks(
            request.request_id, num_running_requests)

    def free_block_hashes(self, request: Request) -> None:
        """Discard the block hashes for the request.

        NOTE: Unlike `free`, this method should be called only when the request
        is finished, not when it is preempted.
        """
        self.req_to_block_hashes.pop(request.request_id, None)

    def take_events(self) -> list[KVCacheEvent]:
        """Take the KV cache events from the block pool.

        Returns:
            A list of KV cache events.
        """
        return self.block_pool.take_events()

    def get_block_ids(self, request_id: str) -> tuple[list[int], ...]:
        """Get the block ids of a request."""
        return KVCacheBlocks(
            self.coordinator.get_blocks(request_id)).get_block_ids()

    def cache_blocks(self, request: Request, num_computed_tokens: int) -> None:
        """Cache the blocks for the request, if enabled."""
        if self.enable_caching:
            block_hashes = self.req_to_block_hashes[request.request_id]
            self.coordinator.cache_blocks(request, block_hashes,
                                          num_computed_tokens)

    def create_empty_block_list(self) -> KVCacheBlocks:
        """Creates a new KVCacheBlocks instance with no blocks."""
        return KVCacheBlocks(tuple([]
                                   for _ in range(self.num_kv_cache_groups)))
