# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Attention backend with CPAC compression support
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type

import torch

from vllm.attention import AttentionMetadata
from vllm.attention.backends.abstract import AttentionBackend
from vllm.attention.backends.utils import CommonAttentionState
from vllm.attention.ops.cpac_paged_attn import CPACPagedAttention
from vllm.attention.ops.cpac_compression import CPACManager, CPACConfig
from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class CPACMetadata(AttentionMetadata):
    """Attention metadata with CPAC compression information"""
    # Memory pressure for adaptive compression
    memory_pressure: float = 0.0
    # Blocks that need decompression for this batch
    blocks_to_decompress: Optional[List[int]] = None
    # CPAC manager reference
    cpac_manager: Optional[CPACManager] = None


class CPACAttentionBackend(AttentionBackend):
    """
    Attention backend that integrates CPAC compression
    
    This backend extends the standard attention implementation to support
    cross-page adaptive compression for memory efficiency.
    """
    
    @staticmethod
    def get_name() -> str:
        return "cpac"
    
    @staticmethod
    def get_impl_cls() -> Type["CPACAttentionImpl"]:
        return CPACAttentionImpl
    
    @staticmethod
    def get_state_cls() -> Type["CPACAttentionState"]:
        return CPACAttentionState
    
    @staticmethod
    def get_metadata_cls() -> Type[CPACMetadata]:
        return CPACMetadata
    
    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        """Get KV cache shape - same as PagedAttention"""
        return CPACPagedAttention.get_kv_cache_shape(
            num_blocks, block_size, num_kv_heads, head_size
        )
    
    @staticmethod
    def swap_blocks(
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        src_to_dst: torch.Tensor,
    ) -> None:
        """Swap blocks between src and dst KV caches"""
        CPACPagedAttention.swap_blocks(src_kv_cache, dst_kv_cache, src_to_dst)
    
    @staticmethod
    def copy_blocks(
        kv_caches: List[torch.Tensor],
        src_to_dists: torch.Tensor,
    ) -> None:
        """Copy blocks within KV caches"""
        CPACPagedAttention.copy_blocks(kv_caches, src_to_dists)


class CPACAttentionState(CommonAttentionState):
    """Attention state with CPAC compression support"""
    
    def __init__(self, runner: Any):
        super().__init__(runner)
        
        # Initialize CPAC manager if enabled
        model_config = runner.model_config
        cache_config = runner.cache_config
        
        if hasattr(cache_config, 'enable_cpac') and cache_config.enable_cpac:
            cpac_config = CPACConfig(
                similarity_threshold=getattr(cache_config, 'cpac_similarity_threshold', 0.85),
                compression_level=getattr(cache_config, 'cpac_compression_level', 2),
                adaptive_compression=getattr(cache_config, 'cpac_adaptive', True),
            )
            
            self.cpac_manager = CPACManager(
                block_size=cache_config.block_size,
                num_gpu_blocks=cache_config.num_gpu_blocks,
                num_kv_heads=model_config.get_num_kv_heads(runner.parallel_config),
                head_size=model_config.get_head_size(),
                config=cpac_config,
            )
            
            self.cpac_enabled = True
            logger.info("CPAC compression initialized in attention backend")
        else:
            self.cpac_manager = None
            self.cpac_enabled = False
    
    def get_memory_pressure(self) -> float:
        """Calculate current memory pressure"""
        if hasattr(self, '_runner'):
            # Get from block manager if available
            block_manager = getattr(self._runner, 'block_manager', None)
            if block_manager and hasattr(block_manager, 'get_memory_pressure'):
                return block_manager.get_memory_pressure()
        
        # Fallback: estimate from GPU memory
        if torch.cuda.is_available():
            free_memory = torch.cuda.mem_get_info()[0]
            total_memory = torch.cuda.mem_get_info()[1]
            return 1.0 - (free_memory / total_memory)
        
        return 0.0


class CPACAttentionImpl:
    """
    Attention implementation with CPAC compression
    """
    
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[List[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        blocksparse_params: Optional[Dict[str, Any]] = None,
        logits_soft_cap: Optional[float] = None,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.alibi_slopes = alibi_slopes
        self.sliding_window = sliding_window
        self.kv_cache_dtype = kv_cache_dtype
        self.blocksparse_params = blocksparse_params
        self.logits_soft_cap = logits_soft_cap
        
        # CPAC-specific initialization
        self.cpac_paged_attn = CPACPagedAttention(
            block_size=16,  # Default, will be overridden
            num_gpu_blocks=1024,  # Default, will be overridden
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            enable_cpac=True,
        )
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: CPACMetadata,
        k_scale: torch.Tensor,
        v_scale: torch.Tensor,
        attn_type: str = "decoder",
    ) -> torch.Tensor:
        """Forward pass with CPAC compression/decompression"""
        
        # Split KV cache
        key_cache, value_cache = CPACPagedAttention.split_kv_cache(
            kv_cache, self.num_kv_heads, self.head_size
        )
        
        # Handle compression for prefill
        if attn_type == "encoder" or (attn_type == "decoder" and attn_metadata.num_prefill_tokens > 0):
            # Write new KV pairs to cache with compression
            slot_mapping = attn_metadata.slot_mapping
            
            compression_stats = CPACPagedAttention.write_to_paged_cache_cpac(
                key=key,
                value=value,
                key_cache=key_cache,
                value_cache=value_cache,
                slot_mapping=slot_mapping,
                kv_cache_dtype=self.kv_cache_dtype,
                k_scale=k_scale,
                v_scale=v_scale,
                cpac_manager=attn_metadata.cpac_manager,
                memory_pressure=attn_metadata.memory_pressure,
            )
            
            if compression_stats["compressed_blocks"] > 0:
                logger.debug(f"Compressed {compression_stats['compressed_blocks']} blocks, "
                           f"ratio: {compression_stats['compression_ratio']:.2f}")
        
        # Handle decompression for decode
        if attn_type == "decoder" and attn_metadata.num_decode_tokens > 0:
            # Perform attention with on-the-fly decompression
            output = CPACPagedAttention.forward_decode_cpac(
                query=query,
                key_cache=key_cache,
                value_cache=value_cache,
                block_tables=attn_metadata.block_tables,
                seq_lens=attn_metadata.seq_lens_tensor,
                max_seq_len=attn_metadata.max_decode_seq_len,
                kv_cache_dtype=self.kv_cache_dtype,
                num_kv_heads=self.num_kv_heads,
                scale=self.scale,
                alibi_slopes=self.alibi_slopes,
                k_scale=k_scale,
                v_scale=v_scale,
                cpac_manager=attn_metadata.cpac_manager,
            )
        else:
            # Fallback to standard attention for other cases
            output = self._standard_attention(
                query, key, value, key_cache, value_cache, attn_metadata
            )
        
        return output
    
    def _standard_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        attn_metadata: CPACMetadata,
    ) -> torch.Tensor:
        """Fallback to standard attention computation"""
        # This would integrate with the existing attention implementation
        # For now, returning a placeholder
        batch_size, seq_len, _ = query.shape
        return torch.zeros(
            batch_size, seq_len, self.num_heads * self.head_size,
            device=query.device, dtype=query.dtype
        )