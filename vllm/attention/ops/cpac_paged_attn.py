# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Integration of CPAC compression with PagedAttention
"""

from typing import Optional, Tuple, List, Dict
import torch
import torch.nn.functional as F

from vllm import _custom_ops as ops
from vllm.attention.ops.paged_attn import PagedAttention, PagedAttentionMetadata
from vllm.attention.ops.cpac_compression import CPACManager, CPACConfig
from vllm.logger import init_logger

logger = init_logger(__name__)

# Import CPAC CUDA kernels
try:
    import vllm._C  # This would need to be extended to include CPAC kernels
    HAS_CPAC_CUDA = True
except ImportError:
    HAS_CPAC_CUDA = False
    logger.warning("CPAC CUDA kernels not available, using fallback implementation")


class CPACPagedAttention(PagedAttention):
    """
    PagedAttention with Cross-Page Adaptive Compression (CPAC)
    
    This extends the standard PagedAttention implementation to support
    cross-page compression for reduced memory usage.
    """
    
    def __init__(self, block_size: int, num_gpu_blocks: int,
                 num_kv_heads: int, head_size: int,
                 enable_cpac: bool = True,
                 cpac_config: Optional[CPACConfig] = None):
        self.block_size = block_size
        self.num_gpu_blocks = num_gpu_blocks
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.enable_cpac = enable_cpac and HAS_CPAC_CUDA
        
        if self.enable_cpac:
            self.cpac_manager = CPACManager(
                block_size=block_size,
                num_gpu_blocks=num_gpu_blocks,
                num_kv_heads=num_kv_heads,
                head_size=head_size,
                config=cpac_config or CPACConfig()
            )
            logger.info(f"CPAC compression enabled with config: {cpac_config}")
        else:
            self.cpac_manager = None
            if enable_cpac and not HAS_CPAC_CUDA:
                logger.warning("CPAC requested but CUDA kernels not available")
    
    @staticmethod
    def write_to_paged_cache_cpac(
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: torch.Tensor,
        v_scale: torch.Tensor,
        cpac_manager: Optional[CPACManager] = None,
        memory_pressure: float = 0.0,
    ) -> Dict[str, float]:
        """
        Write to paged cache with optional CPAC compression
        
        Returns compression statistics
        """
        # First, perform standard cache write
        PagedAttention.write_to_paged_cache(
            key, value, key_cache, value_cache,
            slot_mapping, kv_cache_dtype, k_scale, v_scale
        )
        
        stats = {"compressed_blocks": 0, "compression_ratio": 0.0}
        
        if cpac_manager is None:
            return stats
        
        # Determine which blocks to compress based on memory pressure
        unique_blocks = torch.unique(slot_mapping // cpac_manager.block_size)
        
        for block_idx in unique_blocks:
            block_idx_int = block_idx.item()
            
            if cpac_manager.should_compress_block(block_idx_int, memory_pressure):
                # Extract block data
                block_start = block_idx_int * cpac_manager.block_size
                block_end = block_start + cpac_manager.block_size
                
                # Get slots for this block
                block_slots = torch.where(
                    (slot_mapping >= block_start) & (slot_mapping < block_end)
                )[0]
                
                if len(block_slots) > 0:
                    # Extract key and value data for this block
                    block_key_data = key_cache[block_idx_int]
                    block_value_data = value_cache[block_idx_int]
                    
                    # Attempt compression
                    if cpac_manager.compress_block(
                        block_idx_int, block_key_data, block_value_data
                    ):
                        stats["compressed_blocks"] += 1
        
        # Update statistics
        compression_stats = cpac_manager.get_compression_stats()
        stats["compression_ratio"] = compression_stats["average_ratio"]
        
        return stats
    
    @staticmethod
    def forward_decode_cpac(
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        max_seq_len: int,
        kv_cache_dtype: str,
        num_kv_heads: int,
        scale: float,
        alibi_slopes: Optional[torch.Tensor],
        k_scale: torch.Tensor,
        v_scale: torch.Tensor,
        cpac_manager: Optional[CPACManager] = None,
        tp_rank: int = 0,
        blocksparse_local_blocks: int = 0,
        blocksparse_vert_stride: int = 0,
        blocksparse_block_size: int = 64,
        blocksparse_head_sliding_step: int = 0,
    ) -> torch.Tensor:
        """
        Forward decode with CPAC decompression on-the-fly
        """
        if cpac_manager is None:
            # Fall back to standard PagedAttention
            return PagedAttention.forward_decode(
                query, key_cache, value_cache, block_tables, seq_lens,
                max_seq_len, kv_cache_dtype, num_kv_heads, scale,
                alibi_slopes, k_scale, v_scale, tp_rank,
                blocksparse_local_blocks, blocksparse_vert_stride,
                blocksparse_block_size, blocksparse_head_sliding_step
            )
        
        # Identify which blocks need decompression
        num_seqs = block_tables.shape[0]
        blocks_to_decompress = set()
        
        for seq_idx in range(num_seqs):
            seq_len = seq_lens[seq_idx].item()
            num_blocks_needed = (seq_len + cpac_manager.block_size - 1) // cpac_manager.block_size
            
            for block_idx in range(num_blocks_needed):
                if block_idx < block_tables.shape[1]:
                    physical_block = block_tables[seq_idx, block_idx].item()
                    if physical_block >= 0 and physical_block in cpac_manager.compressed_blocks:
                        blocks_to_decompress.add(physical_block)
        
        # Create temporary decompressed cache if needed
        if blocks_to_decompress:
            # Clone caches for decompression (in practice, use a pool)
            temp_key_cache = key_cache.clone()
            temp_value_cache = value_cache.clone()
            
            # Decompress required blocks
            for block_idx in blocks_to_decompress:
                cpac_manager.decompress_block(
                    block_idx,
                    temp_key_cache[block_idx],
                    temp_value_cache[block_idx]
                )
            
            # Use decompressed caches for attention
            result = PagedAttention.forward_decode(
                query, temp_key_cache, temp_value_cache, block_tables, seq_lens,
                max_seq_len, kv_cache_dtype, num_kv_heads, scale,
                alibi_slopes, k_scale, v_scale, tp_rank,
                blocksparse_local_blocks, blocksparse_vert_stride,
                blocksparse_block_size, blocksparse_head_sliding_step
            )
        else:
            # No decompression needed
            result = PagedAttention.forward_decode(
                query, key_cache, value_cache, block_tables, seq_lens,
                max_seq_len, kv_cache_dtype, num_kv_heads, scale,
                alibi_slopes, k_scale, v_scale, tp_rank,
                blocksparse_local_blocks, blocksparse_vert_stride,
                blocksparse_block_size, blocksparse_head_sliding_step
            )
        
        return result
    
    @staticmethod
    def analyze_cross_page_similarity(
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_indices: List[int],
        cpac_manager: CPACManager
    ) -> Dict[str, float]:
        """
        Analyze cross-page similarity for compression opportunities
        
        This is useful for understanding compression potential and tuning
        """
        if not block_indices:
            return {}
        
        similarity_matrix = torch.zeros((len(block_indices), len(block_indices)))
        
        for i, block_i in enumerate(block_indices):
            for j, block_j in enumerate(block_indices):
                if i < j:  # Only compute upper triangle
                    # Extract block data
                    key_i = key_cache[block_i]
                    key_j = key_cache[block_j]
                    
                    # Compute features (using CUDA kernel if available)
                    if HAS_CPAC_CUDA:
                        feat_i = vllm._C.compute_page_features_cuda(
                            key_i, cpac_manager.config.similarity_threshold
                        )
                        feat_j = vllm._C.compute_page_features_cuda(
                            key_j, cpac_manager.config.similarity_threshold
                        )
                        sim = vllm._C.compute_similarity_cuda(feat_i, feat_j).item()
                    else:
                        # Fallback CPU implementation
                        feat_i = cpac_manager.compressor.similarity_tracker.compute_page_features(key_i)
                        feat_j = cpac_manager.compressor.similarity_tracker.compute_page_features(key_j)
                        sim = F.cosine_similarity(feat_i.unsqueeze(0), feat_j.unsqueeze(0)).item()
                    
                    similarity_matrix[i, j] = sim
                    similarity_matrix[j, i] = sim
        
        # Compute statistics
        stats = {
            "mean_similarity": similarity_matrix.mean().item(),
            "max_similarity": similarity_matrix.max().item(),
            "similarity_std": similarity_matrix.std().item(),
            "high_similarity_pairs": (similarity_matrix > cpac_manager.config.similarity_threshold).sum().item() / 2,
            "compression_potential": (similarity_matrix > cpac_manager.config.similarity_threshold).float().mean().item()
        }
        
        return stats
    
    def get_memory_usage_with_compression(self) -> Dict[str, float]:
        """
        Get memory usage statistics including compression savings
        """
        if self.cpac_manager is None:
            return {
                "total_blocks": self.num_gpu_blocks,
                "compressed_blocks": 0,
                "compression_ratio": 1.0,
                "memory_saved_mb": 0.0
            }
        
        stats = self.cpac_manager.get_compression_stats()
        
        # Calculate memory savings
        bytes_per_block = self.block_size * self.num_kv_heads * self.head_size * 2 * 2  # K+V, FP16
        compressed_blocks = len(self.cpac_manager.compressed_blocks)
        avg_compression_ratio = stats["average_ratio"] if stats["average_ratio"] > 0 else 1.0
        
        memory_saved_bytes = compressed_blocks * bytes_per_block * (1 - 1/avg_compression_ratio)
        memory_saved_mb = memory_saved_bytes / (1024 * 1024)
        
        return {
            "total_blocks": self.num_gpu_blocks,
            "compressed_blocks": compressed_blocks,
            "compression_ratio": avg_compression_ratio,
            "memory_saved_mb": memory_saved_mb,
            "successful_compressions": stats["successful_compressions"],
            "total_compressions": stats["total_compressions"]
        }