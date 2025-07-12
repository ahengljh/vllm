# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Block manager extension for CPAC compression
"""

from typing import Dict, List, Optional, Set, Tuple
import torch
import time

from vllm.core.block_manager import SelfAttnBlockSpaceManager
from vllm.core.block.block_table import BlockTable
from vllm.core.interfaces import AllocStatus
from vllm.sequence import Sequence, SequenceGroup
from vllm.attention.ops.cpac_compression import CPACManager, CPACConfig
from vllm.attention.ops.cpac_ops import cpac_kernel
from vllm.logger import init_logger

logger = init_logger(__name__)


class CPACBlockSpaceManager(SelfAttnBlockSpaceManager):
    """
    Extended block manager with CPAC compression support
    
    This manager tracks block access patterns and applies cross-page
    compression to reduce memory usage.
    """
    
    def __init__(
        self,
        block_size: int,
        num_gpu_blocks: int,
        num_cpu_blocks: int,
        watermark: float = 0.01,
        sliding_window: Optional[int] = None,
        enable_caching: bool = False,
        enable_cpac: bool = True,
        cpac_config: Optional[CPACConfig] = None,
        num_kv_heads: int = 32,
        head_size: int = 128,
    ) -> None:
        super().__init__(
            block_size=block_size,
            num_gpu_blocks=num_gpu_blocks,
            num_cpu_blocks=num_cpu_blocks,
            watermark=watermark,
            sliding_window=sliding_window,
            enable_caching=enable_caching,
        )
        
        self.enable_cpac = enable_cpac
        
        if enable_cpac:
            self.cpac_manager = CPACManager(
                block_size=block_size,
                num_gpu_blocks=num_gpu_blocks,
                num_kv_heads=num_kv_heads,
                head_size=head_size,
                config=cpac_config or CPACConfig(),
            )
            
            # Additional tracking for compression
            self.block_access_counts: Dict[int, int] = {}
            self.block_last_access: Dict[int, float] = {}
            self.compression_candidates: Set[int] = set()
            self.compression_stats = {
                "total_compressions": 0,
                "successful_compressions": 0,
                "memory_saved_mb": 0.0,
                "compression_time_ms": 0.0,
            }
            
            logger.info(f"CPAC compression enabled with config: {cpac_config}")
        else:
            self.cpac_manager = None
    
    def allocate(self, seq_group: SequenceGroup) -> None:
        """Allocate blocks with compression tracking"""
        super().allocate(seq_group)
        
        if not self.enable_cpac:
            return
        
        # Track newly allocated blocks
        for seq in seq_group.get_seqs():
            if seq.seq_id in self.block_tables:
                block_table = self.block_tables[seq.seq_id]
                for block in block_table.blocks:
                    if block is not None:
                        block_idx = block.block_number
                        self.block_access_counts[block_idx] = 0
                        self.block_last_access[block_idx] = time.time()
    
    def access_block(self, block_idx: int) -> None:
        """Track block access for compression decisions"""
        if not self.enable_cpac:
            return
        
        self.block_access_counts[block_idx] = self.block_access_counts.get(block_idx, 0) + 1
        self.block_last_access[block_idx] = time.time()
    
    def get_memory_pressure(self) -> float:
        """Calculate current memory pressure"""
        num_free_gpu_blocks = self.block_allocator.get_num_free_blocks(
            device=torch.device("cuda")
        )
        used_blocks = self.num_total_gpu_blocks - num_free_gpu_blocks
        return used_blocks / self.num_total_gpu_blocks
    
    def identify_compression_candidates(self, 
                                      min_age_seconds: float = 10.0,
                                      max_access_count: int = 5) -> List[int]:
        """Identify blocks suitable for compression"""
        if not self.enable_cpac:
            return []
        
        current_time = time.time()
        candidates = []
        
        for block_idx, last_access in self.block_last_access.items():
            # Skip already compressed blocks
            if block_idx in self.cpac_manager.compressed_blocks:
                continue
            
            age = current_time - last_access
            access_count = self.block_access_counts.get(block_idx, 0)
            
            # Cold blocks are good candidates
            if age > min_age_seconds and access_count < max_access_count:
                candidates.append(block_idx)
        
        # Sort by age (oldest first)
        candidates.sort(key=lambda idx: self.block_last_access[idx])
        
        return candidates
    
    def compress_blocks(self, 
                       key_cache: torch.Tensor,
                       value_cache: torch.Tensor,
                       max_blocks_to_compress: int = 10) -> Dict[str, float]:
        """Compress eligible blocks to save memory"""
        if not self.enable_cpac:
            return {}
        
        start_time = time.time()
        memory_pressure = self.get_memory_pressure()
        
        # Get compression candidates
        candidates = self.identify_compression_candidates()
        
        if not candidates:
            return {"compressed": 0, "memory_saved_mb": 0.0}
        
        # Limit number of compressions per batch
        candidates = candidates[:max_blocks_to_compress]
        
        compressed_count = 0
        for block_idx in candidates:
            if self.cpac_manager.should_compress_block(block_idx, memory_pressure):
                # Extract block data
                block_key = key_cache[block_idx]
                block_value = value_cache[block_idx]
                
                # Attempt compression
                if self.cpac_manager.compress_block(block_idx, block_key, block_value):
                    compressed_count += 1
                    self.compression_stats["successful_compressions"] += 1
                
                self.compression_stats["total_compressions"] += 1
        
        # Calculate memory saved
        bytes_per_block = (self.block_size * self.cpac_manager.num_kv_heads * 
                          self.cpac_manager.head_size * 2 * 2)  # K+V, FP16
        
        avg_ratio = self.cpac_manager.get_compression_stats()["average_ratio"]
        if avg_ratio > 1:
            memory_saved_bytes = compressed_count * bytes_per_block * (1 - 1/avg_ratio)
            memory_saved_mb = memory_saved_bytes / (1024 * 1024)
        else:
            memory_saved_mb = 0.0
        
        compression_time_ms = (time.time() - start_time) * 1000
        self.compression_stats["compression_time_ms"] += compression_time_ms
        self.compression_stats["memory_saved_mb"] += memory_saved_mb
        
        return {
            "compressed": compressed_count,
            "memory_saved_mb": memory_saved_mb,
            "compression_time_ms": compression_time_ms,
            "memory_pressure": memory_pressure,
        }
    
    def decompress_blocks_for_seq(self,
                                 seq_id: int,
                                 key_cache: torch.Tensor,
                                 value_cache: torch.Tensor) -> List[int]:
        """Decompress blocks needed for a sequence"""
        if not self.enable_cpac or seq_id not in self.block_tables:
            return []
        
        block_table = self.block_tables[seq_id]
        decompressed_blocks = []
        
        for block in block_table.blocks:
            if block is not None:
                block_idx = block.block_number
                if block_idx in self.cpac_manager.compressed_blocks:
                    # Decompress block
                    self.cpac_manager.decompress_block(
                        block_idx,
                        key_cache[block_idx],
                        value_cache[block_idx]
                    )
                    decompressed_blocks.append(block_idx)
                    
                    # Update access tracking
                    self.access_block(block_idx)
        
        return decompressed_blocks
    
    def get_compression_stats(self) -> Dict[str, float]:
        """Get detailed compression statistics"""
        if not self.enable_cpac:
            return {
                "compression_enabled": False,
                "compressed_blocks": 0,
                "total_blocks": self.num_total_gpu_blocks,
            }
        
        cpac_stats = self.cpac_manager.get_compression_stats()
        
        return {
            "compression_enabled": True,
            "compressed_blocks": len(self.cpac_manager.compressed_blocks),
            "total_blocks": self.num_total_gpu_blocks,
            "compression_ratio": cpac_stats["average_ratio"],
            "memory_saved_mb": self.compression_stats["memory_saved_mb"],
            "total_compressions": self.compression_stats["total_compressions"],
            "successful_compressions": self.compression_stats["successful_compressions"],
            "compression_success_rate": (
                self.compression_stats["successful_compressions"] / 
                max(1, self.compression_stats["total_compressions"])
            ),
            "avg_compression_time_ms": (
                self.compression_stats["compression_time_ms"] / 
                max(1, self.compression_stats["total_compressions"])
            ),
            "memory_pressure": self.get_memory_pressure(),
        }
    
    def analyze_compression_opportunities(self,
                                        key_cache: torch.Tensor,
                                        value_cache: torch.Tensor,
                                        sample_size: int = 100) -> Dict[str, float]:
        """Analyze potential compression opportunities in current cache"""
        if not self.enable_cpac:
            return {}
        
        # Sample blocks for analysis
        all_blocks = list(self.block_access_counts.keys())
        sample_blocks = all_blocks[:min(sample_size, len(all_blocks))]
        
        if len(sample_blocks) < 2:
            return {"analysis": "Not enough blocks to analyze"}
        
        # Compute features for sampled blocks
        features_list = []
        for block_idx in sample_blocks:
            block_key = key_cache[block_idx]
            features = cpac_kernel.compute_features(block_key)
            features_list.append(features)
        
        features_tensor = torch.stack(features_list)
        
        # Compute pairwise similarities
        from vllm.attention.ops.cpac_ops import batch_compute_similarities
        similarities, indices = batch_compute_similarities(features_tensor, top_k=4)
        
        # Analyze results
        high_similarity_threshold = self.cpac_manager.config.similarity_threshold
        high_similarity_pairs = (similarities > high_similarity_threshold).sum().item()
        total_pairs = len(sample_blocks) * (len(sample_blocks) - 1) / 2
        
        return {
            "sampled_blocks": len(sample_blocks),
            "mean_similarity": similarities.mean().item(),
            "max_similarity": similarities.max().item(),
            "high_similarity_ratio": high_similarity_pairs / max(1, total_pairs),
            "estimated_compressible_blocks": int(
                self.num_total_gpu_blocks * high_similarity_pairs / max(1, total_pairs)
            ),
            "potential_memory_savings_mb": (
                self.num_total_gpu_blocks * high_similarity_pairs / max(1, total_pairs) *
                self.block_size * self.cpac_manager.num_kv_heads * 
                self.cpac_manager.head_size * 2 * 2 * 0.5 / (1024 * 1024)
            ),
        }