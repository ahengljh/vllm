"""Cross-Page KV Cache Compression for vLLM

This module implements cross-page KV cache compression using magnitude-direction
decomposition inspired by MiniCache. The core idea is to identify and exploit
redundancy across different pages within the same layer.
"""

import math
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn.functional as F
from torch import Tensor

from vllm.logger import init_logger

logger = init_logger(__name__)


class CompressionState(Enum):
    """States for page compression"""
    UNCOMPRESSED = "uncompressed"
    COMPRESSED = "compressed"
    MERGEABLE = "mergeable"
    OUTLIER = "outlier"


@dataclass
class CompressedPageInfo:
    """Information about a compressed page"""
    page_id: int
    magnitude: Tensor  # Shape: [block_size, num_heads, head_dim]
    direction: Tensor  # Shape: [block_size, num_heads, head_dim] (normalized)
    merged_with: Optional[List[int]] = None  # List of page IDs merged with this one
    similarity_scores: Optional[Dict[int, float]] = None
    compression_ratio: float = 1.0
    state: CompressionState = CompressionState.UNCOMPRESSED


class MagnitudeDirectionDecomposer:
    """
    Decomposes KV cache vectors into magnitude and direction components
    following MiniCache's approach.
    """
    
    def __init__(self, epsilon: float = 1e-8):
        self.epsilon = epsilon
    
    def decompose(self, kv_vectors: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Decompose KV vectors into magnitude and direction components.
        
        Args:
            kv_vectors: Input tensor of shape [block_size, num_heads, head_dim]
            
        Returns:
            magnitude: Scalar values representing vector norms [block_size, num_heads, 1]
            direction: Normalized vectors [block_size, num_heads, head_dim]
        """
        # Compute L2 norm along the last dimension (head_dim)
        magnitude = torch.norm(kv_vectors, p=2, dim=-1, keepdim=True)
        
        # Avoid division by zero
        safe_magnitude = torch.clamp(magnitude, min=self.epsilon)
        
        # Normalize to get direction
        direction = kv_vectors / safe_magnitude
        
        return magnitude, direction
    
    def recompose(self, magnitude: Tensor, direction: Tensor) -> Tensor:
        """
        Recompose magnitude and direction back to original vectors.
        
        Args:
            magnitude: Scalar values [block_size, num_heads, 1]
            direction: Normalized vectors [block_size, num_heads, head_dim]
            
        Returns:
            Reconstructed vectors [block_size, num_heads, head_dim]
        """
        return magnitude * direction


class CrossPageSimilarityAnalyzer:
    """
    Analyzes similarity between pages to identify mergeable candidates.
    """
    
    def __init__(self, 
                 cosine_threshold: float = 0.85,
                 attention_threshold: float = 0.8,
                 temporal_window: int = 100):
        self.cosine_threshold = cosine_threshold
        self.attention_threshold = attention_threshold
        self.temporal_window = temporal_window
    
    def compute_directional_similarity(self, 
                                     direction1: Tensor, 
                                     direction2: Tensor) -> float:
        """
        Compute cosine similarity between directional components.
        
        Args:
            direction1: First page directions [block_size, num_heads, head_dim]
            direction2: Second page directions [block_size, num_heads, head_dim]
            
        Returns:
            Average cosine similarity score
        """
        # Flatten to [block_size * num_heads, head_dim]
        flat_dir1 = direction1.reshape(-1, direction1.shape[-1])
        flat_dir2 = direction2.reshape(-1, direction2.shape[-1])
        
        # Compute cosine similarity
        similarities = F.cosine_similarity(flat_dir1, flat_dir2, dim=-1)
        
        # Return average similarity
        return similarities.mean().item()
    
    def compute_attention_pattern_similarity(self, 
                                           page1_attns: Optional[Tensor], 
                                           page2_attns: Optional[Tensor]) -> float:
        """
        Compute similarity based on attention patterns.
        
        Args:
            page1_attns: Attention patterns for page 1 [num_heads, seq_len]
            page2_attns: Attention patterns for page 2 [num_heads, seq_len]
            
        Returns:
            Attention pattern similarity score
        """
        if page1_attns is None or page2_attns is None:
            return 0.0
        
        # Normalize attention patterns
        norm_attn1 = F.softmax(page1_attns, dim=-1)
        norm_attn2 = F.softmax(page2_attns, dim=-1)
        
        # Compute KL divergence as similarity measure
        kl_div = F.kl_div(norm_attn1.log(), norm_attn2, reduction='mean')
        
        # Convert to similarity score (0-1 range)
        similarity = torch.exp(-kl_div).item()
        
        return similarity
    
    def is_temporal_neighbor(self, page1_id: int, page2_id: int) -> bool:
        """Check if two pages are temporal neighbors."""
        return abs(page1_id - page2_id) <= self.temporal_window
    
    def find_similar_pages(self, 
                          target_page: CompressedPageInfo,
                          candidate_pages: List[CompressedPageInfo],
                          attention_patterns: Optional[Dict[int, Tensor]] = None) -> List[Tuple[int, float]]:
        """
        Find pages similar to the target page.
        
        Args:
            target_page: Page to find similarities for
            candidate_pages: List of candidate pages
            attention_patterns: Optional attention patterns for each page
            
        Returns:
            List of (page_id, similarity_score) tuples sorted by similarity
        """
        similarities = []
        
        for candidate in candidate_pages:
            if candidate.page_id == target_page.page_id:
                continue
                
            # Skip if not temporal neighbors (for efficiency)
            if not self.is_temporal_neighbor(target_page.page_id, candidate.page_id):
                continue
            
            # Compute directional similarity
            dir_sim = self.compute_directional_similarity(
                target_page.direction, candidate.direction
            )
            
            # Compute attention pattern similarity if available
            attn_sim = 0.0
            if attention_patterns:
                target_attn = attention_patterns.get(target_page.page_id)
                candidate_attn = attention_patterns.get(candidate.page_id)
                attn_sim = self.compute_attention_pattern_similarity(
                    target_attn, candidate_attn
                )
            
            # Combine similarities (weighted average)
            combined_sim = 0.7 * dir_sim + 0.3 * attn_sim
            
            # Only consider if above threshold
            if combined_sim >= self.cosine_threshold:
                similarities.append((candidate.page_id, combined_sim))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities


class SelectiveMergingStrategy:
    """
    Implements selective merging strategy with outlier detection
    and importance scoring.
    """
    
    def __init__(self, 
                 max_merge_ratio: float = 4.0,
                 importance_threshold: float = 0.9,
                 outlier_threshold: float = 0.3,
                 memory_pressure_threshold: float = 0.8):
        self.max_merge_ratio = max_merge_ratio
        self.importance_threshold = importance_threshold
        self.outlier_threshold = outlier_threshold
        self.memory_pressure_threshold = memory_pressure_threshold
    
    def compute_page_importance(self, page_info: CompressedPageInfo) -> float:
        """
        Compute importance score for a page.
        
        Args:
            page_info: Page information
            
        Returns:
            Importance score (0-1, higher is more important)
        """
        # Base importance on magnitude variance (higher variance = more important)
        magnitude_var = torch.var(page_info.magnitude).item()
        
        # Normalize importance score
        importance = min(magnitude_var / 100.0, 1.0)  # Adjust scaling as needed
        
        return importance
    
    def is_outlier(self, page_info: CompressedPageInfo, 
                   candidate_pages: List[CompressedPageInfo]) -> bool:
        """
        Determine if a page is an outlier that should not be compressed.
        
        Args:
            page_info: Page to check
            candidate_pages: Other pages for comparison
            
        Returns:
            True if page is an outlier
        """
        if not candidate_pages:
            return True
        
        # Compute average similarity with all other pages
        total_similarity = 0.0
        count = 0
        
        for candidate in candidate_pages:
            if candidate.page_id != page_info.page_id:
                # Simple direction similarity check
                sim = F.cosine_similarity(
                    page_info.direction.flatten(),
                    candidate.direction.flatten(),
                    dim=0
                ).item()
                total_similarity += sim
                count += 1
        
        if count == 0:
            return True
        
        avg_similarity = total_similarity / count
        
        # Mark as outlier if average similarity is below threshold
        return avg_similarity < self.outlier_threshold
    
    def should_merge(self, 
                    target_page: CompressedPageInfo,
                    candidate_page: CompressedPageInfo,
                    similarity_score: float,
                    memory_pressure: float) -> bool:
        """
        Determine if two pages should be merged.
        
        Args:
            target_page: Target page for merging
            candidate_page: Candidate page to merge with
            similarity_score: Similarity score between pages
            memory_pressure: Current memory pressure (0-1)
            
        Returns:
            True if pages should be merged
        """
        # Don't merge outliers
        if (target_page.state == CompressionState.OUTLIER or 
            candidate_page.state == CompressionState.OUTLIER):
            return False
        
        # Don't merge high importance pages unless under high memory pressure
        target_importance = self.compute_page_importance(target_page)
        candidate_importance = self.compute_page_importance(candidate_page)
        
        if (max(target_importance, candidate_importance) > self.importance_threshold 
            and memory_pressure < self.memory_pressure_threshold):
            return False
        
        # Check if already compressed too much
        current_ratio = max(target_page.compression_ratio, 
                          candidate_page.compression_ratio)
        if current_ratio >= self.max_merge_ratio:
            return False
        
        return True
    
    def merge_pages(self, 
                   target_page: CompressedPageInfo,
                   source_pages: List[CompressedPageInfo]) -> CompressedPageInfo:
        """
        Merge multiple pages into a single compressed representation.
        
        Args:
            target_page: Primary page to merge into
            source_pages: Pages to merge into target
            
        Returns:
            Updated target page with merged information
        """
        all_pages = [target_page] + source_pages
        
        # Average the directions (weighted by magnitude)
        total_weight = torch.zeros_like(target_page.magnitude)
        weighted_direction = torch.zeros_like(target_page.direction)
        
        for page in all_pages:
            weight = page.magnitude
            total_weight += weight
            weighted_direction += weight * page.direction
        
        # Avoid division by zero
        safe_total_weight = torch.clamp(total_weight, min=1e-8)
        merged_direction = weighted_direction / safe_total_weight
        
        # Normalize the direction
        merged_direction = F.normalize(merged_direction, p=2, dim=-1)
        
        # Use the maximum magnitude as representative
        merged_magnitude = torch.max(
            torch.stack([page.magnitude for page in all_pages]), dim=0
        )[0]
        
        # Update target page
        target_page.direction = merged_direction
        target_page.magnitude = merged_magnitude
        target_page.merged_with = [page.page_id for page in source_pages]
        target_page.compression_ratio = len(all_pages)
        target_page.state = CompressionState.COMPRESSED
        
        return target_page


class KVCacheCompressor:
    """
    Main class that coordinates KV cache compression operations.
    """
    
    def __init__(self, 
                 cosine_threshold: float = 0.85,
                 max_merge_ratio: float = 4.0,
                 memory_pressure_threshold: float = 0.8,
                 enable_compression: bool = True):
        
        self.enable_compression = enable_compression
        self.decomposer = MagnitudeDirectionDecomposer()
        self.similarity_analyzer = CrossPageSimilarityAnalyzer(
            cosine_threshold=cosine_threshold
        )
        self.merging_strategy = SelectiveMergingStrategy(
            max_merge_ratio=max_merge_ratio,
            memory_pressure_threshold=memory_pressure_threshold
        )
        
        # Cache for compressed pages
        self.compressed_pages: Dict[int, CompressedPageInfo] = {}
        self.uncompressed_pages: Set[int] = set()
        
    def compress_page(self, 
                     page_id: int, 
                     key_cache: Tensor, 
                     value_cache: Tensor) -> CompressedPageInfo:
        """
        Compress a single page using magnitude-direction decomposition.
        
        Args:
            page_id: Unique identifier for the page
            key_cache: Key cache tensor [block_size, num_heads, head_dim]
            value_cache: Value cache tensor [block_size, num_heads, head_dim]
            
        Returns:
            Compressed page information
        """
        if not self.enable_compression:
            # Return uncompressed info
            return CompressedPageInfo(
                page_id=page_id,
                magnitude=torch.norm(key_cache, p=2, dim=-1, keepdim=True),
                direction=F.normalize(key_cache, p=2, dim=-1),
                state=CompressionState.UNCOMPRESSED
            )
        
        # Decompose key and value caches
        key_magnitude, key_direction = self.decomposer.decompose(key_cache)
        value_magnitude, value_direction = self.decomposer.decompose(value_cache)
        
        # For now, store key information (can be extended to handle both)
        page_info = CompressedPageInfo(
            page_id=page_id,
            magnitude=key_magnitude,
            direction=key_direction,
            state=CompressionState.MERGEABLE
        )
        
        self.compressed_pages[page_id] = page_info
        
        return page_info
    
    def find_and_merge_similar_pages(self, 
                                   target_page_id: int,
                                   memory_pressure: float = 0.5) -> Optional[int]:
        """
        Find similar pages and merge them if appropriate.
        
        Args:
            target_page_id: ID of the page to find similarities for
            memory_pressure: Current memory pressure (0-1)
            
        Returns:
            Number of pages merged, or None if no merging occurred
        """
        if not self.enable_compression or target_page_id not in self.compressed_pages:
            return None
        
        target_page = self.compressed_pages[target_page_id]
        
        # Check if this page is an outlier
        candidate_pages = list(self.compressed_pages.values())
        if self.merging_strategy.is_outlier(target_page, candidate_pages):
            target_page.state = CompressionState.OUTLIER
            return None
        
        # Find similar pages
        similar_pages = self.similarity_analyzer.find_similar_pages(
            target_page, candidate_pages
        )
        
        if not similar_pages:
            return None
        
        # Select pages to merge
        pages_to_merge = []
        for page_id, similarity in similar_pages:
            candidate_page = self.compressed_pages[page_id]
            
            if self.merging_strategy.should_merge(
                target_page, candidate_page, similarity, memory_pressure
            ):
                pages_to_merge.append(candidate_page)
                
                # Limit number of merges per operation
                if len(pages_to_merge) >= 3:  # Max 4 pages total (target + 3)
                    break
        
        if not pages_to_merge:
            return None
        
        # Perform the merge
        merged_page = self.merging_strategy.merge_pages(target_page, pages_to_merge)
        
        # Remove merged pages from cache
        for page in pages_to_merge:
            if page.page_id in self.compressed_pages:
                del self.compressed_pages[page.page_id]
        
        # Update the merged page
        self.compressed_pages[target_page_id] = merged_page
        
        logger.info(f"Merged {len(pages_to_merge)} pages into page {target_page_id}, "
                   f"compression ratio: {merged_page.compression_ratio:.2f}")
        
        return len(pages_to_merge)
    
    def decompress_page(self, page_id: int) -> Optional[Tuple[Tensor, Tensor]]:
        """
        Decompress a page back to its original KV cache representation.
        
        Args:
            page_id: ID of the page to decompress
            
        Returns:
            Tuple of (key_cache, value_cache) or None if not found
        """
        if page_id not in self.compressed_pages:
            return None
        
        page_info = self.compressed_pages[page_id]
        
        # Recompose the key cache
        key_cache = self.decomposer.recompose(
            page_info.magnitude, page_info.direction
        )
        
        # For now, use the same for value cache (can be improved)
        value_cache = key_cache.clone()
        
        return key_cache, value_cache
    
    def get_compression_stats(self) -> Dict[str, float]:
        """Get compression statistics."""
        if not self.compressed_pages:
            return {
                "total_pages": 0,
                "compressed_pages": 0,
                "average_compression_ratio": 1.0,
                "memory_saved_ratio": 0.0
            }
        
        total_pages = len(self.compressed_pages)
        compressed_pages = sum(
            1 for page in self.compressed_pages.values() 
            if page.state == CompressionState.COMPRESSED
        )
        
        total_ratio = sum(
            page.compression_ratio for page in self.compressed_pages.values()
        )
        avg_compression_ratio = total_ratio / total_pages if total_pages > 0 else 1.0
        
        # Estimate memory saved (simplified calculation)
        memory_saved_ratio = 1.0 - (1.0 / avg_compression_ratio) if avg_compression_ratio > 1 else 0.0
        
        return {
            "total_pages": total_pages,
            "compressed_pages": compressed_pages,
            "average_compression_ratio": avg_compression_ratio,
            "memory_saved_ratio": memory_saved_ratio
        }
    
    def clear_cache(self):
        """Clear all compressed page information."""
        self.compressed_pages.clear()
        self.uncompressed_pages.clear() 