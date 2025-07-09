"""KV Cache Compression Manager for vLLM

This module provides the high-level interface for coordinating KV cache compression
operations across the system. It integrates with the existing KV cache management
infrastructure and provides memory optimization through cross-page compression.
"""

import time
from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass, field
from collections import defaultdict
import threading

import torch
from torch import Tensor

from vllm.attention.ops.kv_compression import (
    KVCacheCompressor, CompressedPageInfo, CompressionState
)
from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class CompressionMetrics:
    """Metrics for tracking compression performance."""
    total_pages_processed: int = 0
    total_pages_compressed: int = 0
    total_pages_merged: int = 0
    total_memory_saved_bytes: int = 0
    average_compression_ratio: float = 1.0
    compression_time_ms: float = 0.0
    decompression_time_ms: float = 0.0
    outlier_pages: int = 0
    
    def update_compression_stats(self, 
                               pages_processed: int,
                               pages_compressed: int,
                               pages_merged: int,
                               memory_saved: int,
                               compression_time: float):
        """Update compression statistics."""
        self.total_pages_processed += pages_processed
        self.total_pages_compressed += pages_compressed
        self.total_pages_merged += pages_merged
        self.total_memory_saved_bytes += memory_saved
        self.compression_time_ms += compression_time * 1000  # Convert to ms
        
        if self.total_pages_processed > 0:
            self.average_compression_ratio = (
                (self.total_pages_processed + self.total_pages_merged) / 
                self.total_pages_processed
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for logging."""
        return {
            "total_pages_processed": self.total_pages_processed,
            "total_pages_compressed": self.total_pages_compressed,
            "total_pages_merged": self.total_pages_merged,
            "total_memory_saved_mb": self.total_memory_saved_bytes / (1024 * 1024),
            "average_compression_ratio": self.average_compression_ratio,
            "compression_time_ms": self.compression_time_ms,
            "decompression_time_ms": self.decompression_time_ms,
            "outlier_pages": self.outlier_pages,
            "compression_efficiency": (
                self.total_memory_saved_bytes / max(1, self.total_pages_processed)
            )
        }


@dataclass 
class PageAccessInfo:
    """Information about page access patterns."""
    page_id: int
    last_access_time: float = field(default_factory=time.time)
    access_count: int = 0
    request_ids: Set[str] = field(default_factory=set)
    
    def update_access(self, request_id: str):
        """Update access information."""
        self.last_access_time = time.time()
        self.access_count += 1
        self.request_ids.add(request_id)


class KVCacheCompressionManager:
    """
    Manages KV cache compression across the entire vLLM system.
    
    This class provides the high-level interface for compression operations,
    integrating with existing KV cache managers and attention backends.
    """
    
    def __init__(self, 
                 enable_compression: bool = True,
                 cosine_threshold: float = 0.85,
                 max_merge_ratio: float = 4.0,
                 memory_pressure_threshold: float = 0.8,
                 temporal_window: int = 100,
                 outlier_threshold: float = 0.3,
                 importance_threshold: float = 0.9,
                 compression_interval: float = 1.0,
                 max_compression_batch_size: int = 64):
        """
        Initialize the compression manager.
        
        Args:
            enable_compression: Whether to enable compression
            cosine_threshold: Cosine similarity threshold for merging
            max_merge_ratio: Maximum compression ratio per page
            memory_pressure_threshold: Memory pressure for aggressive compression  
            temporal_window: Window for temporal locality analysis
            outlier_threshold: Threshold for outlier detection
            importance_threshold: Threshold for importance scoring
            compression_interval: Interval between compression runs (seconds)
            max_compression_batch_size: Maximum pages to process per batch
        """
        self.enable_compression = enable_compression
        self.compression_interval = compression_interval
        self.max_compression_batch_size = max_compression_batch_size
        
        # Initialize the core compressor
        self.compressor = KVCacheCompressor(
            cosine_threshold=cosine_threshold,
            max_merge_ratio=max_merge_ratio,
            memory_pressure_threshold=memory_pressure_threshold,
            enable_compression=enable_compression
        )
        
        # Update compressor thresholds
        self.compressor.similarity_analyzer.temporal_window = temporal_window
        self.compressor.merging_strategy.outlier_threshold = outlier_threshold
        self.compressor.merging_strategy.importance_threshold = importance_threshold
        
        # Tracking structures
        self.page_access_info: Dict[int, PageAccessInfo] = {}
        self.layer_compressors: Dict[str, KVCacheCompressor] = {}
        self.metrics = CompressionMetrics()
        
        # Request to page mapping for cleanup
        self.request_to_pages: Dict[str, Set[int]] = defaultdict(set)
        
        # Memory monitoring
        self.last_memory_usage = 0.0
        self.memory_usage_history: List[float] = []
        
        # Threading for background compression
        self._compression_lock = threading.RLock()
        self._background_thread: Optional[threading.Thread] = None
        self._stop_background = threading.Event()
        
        if enable_compression:
            self._start_background_compression()
    
    def _start_background_compression(self):
        """Start background compression thread."""
        if self._background_thread is None or not self._background_thread.is_alive():
            self._stop_background.clear()
            self._background_thread = threading.Thread(
                target=self._background_compression_loop,
                daemon=True
            )
            self._background_thread.start()
            logger.info("Started KV cache compression background thread")
    
    def _background_compression_loop(self):
        """Background thread for periodic compression."""
        while not self._stop_background.wait(self.compression_interval):
            try:
                if self.enable_compression:
                    self._perform_periodic_compression()
            except Exception as e:
                logger.warning(f"Error in background compression: {e}")
    
    def _perform_periodic_compression(self):
        """Perform periodic compression of eligible pages."""
        with self._compression_lock:
            current_memory_pressure = self._calculate_memory_pressure()
            
            # Get pages eligible for compression
            eligible_pages = self._get_eligible_pages_for_compression()
            
            if len(eligible_pages) < 2:
                return  # Need at least 2 pages to merge
            
            # Batch process pages to avoid overwhelming the system
            batch_size = min(self.max_compression_batch_size, len(eligible_pages))
            batch_pages = eligible_pages[:batch_size]
            
            start_time = time.time()
            compressed_count = 0
            merged_count = 0
            
            for page_id in batch_pages:
                if page_id in self.compressor.compressed_pages:
                    result = self.compressor.find_and_merge_similar_pages(
                        page_id, current_memory_pressure
                    )
                    if result is not None:
                        compressed_count += 1
                        merged_count += result
            
            compression_time = time.time() - start_time
            
            # Update metrics
            self.metrics.update_compression_stats(
                pages_processed=len(batch_pages),
                pages_compressed=compressed_count,
                pages_merged=merged_count,
                memory_saved=self._estimate_memory_saved(merged_count),
                compression_time=compression_time
            )
            
            if compressed_count > 0:
                logger.debug(f"Compressed {compressed_count} pages, "
                           f"merged {merged_count} total pages in "
                           f"{compression_time*1000:.2f}ms")
    
    def _get_eligible_pages_for_compression(self) -> List[int]:
        """Get pages eligible for compression based on access patterns."""
        current_time = time.time()
        eligible_pages = []
        
        for page_id, access_info in self.page_access_info.items():
            # Check if page hasn't been accessed recently
            time_since_access = current_time - access_info.last_access_time
            
            # Pages that haven't been accessed for a while are eligible
            if time_since_access > self.compression_interval * 2:
                eligible_pages.append(page_id)
        
        # Sort by last access time (oldest first)
        eligible_pages.sort(
            key=lambda pid: self.page_access_info[pid].last_access_time
        )
        
        return eligible_pages
    
    def _calculate_memory_pressure(self) -> float:
        """Calculate current memory pressure."""
        try:
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated()
                reserved = torch.cuda.memory_reserved()
                if reserved > 0:
                    pressure = allocated / reserved
                    self.last_memory_usage = pressure
                    self.memory_usage_history.append(pressure)
                    
                    # Keep only recent history
                    if len(self.memory_usage_history) > 100:
                        self.memory_usage_history = self.memory_usage_history[-100:]
                    
                    return pressure
        except Exception as e:
            logger.debug(f"Could not calculate memory pressure: {e}")
        
        return self.last_memory_usage
    
    def _estimate_memory_saved(self, merged_pages: int) -> int:
        """Estimate memory saved from merging pages."""
        # Rough estimate: each merged page saves ~75% of its memory
        # This is a simplification - actual savings depend on page content
        estimated_page_size = 64 * 1024  # 64KB per page estimate
        return int(merged_pages * estimated_page_size * 0.75)
    
    def register_page(self, 
                     page_id: int, 
                     layer_name: str,
                     key_cache: Tensor, 
                     value_cache: Tensor,
                     request_id: str) -> CompressedPageInfo:
        """
        Register a new page for compression tracking.
        
        Args:
            page_id: Unique identifier for the page
            layer_name: Name of the attention layer
            key_cache: Key cache tensor
            value_cache: Value cache tensor
            request_id: Request ID this page belongs to
            
        Returns:
            Compressed page information
        """
        if not self.enable_compression:
            return CompressedPageInfo(
                page_id=page_id,
                magnitude=torch.norm(key_cache, p=2, dim=-1, keepdim=True),
                direction=torch.nn.functional.normalize(key_cache, p=2, dim=-1),
                state=CompressionState.UNCOMPRESSED
            )
        
        with self._compression_lock:
            # Get or create layer-specific compressor
            if layer_name not in self.layer_compressors:
                self.layer_compressors[layer_name] = KVCacheCompressor(
                    cosine_threshold=self.compressor.similarity_analyzer.cosine_threshold,
                    max_merge_ratio=self.compressor.merging_strategy.max_merge_ratio,
                    memory_pressure_threshold=self.compressor.merging_strategy.memory_pressure_threshold,
                    enable_compression=True
                )
            
            compressor = self.layer_compressors[layer_name]
            
            # Compress the page
            page_info = compressor.compress_page(page_id, key_cache, value_cache)
            
            # Update access tracking
            if page_id not in self.page_access_info:
                self.page_access_info[page_id] = PageAccessInfo(page_id)
            
            self.page_access_info[page_id].update_access(request_id)
            self.request_to_pages[request_id].add(page_id)
            
            return page_info
    
    def access_page(self, page_id: int, request_id: str) -> Optional[Tuple[Tensor, Tensor]]:
        """
        Access a page and return decompressed KV cache.
        
        Args:
            page_id: Page identifier
            request_id: Request accessing the page
            
        Returns:
            Tuple of (key_cache, value_cache) or None if not found
        """
        if not self.enable_compression:
            return None
        
        with self._compression_lock:
            # Update access tracking
            if page_id in self.page_access_info:
                self.page_access_info[page_id].update_access(request_id)
                self.request_to_pages[request_id].add(page_id)
            
            # Try to decompress from any layer compressor
            for compressor in self.layer_compressors.values():
                result = compressor.decompress_page(page_id)
                if result is not None:
                    start_time = time.time()
                    decompression_time = time.time() - start_time
                    self.metrics.decompression_time_ms += decompression_time * 1000
                    return result
            
            return None
    
    def remove_request_pages(self, request_id: str):
        """
        Remove all pages associated with a request.
        
        Args:
            request_id: Request identifier
        """
        if not self.enable_compression:
            return
        
        with self._compression_lock:
            if request_id in self.request_to_pages:
                page_ids = self.request_to_pages[request_id].copy()
                
                for page_id in page_ids:
                    # Update access info
                    if page_id in self.page_access_info:
                        access_info = self.page_access_info[page_id]
                        access_info.request_ids.discard(request_id)
                        
                        # Remove page if no more requests reference it
                        if not access_info.request_ids:
                            del self.page_access_info[page_id]
                            
                            # Remove from all layer compressors
                            for compressor in self.layer_compressors.values():
                                if page_id in compressor.compressed_pages:
                                    del compressor.compressed_pages[page_id]
                
                del self.request_to_pages[request_id]
    
    def get_compression_stats(self) -> Dict[str, Any]:
        """Get comprehensive compression statistics."""
        stats = self.metrics.to_dict()
        
        # Add per-layer stats
        layer_stats = {}
        for layer_name, compressor in self.layer_compressors.items():
            layer_stats[layer_name] = compressor.get_compression_stats()
        
        stats["per_layer_stats"] = layer_stats
        stats["total_tracked_pages"] = len(self.page_access_info)
        stats["active_requests"] = len(self.request_to_pages)
        stats["current_memory_pressure"] = self._calculate_memory_pressure()
        
        return stats
    
    def force_compression(self, layer_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Force immediate compression of eligible pages.
        
        Args:
            layer_name: Optional layer to target, None for all layers
            
        Returns:
            Compression results
        """
        if not self.enable_compression:
            return {"message": "Compression is disabled"}
        
        with self._compression_lock:
            results = {}
            current_memory_pressure = self._calculate_memory_pressure()
            
            if layer_name and layer_name in self.layer_compressors:
                compressors = [(layer_name, self.layer_compressors[layer_name])]
            else:
                compressors = list(self.layer_compressors.items())
            
            for layer, compressor in compressors:
                layer_results = {"compressed": 0, "merged": 0}
                
                page_ids = list(compressor.compressed_pages.keys())
                for page_id in page_ids:
                    result = compressor.find_and_merge_similar_pages(
                        page_id, current_memory_pressure
                    )
                    if result is not None:
                        layer_results["compressed"] += 1
                        layer_results["merged"] += result
                
                results[layer] = layer_results
            
            return results
    
    def clear_all_cache(self):
        """Clear all compression cache and reset state."""
        with self._compression_lock:
            # Clear all layer compressors
            for compressor in self.layer_compressors.values():
                compressor.clear_cache()
            
            self.layer_compressors.clear()
            self.page_access_info.clear()
            self.request_to_pages.clear()
            self.memory_usage_history.clear()
            
            # Reset metrics
            self.metrics = CompressionMetrics()
            
            logger.info("Cleared all KV cache compression state")
    
    def shutdown(self):
        """Shutdown the compression manager and cleanup resources."""
        self._stop_background.set()
        
        if self._background_thread and self._background_thread.is_alive():
            self._background_thread.join(timeout=5.0)
        
        self.clear_all_cache()
        logger.info("KV cache compression manager shut down")
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        try:
            self.shutdown()
        except Exception:
            pass  # Ignore errors during cleanup 