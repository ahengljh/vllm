"""Memory Management for Compressed KV Cache

This module provides memory management and eviction policies for compressed
KV cache pages, ensuring optimal memory usage and performance under
varying memory pressure conditions.
"""

import time
import threading
from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass, field
from enum import Enum
from collections import OrderedDict
import heapq

import torch
from torch import Tensor

from vllm.attention.ops.kv_compression import CompressedPageInfo, CompressionState
from vllm.logger import init_logger

logger = init_logger(__name__)


class EvictionPolicy(Enum):
    """Eviction policies for compressed pages."""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used  
    IMPORTANCE_BASED = "importance_based"  # Based on page importance scores
    HYBRID = "hybrid"  # Combination of access patterns and importance


@dataclass
class MemoryStats:
    """Memory usage statistics."""
    total_memory_bytes: int = 0
    compressed_memory_bytes: int = 0
    uncompressed_memory_bytes: int = 0
    memory_saved_bytes: int = 0
    num_compressed_pages: int = 0
    num_uncompressed_pages: int = 0
    compression_ratio: float = 1.0
    memory_pressure: float = 0.0
    evicted_pages: int = 0
    cache_hit_rate: float = 0.0


@dataclass 
class PageMemoryInfo:
    """Memory information for a page."""
    page_id: int
    size_bytes: int
    compressed_size_bytes: int
    last_access_time: float = field(default_factory=time.time)
    access_count: int = 0
    importance_score: float = 0.0
    eviction_priority: float = 0.0
    is_compressed: bool = False
    request_ids: Set[str] = field(default_factory=set)
    
    def update_access(self, request_id: str):
        """Update access information."""
        self.last_access_time = time.time()
        self.access_count += 1
        self.request_ids.add(request_id)
    
    def calculate_eviction_priority(self, policy: EvictionPolicy, current_time: float) -> float:
        """Calculate eviction priority based on policy."""
        if policy == EvictionPolicy.LRU:
            # Higher priority = more likely to be evicted
            return current_time - self.last_access_time
        elif policy == EvictionPolicy.LFU:
            # Lower access count = higher priority for eviction
            return 1.0 / max(1, self.access_count)
        elif policy == EvictionPolicy.IMPORTANCE_BASED:
            # Lower importance = higher priority for eviction
            return 1.0 - self.importance_score
        elif policy == EvictionPolicy.HYBRID:
            # Combine multiple factors
            time_factor = (current_time - self.last_access_time) / 3600.0  # Hours
            freq_factor = 1.0 / max(1, self.access_count)
            importance_factor = 1.0 - self.importance_score
            return 0.4 * time_factor + 0.3 * freq_factor + 0.3 * importance_factor
        else:
            return 0.0


class CompressedPageCache:
    """
    Cache for compressed pages with LRU eviction.
    """
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache: OrderedDict[int, CompressedPageInfo] = OrderedDict()
        self.access_lock = threading.RLock()
    
    def get(self, page_id: int) -> Optional[CompressedPageInfo]:
        """Get a page from cache, updating LRU order."""
        with self.access_lock:
            if page_id in self.cache:
                # Move to end (most recently used)
                page = self.cache.pop(page_id)
                self.cache[page_id] = page
                return page
            return None
    
    def put(self, page_id: int, page_info: CompressedPageInfo):
        """Add a page to cache, evicting if necessary."""
        with self.access_lock:
            if page_id in self.cache:
                # Update existing entry
                self.cache.pop(page_id)
            elif len(self.cache) >= self.max_size:
                # Evict least recently used
                evicted_id, _ = self.cache.popitem(last=False)
                logger.debug(f"Evicted compressed page {evicted_id} from cache")
            
            self.cache[page_id] = page_info
    
    def remove(self, page_id: int) -> Optional[CompressedPageInfo]:
        """Remove a page from cache."""
        with self.access_lock:
            return self.cache.pop(page_id, None)
    
    def clear(self):
        """Clear all cached pages."""
        with self.access_lock:
            self.cache.clear()
    
    def size(self) -> int:
        """Get current cache size."""
        return len(self.cache)
    
    def get_all_page_ids(self) -> List[int]:
        """Get all page IDs in cache."""
        with self.access_lock:
            return list(self.cache.keys())


class KVMemoryManager:
    """
    Memory manager for compressed KV cache with eviction policies.
    """
    
    def __init__(self,
                 max_memory_bytes: int = 1024 * 1024 * 1024,  # 1GB default
                 eviction_policy: EvictionPolicy = EvictionPolicy.HYBRID,
                 memory_pressure_threshold: float = 0.8,
                 eviction_batch_size: int = 32,
                 enable_background_eviction: bool = True):
        """
        Initialize memory manager.
        
        Args:
            max_memory_bytes: Maximum memory for compressed pages
            eviction_policy: Policy for choosing pages to evict
            memory_pressure_threshold: Threshold for triggering eviction
            eviction_batch_size: Number of pages to evict at once
            enable_background_eviction: Whether to run background eviction
        """
        self.max_memory_bytes = max_memory_bytes
        self.eviction_policy = eviction_policy
        self.memory_pressure_threshold = memory_pressure_threshold
        self.eviction_batch_size = eviction_batch_size
        self.enable_background_eviction = enable_background_eviction
        
        # Memory tracking
        self.page_memory_info: Dict[int, PageMemoryInfo] = {}
        self.total_memory_used = 0
        self.stats = MemoryStats()
        
        # Compressed page cache
        self.compressed_cache = CompressedPageCache(max_size=10000)
        
        # Access tracking for cache hit rate
        self.total_accesses = 0
        self.cache_hits = 0
        
        # Threading for background tasks
        self.memory_lock = threading.RLock()
        self.background_thread: Optional[threading.Thread] = None
        self.stop_background = threading.Event()
        
        if enable_background_eviction:
            self._start_background_eviction()
    
    def _start_background_eviction(self):
        """Start background eviction thread."""
        if self.background_thread is None or not self.background_thread.is_alive():
            self.stop_background.clear()
            self.background_thread = threading.Thread(
                target=self._background_eviction_loop,
                daemon=True
            )
            self.background_thread.start()
            logger.info("Started KV cache memory manager background thread")
    
    def _background_eviction_loop(self):
        """Background thread for periodic memory management."""
        while not self.stop_background.wait(5.0):  # Check every 5 seconds
            try:
                self._periodic_memory_management()
            except Exception as e:
                logger.warning(f"Error in background memory management: {e}")
    
    def _periodic_memory_management(self):
        """Perform periodic memory management tasks."""
        with self.memory_lock:
            # Update memory pressure
            self._update_memory_pressure()
            
            # Trigger eviction if needed
            if self.stats.memory_pressure > self.memory_pressure_threshold:
                self._evict_pages_if_needed()
            
            # Update cache hit rate
            self._update_cache_hit_rate()
            
            # Log stats periodically
            if self.total_accesses > 0 and self.total_accesses % 1000 == 0:
                logger.debug(f"Memory stats: {self.get_memory_stats()}")
    
    def register_page(self, 
                     page_id: int,
                     page_info: CompressedPageInfo,
                     size_bytes: int,
                     request_id: str) -> bool:
        """
        Register a page in memory management.
        
        Args:
            page_id: Page identifier
            page_info: Compressed page information
            size_bytes: Original size in bytes
            request_id: Request identifier
            
        Returns:
            True if page was registered successfully
        """
        with self.memory_lock:
            # Calculate compressed size
            compressed_size = self._estimate_compressed_size(page_info, size_bytes)
            
            # Check if we have enough memory
            if not self._can_allocate(compressed_size):
                # Try to free memory by evicting pages
                if not self._evict_pages_for_allocation(compressed_size):
                    logger.warning(f"Cannot allocate memory for page {page_id}")
                    return False
            
            # Create memory info
            memory_info = PageMemoryInfo(
                page_id=page_id,
                size_bytes=size_bytes,
                compressed_size_bytes=compressed_size,
                importance_score=self._calculate_page_importance(page_info),
                is_compressed=(page_info.state == CompressionState.COMPRESSED)
            )
            memory_info.update_access(request_id)
            
            # Register the page
            self.page_memory_info[page_id] = memory_info
            self.total_memory_used += compressed_size
            
            # Add to compressed cache if compressed
            if memory_info.is_compressed:
                self.compressed_cache.put(page_id, page_info)
                self.stats.num_compressed_pages += 1
                self.stats.compressed_memory_bytes += compressed_size
                self.stats.memory_saved_bytes += max(0, size_bytes - compressed_size)
            else:
                self.stats.num_uncompressed_pages += 1
                self.stats.uncompressed_memory_bytes += size_bytes
            
            self.stats.total_memory_bytes += size_bytes
            
            return True
    
    def access_page(self, page_id: int, request_id: str) -> Optional[CompressedPageInfo]:
        """
        Access a page, updating access statistics.
        
        Args:
            page_id: Page identifier
            request_id: Request identifier
            
        Returns:
            Compressed page info if found
        """
        with self.memory_lock:
            self.total_accesses += 1
            
            # Check compressed cache first
            page_info = self.compressed_cache.get(page_id)
            if page_info is not None:
                self.cache_hits += 1
                
                # Update memory info
                if page_id in self.page_memory_info:
                    self.page_memory_info[page_id].update_access(request_id)
                
                return page_info
            
            # Page not in compressed cache
            return None
    
    def remove_page(self, page_id: int):
        """
        Remove a page from memory management.
        
        Args:
            page_id: Page identifier
        """
        with self.memory_lock:
            if page_id in self.page_memory_info:
                memory_info = self.page_memory_info[page_id]
                
                # Update statistics
                if memory_info.is_compressed:
                    self.stats.num_compressed_pages -= 1
                    self.stats.compressed_memory_bytes -= memory_info.compressed_size_bytes
                    self.stats.memory_saved_bytes -= max(0, 
                        memory_info.size_bytes - memory_info.compressed_size_bytes)
                else:
                    self.stats.num_uncompressed_pages -= 1
                    self.stats.uncompressed_memory_bytes -= memory_info.size_bytes
                
                self.stats.total_memory_bytes -= memory_info.size_bytes
                self.total_memory_used -= memory_info.compressed_size_bytes
                
                # Remove from cache and tracking
                self.compressed_cache.remove(page_id)
                del self.page_memory_info[page_id]
    
    def remove_request_pages(self, request_id: str):
        """
        Remove all pages associated with a request.
        
        Args:
            request_id: Request identifier
        """
        with self.memory_lock:
            pages_to_remove = []
            
            for page_id, memory_info in self.page_memory_info.items():
                if request_id in memory_info.request_ids:
                    memory_info.request_ids.discard(request_id)
                    
                    # Remove page if no more requests reference it
                    if not memory_info.request_ids:
                        pages_to_remove.append(page_id)
            
            for page_id in pages_to_remove:
                self.remove_page(page_id)
    
    def _can_allocate(self, size_bytes: int) -> bool:
        """Check if we can allocate the requested memory."""
        return (self.total_memory_used + size_bytes) <= self.max_memory_bytes
    
    def _evict_pages_for_allocation(self, required_bytes: int) -> bool:
        """
        Evict pages to make space for new allocation.
        
        Args:
            required_bytes: Bytes needed for allocation
            
        Returns:
            True if enough space was freed
        """
        freed_bytes = 0
        eviction_candidates = self._get_eviction_candidates()
        
        for page_id in eviction_candidates:
            if page_id in self.page_memory_info:
                memory_info = self.page_memory_info[page_id]
                freed_bytes += memory_info.compressed_size_bytes
                
                logger.debug(f"Evicting page {page_id} to free memory")
                self.remove_page(page_id)
                self.stats.evicted_pages += 1
                
                if freed_bytes >= required_bytes:
                    return True
                
                # Don't evict too many pages at once
                if len(eviction_candidates) > self.eviction_batch_size:
                    break
        
        return freed_bytes >= required_bytes
    
    def _evict_pages_if_needed(self):
        """Evict pages if memory pressure is too high."""
        if self.stats.memory_pressure <= self.memory_pressure_threshold:
            return
        
        # Calculate target memory to free (10% of max memory)
        target_free_bytes = int(self.max_memory_bytes * 0.1)
        
        eviction_candidates = self._get_eviction_candidates()[:self.eviction_batch_size]
        freed_bytes = 0
        
        for page_id in eviction_candidates:
            if page_id in self.page_memory_info:
                memory_info = self.page_memory_info[page_id]
                freed_bytes += memory_info.compressed_size_bytes
                
                self.remove_page(page_id)
                self.stats.evicted_pages += 1
                
                if freed_bytes >= target_free_bytes:
                    break
        
        if freed_bytes > 0:
            logger.debug(f"Freed {freed_bytes} bytes by evicting {len(eviction_candidates)} pages")
    
    def _get_eviction_candidates(self) -> List[int]:
        """
        Get list of page IDs ordered by eviction priority.
        
        Returns:
            List of page IDs sorted by eviction priority (highest first)
        """
        current_time = time.time()
        candidates = []
        
        for page_id, memory_info in self.page_memory_info.items():
            priority = memory_info.calculate_eviction_priority(
                self.eviction_policy, current_time
            )
            candidates.append((priority, page_id))
        
        # Sort by priority (highest first)
        candidates.sort(reverse=True)
        
        return [page_id for _, page_id in candidates]
    
    def _estimate_compressed_size(self, page_info: CompressedPageInfo, original_size: int) -> int:
        """Estimate compressed size of a page."""
        if page_info.state == CompressionState.COMPRESSED:
            # Estimate based on compression ratio
            compression_ratio = max(page_info.compression_ratio, 1.0)
            return int(original_size / compression_ratio)
        else:
            return original_size
    
    def _calculate_page_importance(self, page_info: CompressedPageInfo) -> float:
        """Calculate importance score for a page."""
        # Use magnitude variance as importance indicator
        if page_info.magnitude is not None:
            return min(torch.var(page_info.magnitude).item() / 100.0, 1.0)
        return 0.5  # Default importance
    
    def _update_memory_pressure(self):
        """Update current memory pressure."""
        if self.max_memory_bytes > 0:
            self.stats.memory_pressure = self.total_memory_used / self.max_memory_bytes
        else:
            self.stats.memory_pressure = 0.0
    
    def _update_cache_hit_rate(self):
        """Update cache hit rate statistics."""
        if self.total_accesses > 0:
            self.stats.cache_hit_rate = self.cache_hits / self.total_accesses
        else:
            self.stats.cache_hit_rate = 0.0
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics."""
        with self.memory_lock:
            # Update derived statistics
            if self.stats.total_memory_bytes > 0:
                self.stats.compression_ratio = (
                    self.stats.total_memory_bytes / 
                    max(1, self.stats.total_memory_bytes - self.stats.memory_saved_bytes)
                )
            
            return {
                "total_memory_mb": self.stats.total_memory_bytes / (1024 * 1024),
                "compressed_memory_mb": self.stats.compressed_memory_bytes / (1024 * 1024),
                "uncompressed_memory_mb": self.stats.uncompressed_memory_bytes / (1024 * 1024),
                "memory_saved_mb": self.stats.memory_saved_bytes / (1024 * 1024),
                "num_compressed_pages": self.stats.num_compressed_pages,
                "num_uncompressed_pages": self.stats.num_uncompressed_pages,
                "compression_ratio": self.stats.compression_ratio,
                "memory_pressure": self.stats.memory_pressure,
                "cache_hit_rate": self.stats.cache_hit_rate,
                "evicted_pages": self.stats.evicted_pages,
                "total_accesses": self.total_accesses,
                "cache_hits": self.cache_hits,
                "eviction_policy": self.eviction_policy.value,
                "max_memory_mb": self.max_memory_bytes / (1024 * 1024),
                "memory_utilization": self.total_memory_used / max(1, self.max_memory_bytes)
            }
    
    def force_eviction(self, num_pages: int = None) -> int:
        """
        Force eviction of pages.
        
        Args:
            num_pages: Number of pages to evict, None for batch size
            
        Returns:
            Number of pages actually evicted
        """
        with self.memory_lock:
            if num_pages is None:
                num_pages = self.eviction_batch_size
            
            eviction_candidates = self._get_eviction_candidates()[:num_pages]
            evicted_count = 0
            
            for page_id in eviction_candidates:
                if page_id in self.page_memory_info:
                    self.remove_page(page_id)
                    self.stats.evicted_pages += 1
                    evicted_count += 1
            
            logger.info(f"Force evicted {evicted_count} pages")
            return evicted_count
    
    def clear_all_memory(self):
        """Clear all memory and reset statistics."""
        with self.memory_lock:
            self.page_memory_info.clear()
            self.compressed_cache.clear()
            self.total_memory_used = 0
            self.total_accesses = 0
            self.cache_hits = 0
            self.stats = MemoryStats()
            logger.info("Cleared all KV cache memory")
    
    def shutdown(self):
        """Shutdown the memory manager."""
        self.stop_background.set()
        
        if self.background_thread and self.background_thread.is_alive():
            self.background_thread.join(timeout=5.0)
        
        self.clear_all_memory()
        logger.info("KV cache memory manager shut down")
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        try:
            self.shutdown()
        except Exception:
            pass  # Ignore errors during cleanup 