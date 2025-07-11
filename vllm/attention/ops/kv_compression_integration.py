"""KV Cache Compression Integration for vLLM

This module provides integration between the KV cache compression system
and the existing vLLM attention backends and cache management infrastructure.
"""

from typing import Dict, Optional, Tuple, Any
import torch
from torch import Tensor

from vllm.attention.ops.kv_compression_manager import KVCacheCompressionManager
from vllm.config import CacheConfig
from vllm.logger import init_logger

logger = init_logger(__name__)

# Global compression manager instance
_global_compression_manager: Optional[KVCacheCompressionManager] = None


def get_compression_manager() -> Optional[KVCacheCompressionManager]:
    """Get the global compression manager instance."""
    return _global_compression_manager


def initialize_compression_manager(cache_config: CacheConfig) -> Optional[KVCacheCompressionManager]:
    """
    Initialize the global compression manager from cache configuration.
    
    Args:
        cache_config: vLLM cache configuration
        
    Returns:
        The compression manager if enabled, None otherwise
    """
    global _global_compression_manager
    
    if not cache_config.enable_kv_cache_compression:
        logger.info("KV cache compression is disabled")
        return None
    
    try:
        _global_compression_manager = KVCacheCompressionManager(
            enable_compression=cache_config.enable_kv_cache_compression,
            cosine_threshold=cache_config.kv_compression_cosine_threshold,
            max_merge_ratio=cache_config.kv_compression_max_merge_ratio,
            memory_pressure_threshold=cache_config.kv_compression_memory_pressure_threshold,
            temporal_window=cache_config.kv_compression_temporal_window,
            outlier_threshold=cache_config.kv_compression_outlier_threshold,
            importance_threshold=cache_config.kv_compression_importance_threshold,
        )
        
        logger.info("KV cache compression initialized with settings: "
                   f"cosine_threshold={cache_config.kv_compression_cosine_threshold}, "
                   f"max_merge_ratio={cache_config.kv_compression_max_merge_ratio}, "
                   f"memory_pressure_threshold={cache_config.kv_compression_memory_pressure_threshold}")
        
        return _global_compression_manager
        
    except Exception as e:
        logger.error(f"Failed to initialize KV cache compression: {e}")
        return None


def shutdown_compression_manager():
    """Shutdown the global compression manager."""
    global _global_compression_manager
    
    if _global_compression_manager is not None:
        _global_compression_manager.shutdown()
        _global_compression_manager = None
        logger.info("KV cache compression manager shut down")


def compress_kv_cache(
    layer_name: str,
    page_id: int,
    key_cache: Tensor,
    value_cache: Tensor,
    request_id: str
) -> bool:
    """
    Compress a KV cache page.
    
    Args:
        layer_name: Name of the attention layer
        page_id: Unique page identifier  
        key_cache: Key cache tensor
        value_cache: Value cache tensor
        request_id: Request identifier
        
    Returns:
        True if compression was applied, False otherwise
    """
    if _global_compression_manager is None:
        return False
    
    try:
        _global_compression_manager.register_page(
            page_id=page_id,
            layer_name=layer_name,
            key_cache=key_cache,
            value_cache=value_cache,
            request_id=request_id
        )
        return True
        
    except Exception as e:
        logger.debug(f"Failed to compress KV cache page {page_id}: {e}")
        return False


def decompress_kv_cache(
    page_id: int,
    request_id: str
) -> Optional[Tuple[Tensor, Tensor]]:
    """
    Decompress a KV cache page.
    
    Args:
        page_id: Page identifier
        request_id: Request identifier
        
    Returns:
        Tuple of (key_cache, value_cache) if successful, None otherwise
    """
    if _global_compression_manager is None:
        return None
    
    try:
        return _global_compression_manager.access_page(page_id, request_id)
        
    except Exception as e:
        logger.debug(f"Failed to decompress KV cache page {page_id}: {e}")
        return None


def cleanup_request_compression(request_id: str):
    """
    Clean up compression data for a finished request.
    
    Args:
        request_id: Request identifier
    """
    if _global_compression_manager is not None:
        try:
            _global_compression_manager.remove_request_pages(request_id)
        except Exception as e:
            logger.debug(f"Failed to cleanup compression for request {request_id}: {e}")


def get_compression_stats() -> Dict[str, Any]:
    """
    Get compression statistics.
    
    Returns:
        Dictionary of compression statistics
    """
    if _global_compression_manager is None:
        return {
            "compression_enabled": False,
            "message": "Compression is not enabled"
        }
    
    try:
        stats = _global_compression_manager.get_compression_stats()
        stats["compression_enabled"] = True
        return stats
        
    except Exception as e:
        logger.debug(f"Failed to get compression stats: {e}")
        return {
            "compression_enabled": True,
            "error": str(e)
        }


def force_compression(layer_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Force immediate compression of eligible pages.
    
    Args:
        layer_name: Optional layer name to target
        
    Returns:
        Compression results
    """
    if _global_compression_manager is None:
        return {"message": "Compression is not enabled"}
    
    try:
        return _global_compression_manager.force_compression(layer_name)
        
    except Exception as e:
        logger.debug(f"Failed to force compression: {e}")
        return {"error": str(e)}


class KVCacheCompressionWrapper:
    """
    Wrapper to integrate compression with existing KV cache operations.
    
    This can be used by attention backends to seamlessly add compression
    without major architectural changes.
    """
    
    def __init__(self, cache_config: CacheConfig):
        """
        Initialize the compression wrapper.
        
        Args:
            cache_config: vLLM cache configuration
        """
        self.compression_enabled = cache_config.enable_kv_cache_compression
        self.cache_config = cache_config
        
        if self.compression_enabled and _global_compression_manager is None:
            initialize_compression_manager(cache_config)
    
    def write_to_cache(self,
                      layer_name: str,
                      page_id: int,
                      key_cache: Tensor,
                      value_cache: Tensor,
                      request_id: str,
                      original_write_fn: Optional[callable] = None) -> bool:
        """
        Write KV cache with optional compression.
        
        Args:
            layer_name: Attention layer name
            page_id: Page identifier
            key_cache: Key cache tensor
            value_cache: Value cache tensor
            request_id: Request identifier
            original_write_fn: Original write function to call
            
        Returns:
            True if write was successful
        """
        # Call original write function if provided
        if original_write_fn is not None:
            try:
                original_write_fn(key_cache, value_cache)
            except Exception as e:
                logger.debug(f"Original write function failed: {e}")
                return False
        
        # Apply compression if enabled
        if self.compression_enabled:
            return compress_kv_cache(
                layer_name=layer_name,
                page_id=page_id,
                key_cache=key_cache,
                value_cache=value_cache,
                request_id=request_id
            )
        
        return True
    
    def read_from_cache(self,
                       page_id: int,
                       request_id: str,
                       original_read_fn: Optional[callable] = None) -> Optional[Tuple[Tensor, Tensor]]:
        """
        Read KV cache with optional decompression.
        
        Args:
            page_id: Page identifier
            request_id: Request identifier
            original_read_fn: Original read function to call
            
        Returns:
            Tuple of (key_cache, value_cache) if successful
        """
        # Try compression first if enabled
        if self.compression_enabled:
            compressed_result = decompress_kv_cache(page_id, request_id)
            if compressed_result is not None:
                return compressed_result
        
        # Fall back to original read function
        if original_read_fn is not None:
            try:
                return original_read_fn()
            except Exception as e:
                logger.debug(f"Original read function failed: {e}")
        
        return None
    
    def cleanup_request(self, request_id: str):
        """
        Clean up resources for a finished request.
        
        Args:
            request_id: Request identifier
        """
        if self.compression_enabled:
            cleanup_request_compression(request_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get compression statistics."""
        return get_compression_stats()
    
    def force_compress(self, layer_name: Optional[str] = None) -> Dict[str, Any]:
        """Force immediate compression."""
        return force_compression(layer_name)


# Utility functions for integration with existing attention backends

def create_compression_wrapper(cache_config: CacheConfig) -> KVCacheCompressionWrapper:
    """
    Create a compression wrapper from cache configuration.
    
    Args:
        cache_config: vLLM cache configuration
        
    Returns:
        Compression wrapper instance
    """
    return KVCacheCompressionWrapper(cache_config)


def is_compression_enabled() -> bool:
    """Check if compression is currently enabled."""
    return _global_compression_manager is not None


def get_compression_memory_usage() -> Dict[str, float]:
    """
    Get memory usage statistics related to compression.
    
    Returns:
        Dictionary with memory usage information
    """
    if _global_compression_manager is None:
        return {"compression_enabled": False}
    
    try:
        stats = _global_compression_manager.get_compression_stats()
        
        # Calculate memory usage metrics
        memory_saved_mb = stats.get("total_memory_saved_mb", 0)
        compression_ratio = stats.get("average_compression_ratio", 1.0)
        
        return {
            "compression_enabled": True,
            "memory_saved_mb": memory_saved_mb,
            "compression_ratio": compression_ratio,
            "memory_efficiency": memory_saved_mb / max(1, stats.get("total_pages_processed", 1)),
            "current_memory_pressure": stats.get("current_memory_pressure", 0.0)
        }
        
    except Exception as e:
        logger.debug(f"Failed to get compression memory usage: {e}")
        return {
            "compression_enabled": True,
            "error": str(e)
        } 