import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
import time
import os
from dataclasses import dataclass
import threading
from collections import defaultdict

# This is a monkey-patch script that hooks into vLLM internals to monitor KV-cache pages
# It should be imported before initializing vLLM models

original_reshape_and_cache = None
original_swap_blocks = None
original_copy_blocks = None

# Global storage for captured cache data
cache_captures = defaultdict(list)
cache_access_patterns = defaultdict(int)
page_similarity_map = {}


@dataclass
class CachePageSnapshot:
    """Snapshot of a cache page at a specific time"""
    timestamp: float
    layer_idx: int
    block_idx: int
    cache_type: str  # 'key' or 'value'
    data: np.ndarray  # Actual cache data
    sequence_id: Optional[int] = None
    token_position: Optional[int] = None


def hook_reshape_and_cache(key, value, key_cache, value_cache, slot_mapping, kv_cache_dtype, k_scale, v_scale):
    """Hook for reshape_and_cache operation to capture KV data"""
    global original_reshape_and_cache, cache_captures
    
    # Call original function
    result = original_reshape_and_cache(key, value, key_cache, value_cache, 
                                       slot_mapping, kv_cache_dtype, k_scale, v_scale)
    
    # Capture cache state
    timestamp = time.time()
    
    # Extract layer information from the tensor shapes
    # Assuming key shape is [num_tokens, num_heads, head_size]
    if key is not None and slot_mapping is not None:
        try:
            # Get block indices from slot mapping
            block_indices = (slot_mapping // 16).unique()  # Assuming block_size=16
            
            for block_idx in block_indices:
                # Capture key cache page
                if key_cache is not None:
                    key_snapshot = CachePageSnapshot(
                        timestamp=timestamp,
                        layer_idx=0,  # Would need to track this from model context
                        block_idx=block_idx.item(),
                        cache_type='key',
                        data=key_cache[0, block_idx].cpu().numpy().copy()
                    )
                    cache_captures['key'].append(key_snapshot)
                
                # Capture value cache page
                if value_cache is not None:
                    value_snapshot = CachePageSnapshot(
                        timestamp=timestamp,
                        layer_idx=0,  # Would need to track this from model context
                        block_idx=block_idx.item(),
                        cache_type='value',
                        data=value_cache[0, block_idx].cpu().numpy().copy()
                    )
                    cache_captures['value'].append(value_snapshot)
        except Exception as e:
            print(f"Error capturing cache data: {e}")
    
    return result


def compute_page_similarity_realtime(page1: np.ndarray, page2: np.ndarray) -> float:
    """Compute cosine similarity between two cache pages"""
    p1_flat = page1.flatten()
    p2_flat = page2.flatten()
    
    # Normalize
    p1_norm = p1_flat / (np.linalg.norm(p1_flat) + 1e-8)
    p2_norm = p2_flat / (np.linalg.norm(p2_flat) + 1e-8)
    
    # Cosine similarity
    similarity = np.dot(p1_norm, p2_norm)
    return similarity


def analyze_cache_patterns():
    """Analyze captured cache data for patterns"""
    global cache_captures, page_similarity_map
    
    results = {
        'timestamp': time.time(),
        'total_captures': {k: len(v) for k, v in cache_captures.items()},
        'similarity_analysis': {},
        'redundancy_stats': {}
    }
    
    # Analyze each cache type
    for cache_type in ['key', 'value']:
        snapshots = cache_captures[cache_type]
        if len(snapshots) < 2:
            continue
        
        similarities = []
        high_similarity_pairs = []
        
        # Compare recent snapshots
        recent_snapshots = snapshots[-100:]  # Last 100 snapshots
        
        for i in range(len(recent_snapshots)):
            for j in range(i + 1, len(recent_snapshots)):
                snap1, snap2 = recent_snapshots[i], recent_snapshots[j]
                
                # Only compare if same layer
                if snap1.layer_idx == snap2.layer_idx:
                    similarity = compute_page_similarity_realtime(snap1.data, snap2.data)
                    similarities.append(similarity)
                    
                    if similarity > 0.95:  # High similarity threshold
                        high_similarity_pairs.append({
                            'block1': snap1.block_idx,
                            'block2': snap2.block_idx,
                            'similarity': similarity,
                            'layer': snap1.layer_idx
                        })
        
        if similarities:
            results['similarity_analysis'][cache_type] = {
                'mean_similarity': np.mean(similarities),
                'max_similarity': np.max(similarities),
                'min_similarity': np.min(similarities),
                'high_similarity_count': len(high_similarity_pairs),
                'high_similarity_ratio': len(high_similarity_pairs) / len(similarities)
            }
            
            # Estimate redundancy
            redundancy_ratio = len([s for s in similarities if s > 0.9]) / len(similarities)
            potential_compression = redundancy_ratio * 0.5  # Assume 50% compression for redundant pages
            
            results['redundancy_stats'][cache_type] = {
                'redundancy_ratio': redundancy_ratio,
                'potential_compression_ratio': potential_compression,
                'estimated_memory_savings_percent': potential_compression * 100
            }
    
    return results


def install_hooks():
    """Install monitoring hooks into vLLM cache operations"""
    global original_reshape_and_cache
    
    try:
        # Import vLLM cache operations
        from vllm._C import cache_ops
        
        # Save original functions
        original_reshape_and_cache = cache_ops.reshape_and_cache
        
        # Install hooks
        cache_ops.reshape_and_cache = hook_reshape_and_cache
        
        print("KV-cache monitoring hooks installed successfully")
        return True
    except Exception as e:
        print(f"Failed to install hooks: {e}")
        return False


def save_analysis_results(output_dir: str = "kv_cache_monitoring"):
    """Save analysis results to disk"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Analyze current captures
    analysis = analyze_cache_patterns()
    
    # Save analysis
    with open(f"{output_dir}/realtime_analysis_{int(time.time())}.json", 'w') as f:
        json.dump(analysis, f, indent=2)
    
    # Save summary statistics
    summary = {
        'monitoring_duration': time.time() - cache_captures['key'][0].timestamp if cache_captures['key'] else 0,
        'total_key_captures': len(cache_captures['key']),
        'total_value_captures': len(cache_captures['value']),
        'analysis': analysis
    }
    
    with open(f"{output_dir}/monitoring_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Analysis results saved to {output_dir}/")
    return summary


def start_periodic_analysis(interval: int = 60):
    """Start periodic analysis in background thread"""
    def analyze_loop():
        while True:
            time.sleep(interval)
            try:
                results = analyze_cache_patterns()
                print(f"Periodic analysis at {time.strftime('%Y-%m-%d %H:%M:%S')}:")
                print(f"  Key cache similarity: {results['similarity_analysis'].get('key', {}).get('mean_similarity', 'N/A'):.3f}")
                print(f"  Value cache similarity: {results['similarity_analysis'].get('value', {}).get('mean_similarity', 'N/A'):.3f}")
                if 'key' in results['redundancy_stats']:
                    print(f"  Potential memory savings: {results['redundancy_stats']['key']['estimated_memory_savings_percent']:.1f}%")
            except Exception as e:
                print(f"Error in periodic analysis: {e}")
    
    thread = threading.Thread(target=analyze_loop, daemon=True)
    thread.start()
    print(f"Started periodic analysis with {interval}s interval")


# Auto-install hooks when module is imported
if __name__ != "__main__":
    install_hooks()
    start_periodic_analysis(60)  # Analyze every minute