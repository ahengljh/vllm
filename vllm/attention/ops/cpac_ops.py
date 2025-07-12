# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Python bindings for CPAC CUDA operations
"""

import torch
from typing import Tuple, Optional

# Try to import the compiled CUDA extensions
try:
    import vllm._C
    # Check if CPAC ops are available in the compiled extension
    if hasattr(vllm._C, 'cpac_ops'):
        cpac_ops = vllm._C.cpac_ops
        HAS_CPAC_OPS = True
    else:
        HAS_CPAC_OPS = False
except ImportError:
    HAS_CPAC_OPS = False
    

def compute_page_features(
    page_data: torch.Tensor,
    feature_dim: int = 128
) -> torch.Tensor:
    """
    Compute features for a page to be used in similarity matching
    
    Args:
        page_data: Tensor of shape [num_heads, head_size, block_size]
        feature_dim: Dimension of output feature vector
    
    Returns:
        features: Tensor of shape [feature_dim]
    """
    if HAS_CPAC_OPS and page_data.is_cuda:
        return cpac_ops.compute_page_features_cuda(page_data, feature_dim)
    else:
        # CPU fallback implementation
        num_heads, head_size, block_size = page_data.shape
        features = []
        
        # Mean and variance per head
        reshaped = page_data.view(num_heads, -1)
        mean_per_head = reshaped.mean(dim=1)
        var_per_head = reshaped.var(dim=1)
        features.append(mean_per_head)
        features.append(var_per_head.sqrt())
        
        # Attention pattern signature
        pattern_sig = page_data.abs().mean(dim=1).flatten()
        features.append(pattern_sig[:min(pattern_sig.size(0), feature_dim - 2*num_heads)])
        
        # Pad if necessary
        feature_tensor = torch.cat(features)
        if feature_tensor.size(0) < feature_dim:
            padding = torch.zeros(feature_dim - feature_tensor.size(0), 
                                device=page_data.device, dtype=torch.float32)
            feature_tensor = torch.cat([feature_tensor, padding])
        
        return feature_tensor[:feature_dim]


def compute_similarity(
    features1: torch.Tensor,
    features2: torch.Tensor
) -> torch.Tensor:
    """
    Compute cosine similarity between two feature vectors
    
    Args:
        features1: Feature vector 1
        features2: Feature vector 2
    
    Returns:
        similarity: Scalar similarity value
    """
    if HAS_CPAC_OPS and features1.is_cuda:
        return cpac_ops.compute_similarity_cuda(features1, features2)
    else:
        # CPU fallback
        return torch.nn.functional.cosine_similarity(
            features1.unsqueeze(0), features2.unsqueeze(0)
        )


def compress_delta(
    target_page: torch.Tensor,
    base_page: torch.Tensor,
    delta_bits: int = 8
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compress target page as delta from base page
    
    Args:
        target_page: Page to compress
        base_page: Base page for delta encoding
        delta_bits: Number of bits for quantized delta
    
    Returns:
        quantized_delta: Quantized delta tensor
        scale: Scale factor for dequantization
    """
    if HAS_CPAC_OPS and target_page.is_cuda:
        return cpac_ops.compress_delta_cuda(target_page, base_page, delta_bits)
    else:
        # CPU fallback
        delta = target_page - base_page
        max_val = delta.abs().max().item()
        
        if max_val == 0:
            return torch.zeros_like(delta, dtype=torch.int8), torch.tensor(1.0)
        
        max_int = (1 << (delta_bits - 1)) - 1
        scale = max_val / max_int
        
        quantized = torch.round(delta / scale).clamp(-max_int, max_int).to(torch.int8)
        
        return quantized, torch.tensor(scale)


def decompress_delta(
    quantized_delta: torch.Tensor,
    base_page: torch.Tensor,
    scale: float
) -> torch.Tensor:
    """
    Decompress page from quantized delta and base page
    
    Args:
        quantized_delta: Quantized delta tensor
        base_page: Base page for reconstruction
        scale: Scale factor from compression
    
    Returns:
        decompressed_page: Reconstructed page
    """
    if HAS_CPAC_OPS and quantized_delta.is_cuda:
        return cpac_ops.decompress_delta_cuda(quantized_delta, base_page, scale)
    else:
        # CPU fallback
        delta = quantized_delta.float() * scale
        return base_page + delta.to(base_page.dtype)


def batch_compute_similarities(
    features_list: torch.Tensor,
    top_k: int = 4
) -> torch.Tensor:
    """
    Compute pairwise similarities for a batch of feature vectors
    
    Args:
        features_list: Tensor of shape [num_pages, feature_dim]
        top_k: Number of top similar pages to return per page
    
    Returns:
        top_similarities: Tensor of shape [num_pages, top_k] with similarity scores
        top_indices: Tensor of shape [num_pages, top_k] with page indices
    """
    num_pages = features_list.size(0)
    
    # Compute pairwise similarities
    similarities = torch.mm(features_list, features_list.t())
    
    # Normalize by norms
    norms = torch.norm(features_list, dim=1, keepdim=True)
    similarities = similarities / (norms * norms.t() + 1e-8)
    
    # Mask out self-similarities
    mask = torch.eye(num_pages, device=features_list.device)
    similarities = similarities * (1 - mask) - mask
    
    # Get top-k similarities
    top_similarities, top_indices = torch.topk(similarities, k=min(top_k, num_pages-1), dim=1)
    
    return top_similarities, top_indices


def estimate_compression_ratio(
    target_page: torch.Tensor,
    base_page: torch.Tensor,
    delta_bits: int = 8
) -> float:
    """
    Estimate compression ratio without actually compressing
    
    Args:
        target_page: Page to compress
        base_page: Base page for delta encoding
        delta_bits: Number of bits for quantized delta
    
    Returns:
        estimated_ratio: Estimated compression ratio
    """
    original_bits = target_page.numel() * target_page.element_size() * 8
    compressed_bits = target_page.numel() * delta_bits + 32  # +32 for scale
    
    # Estimate quality loss
    delta = target_page - base_page
    max_val = delta.abs().max().item()
    
    if max_val == 0:
        return float('inf')  # Perfect compression
    
    max_int = (1 << (delta_bits - 1)) - 1
    scale = max_val / max_int
    quantization_error = scale / 2  # Average quantization error
    
    # Penalize high quantization error
    quality_factor = 1.0 / (1.0 + quantization_error)
    
    return (original_bits / compressed_bits) * quality_factor


class CPACKernel:
    """
    Wrapper class for CPAC CUDA kernels with automatic fallback
    """
    
    def __init__(self):
        self.has_cuda = HAS_CPAC_OPS
        if not self.has_cuda:
            import warnings
            warnings.warn("CPAC CUDA kernels not available, using CPU fallback")
    
    def compute_features(self, page_data: torch.Tensor, feature_dim: int = 128) -> torch.Tensor:
        return compute_page_features(page_data, feature_dim)
    
    def compute_similarity(self, feat1: torch.Tensor, feat2: torch.Tensor) -> torch.Tensor:
        return compute_similarity(feat1, feat2)
    
    def compress(self, target: torch.Tensor, base: torch.Tensor, 
                 bits: int = 8) -> Tuple[torch.Tensor, torch.Tensor]:
        return compress_delta(target, base, bits)
    
    def decompress(self, delta: torch.Tensor, base: torch.Tensor, 
                   scale: float) -> torch.Tensor:
        return decompress_delta(delta, base, scale)


# Global kernel instance
cpac_kernel = CPACKernel()