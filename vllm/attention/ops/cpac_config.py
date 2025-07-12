# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
CPAC configuration through environment variables
"""

import os
from typing import Optional
from vllm.attention.ops.cpac_compression import CPACConfig


def get_cpac_config_from_env() -> Optional[CPACConfig]:
    """
    Get CPAC configuration from environment variables
    
    Environment variables:
    - VLLM_CPAC_ENABLED: Enable CPAC compression (true/false)
    - VLLM_CPAC_SIMILARITY_THRESHOLD: Similarity threshold (0.0-1.0)
    - VLLM_CPAC_COMPRESSION_LEVEL: Compression level (1-3)
    - VLLM_CPAC_ADAPTIVE: Enable adaptive compression (true/false)
    - VLLM_CPAC_DELTA_BITS: Bits for delta quantization (4-16)
    """
    
    if os.getenv("VLLM_CPAC_ENABLED", "false").lower() != "true":
        return None
    
    config = CPACConfig()
    
    # Parse environment variables
    if "VLLM_CPAC_SIMILARITY_THRESHOLD" in os.environ:
        config.similarity_threshold = float(os.environ["VLLM_CPAC_SIMILARITY_THRESHOLD"])
    
    if "VLLM_CPAC_COMPRESSION_LEVEL" in os.environ:
        config.compression_level = int(os.environ["VLLM_CPAC_COMPRESSION_LEVEL"])
    
    if "VLLM_CPAC_ADAPTIVE" in os.environ:
        config.adaptive_compression = os.environ["VLLM_CPAC_ADAPTIVE"].lower() == "true"
    
    if "VLLM_CPAC_DELTA_BITS" in os.environ:
        config.delta_bits = int(os.environ["VLLM_CPAC_DELTA_BITS"])
    
    if "VLLM_CPAC_MAX_CLUSTER_SIZE" in os.environ:
        config.max_cluster_size = int(os.environ["VLLM_CPAC_MAX_CLUSTER_SIZE"])
    
    if "VLLM_CPAC_TOP_K_SIMILAR" in os.environ:
        config.top_k_similar = int(os.environ["VLLM_CPAC_TOP_K_SIMILAR"])
    
    return config


# Usage in vLLM initialization
def should_enable_cpac() -> bool:
    """Check if CPAC should be enabled based on environment"""
    return os.getenv("VLLM_CPAC_ENABLED", "false").lower() == "true"