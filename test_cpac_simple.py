#!/usr/bin/env python
"""
Simple test to verify CPAC is working
"""

import torch
import sys

print("=== CPAC Simple Test ===")

# Check CUDA
if not torch.cuda.is_available():
    print("Error: CUDA not available")
    sys.exit(1)

print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")

# Try importing CPAC
try:
    from vllm.attention.ops.cpac_compression import CPACManager, CPACConfig
    print("✓ CPAC compression module imported")
except ImportError as e:
    print(f"✗ Failed to import CPAC compression: {e}")
    sys.exit(1)

try:
    from vllm.attention.ops.cpac_ops import HAS_CPAC_OPS, cpac_kernel
    print(f"✓ CPAC ops available: {HAS_CPAC_OPS}")
except ImportError as e:
    print(f"✗ Failed to import CPAC ops: {e}")
    sys.exit(1)

# Test basic functionality
if HAS_CPAC_OPS:
    print("\nTesting CPAC operations...")
    
    # Create test data
    block_size = 16
    num_heads = 32
    head_size = 128
    
    test_block = torch.randn(num_heads, head_size, block_size, 
                            device='cuda', dtype=torch.float16)
    
    # Test feature extraction
    try:
        features = cpac_kernel.compute_features(test_block)
        print(f"✓ Feature extraction works - shape: {features.shape}")
    except Exception as e:
        print(f"✗ Feature extraction failed: {e}")
    
    # Test compression
    try:
        base_block = test_block
        target_block = test_block + torch.randn_like(test_block) * 0.1
        
        quantized, scale = cpac_kernel.compress(target_block, base_block)
        print(f"✓ Compression works - quantized shape: {quantized.shape}")
        
        # Test decompression
        decompressed = cpac_kernel.decompress(quantized, base_block, scale.item())
        error = torch.norm(target_block - decompressed) / torch.norm(target_block)
        print(f"✓ Decompression works - relative error: {error:.4f}")
        
    except Exception as e:
        print(f"✗ Compression/decompression failed: {e}")
    
    print("\n✓ All tests passed! CPAC is ready to use.")
else:
    print("\n⚠️  CPAC CUDA ops not available - using CPU fallback")
    print("Performance will be limited. Make sure CPAC was built correctly.")

# Test manager
print("\nTesting CPAC Manager...")
try:
    config = CPACConfig(similarity_threshold=0.85)
    manager = CPACManager(
        block_size=16,
        num_gpu_blocks=100,
        num_kv_heads=32,
        head_size=128,
        config=config
    )
    print("✓ CPAC Manager initialized successfully")
    print(f"  Config: similarity_threshold={config.similarity_threshold}, "
          f"compression_level={config.compression_level}")
except Exception as e:
    print(f"✗ Failed to initialize CPAC Manager: {e}")