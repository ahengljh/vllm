#!/usr/bin/env python3
"""
Example demonstrating the unified structured output backend system.

This shows how to use the new unified backend with automatic selection,
caching, and optimized batch processing.
"""

import asyncio
import json
import time
from typing import List, Dict, Any

# Example 1: Basic Usage with Automatic Backend Selection
def example_basic_usage():
    """Basic example showing automatic backend selection."""
    from vllm.config import VllmConfig
    from vllm.v1.structured_output.unified_backend import UnifiedBackendManager
    from vllm.v1.structured_output.backend_types import StructuredOutputOptions
    
    # Mock config and tokenizer for demonstration
    class MockConfig:
        pass
    
    class MockTokenizer:
        def get_vocab_size(self):
            return 32000
    
    config = MockConfig()
    tokenizer = MockTokenizer()
    
    # Create the unified backend manager
    manager = UnifiedBackendManager(
        vllm_config=config,
        tokenizer=tokenizer,
        vocab_size=32000,
        preferred_backend="xgrammar",  # Optional: prefer xgrammar when available
        cache_size=1000  # LRU cache for compiled grammars
    )
    
    # Example 1: JSON Schema - will automatically use best backend
    json_schema = json.dumps({
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "email": {"type": "string", "format": "email"}
        },
        "required": ["name", "age"]
    })
    
    grammar1 = manager.compile_grammar(
        StructuredOutputOptions.JSON,
        json_schema
    )
    print(f"✓ Compiled JSON schema grammar")
    
    # Example 2: Regex pattern - will select backend that supports regex
    phone_regex = r"(\+1-)?\d{3}-\d{3}-\d{4}"
    grammar2 = manager.compile_grammar(
        StructuredOutputOptions.REGEX,
        phone_regex
    )
    print(f"✓ Compiled regex grammar")
    
    # Example 3: Choice constraint
    choices = '["red", "green", "blue"]'
    grammar3 = manager.compile_grammar(
        StructuredOutputOptions.CHOICE,
        choices
    )
    print(f"✓ Compiled choice grammar")
    
    # The second compilation of the same grammar will be cached!
    start = time.time()
    grammar1_cached = manager.compile_grammar(
        StructuredOutputOptions.JSON,
        json_schema
    )
    cache_time = time.time() - start
    print(f"✓ Retrieved cached grammar in {cache_time*1000:.2f}ms")
    
    # Get statistics
    stats = manager.get_backend_stats()
    print(f"\nBackend Statistics:")
    print(f"  Active backends: {stats['active_backends']}")
    print(f"  Cache size: {stats['cache_size']}/{stats['cache_max_size']}")
    print(f"  Preferred backend: {stats['preferred_backend']}")
    
    # Cleanup
    manager.destroy()


# Example 2: Optimized Batch Processing
def example_batch_processing():
    """Example showing optimized batch processing for production use."""
    from vllm.config import VllmConfig
    from vllm.v1.structured_output.optimized_manager import (
        OptimizedStructuredOutputManager,
        BatchProcessingConfig
    )
    
    # Configure batch processing
    batch_config = BatchProcessingConfig(
        parallel_threshold=10,  # Use parallel processing for >10 requests
        batch_size=4,           # Process in batches of 4
        max_workers=4,          # Use 4 worker threads
        adaptive_batching=True, # Dynamically adjust batch size
        cache_warmup_size=5     # Pre-compile 5 common patterns
    )
    
    # Mock config
    class MockConfig:
        class SchedulerConfig:
            max_num_seqs = 100
        
        class DecodeConfig:
            disable_any_whitespace = False
            disable_additional_properties = False
            reasoning_config = None
        
        class ModelConfig:
            supported_reasoning = None
        
        scheduler_config = SchedulerConfig()
        decoding_config = DecodeConfig()
        model_config = ModelConfig()
        tokenizer_config = None
        parallel_config = None
        lora_config = None
        speculative_config = None
    
    config = MockConfig()
    
    # Create optimized manager
    manager = OptimizedStructuredOutputManager(
        vllm_config=config,
        preferred_backend="xgrammar",
        batch_config=batch_config
    )
    
    print("Optimized Manager initialized with:")
    print(f"  Parallel threshold: {batch_config.parallel_threshold}")
    print(f"  Batch size: {batch_config.batch_size}")
    print(f"  Adaptive batching: {batch_config.adaptive_batching}")
    
    # Simulate batch request processing
    class MockRequest:
        def __init__(self, request_id: str, schema: str):
            self.request_id = request_id
            self.structured_output_request = MockStructuredOutputRequest(schema)
    
    class MockStructuredOutputRequest:
        def __init__(self, schema: str):
            self.structured_output_key = (StructuredOutputOptions.JSON, schema)
            self.grammar = None
    
    # Create multiple requests
    requests = []
    for i in range(20):
        schema = json.dumps({
            "type": "object",
            "properties": {
                f"field_{i}": {"type": "string"}
            }
        })
        requests.append(MockRequest(f"req_{i}", schema))
    
    # Process requests (would normally be done asynchronously)
    print(f"\nProcessing {len(requests)} requests...")
    start = time.time()
    
    for request in requests:
        manager.create_grammar(request)
    
    # Wait for all grammars to be compiled
    for request in requests:
        if hasattr(request.structured_output_request.grammar, 'result'):
            # It's a Future, wait for it
            request.structured_output_request.grammar.result()
    
    elapsed = time.time() - start
    print(f"✓ Processed {len(requests)} requests in {elapsed*1000:.2f}ms")
    print(f"  Average: {elapsed*1000/len(requests):.2f}ms per request")
    
    # Get performance statistics
    stats = manager.get_stats()
    print(f"\nPerformance Statistics:")
    print(f"  Cache hit rate: {stats['cache_hit_rate']:.1%}")
    print(f"  Cache hits: {stats['cache_hits']}")
    print(f"  Cache misses: {stats['cache_misses']}")
    if stats['avg_batch_processing_time_ms'] > 0:
        print(f"  Avg batch time: {stats['avg_batch_processing_time_ms']:.2f}ms")
    
    # Cleanup
    manager.destroy()


# Example 3: Migration from Old to New System
def example_migration():
    """Example showing how to migrate from the old system."""
    print("\n=== Migration Example ===\n")
    
    # OLD WAY (multiple if-else, manual imports)
    print("OLD APPROACH:")
    print("""
    if backend == "xgrammar":
        from vllm.v1.structured_output.backend_xgrammar import XgrammarBackend
        self.backend = XgrammarBackend(config, tokenizer, vocab_size)
    elif backend == "guidance":
        from vllm.v1.structured_output.backend_guidance import GuidanceBackend
        self.backend = GuidanceBackend(config, tokenizer, vocab_size)
    elif backend == "outlines":
        from vllm.v1.structured_output.backend_outlines import OutlinesBackend
        self.backend = OutlinesBackend(config, tokenizer, vocab_size)
    
    # No caching
    grammar = self.backend.compile_grammar(request_type, grammar_spec)
    """)
    
    # NEW WAY (unified, automatic)
    print("\nNEW APPROACH:")
    print("""
    # Single line initialization with automatic backend management
    self.backend_manager = UnifiedBackendManager(
        vllm_config=config,
        tokenizer=tokenizer,
        vocab_size=vocab_size,
        preferred_backend=backend,  # Optional, can be None for auto-selection
        cache_size=1000
    )
    
    # Automatic backend selection, caching, and lazy loading
    grammar = self.backend_manager.compile_grammar(request_type, grammar_spec)
    """)
    
    print("\nKEY BENEFITS:")
    print("✓ No manual imports - backends loaded lazily")
    print("✓ Automatic fallback if preferred backend doesn't support feature")
    print("✓ Built-in LRU caching for compiled grammars")
    print("✓ Performance statistics and monitoring")
    print("✓ Thread-safe operations")
    print("✓ Cleaner, more maintainable code")


# Example 4: Custom Backend Registration
def example_custom_backend():
    """Example showing how to register a custom backend."""
    from vllm.v1.structured_output.unified_backend import (
        BackendRegistry,
        BackendCapability
    )
    from vllm.v1.structured_output.backend_types import (
        StructuredOutputBackend,
        StructuredOutputGrammar,
        StructuredOutputOptions
    )
    
    print("\n=== Custom Backend Registration ===\n")
    
    # Define a custom backend
    class MyCustomBackend(StructuredOutputBackend):
        def compile_grammar(self, request_type: StructuredOutputOptions, 
                          grammar_spec: str) -> StructuredOutputGrammar:
            print(f"  MyCustomBackend compiling: {request_type}")
            # Return a mock grammar
            return type('MockGrammar', (), {
                'accept_tokens': lambda *args: True,
                'validate_tokens': lambda *args: args[1],
                'rollback': lambda *args: None,
                'fill_bitmask': lambda *args: None,
                'is_terminated': lambda: False,
                'reset': lambda: None
            })()
        
        def allocate_token_bitmask(self, max_num_seqs: int):
            import torch
            return torch.zeros((max_num_seqs, 1000), dtype=torch.int32)
        
        def destroy(self):
            pass
    
    # Get the registry and register custom backend
    registry = BackendRegistry()
    registry.register(
        name="my_custom_backend",
        backend_class=MyCustomBackend,
        capabilities=(
            BackendCapability.JSON |
            BackendCapability.REGEX |
            BackendCapability.JUMP_DECODING  # Special capability!
        ),
        priority=100  # High priority
    )
    
    print("✓ Registered custom backend with capabilities:")
    print("  - JSON support")
    print("  - REGEX support")
    print("  - JUMP_DECODING (special feature)")
    print(f"  - Priority: 100 (will be preferred)")
    
    # Now when using UnifiedBackendManager, it will consider your backend
    from vllm.v1.structured_output.unified_backend import UnifiedBackendManager
    
    class MockConfig:
        pass
    
    class MockTokenizer:
        def get_vocab_size(self):
            return 32000
    
    manager = UnifiedBackendManager(
        vllm_config=MockConfig(),
        tokenizer=MockTokenizer(),
        vocab_size=32000,
        preferred_backend="my_custom_backend"  # Use our custom backend
    )
    
    # This will use our custom backend
    grammar = manager.compile_grammar(
        StructuredOutputOptions.JSON,
        '{"type": "string"}'
    )
    
    print("\n✓ Successfully used custom backend for compilation")
    
    # List all available backends
    all_backends = registry.list_backends()
    print(f"\nAll registered backends: {list(all_backends.keys())}")


if __name__ == "__main__":
    print("=" * 60)
    print("UNIFIED STRUCTURED OUTPUT BACKEND EXAMPLES")
    print("=" * 60)
    
    try:
        print("\n1. BASIC USAGE")
        print("-" * 40)
        example_basic_usage()
    except Exception as e:
        print(f"Note: Basic example requires full vLLM environment: {e}")
    
    try:
        print("\n2. BATCH PROCESSING")
        print("-" * 40)
        example_batch_processing()
    except Exception as e:
        print(f"Note: Batch example requires full vLLM environment: {e}")
    
    print("\n3. MIGRATION GUIDE")
    print("-" * 40)
    example_migration()
    
    try:
        print("\n4. CUSTOM BACKEND")
        print("-" * 40)
        example_custom_backend()
    except Exception as e:
        print(f"Note: Custom backend example requires full vLLM environment: {e}")
    
    print("\n" + "=" * 60)
    print("For production use, see the documentation at:")
    print("docs/unified_structured_output.md")
    print("=" * 60)