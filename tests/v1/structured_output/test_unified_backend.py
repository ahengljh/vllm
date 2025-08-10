# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Tests for the unified structured output backend system."""

import json
import threading
import time
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch

from vllm.v1.structured_output.backend_types import (
    StructuredOutputGrammar,
    StructuredOutputOptions,
)
from vllm.v1.structured_output.unified_backend import (
    BackendCapability,
    BackendMetadata,
    BackendRegistry,
    GrammarCache,
    UnifiedBackendManager,
)


class MockGrammar(StructuredOutputGrammar):
    """Mock grammar for testing."""
    
    def __init__(self, spec: str):
        self.spec = spec
        self.tokens_accepted = []
        self.terminated = False
    
    def accept_tokens(self, request_id: str, tokens: list[int]) -> bool:
        self.tokens_accepted.extend(tokens)
        return True
    
    def validate_tokens(self, tokens: list[int]) -> list[int]:
        return tokens
    
    def rollback(self, num_tokens: int) -> None:
        self.tokens_accepted = self.tokens_accepted[:-num_tokens]
    
    def fill_bitmask(self, bitmask: torch.Tensor, batch_index: int) -> None:
        pass
    
    def is_terminated(self) -> bool:
        return self.terminated
    
    def reset(self):
        self.tokens_accepted = []
        self.terminated = False


class MockBackend:
    """Mock backend for testing."""
    
    def __init__(self, vllm_config, tokenizer, vocab_size):
        self.vllm_config = vllm_config
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.compile_count = 0
    
    def compile_grammar(self, request_type: StructuredOutputOptions, grammar_spec: str):
        self.compile_count += 1
        return MockGrammar(f"{request_type}:{grammar_spec}")
    
    def allocate_token_bitmask(self, max_num_seqs: int):
        return torch.zeros((max_num_seqs, (self.vocab_size + 31) // 32), dtype=torch.int32)
    
    def destroy(self):
        pass


class TestBackendRegistry:
    """Tests for the BackendRegistry class."""
    
    def setup_method(self):
        """Reset the registry before each test."""
        # Clean slate for each test
        BackendRegistry._instance = None
    
    def teardown_method(self):
        """Clean up after each test."""
        BackendRegistry._instance = None
    
    def test_singleton_pattern(self):
        """Test that BackendRegistry follows singleton pattern."""
        registry1 = BackendRegistry()
        registry2 = BackendRegistry()
        assert registry1 is registry2
    
    def test_register_backend(self):
        """Test registering a new backend."""
        registry = BackendRegistry()
        
        # Clean up if test backend already exists
        if "test_backend" in registry._backends:
            del registry._backends["test_backend"]
        
        registry.register(
            "test_backend",
            MockBackend,
            BackendCapability.JSON | BackendCapability.REGEX,
            priority=15
        )
        
        assert "test_backend" in registry._backends
        metadata = registry._backends["test_backend"]
        assert metadata.name == "test_backend"
        assert metadata.backend_class == MockBackend
        assert metadata.priority == 15
        assert BackendCapability.JSON in metadata.capabilities
        assert BackendCapability.REGEX in metadata.capabilities
    
    def test_get_best_backend_for_option(self):
        """Test automatic backend selection based on capabilities and priority."""
        # Fresh registry
        BackendRegistry._instance = None
        registry = BackendRegistry()
        
        # Remove defaults for clean test
        registry._backends.clear()
        
        # Register multiple backends with different capabilities and priorities
        registry.register(
            "backend_a",
            MockBackend,
            BackendCapability.JSON,
            priority=5
        )
        registry.register(
            "backend_b",
            MockBackend,
            BackendCapability.JSON | BackendCapability.REGEX,
            priority=10
        )
        registry.register(
            "backend_c",
            MockBackend,
            BackendCapability.REGEX,
            priority=15
        )
        
        # backend_b wins for JSON (priority 10 > 5)
        assert registry.get_best_backend_for_option(StructuredOutputOptions.JSON) == "backend_b"
        
        # backend_c wins for REGEX (priority 15)
        assert registry.get_best_backend_for_option(StructuredOutputOptions.REGEX) == "backend_c"
        
        # Test with preferred backend
        assert registry.get_best_backend_for_option(
            StructuredOutputOptions.JSON, 
            preferred_backend="backend_a"
        ) == "backend_a"
    
    def test_backend_metadata_supports(self):
        """Test BackendMetadata.supports() method."""
        metadata = BackendMetadata(
            name="test",
            backend_class=MockBackend,
            capabilities=BackendCapability.JSON | BackendCapability.REGEX
        )
        
        assert metadata.supports(StructuredOutputOptions.JSON)
        assert metadata.supports(StructuredOutputOptions.REGEX)
        assert not metadata.supports(StructuredOutputOptions.GRAMMAR)


class TestGrammarCache:
    """Tests for the GrammarCache class."""
    
    def test_cache_basic_operations(self):
        """Test basic cache operations: get, put, size."""
        cache = GrammarCache(max_size=3)
        
        # Test empty cache
        assert cache.size() == 0
        assert cache.get("backend1", StructuredOutputOptions.JSON, "spec1") is None
        
        # Test putting items
        grammar1 = MockGrammar("grammar1")
        cache.put("backend1", StructuredOutputOptions.JSON, "spec1", grammar1)
        assert cache.size() == 1
        
        # Test getting items
        cached_grammar = cache.get("backend1", StructuredOutputOptions.JSON, "spec1")
        assert cached_grammar is grammar1
        
        # Test different keys
        grammar2 = MockGrammar("grammar2")
        cache.put("backend2", StructuredOutputOptions.REGEX, "spec2", grammar2)
        assert cache.size() == 2
    
    def test_cache_eviction(self):
        """Test LRU eviction when cache is full."""
        cache = GrammarCache(max_size=2)
        
        grammar1 = MockGrammar("grammar1")
        grammar2 = MockGrammar("grammar2")
        grammar3 = MockGrammar("grammar3")
        
        # Fill cache
        cache.put("backend", StructuredOutputOptions.JSON, "spec1", grammar1)
        cache.put("backend", StructuredOutputOptions.JSON, "spec2", grammar2)
        assert cache.size() == 2
        
        # Third item evicts oldest (grammar1)
        cache.put("backend", StructuredOutputOptions.JSON, "spec3", grammar3)
        assert cache.size() == 2
        assert cache.get("backend", StructuredOutputOptions.JSON, "spec1") is None  # evicted
        assert cache.get("backend", StructuredOutputOptions.JSON, "spec2") is grammar2
        assert cache.get("backend", StructuredOutputOptions.JSON, "spec3") is grammar3
    
    def test_cache_lru_update(self):
        """Test that accessing items updates LRU order."""
        cache = GrammarCache(max_size=2)
        
        grammar1 = MockGrammar("grammar1")
        grammar2 = MockGrammar("grammar2")
        grammar3 = MockGrammar("grammar3")
        
        cache.put("backend", StructuredOutputOptions.JSON, "spec1", grammar1)
        cache.put("backend", StructuredOutputOptions.JSON, "spec2", grammar2)
        
        # Touch grammar1 to make it recent
        cache.get("backend", StructuredOutputOptions.JSON, "spec1")
        
        # Now grammar2 gets evicted
        cache.put("backend", StructuredOutputOptions.JSON, "spec3", grammar3)
        assert cache.get("backend", StructuredOutputOptions.JSON, "spec1") is grammar1
        assert cache.get("backend", StructuredOutputOptions.JSON, "spec2") is None
        assert cache.get("backend", StructuredOutputOptions.JSON, "spec3") is grammar3
    
    def test_cache_clear(self):
        """Test clearing the cache."""
        cache = GrammarCache(max_size=5)
        
        for i in range(3):
            cache.put("backend", StructuredOutputOptions.JSON, f"spec{i}", MockGrammar(f"g{i}"))
        
        assert cache.size() == 3
        cache.clear()
        assert cache.size() == 0
        assert cache.get("backend", StructuredOutputOptions.JSON, "spec0") is None
    
    def test_cache_thread_safety(self):
        """Test that cache operations are thread-safe."""
        cache = GrammarCache(max_size=100)
        errors = []
        
        def writer_thread(thread_id):
            try:
                for i in range(10):
                    cache.put(
                        f"backend{thread_id}",
                        StructuredOutputOptions.JSON,
                        f"spec{i}",
                        MockGrammar(f"grammar_{thread_id}_{i}")
                    )
            except Exception as e:
                errors.append(e)
        
        def reader_thread(thread_id):
            try:
                for i in range(10):
                    cache.get(
                        f"backend{thread_id % 5}",
                        StructuredOutputOptions.JSON,
                        f"spec{i}"
                    )
            except Exception as e:
                errors.append(e)
        
        threads = []
        for i in range(5):
            threads.append(threading.Thread(target=writer_thread, args=(i,)))
            threads.append(threading.Thread(target=reader_thread, args=(i,)))
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join()
        
        assert len(errors) == 0, f"Thread safety errors: {errors}"


class TestUnifiedBackendManager:
    """Tests for the UnifiedBackendManager class."""
    
    def setup_method(self):
        """Reset registry before each test."""
        BackendRegistry._instance = None
    
    def teardown_method(self):
        """Clean up after each test."""
        BackendRegistry._instance = None
    
    @pytest.fixture
    def mock_vllm_config(self):
        """Create a mock VLLM configuration."""
        config = MagicMock()
        config.decoding_config.disable_any_whitespace = False
        config.decoding_config.disable_additional_properties = False
        config.speculative_config = None
        return config
    
    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer with proper vocabulary."""
        tokenizer = MagicMock()
        tokenizer.get_vocab_size.return_value = 32000
        # xgrammar needs non-empty vocab
        tokenizer.get_vocab.return_value = {f"token_{i}": i for i in range(100)}
        tokenizer.eos_token_id = 2
        return tokenizer
    
    def test_manager_initialization(self, mock_vllm_config, mock_tokenizer):
        """Test UnifiedBackendManager initialization."""
        manager = UnifiedBackendManager(
            vllm_config=mock_vllm_config,
            tokenizer=mock_tokenizer,
            vocab_size=32000,
            preferred_backend="xgrammar",
            cache_size=500
        )
        
        assert manager.vllm_config == mock_vllm_config
        assert manager.tokenizer == mock_tokenizer
        assert manager.vocab_size == 32000
        assert manager.preferred_backend == "xgrammar"
        assert manager.grammar_cache.max_size == 500
    
    def test_backend_creation_lazy(self, mock_vllm_config, mock_tokenizer):
        """Test that backends are created lazily."""
        # Fresh registry, no defaults
        registry = BackendRegistry()
        registry._backends.clear()
        
        manager = UnifiedBackendManager(
            vllm_config=mock_vllm_config,
            tokenizer=mock_tokenizer,
            vocab_size=32000
        )
        
        # Register a test backend
        manager.registry.register(
            "test_backend",
            MockBackend,
            BackendCapability.JSON,
            priority=20
        )
        
        # No backends should be created yet
        assert len(manager._backends) == 0
        
        # Compile a grammar
        grammar = manager.compile_grammar(
            StructuredOutputOptions.JSON,
            '{"type": "string"}',
            backend_name="test_backend"
        )
        
        # Now the backend should be created
        assert "test_backend" in manager._backends
        assert isinstance(manager._backends["test_backend"], MockBackend)
    
    def test_grammar_compilation_with_cache(self, mock_vllm_config, mock_tokenizer):
        """Test grammar compilation with caching."""
        # Create fresh registry
        registry = BackendRegistry()
        registry._backends.clear()
        
        manager = UnifiedBackendManager(
            vllm_config=mock_vllm_config,
            tokenizer=mock_tokenizer,
            vocab_size=32000
        )
        
        # Register a test backend
        manager.registry.register(
            "test_backend",
            MockBackend,
            BackendCapability.JSON,
            priority=20
        )
        
        # First compilation
        grammar1 = manager.compile_grammar(
            StructuredOutputOptions.JSON,
            '{"type": "string"}',
            backend_name="test_backend"
        )
        
        backend = manager._backends["test_backend"]
        assert backend.compile_count == 1
        
        # Second compilation of same grammar should use cache
        grammar2 = manager.compile_grammar(
            StructuredOutputOptions.JSON,
            '{"type": "string"}',
            backend_name="test_backend"
        )
        
        # Still 1 - used cache
        assert backend.compile_count == 1
        
        # Different grammar needs compilation
        grammar3 = manager.compile_grammar(
            StructuredOutputOptions.JSON,
            '{"type": "number"}',
            backend_name="test_backend"
        )
        
        assert backend.compile_count == 2
    
    def test_automatic_backend_selection(self, mock_vllm_config, mock_tokenizer):
        """Test automatic backend selection based on request type."""
        # Create fresh registry and manager
        registry = BackendRegistry()
        registry._backends.clear()
        
        manager = UnifiedBackendManager(
            vllm_config=mock_vllm_config,
            tokenizer=mock_tokenizer,
            vocab_size=32000
        )
        
        # Register backends with different capabilities
        manager.registry.register(
            "json_only",
            MockBackend,
            BackendCapability.JSON,
            priority=10
        )
        manager.registry.register(
            "regex_only",
            MockBackend,
            BackendCapability.REGEX,
            priority=10
        )
        
        # Auto-picks json_only for JSON
        grammar = manager.compile_grammar(
            StructuredOutputOptions.JSON,
            '{"type": "string"}'
        )
        assert "json_only" in manager._backends
        
        # Auto-picks regex_only for REGEX
        grammar = manager.compile_grammar(
            StructuredOutputOptions.REGEX,
            r'\d+'
        )
        assert "regex_only" in manager._backends
    
    def test_backend_stats(self, mock_vllm_config, mock_tokenizer):
        """Test getting backend statistics."""
        # Create fresh registry
        registry = BackendRegistry()
        registry._backends.clear()
        
        manager = UnifiedBackendManager(
            vllm_config=mock_vllm_config,
            tokenizer=mock_tokenizer,
            vocab_size=32000,
            preferred_backend="test_backend"
        )
        
        manager.registry.register(
            "test_backend",
            MockBackend,
            BackendCapability.JSON,
            priority=10
        )
        
        # Compile some grammars
        for i in range(3):
            manager.compile_grammar(
                StructuredOutputOptions.JSON,
                f'{{"type": "string{i}"}}',
                backend_name="test_backend"
            )
        
        stats = manager.get_backend_stats()
        
        assert "test_backend" in stats["registered_backends"]
        assert "test_backend" in stats["active_backends"]
        assert stats["cache_size"] == 3
        assert stats["preferred_backend"] == "test_backend"
    
    def test_manager_destroy(self, mock_vllm_config, mock_tokenizer):
        """Test cleanup when destroying manager."""
        # Create fresh registry
        registry = BackendRegistry()
        registry._backends.clear()
        
        manager = UnifiedBackendManager(
            vllm_config=mock_vllm_config,
            tokenizer=mock_tokenizer,
            vocab_size=32000
        )
        
        manager.registry.register(
            "mock_test_backend",  # unique name
            MockBackend,
            BackendCapability.JSON,
            priority=10
        )
        
        # Create a backend and cache some grammars
        manager.compile_grammar(
            StructuredOutputOptions.JSON, 
            '{"type": "string"}',
            backend_name="mock_test_backend"
        )
        
        assert len(manager._backends) == 1
        assert manager.grammar_cache.size() == 1
        
        # Destroy should clean everything
        manager.destroy()
        
        assert len(manager._backends) == 0
        assert manager.grammar_cache.size() == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])