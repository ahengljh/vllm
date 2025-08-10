# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unified backend system for structured output with registry pattern and caching."""

from __future__ import annotations

import enum
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Type

import torch

from vllm.logger import init_logger
from vllm.v1.structured_output.backend_types import (
    StructuredOutputBackend,
    StructuredOutputGrammar,
    StructuredOutputOptions,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.transformers_utils.tokenizer import AnyTokenizer

logger = init_logger(__name__)


class BackendCapability(enum.Flag):
    """Backend capabilities."""
    JSON = enum.auto()
    JSON_OBJECT = enum.auto()
    REGEX = enum.auto()
    GRAMMAR = enum.auto()
    CHOICE = enum.auto()
    STRUCTURAL_TAG = enum.auto()
    JUMP_DECODING = enum.auto()
    PARALLEL_COMPILATION = enum.auto()
    INCREMENTAL_PARSING = enum.auto()


@dataclass
class BackendMetadata:
    """Metadata for a registered backend."""
    name: str
    backend_class: Type[StructuredOutputBackend]
    capabilities: BackendCapability
    priority: int = 0  # Higher priority backends are preferred
    lazy_import: Optional[Callable[[], None]] = None
    
    def supports(self, option: StructuredOutputOptions) -> bool:
        """Check if this backend supports a specific option."""
        capability_map = {
            StructuredOutputOptions.JSON: BackendCapability.JSON,
            StructuredOutputOptions.JSON_OBJECT: BackendCapability.JSON_OBJECT,
            StructuredOutputOptions.REGEX: BackendCapability.REGEX,
            StructuredOutputOptions.GRAMMAR: BackendCapability.GRAMMAR,
            StructuredOutputOptions.CHOICE: BackendCapability.CHOICE,
            StructuredOutputOptions.STRUCTURAL_TAG: BackendCapability.STRUCTURAL_TAG,
        }
        required_capability = capability_map.get(option)
        return required_capability is not None and required_capability in self.capabilities


class BackendRegistry:
    """Registry for structured output backends."""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._backends: Dict[str, BackendMetadata] = {}
        self._initialized = True
        self._register_default_backends()
    
    def _register_default_backends(self):
        """Register the default backends."""
        # XGrammar - best overall support
        self.register(
            "xgrammar",
            backend_class=None,  # lazy load
            capabilities=(
                BackendCapability.JSON |
                BackendCapability.JSON_OBJECT |
                BackendCapability.REGEX |
                BackendCapability.GRAMMAR |
                BackendCapability.STRUCTURAL_TAG
            ),
            priority=10,
            lazy_import=self._lazy_import_xgrammar
        )
        
        # Guidance - good for complex patterns
        self.register(
            "guidance",
            backend_class=None,
            capabilities=(
                BackendCapability.JSON |
                BackendCapability.JSON_OBJECT |
                BackendCapability.REGEX |
                BackendCapability.GRAMMAR |
                BackendCapability.CHOICE
            ),
            priority=5,
            lazy_import=self._lazy_import_guidance
        )
        
        # Outlines - fast for simple patterns
        self.register(
            "outlines",
            backend_class=None,
            capabilities=(
                BackendCapability.JSON |
                BackendCapability.REGEX |
                BackendCapability.CHOICE
            ),
            priority=3,
            lazy_import=self._lazy_import_outlines
        )
    
    def _lazy_import_xgrammar(self) -> Type[StructuredOutputBackend]:
        from vllm.v1.structured_output.backend_xgrammar import XgrammarBackend
        return XgrammarBackend
    
    def _lazy_import_guidance(self) -> Type[StructuredOutputBackend]:
        from vllm.v1.structured_output.backend_guidance import GuidanceBackend
        return GuidanceBackend
    
    def _lazy_import_outlines(self) -> Type[StructuredOutputBackend]:
        from vllm.v1.structured_output.backend_outlines import OutlinesBackend
        return OutlinesBackend
    
    def register(
        self,
        name: str,
        backend_class: Optional[Type[StructuredOutputBackend]],
        capabilities: BackendCapability,
        priority: int = 0,
        lazy_import: Optional[Callable[[], Type[StructuredOutputBackend]]] = None
    ):
        """Register a new backend."""
        if name in self._backends:
            logger.warning(f"Backend '{name}' is already registered. Overwriting.")
        
        self._backends[name] = BackendMetadata(
            name=name,
            backend_class=backend_class,
            capabilities=capabilities,
            priority=priority,
            lazy_import=lazy_import
        )
    
    def get_backend_class(self, name: str) -> Type[StructuredOutputBackend]:
        """Get a backend class by name."""
        if name not in self._backends:
            raise ValueError(f"Backend '{name}' is not registered")
        
        metadata = self._backends[name]
        if metadata.backend_class is None:
            if metadata.lazy_import is not None:
                metadata.backend_class = metadata.lazy_import()
            else:
                raise ValueError(f"Backend '{name}' has no implementation")
        
        return metadata.backend_class
    
    def get_best_backend_for_option(
        self,
        option: StructuredOutputOptions,
        preferred_backend: Optional[str] = None
    ) -> str:
        """Get the best backend for a specific option."""
        # Try preferred backend first
        if preferred_backend and preferred_backend in self._backends:
            metadata = self._backends[preferred_backend]
            if metadata.supports(option):
                return preferred_backend
            logger.warning(
                f"Preferred backend '{preferred_backend}' does not support "
                f"{option}. Falling back to automatic selection."
            )
        
        # Find compatible backends
        supporting_backends = [
            (name, metadata)
            for name, metadata in self._backends.items()
            if metadata.supports(option)
        ]
        
        if not supporting_backends:
            raise ValueError(f"No backend supports {option}")
        
        # Pick highest priority
        supporting_backends.sort(key=lambda x: x[1].priority, reverse=True)
        
        return supporting_backends[0][0]
    
    def list_backends(self) -> Dict[str, BackendCapability]:
        """List all registered backends and their capabilities."""
        return {
            name: metadata.capabilities
            for name, metadata in self._backends.items()
        }


@dataclass
class GrammarCache:
    """LRU cache for compiled grammars."""
    
    max_size: int = 1000
    _cache: Dict[tuple, StructuredOutputGrammar] = field(default_factory=dict)
    _access_order: list = field(default_factory=list)  # oldest to newest
    _lock: threading.Lock = field(default_factory=threading.Lock)
    
    def get(
        self,
        backend_name: str,
        request_type: StructuredOutputOptions,
        grammar_spec: str
    ) -> Optional[StructuredOutputGrammar]:
        """Get a cached grammar if it exists."""
        key = (backend_name, request_type, grammar_spec)
        with self._lock:
            if key in self._cache:
                # Update LRU order
                self._access_order.remove(key)
                self._access_order.append(key)
                return self._cache[key]
        return None
    
    def put(
        self,
        backend_name: str,
        request_type: StructuredOutputOptions,
        grammar_spec: str,
        grammar: StructuredOutputGrammar
    ):
        """Store a compiled grammar in the cache."""
        key = (backend_name, request_type, grammar_spec)
        with self._lock:
            if key in self._cache:
                # Just update LRU
                self._access_order.remove(key)
                self._access_order.append(key)
                return
            
            # Evict if full
            if len(self._cache) >= self.max_size:
                oldest_key = self._access_order.pop(0)
                del self._cache[oldest_key]
            
            self._cache[key] = grammar
            self._access_order.append(key)
    
    def clear(self):
        """Clear the entire cache."""
        with self._lock:
            self._cache.clear()
            self._access_order.clear()
    
    def size(self) -> int:
        """Get the current cache size."""
        with self._lock:
            return len(self._cache)


class UnifiedBackendManager:
    """Manager for structured output backends with caching."""
    
    def __init__(
        self,
        vllm_config: VllmConfig,
        tokenizer: AnyTokenizer,
        vocab_size: int,
        preferred_backend: Optional[str] = None,
        cache_size: int = 1000
    ):
        self.vllm_config = vllm_config
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.preferred_backend = preferred_backend
        
        self.registry = BackendRegistry()
        self.grammar_cache = GrammarCache(max_size=cache_size)
        self._backends: Dict[str, StructuredOutputBackend] = {}
        self._backend_locks: Dict[str, threading.Lock] = {}
    
    def _get_or_create_backend(self, backend_name: str) -> StructuredOutputBackend:
        """Get or create a backend instance."""
        if backend_name not in self._backends:
            if backend_name not in self._backend_locks:
                self._backend_locks[backend_name] = threading.Lock()
            
            with self._backend_locks[backend_name]:
                # Double-check pattern
                if backend_name not in self._backends:
                    backend_class = self.registry.get_backend_class(backend_name)
                    self._backends[backend_name] = backend_class(
                        vllm_config=self.vllm_config,
                        tokenizer=self.tokenizer,
                        vocab_size=self.vocab_size
                    )
        
        return self._backends[backend_name]
    
    def compile_grammar(
        self,
        request_type: StructuredOutputOptions,
        grammar_spec: str,
        backend_name: Optional[str] = None
    ) -> StructuredOutputGrammar:
        """Compile a grammar with caching and automatic backend selection."""
        # Pick backend
        if backend_name is None:
            backend_name = self.registry.get_best_backend_for_option(
                request_type,
                preferred_backend=self.preferred_backend
            )
        
        # Try cache
        cached_grammar = self.grammar_cache.get(backend_name, request_type, grammar_spec)
        if cached_grammar is not None:
            logger.debug(f"Cache hit: {backend_name}:{request_type}")
            cached_grammar.reset()  # reset state
            return cached_grammar
        
        # Compile new
        backend = self._get_or_create_backend(backend_name)
        grammar = backend.compile_grammar(request_type, grammar_spec)
        
        # Store in cache
        self.grammar_cache.put(backend_name, request_type, grammar_spec, grammar)
        
        logger.debug(f"Compiled: {backend_name} for {request_type}")
        return grammar
    
    def allocate_token_bitmask(
        self,
        max_num_seqs: int,
        backend_name: Optional[str] = None
    ) -> torch.Tensor:
        """Allocate a token bitmask for the specified backend."""
        if backend_name is None:
            # Use the first available backend
            backend_name = next(iter(self.registry.list_backends().keys()))
        
        backend = self._get_or_create_backend(backend_name)
        return backend.allocate_token_bitmask(max_num_seqs)
    
    def destroy(self):
        """Clean up all backends and clear caches."""
        for backend in self._backends.values():
            backend.destroy()
        self._backends.clear()
        self.grammar_cache.clear()
    
    def get_backend_stats(self) -> Dict[str, Any]:
        """Get statistics about backend usage and cache performance."""
        return {
            "registered_backends": list(self.registry.list_backends().keys()),
            "active_backends": list(self._backends.keys()),
            "cache_size": self.grammar_cache.size(),
            "cache_max_size": self.grammar_cache.max_size,
            "preferred_backend": self.preferred_backend,
        }