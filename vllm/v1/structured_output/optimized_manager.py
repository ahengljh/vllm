# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Optimized structured output manager with enhanced batch processing and unified backend."""

from __future__ import annotations

import asyncio
import multiprocessing
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.reasoning import ReasoningParserManager
from vllm.transformers_utils.tokenizer_group import init_tokenizer_from_configs
from vllm.v1.structured_output.backend_types import (
    StructuredOutputGrammar,
    StructuredOutputOptions,
)
from vllm.v1.structured_output.unified_backend import UnifiedBackendManager

if TYPE_CHECKING:
    import numpy.typing as npt

    from vllm.reasoning import ReasoningParser
    from vllm.v1.request import Request

logger = init_logger(__name__)


@dataclass
class BatchProcessingConfig:
    """Configuration for batch processing optimization."""
    
    # Threshold for triggering parallel processing
    parallel_threshold: int = 128
    
    # Size of each batch for parallel processing
    batch_size: int = 16
    
    # Maximum number of worker threads
    max_workers: int = field(default_factory=lambda: max(1, min(multiprocessing.cpu_count() // 2, 8)))
    
    # Enable adaptive batching based on load
    adaptive_batching: bool = True
    
    # Cache warmup size (pre-compile common patterns)
    cache_warmup_size: int = 100


class OptimizedStructuredOutputManager:
    """Enhanced structured output manager with unified backend and optimizations."""
    
    def __init__(
        self,
        vllm_config: VllmConfig,
        preferred_backend: Optional[str] = None,
        batch_config: Optional[BatchProcessingConfig] = None
    ):
        self.vllm_config = vllm_config
        self.batch_config = batch_config or BatchProcessingConfig()
        
        # Initialize tokenizer
        model_config = self.vllm_config.model_config
        tokenizer_config = self.vllm_config.tokenizer_config
        self.tokenizer = init_tokenizer_from_configs(
            model_config=model_config,
            scheduler_config=self.vllm_config.scheduler_config,
            parallel_config=self.vllm_config.parallel_config,
            enable_lora=bool(self.vllm_config.lora_config),
            tokenizer_config=tokenizer_config,
        )
        
        # Determine vocabulary size
        vocab_size = self.tokenizer.get_vocab_size()
        if hasattr(self.tokenizer, "get_added_vocab"):
            vocab_size += len(self.tokenizer.get_added_vocab())
        
        # Initialize unified backend manager
        self.backend_manager = UnifiedBackendManager(
            vllm_config=vllm_config,
            tokenizer=self.tokenizer,
            vocab_size=vocab_size,
            preferred_backend=preferred_backend,
            cache_size=self.batch_config.cache_warmup_size * 10  # Allow for growth
        )
        
        # Initialize reasoning parser if needed
        self.reasoner: Optional[ReasoningParser] = None
        if model_config.supported_reasoning is not None:
            rpm = ReasoningParserManager()
            self.reasoner = rpm.get_reasoning_parser(
                self.tokenizer,
                model_config.supported_reasoning,
                self.vllm_config.decoding_config.reasoning_config,
            )
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=self.batch_config.max_workers)
        self.executor_for_fillmask = ThreadPoolExecutor(max_workers=self.batch_config.max_workers)
        
        # Bitmask management
        self._grammar_bitmask: Optional[torch.Tensor] = None
        self._full_mask = torch.tensor(-1, dtype=torch.int32)
        
        # Performance tracking
        self._batch_processing_times: List[float] = []
        self._cache_hits = 0
        self._cache_misses = 0
        
        # Warm up cache with common patterns if configured
        if self.batch_config.cache_warmup_size > 0:
            self._warmup_cache()
    
    def _warmup_cache(self):
        """Pre-compile common grammar patterns to warm up the cache."""
        common_patterns = [
            (StructuredOutputOptions.JSON_OBJECT, '{"type": "object"}'),
            (StructuredOutputOptions.JSON, '{"type": "string"}'),
            (StructuredOutputOptions.JSON, '{"type": "number"}'),
            (StructuredOutputOptions.JSON, '{"type": "array", "items": {"type": "string"}}'),
            (StructuredOutputOptions.REGEX, r'\d+'),
            (StructuredOutputOptions.REGEX, r'[a-zA-Z]+'),
        ]
        
        for request_type, grammar_spec in common_patterns[:self.batch_config.cache_warmup_size]:
            try:
                self.backend_manager.compile_grammar(request_type, grammar_spec)
                logger.debug(f"Warmed up cache with {request_type}: {grammar_spec[:50]}...")
            except Exception as e:
                logger.debug(f"Failed to warm up pattern: {e}")
    
    def create_grammar(self, request: Request):
        """Create a grammar for a structured output request."""
        if request.structured_output_request is None:
            return
        
        if request.structured_output_request.grammar is not None:
            # Grammar already exists
            return
        
        # Submit grammar compilation to thread pool for async processing
        grammar_future = self.executor.submit(self._async_create_grammar, request)
        request.structured_output_request.grammar = grammar_future  # type: ignore[assignment]
    
    def _async_create_grammar(self, request: Request) -> StructuredOutputGrammar:
        """Asynchronously create a grammar with caching."""
        key = request.structured_output_request.structured_output_key  # type: ignore[union-attr]
        request_type, grammar_spec = key
        
        # Track cache performance
        grammar = self.backend_manager.compile_grammar(request_type, grammar_spec)
        
        # The unified backend already handles caching, but we can track stats
        if hasattr(grammar, '_from_cache'):
            self._cache_hits += 1
        else:
            self._cache_misses += 1
        
        return grammar
    
    def _optimize_batch_size(self, num_requests: int) -> int:
        """Dynamically optimize batch size based on performance history."""
        if not self.batch_config.adaptive_batching:
            return self.batch_config.batch_size
        
        # Simple adaptive algorithm: adjust based on recent processing times
        if len(self._batch_processing_times) < 10:
            return self.batch_config.batch_size
        
        avg_time = sum(self._batch_processing_times[-10:]) / 10
        
        # If processing is fast, increase batch size; if slow, decrease
        if avg_time < 0.01:  # Less than 10ms average
            return min(self.batch_config.batch_size * 2, 64)
        elif avg_time > 0.1:  # More than 100ms average
            return max(self.batch_config.batch_size // 2, 4)
        
        return self.batch_config.batch_size
    
    def _fill_bitmasks_batch(
        self,
        batch: List[Tuple[StructuredOutputGrammar, int, bool]],
    ) -> None:
        """Fill bitmasks for a batch of requests."""
        import time
        start_time = time.time()
        
        assert self._grammar_bitmask is not None
        for grammar, index, apply_bitmask in batch:
            if apply_bitmask and not grammar.is_terminated():
                grammar.fill_bitmask(self._grammar_bitmask, index)
            else:
                self._grammar_bitmask[index].fill_(self._full_mask)
        
        # Track processing time for adaptive batching
        processing_time = time.time() - start_time
        self._batch_processing_times.append(processing_time)
        if len(self._batch_processing_times) > 100:
            self._batch_processing_times.pop(0)
    
    def _async_submit_fill_bitmask(
        self,
        batch: List[Tuple[StructuredOutputGrammar, int, bool]],
    ) -> Future:
        """Submit a batch of bitmask filling operations to the thread pool."""
        return self.executor_for_fillmask.submit(self._fill_bitmasks_batch, batch)
    
    def grammar_bitmask(
        self,
        requests: Dict[str, Request],
        structured_output_request_ids: Dict[str, int],
        scheduled_spec_decode_tokens: Dict[str, List[int]],
    ) -> Optional[npt.NDArray[np.int32]]:
        """Generate optimized grammar bitmasks for a batch of requests."""
        if not structured_output_request_ids:
            return None
        
        max_num_spec_tokens = 0
        if self.vllm_config.speculative_config is not None:
            max_num_spec_tokens = self.vllm_config.speculative_config.num_speculative_tokens
        
        # Allocate bitmask if needed
        if self._grammar_bitmask is None:
            max_batch_size = self.vllm_config.scheduler_config.max_num_seqs
            self._grammar_bitmask = self.backend_manager.allocate_token_bitmask(
                max_batch_size * (1 + max_num_spec_tokens)
            )
        
        # Sort requests for consistent ordering
        ordered_seq = sorted(structured_output_request_ids.items(), key=lambda x: x[1])
        cumulative_index = 0
        
        # Determine optimal batch size
        num_requests = len(ordered_seq)
        optimal_batch_size = self._optimize_batch_size(num_requests)
        
        # Use parallel processing for large batches without speculative decoding
        if num_requests > self.batch_config.parallel_threshold and max_num_spec_tokens == 0:
            promises = []
            batch = []
            
            for req_id, _ in ordered_seq:
                request = requests[req_id]
                structured_output_request = request.structured_output_request
                
                if TYPE_CHECKING:
                    assert structured_output_request is not None
                    assert structured_output_request.grammar is not None
                
                # Resolve future if needed
                grammar = structured_output_request.grammar
                if isinstance(grammar, Future):
                    grammar = grammar.result()
                    structured_output_request.grammar = grammar
                
                apply_bitmask = self._should_fill_bitmask(request)
                batch.append((grammar, cumulative_index, apply_bitmask))
                
                if len(batch) >= optimal_batch_size:
                    promises.append(self._async_submit_fill_bitmask(batch))
                    batch = []
                
                cumulative_index += 1
            
            # Submit remaining batch
            if batch:
                promises.append(self._async_submit_fill_bitmask(batch))
            
            # Wait for all parallel operations to complete
            for promise in promises:
                promise.result()
        else:
            # Serial processing for small batches or speculative decoding
            for req_id, _ in ordered_seq:
                request = requests[req_id]
                structured_output_request = request.structured_output_request
                
                if TYPE_CHECKING:
                    assert structured_output_request is not None
                    assert structured_output_request.grammar is not None
                
                # Resolve future if needed
                grammar = structured_output_request.grammar
                if isinstance(grammar, Future):
                    grammar = grammar.result()
                    structured_output_request.grammar = grammar
                
                apply_bitmask = self._should_fill_bitmask(request)
                
                # Handle speculative tokens
                state_advancements = 0
                req_tokens = scheduled_spec_decode_tokens.get(req_id, [])
                for token in req_tokens + [None]:
                    self._fill_bitmasks_batch([(grammar, cumulative_index, apply_bitmask)])
                    
                    if apply_bitmask and token is not None and not grammar.is_terminated():
                        assert grammar.accept_tokens(req_id, [token])
                        state_advancements += 1
                    cumulative_index += 1
                
                # Rollback speculative advances
                if state_advancements > 0:
                    grammar.rollback(state_advancements)
        
        # Return the appropriate slice of the bitmask
        bitmask_tensor = self._grammar_bitmask
        if cumulative_index < bitmask_tensor.shape[0]:
            bitmask_tensor = bitmask_tensor[:cumulative_index]
        
        return bitmask_tensor.numpy()
    
    def _should_fill_bitmask(self, request: Request) -> bool:
        """Determine if a bitmask should be filled for a request."""
        # This is a simplified version - you would implement the actual logic
        return request.structured_output_request is not None
    
    def process_outputs(
        self,
        request: Request,
        generated_token_ids: List[int]
    ) -> bool:
        """Process generated tokens through the grammar."""
        if request.structured_output_request is None:
            return True
        
        grammar = request.structured_output_request.grammar
        if isinstance(grammar, Future):
            grammar = grammar.result()
            request.structured_output_request.grammar = grammar
        
        return grammar.accept_tokens(request.request_id, generated_token_ids)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        cache_hit_rate = 0.0
        if self._cache_hits + self._cache_misses > 0:
            cache_hit_rate = self._cache_hits / (self._cache_hits + self._cache_misses)
        
        avg_batch_time = 0.0
        if self._batch_processing_times:
            avg_batch_time = sum(self._batch_processing_times) / len(self._batch_processing_times)
        
        return {
            **self.backend_manager.get_backend_stats(),
            "cache_hit_rate": cache_hit_rate,
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "avg_batch_processing_time_ms": avg_batch_time * 1000,
            "adaptive_batching": self.batch_config.adaptive_batching,
        }
    
    def destroy(self):
        """Clean up resources."""
        self.executor.shutdown(wait=True)
        self.executor_for_fillmask.shutdown(wait=True)
        self.backend_manager.destroy()
        if self._grammar_bitmask is not None:
            del self._grammar_bitmask