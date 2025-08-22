# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Essential Development Commands

### Installation & Build
- **Install from source**: `pip install -e .`
- **Development dependencies**: `pip install -r requirements/dev.txt`
- **Lint dependencies**: `pip install -r requirements/lint.txt`

### Testing
- **Run all tests**: `pytest`
- **Run a single test**: `pytest tests/path/to/test_file.py::TestClass::test_method`
- **Run core model tests**: `pytest -m core_model`
- **Run distributed tests**: `pytest -m distributed`
- **Run CPU tests**: `pytest -m cpu_model`
- **Run with optional tests**: `pytest --optional`

### Linting & Formatting
- **Setup pre-commit**: `pre-commit install`
- **Run all pre-commit checks**: `pre-commit run --all-files`
- **Type checking**: `tools/mypy.sh`
- **Format Python code**: `yapf -i <file>` or `ruff format <file>`
- **Sort imports**: `isort <file>`

## High-Level Architecture

### Core Components

**Engine Layer** (`vllm/engine/`)
- `LLMEngine`: Core synchronous inference engine that manages request scheduling, batching, and model execution
- `AsyncLLMEngine`: Asynchronous wrapper for serving applications
- Request processing pipeline: tokenization → scheduling → execution → detokenization

**Model Executor** (`vllm/model_executor/`)
- Handles model loading, weight management, and distributed execution
- Supports multiple quantization formats (GPTQ, AWQ, FP8, SqueezeLLM, etc.)
- Model registry system for automatic model discovery and loading

**Attention Mechanisms** (`vllm/attention/`)
- PagedAttention: Core memory-efficient attention implementation
- Multiple backend support: FlashAttention, xFormers, FlashInfer, Torch SDPA
- Automatic backend selection based on hardware and model requirements

**Memory Management** (`vllm/core/`)
- Block-based memory allocation for KV cache
- Scheduler manages request priorities and memory allocation
- Supports preemption and swapping for handling memory pressure

**Worker System** (`vllm/worker/`)
- Worker processes execute model computations
- Supports different worker types: GPU, CPU, TPU, Neuron
- Handles model parallel and pipeline parallel execution

**V1 Architecture** (`vllm/v1/`)
- New experimental architecture (alpha) with 1.7x speedup
- Cleaner separation of concerns and improved performance
- Currently supports limited models (check v1/models/)

### Request Flow
1. Request enters through entrypoint (API server or offline LLM)
2. Engine tokenizes input and creates request object
3. Scheduler allocates memory blocks and schedules request
4. Worker executes model forward pass with batched requests
5. Output tokens are generated iteratively until completion
6. Results are detokenized and returned to client

### Key Design Patterns
- **Plugin Architecture**: Extensible system for custom models, attention backends, and executors
- **Lazy Initialization**: Models and kernels loaded on-demand to reduce memory footprint
- **Unified Interface**: Common abstractions across different hardware backends
- **Async-First Design**: Built for high-throughput serving with async/await patterns

## Important Development Considerations

- **Multi-Platform Support**: Code may behave differently on CUDA, ROCm, CPU, TPU. Check platform-specific code in `vllm/platforms/`
- **Performance Critical**: Changes to core execution paths should be benchmarked. Use `benchmarks/` for performance testing
- **Memory Efficiency**: vLLM optimizes for memory usage. Consider memory implications when modifying core components
- **Distributed Execution**: Many components support tensor/pipeline parallelism. Test with distributed setups when modifying these areas
- **Kernel Compilation**: C++/CUDA kernels in `csrc/` require recompilation. Clean builds may be needed after modifications
- **Pre-commit Hooks**: Always run pre-commit before pushing. The project enforces strict formatting and linting rules
- **Type Annotations**: New code should include type hints. MyPy checks are enforced in CI