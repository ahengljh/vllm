# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# CMake configuration for CPAC extension

message(STATUS "Configuring CPAC (Cross-Page Adaptive Compression) extension")

# CPAC source files
set(CPAC_EXT_SRC
    "csrc/cpac_kernels.cu"
    "csrc/cpac_ops.cpp")

# Add CPAC sources to main extension if CUDA is available
if(VLLM_GPU_LANG STREQUAL "CUDA")
    message(STATUS "Adding CPAC CUDA kernels to build")
    list(APPEND VLLM_EXT_SRC ${CPAC_EXT_SRC})
    
    # Add compile definitions for CPAC
    add_compile_definitions(VLLM_CPAC_ENABLED)
endif()