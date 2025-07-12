#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "cuda_utils.h"
#include "dispatch_utils.h"

namespace vllm {
namespace cpac {

// Constants for CPAC compression
constexpr int WARP_SIZE = 32;
constexpr int MAX_BLOCK_SIZE = 16;  // Max tokens per block
constexpr int SIMILARITY_FEATURE_DIM = 128;

// Kernel for computing page features for similarity matching
template<typename scalar_t>
__global__ void compute_page_features_kernel(
    const scalar_t* __restrict__ page_data,  // [num_heads, head_size, block_size]
    float* __restrict__ features,            // [feature_dim]
    const int num_heads,
    const int head_size,
    const int block_size,
    const int feature_dim) {
    
    extern __shared__ float shared_mem[];
    float* local_sums = shared_mem;
    float* local_vars = &shared_mem[num_heads];
    
    const int tid = threadIdx.x;
    const int head_idx = tid % num_heads;
    const int total_threads = blockDim.x;
    
    // Initialize shared memory
    if (tid < num_heads * 2) {
        local_sums[tid] = 0.0f;
        if (tid >= num_heads) {
            local_vars[tid - num_heads] = 0.0f;
        }
    }
    __syncthreads();
    
    // Step 1: Compute mean per head
    float thread_sum = 0.0f;
    int elements_per_head = head_size * block_size;
    int elements_per_thread = (elements_per_head + total_threads - 1) / total_threads;
    
    for (int i = 0; i < elements_per_thread; i++) {
        int elem_idx = tid * elements_per_thread + i;
        if (elem_idx < elements_per_head && head_idx < num_heads) {
            int data_idx = head_idx * elements_per_head + elem_idx;
            thread_sum += static_cast<float>(page_data[data_idx]);
        }
    }
    
    // Reduce within warp
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    }
    
    // Store warp reduction results
    if (tid % WARP_SIZE == 0 && head_idx < num_heads) {
        atomicAdd(&local_sums[head_idx], thread_sum);
    }
    __syncthreads();
    
    // Compute means
    if (tid < num_heads) {
        local_sums[tid] /= elements_per_head;
        features[tid] = local_sums[tid];  // Store mean features
    }
    __syncthreads();
    
    // Step 2: Compute variance per head
    float thread_var = 0.0f;
    float head_mean = tid < num_heads ? local_sums[head_idx] : 0.0f;
    
    for (int i = 0; i < elements_per_thread; i++) {
        int elem_idx = tid * elements_per_thread + i;
        if (elem_idx < elements_per_head && head_idx < num_heads) {
            int data_idx = head_idx * elements_per_head + elem_idx;
            float diff = static_cast<float>(page_data[data_idx]) - head_mean;
            thread_var += diff * diff;
        }
    }
    
    // Reduce variance within warp
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        thread_var += __shfl_down_sync(0xffffffff, thread_var, offset);
    }
    
    if (tid % WARP_SIZE == 0 && head_idx < num_heads) {
        atomicAdd(&local_vars[head_idx], thread_var);
    }
    __syncthreads();
    
    // Store variance features
    if (tid < num_heads) {
        local_vars[tid] /= elements_per_head;
        features[num_heads + tid] = sqrtf(local_vars[tid]);  // Store std dev
    }
    
    // Step 3: Compute attention pattern signature (simplified)
    // Average absolute values across head dimension
    if (tid < block_size) {
        float pattern_sum = 0.0f;
        for (int h = 0; h < num_heads; h++) {
            for (int d = 0; d < head_size; d++) {
                int idx = h * head_size * block_size + d * block_size + tid;
                pattern_sum += fabsf(static_cast<float>(page_data[idx]));
            }
        }
        features[2 * num_heads + tid] = pattern_sum / (num_heads * head_size);
    }
}

// Kernel for computing similarity between two page feature vectors
__global__ void compute_similarity_kernel(
    const float* __restrict__ features1,
    const float* __restrict__ features2,
    float* __restrict__ similarity,
    const int feature_dim) {
    
    extern __shared__ float shared_mem[];
    float* dot_product = shared_mem;
    float* norm1_sq = &shared_mem[1];
    float* norm2_sq = &shared_mem[2];
    
    const int tid = threadIdx.x;
    
    // Initialize shared memory
    if (tid < 3) {
        shared_mem[tid] = 0.0f;
    }
    __syncthreads();
    
    // Compute dot product and norms
    float local_dot = 0.0f;
    float local_norm1 = 0.0f;
    float local_norm2 = 0.0f;
    
    for (int i = tid; i < feature_dim; i += blockDim.x) {
        float f1 = features1[i];
        float f2 = features2[i];
        local_dot += f1 * f2;
        local_norm1 += f1 * f1;
        local_norm2 += f2 * f2;
    }
    
    // Reduce within block
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        local_dot += __shfl_down_sync(0xffffffff, local_dot, offset);
        local_norm1 += __shfl_down_sync(0xffffffff, local_norm1, offset);
        local_norm2 += __shfl_down_sync(0xffffffff, local_norm2, offset);
    }
    
    if (tid == 0) {
        atomicAdd(dot_product, local_dot);
        atomicAdd(norm1_sq, local_norm1);
        atomicAdd(norm2_sq, local_norm2);
    }
    __syncthreads();
    
    // Compute cosine similarity
    if (tid == 0) {
        float norm_prod = sqrtf(*norm1_sq * *norm2_sq);
        *similarity = norm_prod > 0.0f ? *dot_product / norm_prod : 0.0f;
    }
}

// Kernel for delta compression between pages
template<typename scalar_t>
__global__ void compute_delta_compression_kernel(
    const scalar_t* __restrict__ target_page,
    const scalar_t* __restrict__ base_page,
    int8_t* __restrict__ quantized_delta,
    float* __restrict__ scale,
    const int num_elements,
    const int delta_bits) {
    
    extern __shared__ float shared_max[];
    
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int total_threads = blockDim.x * gridDim.x;
    const int thread_id = bid * blockDim.x + tid;
    
    // Step 1: Find maximum absolute delta
    float local_max = 0.0f;
    for (int i = thread_id; i < num_elements; i += total_threads) {
        float delta = static_cast<float>(target_page[i]) - static_cast<float>(base_page[i]);
        local_max = fmaxf(local_max, fabsf(delta));
    }
    
    // Block-level reduction
    shared_max[tid] = local_max;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s /= 2) {
        if (tid < s) {
            shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + s]);
        }
        __syncthreads();
    }
    
    // Global reduction (simplified - in practice, use atomic max)
    if (tid == 0 && bid == 0) {
        float global_max = shared_max[0];
        // Compute scale
        int max_int = (1 << (delta_bits - 1)) - 1;
        *scale = global_max / max_int;
    }
    __syncthreads();
    
    // Step 2: Quantize deltas
    float inv_scale = 1.0f / (*scale);
    for (int i = thread_id; i < num_elements; i += total_threads) {
        float delta = static_cast<float>(target_page[i]) - static_cast<float>(base_page[i]);
        int quantized = static_cast<int>(roundf(delta * inv_scale));
        quantized = max(-128, min(127, quantized));  // Clamp to int8
        quantized_delta[i] = static_cast<int8_t>(quantized);
    }
}

// Kernel for delta decompression
template<typename scalar_t>
__global__ void decompress_delta_kernel(
    const int8_t* __restrict__ quantized_delta,
    const scalar_t* __restrict__ base_page,
    scalar_t* __restrict__ decompressed_page,
    const float scale,
    const int num_elements) {
    
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int total_threads = blockDim.x * gridDim.x;
    
    for (int i = tid; i < num_elements; i += total_threads) {
        float delta = static_cast<float>(quantized_delta[i]) * scale;
        float base_val = static_cast<float>(base_page[i]);
        decompressed_page[i] = static_cast<scalar_t>(base_val + delta);
    }
}

// Kernel for adaptive compression decision based on similarity scores
__global__ void adaptive_compression_decision_kernel(
    const float* __restrict__ similarity_scores,  // [num_pages, top_k]
    const float* __restrict__ page_importance,    // [num_pages]
    int* __restrict__ compression_decisions,      // [num_pages]
    const int num_pages,
    const int top_k,
    const float similarity_threshold,
    const float importance_threshold) {
    
    const int page_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (page_idx < num_pages) {
        // Check if page has high similarity with any other page
        bool should_compress = false;
        float max_similarity = 0.0f;
        
        for (int k = 0; k < top_k; k++) {
            float sim = similarity_scores[page_idx * top_k + k];
            max_similarity = fmaxf(max_similarity, sim);
            if (sim >= similarity_threshold) {
                should_compress = true;
                break;
            }
        }
        
        // Consider importance - don't compress highly important pages
        if (page_importance[page_idx] > importance_threshold) {
            should_compress = false;
        }
        
        compression_decisions[page_idx] = should_compress ? 1 : 0;
    }
}

}  // namespace cpac
}  // namespace vllm

// C++ interface functions
torch::Tensor compute_page_features_cuda(
    torch::Tensor page_data,
    int feature_dim) {
    
    const int num_heads = page_data.size(0);
    const int head_size = page_data.size(1);
    const int block_size = page_data.size(2);
    
    auto features = torch::zeros({feature_dim}, page_data.options().dtype(torch::kFloat32));
    
    const int threads = 256;
    const int shared_mem_size = num_heads * 2 * sizeof(float);
    
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        page_data.scalar_type(), "compute_page_features", ([&] {
            vllm::cpac::compute_page_features_kernel<scalar_t><<<1, threads, shared_mem_size>>>(
                page_data.data_ptr<scalar_t>(),
                features.data_ptr<float>(),
                num_heads,
                head_size,
                block_size,
                feature_dim
            );
        }));
    
    return features;
}

torch::Tensor compute_similarity_cuda(
    torch::Tensor features1,
    torch::Tensor features2) {
    
    const int feature_dim = features1.size(0);
    auto similarity = torch::zeros({1}, features1.options());
    
    const int threads = 256;
    const int shared_mem_size = 3 * sizeof(float);
    
    vllm::cpac::compute_similarity_kernel<<<1, threads, shared_mem_size>>>(
        features1.data_ptr<float>(),
        features2.data_ptr<float>(),
        similarity.data_ptr<float>(),
        feature_dim
    );
    
    return similarity;
}

std::tuple<torch::Tensor, torch::Tensor> compress_delta_cuda(
    torch::Tensor target_page,
    torch::Tensor base_page,
    int delta_bits) {
    
    const int num_elements = target_page.numel();
    auto quantized_delta = torch::zeros_like(target_page, torch::kInt8);
    auto scale = torch::zeros({1}, target_page.options().dtype(torch::kFloat32));
    
    const int threads = 256;
    const int blocks = (num_elements + threads - 1) / threads;
    const int shared_mem_size = threads * sizeof(float);
    
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        target_page.scalar_type(), "compute_delta_compression", ([&] {
            vllm::cpac::compute_delta_compression_kernel<scalar_t><<<blocks, threads, shared_mem_size>>>(
                target_page.data_ptr<scalar_t>(),
                base_page.data_ptr<scalar_t>(),
                quantized_delta.data_ptr<int8_t>(),
                scale.data_ptr<float>(),
                num_elements,
                delta_bits
            );
        }));
    
    return std::make_tuple(quantized_delta, scale);
}

torch::Tensor decompress_delta_cuda(
    torch::Tensor quantized_delta,
    torch::Tensor base_page,
    float scale) {
    
    const int num_elements = quantized_delta.numel();
    auto decompressed = torch::empty_like(base_page);
    
    const int threads = 256;
    const int blocks = (num_elements + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        base_page.scalar_type(), "decompress_delta", ([&] {
            vllm::cpac::decompress_delta_kernel<scalar_t><<<blocks, threads>>>(
                quantized_delta.data_ptr<int8_t>(),
                base_page.data_ptr<scalar_t>(),
                decompressed.data_ptr<scalar_t>(),
                scale,
                num_elements
            );
        }));
    
    return decompressed;
}