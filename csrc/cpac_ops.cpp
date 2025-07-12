#include <torch/extension.h>
#include <vector>

// Forward declarations of CUDA functions
torch::Tensor compute_page_features_cuda(torch::Tensor page_data, int feature_dim);
torch::Tensor compute_similarity_cuda(torch::Tensor features1, torch::Tensor features2);
std::tuple<torch::Tensor, torch::Tensor> compress_delta_cuda(
    torch::Tensor target_page, torch::Tensor base_page, int delta_bits);
torch::Tensor decompress_delta_cuda(
    torch::Tensor quantized_delta, torch::Tensor base_page, float scale);

// Wrapper functions with error checking
torch::Tensor cpac_compute_page_features(torch::Tensor page_data, int feature_dim) {
    TORCH_CHECK(page_data.is_cuda(), "page_data must be a CUDA tensor");
    TORCH_CHECK(page_data.dim() == 3, "page_data must be 3-dimensional");
    TORCH_CHECK(feature_dim > 0, "feature_dim must be positive");
    
    return compute_page_features_cuda(page_data, feature_dim);
}

torch::Tensor cpac_compute_similarity(torch::Tensor features1, torch::Tensor features2) {
    TORCH_CHECK(features1.is_cuda(), "features1 must be a CUDA tensor");
    TORCH_CHECK(features2.is_cuda(), "features2 must be a CUDA tensor");
    TORCH_CHECK(features1.size(0) == features2.size(0), 
                "features1 and features2 must have the same size");
    
    return compute_similarity_cuda(features1, features2);
}

std::tuple<torch::Tensor, torch::Tensor> cpac_compress_delta(
    torch::Tensor target_page, torch::Tensor base_page, int delta_bits) {
    TORCH_CHECK(target_page.is_cuda(), "target_page must be a CUDA tensor");
    TORCH_CHECK(base_page.is_cuda(), "base_page must be a CUDA tensor");
    TORCH_CHECK(target_page.sizes() == base_page.sizes(), 
                "target_page and base_page must have the same shape");
    TORCH_CHECK(delta_bits > 0 && delta_bits <= 16, 
                "delta_bits must be between 1 and 16");
    
    return compress_delta_cuda(target_page, base_page, delta_bits);
}

torch::Tensor cpac_decompress_delta(
    torch::Tensor quantized_delta, torch::Tensor base_page, float scale) {
    TORCH_CHECK(quantized_delta.is_cuda(), "quantized_delta must be a CUDA tensor");
    TORCH_CHECK(base_page.is_cuda(), "base_page must be a CUDA tensor");
    TORCH_CHECK(quantized_delta.dtype() == torch::kInt8, 
                "quantized_delta must be int8");
    TORCH_CHECK(scale > 0, "scale must be positive");
    
    return decompress_delta_cuda(quantized_delta, base_page, scale);
}

// Registration function to be called from torch_bindings.cpp
void register_cpac_ops(py::module& m) {
    // Create submodule for CPAC operations
    auto cpac_module = m.def_submodule("cpac_ops", "CPAC compression operations");
    
    cpac_module.def(
        "compute_page_features",
        &cpac_compute_page_features,
        "Compute page features for similarity matching",
        py::arg("page_data"),
        py::arg("feature_dim") = 128
    );
    
    cpac_module.def(
        "compute_similarity",
        &cpac_compute_similarity,
        "Compute cosine similarity between feature vectors",
        py::arg("features1"),
        py::arg("features2")
    );
    
    cpac_module.def(
        "compress_delta",
        &cpac_compress_delta,
        "Compress page using delta encoding",
        py::arg("target_page"),
        py::arg("base_page"),
        py::arg("delta_bits") = 8
    );
    
    cpac_module.def(
        "decompress_delta",
        &cpac_decompress_delta,
        "Decompress page from delta encoding",
        py::arg("quantized_delta"),
        py::arg("base_page"),
        py::arg("scale")
    );
}