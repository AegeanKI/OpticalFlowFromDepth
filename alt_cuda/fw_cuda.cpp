#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <vector>

std::vector<torch::Tensor> forward_warping_cuda(
        torch::Tensor obj,
        torch::Tensor safe_y,
        torch::Tensor safe_x,
        torch::Tensor depth);

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> forward_warping(
        torch::Tensor obj,
        torch::Tensor safe_y,
        torch::Tensor safe_x,
        torch::Tensor depth) {
    CHECK_INPUT(obj);
    CHECK_INPUT(safe_y);
    CHECK_INPUT(safe_x);
    CHECK_INPUT(depth);
    const at::cuda::OptionalCUDAGuard device_guard(device_of(obj));
    return forward_warping_cuda(obj, safe_y, safe_x, depth);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_warping", &forward_warping, "Forward Warping Function (CUDA)");
}
