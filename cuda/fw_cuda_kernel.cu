#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

namespace {
// template <typename scalar_t>
// __device__ __forceinline__ scalar_t sigmoid(scalar_t z) {
//   return 1.0 / (1.0 + exp(-z));
// }

// template <typename scalar_t>
// __device__ __forceinline__ scalar_t d_sigmoid(scalar_t z) {
//   const auto s = sigmoid(z);
//   return (1.0 - s) * s;
// }

// template <typename scalar_t>
// __device__ __forceinline__ scalar_t d_tanh(scalar_t z) {
//   const auto t = tanh(z);
//   return 1 - (t * t);
// }

// template <typename scalar_t>
// __device__ __forceinline__ scalar_t elu(scalar_t z, scalar_t alpha = 1.0) {
//   // return fmaxf(0.0, z) + fminf(0.0, alpha * (exp(z) - 1.0));
//   return fmax((scalar_t) 0.0, z) + fmin((scalar_t) 0.0, alpha * (exp(z) - 1.0));
// }

// template <typename scalar_t>
// __device__ __forceinline__ scalar_t d_elu(scalar_t z, scalar_t alpha = 1.0) {
//   const auto e = exp(z);
//   const auto d_relu = z < 0.0 ? 0.0 : 1.0;
//   return d_relu + (((alpha * (e - 1.0)) < 0.0) ? (alpha * e) : 0.0);
// }

// template <typename scalar_t>
// __global__ void lltm_cuda_forward_kernel(
//     const torch::PackedTensorAccessor32<scalar_t,3,torch::RestrictPtrTraits> gates,
//     const torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> old_cell,
//     torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> new_h,
//     torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> new_cell,
//     torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> input_gate,
//     torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> output_gate,
//     torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> candidate_cell) {
//   //batch index
//   const int n = blockIdx.y;
//   // column index
//   const int c = blockIdx.x * blockDim.x + threadIdx.x;
//   if (c < gates.size(2)){
//     input_gate[n][c] = sigmoid(gates[n][0][c]);
//     output_gate[n][c] = sigmoid(gates[n][1][c]);
//     candidate_cell[n][c] = elu(gates[n][2][c]);
//     new_cell[n][c] =
//         old_cell[n][c] + candidate_cell[n][c] * input_gate[n][c];
//     new_h[n][c] = tanh(new_cell[n][c]) * output_gate[n][c];
//   }
// }

template <typename scalar_t>
__global__ void forward_warping_cuda_kernel(
        const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> obj,
        const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> safe_y,
        const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> safe_x,
        const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> depth,
        torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> output,
        torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> dlut,
        torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> dlut_count,
        double same_range) {
    // printf("hi from blockIdx = (%d, %d), threadIdx = (%d, %d)\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y);
    // auto dim_b = obj.size(0);
    // auto dim_c = obj.size(1);
    auto dim_h = obj.size(2);
    auto dim_w = obj.size(3);

    const int b = blockIdx.x;
    const int c = threadIdx.x;

    for (int j = 0; j < dim_h; j++) {
        for (int i = 0; i < dim_w; i++) {
            auto cur_depth = depth[b][0][j][i];
            auto x = safe_x[b][0][j][i];
            auto y = safe_y[b][0][j][i];

            auto x_f = floor(x);
            auto y_f = floor(y);
            auto x_c = x_f + 1;
            auto y_c = y_f + 1;

            if (x_f >= 0 && x_c < dim_w && y_f >= 0 && y_c < dim_h) {
                auto nw_depth = dlut[b][c][y_f][x_f] / dlut_count[b][c][y_f][x_f];
                auto ne_depth = dlut[b][c][y_f][x_c] / dlut_count[b][c][y_f][x_c];
                auto sw_depth = dlut[b][c][y_c][x_f] / dlut_count[b][c][y_c][x_f];
                auto se_depth = dlut[b][c][y_c][x_c] / dlut_count[b][c][y_c][x_c];

                if (nw_depth == 100.0 || nw_depth > cur_depth + same_range) {
                    output[b][c][y_f][x_f] = obj[b][c][j][i];
                    dlut[b][c][y_f][x_f] = cur_depth;
                    dlut_count[b][c][y_f][x_f] = 1;
                } else if (std::abs(nw_depth - cur_depth) <= same_range) {
                    output[b][c][y_f][x_f] += obj[b][c][j][i];
                    dlut[b][c][y_f][x_f] += cur_depth;
                    dlut_count[b][c][y_f][x_f] += 1;
                }

                if (ne_depth == 100.0 || ne_depth > cur_depth + same_range) {
                    output[b][c][y_f][x_c] = obj[b][c][j][i];
                    dlut[b][c][y_f][x_c] = cur_depth;
                    dlut_count[b][c][y_f][x_c] = 1;
                } else if (std::abs(ne_depth - cur_depth) <= same_range) {
                    output[b][c][y_f][x_c] += obj[b][c][j][i];
                    dlut[b][c][y_f][x_c] += cur_depth;
                    dlut_count[b][c][y_f][x_c] += 1;
                }

                if (sw_depth == 100.0 || sw_depth > cur_depth + same_range) {
                    output[b][c][y_c][x_f] = obj[b][c][j][i];
                    dlut[b][c][y_c][x_f] = cur_depth;
                    dlut_count[b][c][y_c][x_f] = 1;
                } else if (std::abs(sw_depth - cur_depth) <= same_range) {
                    output[b][c][y_c][x_f] += obj[b][c][j][i];
                    dlut[b][c][y_c][x_f] += cur_depth;
                    dlut_count[b][c][y_c][x_f] += 1;
                }

                if (se_depth == 100.0 || se_depth > cur_depth + same_range) {
                    output[b][c][y_c][x_c] = obj[b][c][j][i];
                    dlut[b][c][y_c][x_c] = cur_depth;
                    dlut_count[b][c][y_c][x_c] = 1;
                } else if (std::abs(se_depth - cur_depth) <= same_range) {
                    output[b][c][y_c][x_c] += obj[b][c][j][i];
                    dlut[b][c][y_c][x_c] += cur_depth;
                    dlut_count[b][c][y_c][x_c] += 1;
                }
            }

        }
    }

    for (int j = 0; j < dim_h; j++) {
        for (int i = 0; i < dim_w; i++) {
            if (dlut_count[b][c][j][i] == 0.) continue;

            output[b][c][j][i] = output[b][c][j][i] / dlut_count[b][c][j][i];
        }
    }

}
} // namespace

std::vector<torch::Tensor> forward_warping_cuda(
        torch::Tensor obj,
        torch::Tensor safe_y,
        torch::Tensor safe_x,
        torch::Tensor depth,
        double same_range) {
    auto output = torch::zeros_like(obj);
    auto dlut = torch::ones_like(obj) * 100.;
    auto dlut_count = torch::zeros_like(obj);

    auto dim_b = obj.size(0);
    auto dim_c = obj.size(1);
    auto dim_h = obj.size(2);
    auto dim_w = obj.size(3);

    const int threads = dim_c;
    const dim3 blocks(dim_b);

    AT_DISPATCH_FLOATING_TYPES(obj.scalar_type(), "forward_warping_cuda", ([&] {
        forward_warping_cuda_kernel<scalar_t><<<blocks, threads>>>(
            obj.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            safe_y.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            safe_x.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            depth.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            output.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            dlut.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            dlut_count.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            same_range);
    }));
    return {output};
}
