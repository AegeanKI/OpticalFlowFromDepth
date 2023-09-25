#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

namespace {
template <typename scalar_t>
__global__ void forward_warping_cuda_kernel(
        const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> obj,
        const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> safe_y,
        const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> safe_x,
        const torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> depth,
        torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> output,
        torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> dlut,
        torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> valid,
        torch::PackedTensorAccessor32<scalar_t, 4, torch::RestrictPtrTraits> collision) {
    // printf("hi from blockIdx = (%d, %d), threadIdx = (%d, %d)\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y);
    // auto dim_b = obj.size(0);
    // auto dim_c = obj.size(1);
    auto dim_h = obj.size(2);
    auto dim_w = obj.size(3);

    const int b = blockIdx.x;
    const int c = threadIdx.x;

    for (int j = 0; j < dim_h; j++) {
        for (int i = 0; i < dim_w; i++) {
            // auto cur_depth = depth[b][0][j][i];
            auto x = safe_x[b][0][j][i];
            auto y = safe_y[b][0][j][i];

            if (depth[b][0][j][i] < dlut[b][c][y][x]) {
                output[b][c][y][x] = obj[b][c][j][i];
                dlut[b][c][y][x] = depth[b][0][j][i];
            }
            if (c == 0) {
                valid[b][0][y][x] = 1;
                if (dlut[b][c][y][x] != 1000.) {
                    collision[b][0][y][x] = 0;
                } else {
                    collision[b][0][y][x] = 1;
                }
            }
        }
    }

}
} // namespace

std::vector<torch::Tensor> forward_warping_cuda(
        torch::Tensor obj,
        torch::Tensor safe_y,
        torch::Tensor safe_x,
        torch::Tensor depth) {
    auto output = torch::zeros_like(obj);
    auto dlut = torch::ones_like(obj) * 1000.;
    auto valid = torch::zeros_like(depth);
    auto collision = torch::zeros_like(depth);

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
            valid.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>(),
            collision.packed_accessor32<scalar_t, 4, torch::RestrictPtrTraits>());
    }));

    return {output, valid, collision};
}
