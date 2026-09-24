#include "ops/dynamic_grouped_conv/q4/q4_dynamic_grouped_conv_prepare_kernels.h"
#include "core/device.h"
#include <cuda_bf16.h>

namespace ninfer::ops::detail {
namespace {
constexpr int kCoefficientRows = 1280;

template <int Capacity>
__global__ __launch_bounds__(Capacity * 16, 4) void q4_dynamic_grouped_conv_prepare_reduce_kernel(
    const __nv_bfloat16* base, const __nv_bfloat16* projected, __nv_bfloat16* prepared,
    __nv_bfloat16* finish, int width, int batch_size) {
    __shared__ float coefficients[4][Capacity];
    __shared__ float normalized[Capacity][16];
    const int tid = threadIdx.x, group = blockIdx.x, batch = blockIdx.y;
    if (tid < 4 * Capacity && tid % Capacity < width) {
        const int coefficient = tid / Capacity, position = tid % Capacity,
                  row = coefficient * 320 + group;
        coefficients[coefficient][position] = __bfloat162float(
            projected[(static_cast<std::int64_t>(batch * width + position)) * kCoefficientRows +
                      row]);
    }
    const int position = tid / 16, channel = tid % 16, hidden = group * 16 + channel;
    const std::int64_t offset = static_cast<std::int64_t>(batch * width + position) * 5120 + hidden;
    if (position < width) { normalized[position][channel] = __bfloat162float(prepared[offset]); }
    __syncthreads();
    if (tid < 2 * Capacity && tid % Capacity < width) {
        const int tap = tid / Capacity, pos = tid % Capacity;
        finish[((batch * width + pos) * 2 + tap) * 320 + group] =
            __float2bfloat16_rn(coefficients[2 + tap][pos]);
    }
    if (position < width) {
        float value = (__bfloat162float(base[hidden]) + coefficients[0][position]) *
                      normalized[position][channel];
        if (position > 0)
            value = fmaf(__bfloat162float(base[5120 + hidden]) + coefficients[1][position],
                         normalized[position - 1][channel], value);
        prepared[offset] = __float2bfloat16_rn(value);
    }
}

template <int Capacity>
void launch(const Tensor& projected, const Tensor& base, Tensor& prepared, Tensor& finish,
            cudaStream_t stream) {
    const dim3 grid(320, prepared.ne[2]);
    q4_dynamic_grouped_conv_prepare_reduce_kernel<Capacity><<<grid, Capacity * 16, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(base.data),
        static_cast<const __nv_bfloat16*>(projected.data),
        static_cast<__nv_bfloat16*>(prepared.data), static_cast<__nv_bfloat16*>(finish.data),
        prepared.ne[1], prepared.ne[2]);
    CUDA_CHECK(cudaGetLastError());
}
} // namespace

void q4_dynamic_grouped_conv_prepare_reduce_launch(const Tensor& projected, const Tensor& base,
                                                   Tensor& prepared, Tensor& finish,
                                                   cudaStream_t stream) {
    // Capacity determines the CTA layout; width determines every public tensor address.
    if (prepared.ne[1] <= 8)
        launch<8>(projected, base, prepared, finish, stream);
    else
        launch<16>(projected, base, prepared, finish, stream);
}
} // namespace ninfer::ops::detail
