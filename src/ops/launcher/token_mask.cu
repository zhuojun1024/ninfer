#include "ops/launcher/token_mask.h"

#include "core/device.h"
#include "ops/kernel/token_mask.cuh"

namespace ninfer::ops::detail {

void apply_token_mask_launch(Tensor& logits, const Tensor& mask, cudaStream_t stream) {
    const std::int64_t count = logits.numel();
    if (count == 0) { return; }
    const std::int64_t blocks = (count + kTokenMaskBlock - 1) / kTokenMaskBlock;
    apply_token_mask_kernel<<<static_cast<unsigned int>(blocks), kTokenMaskBlock, 0, stream>>>(
        static_cast<__nv_bfloat16*>(logits.data), static_cast<const std::uint8_t*>(mask.data),
        count);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
