#pragma once

// Implements: include/ninfer/ops/token_mask.h
// Match: contiguous BF16 [rows,columns] logits against a same-shaped contiguous U8 mask, indexed
//        with element (v,c) at v + c*rows.
// Algorithm assumptions: one element per thread; a zero mask element overwrites the logit with
//        BF16 negative infinity.

#include <cuda_bf16.h>
#include <cstdint>
#include <math_constants.h>

namespace ninfer::ops {

inline constexpr int kTokenMaskBlock = 256;

__global__ void apply_token_mask_kernel(__nv_bfloat16* logits, const std::uint8_t* mask,
                                        std::int64_t count) {
    const std::int64_t index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) { return; }
    if (mask[index] == 0) { logits[index] = __float2bfloat16(-CUDART_INF_F); }
}

} // namespace ninfer::ops
