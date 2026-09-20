#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void apply_token_mask_launch(Tensor& logits, const Tensor& mask, cudaStream_t stream);

} // namespace ninfer::ops::detail
