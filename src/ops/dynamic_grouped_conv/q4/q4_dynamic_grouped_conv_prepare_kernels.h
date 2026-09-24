#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

// Applies the shared base kernel and the projected coefficients: reads the contiguous BF16
// [1280, tokens] coefficient matrix produced by the q4 projection, folds it into the normalized
// prepared tensor in place, and writes finish_delta. Mirrors the bf16 reduce with the projection
// already materialized in BF16.
void q4_dynamic_grouped_conv_prepare_reduce_launch(const Tensor& projected_coefficients,
                                                   const Tensor& base_kernel, Tensor& prepared,
                                                   Tensor& finish_delta, cudaStream_t stream);

} // namespace ninfer::ops::detail
