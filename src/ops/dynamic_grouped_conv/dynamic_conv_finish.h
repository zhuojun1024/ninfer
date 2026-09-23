#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

// The dynamic-grouped-convolution finish shared by every materialized projection codec: folds the
// projected z[h,i,b] through the two-tap per-group coefficient state into residual in place.
// projected is the contiguous BF16 [5120,W,B] materialization of z; residual is contiguous BF16
// [5120,W,B] and supplies the token width W and the previous-tap column boundary rule.
void dynamic_conv_finish_launch(const Tensor& projected, const Tensor& base_kernel,
                                const Tensor& finish_delta, Tensor& residual, cudaStream_t stream);

} // namespace ninfer::ops::detail
