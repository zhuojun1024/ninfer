#pragma once

#include "core/weight.h"
#include "core/tensor.h"
#include "ops/linear/q5/q5_launch.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void q5_dynamic_grouped_conv_add_materialized_launch(Q5Launch projection, const Tensor& x,
                                                     const Weight& weight, const Tensor& base_kernel,
                                                     const Tensor& finish_delta, Tensor& residual,
                                                     Tensor& projected, cudaStream_t stream);

// The stable route label of a resolved Q5 projection launch
// ("dynamic_grouped_conv_add.q5.*.materialized_bf16").
[[nodiscard]] const char* q5_projection_route_label(Q5Launch projection);

} // namespace ninfer::ops::detail
