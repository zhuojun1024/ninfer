#pragma once

#include "core/weight.h"
#include "core/tensor.h"
#include "ops/linear/q4/q4_launch.h"

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void q4_dynamic_grouped_conv_add_materialized_launch(Q4Launch projection, const Tensor& x,
                                                     const Weight& weight,
                                                     const Tensor& base_kernel,
                                                     const Tensor& finish_delta,
                                                     Tensor& residual, Tensor& projected,
                                                     cudaStream_t stream);

// The stable route label of a resolved Q4 projection launch
// ("dynamic_grouped_conv_add.q4.*.materialized_bf16").
[[nodiscard]] const char* q4_projection_route_label(Q4Launch projection);

} // namespace ninfer::ops::detail
