#include "core/weight.h"
#include "ops/dynamic_grouped_conv/q4/q4_dynamic_grouped_conv_add_kernels.h"
#include "ops/dynamic_grouped_conv/dynamic_conv_finish.h"
#include "ops/linear/q4/q4_ksplit_launch.cuh"

#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

void q4_dynamic_grouped_conv_add_materialized(Q4Launch projection, const Tensor& x,
                                              const Weight& weight, const Tensor& base,
                                              const Tensor& delta, Tensor& residual,
                                              Tensor& projected, cudaStream_t stream) {
    const int tokens  = x.ne[1] * x.ne[2];
    const Tensor flat = x.view({x.ne[0], tokens});
    Tensor result     = projected.view({weight.n, tokens});
    projection(flat, weight, result, stream);
    dynamic_conv_finish_launch(projected, base, delta, residual, stream);
}

} // namespace

void q4_dynamic_grouped_conv_add_materialized_launch(Q4Launch projection, const Tensor& x,
                                                     const Weight& weight, const Tensor& base,
                                                     const Tensor& delta, Tensor& residual,
                                                     Tensor& projected, cudaStream_t stream) {
    q4_dynamic_grouped_conv_add_materialized(projection, x, weight, base, delta, residual,
                                             projected, stream);
}

const char* q4_projection_route_label(Q4Launch projection) {
    struct NamedLaunch {
        Q4Launch launch;
        const char* label;
    };
    // One label per projection family: both C shapes share their T-bucket kernel, so the label
    // describes the route and the problem's C and T columns disambiguate the instance.
    static constexpr NamedLaunch kRoutes[]{
        {&launch_q4_simt_r8_c4, "dynamic_grouped_conv_add.q4.simt_r8_c4.materialized_bf16"},
        {&launch_q4_ksplit<5120, 4096, 8>,
         "dynamic_grouped_conv_add.q4.ksplit_exact.materialized_bf16"},
        {&launch_q4_ksplit<5120, 17408, 8>,
         "dynamic_grouped_conv_add.q4.ksplit_exact.materialized_bf16"},
        {&launch_q4_simt_r8_c8, "dynamic_grouped_conv_add.q4.simt_r8_c8.materialized_bf16"},
        {&launch_q4_mma_r64_c128, "dynamic_grouped_conv_add.q4.mma_r64_c128.materialized_bf16"},
    };
    for (const NamedLaunch& route : kRoutes) {
        if (route.launch == projection) { return route.label; }
    }
    throw std::logic_error("linear dynamic grouped conv add: unregistered Q4 projection route");
}

} // namespace ninfer::ops::detail
