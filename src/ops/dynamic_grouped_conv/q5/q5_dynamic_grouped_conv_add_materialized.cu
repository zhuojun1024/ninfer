#include "core/weight.h"
#include "ops/dynamic_grouped_conv/q5/q5_dynamic_grouped_conv_add_kernels.h"
#include "ops/dynamic_grouped_conv/dynamic_conv_finish.h"
#include "ops/linear/q5/q5_ksplit_launch.cuh"

#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

void q5_dynamic_grouped_conv_add_materialized(Q5Launch projection, const Tensor& x,
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

void q5_dynamic_grouped_conv_add_materialized_launch(Q5Launch projection, const Tensor& x,
                                                     const Weight& weight, const Tensor& base,
                                                     const Tensor& delta, Tensor& residual,
                                                     Tensor& projected, cudaStream_t stream) {
    q5_dynamic_grouped_conv_add_materialized(projection, x, weight, base, delta, residual, projected,
                                             stream);
}

const char* q5_projection_route_label(Q5Launch projection) {
    struct NamedLaunch {
        Q5Launch launch;
        const char* label;
    };
    // One label per projection family: exact column counts keep their measured launch footprint
    // but describe the same route (the problem's C and T columns disambiguate the instance).
    static constexpr NamedLaunch kRoutes[]{
        {&launch_q5_simt_r8_c4, "dynamic_grouped_conv_add.q5.simt_r8_c4.materialized_bf16"},
        {&launch_q5_ksplit<4096, 2, 2>,
         "dynamic_grouped_conv_add.q5.simt_split2_exact.materialized_bf16"},
        {&launch_q5_ksplit<4096, 3, 2>,
         "dynamic_grouped_conv_add.q5.simt_split2_exact.materialized_bf16"},
        {&launch_q5_ksplit<4096, 4, 2>,
         "dynamic_grouped_conv_add.q5.simt_split2_exact.materialized_bf16"},
        {&launch_q5_ksplit<4096, 5, 2>,
         "dynamic_grouped_conv_add.q5.simt_split2_exact.materialized_bf16"},
        {&launch_q5_ksplit<4096, 6, 2>,
         "dynamic_grouped_conv_add.q5.simt_split2_exact.materialized_bf16"},
        {&launch_q5_ksplit<17408, 2, 2>,
         "dynamic_grouped_conv_add.q5.simt_split2_exact.materialized_bf16"},
        {&launch_q5_ksplit<17408, 3, 2>,
         "dynamic_grouped_conv_add.q5.simt_split2_exact.materialized_bf16"},
        {&launch_q5_ksplit<17408, 4, 2>,
         "dynamic_grouped_conv_add.q5.simt_split2_exact.materialized_bf16"},
        {&launch_q5_ksplit<17408, 5, 2>,
         "dynamic_grouped_conv_add.q5.simt_split2_exact.materialized_bf16"},
        {&launch_q5_ksplit<17408, 6, 2>,
         "dynamic_grouped_conv_add.q5.simt_split2_exact.materialized_bf16"},
        {&launch_q5_simt_r8_c8, "dynamic_grouped_conv_add.q5.simt_r8_c8.materialized_bf16"},
        {&launch_q5_mma_r64_c128, "dynamic_grouped_conv_add.q5.mma_r64_c128.materialized_bf16"},
    };
    for (const NamedLaunch& route : kRoutes) {
        if (route.launch == projection) { return route.label; }
    }
    throw std::logic_error("linear dynamic grouped conv add: unregistered Q5 projection route");
}

} // namespace ninfer::ops::detail
