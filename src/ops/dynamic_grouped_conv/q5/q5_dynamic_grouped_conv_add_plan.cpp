#include "core/weight.h"
#include "ops/dynamic_grouped_conv/q5/q5_dynamic_grouped_conv_add_plan.h"
#include "ops/dynamic_grouped_conv/q5/q5_dynamic_grouped_conv_add_kernels.h"
#include "ops/linear/q5/q5_dispatch.h"
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr std::int32_t kHidden = 5120;

struct Plan {
    Q5Launch projection;
    std::size_t workspace_bytes;
};

Plan resolve_plan(std::int32_t input_rows, std::int32_t width, std::int32_t batch) {
    if (input_rows != 4096 && input_rows != 17408)
        throw std::invalid_argument("linear dynamic grouped conv add: C must be 4096 or 17408");
    if (width < 2 || width > 16 || batch < 1 || batch > 8)
        throw std::invalid_argument("linear dynamic grouped conv add: invalid W/B profile");
    const std::int32_t tokens = width * batch;
    return {select_q5_a16_launch(kHidden, input_rows, tokens),
            static_cast<std::size_t>(kHidden) * static_cast<std::size_t>(tokens) *
                sizeof(std::uint16_t)};
}

} // namespace

std::size_t q5_linear_dynamic_grouped_conv_add_workspace_capacity_bytes(
    std::int32_t input_rows, std::int32_t min_width, std::int32_t max_width, std::int32_t min_batch,
    std::int32_t max_batch) {
    if (min_width < 2 || max_width > 16 || min_width > max_width || min_batch < 1 ||
        max_batch > 8 || min_batch > max_batch)
        throw std::invalid_argument(
            "linear dynamic grouped conv add workspace: invalid W/B interval");
    // Every route shares one projected BF16 matrix; its capacity is monotonic in W and B.
    return resolve_plan(input_rows, max_width, max_batch).workspace_bytes;
}

const char* q5_linear_dynamic_grouped_conv_add_route_name(std::int32_t input_rows,
                                                          std::int32_t width,
                                                          std::int32_t batch_size) {
    return q5_projection_route_label(resolve_plan(input_rows, width, batch_size).projection);
}

void q5_linear_dynamic_grouped_conv_add_dispatch(const Tensor& x, const Weight& weight,
                                                 const Tensor& base, const Tensor& delta,
                                                 Tensor& residual, WorkspaceArena& workspace,
                                                 cudaStream_t stream) {
    const Plan plan          = resolve_plan(x.ne[0], x.ne[1], x.ne[2]);
    auto scope               = workspace.scope();
    const DeviceSpan storage = workspace.alloc_bytes(plan.workspace_bytes);
    Tensor projected(storage.data, DType::BF16, {kHidden, x.ne[1], x.ne[2]});
    q5_dynamic_grouped_conv_add_materialized_launch(plan.projection, x, weight, base, delta,
                                                    residual, projected, stream);
}
} // namespace ninfer::ops::detail
