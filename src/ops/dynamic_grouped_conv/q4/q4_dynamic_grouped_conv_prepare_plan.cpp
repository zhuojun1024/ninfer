#include "core/weight.h"
#include "ops/dynamic_grouped_conv/q4/q4_dynamic_grouped_conv_prepare_plan.h"
#include "ops/dynamic_grouped_conv/q4/q4_dynamic_grouped_conv_prepare_kernels.h"
#include "ops/dynamic_grouped_conv/bf16/bf16_dynamic_grouped_conv_prepare_plan.h"
#include "ops/linear/q4/q4_dispatch.h"
#include "ninfer/ops/rmsnorm.h"
#include <algorithm>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr std::int32_t kHidden          = 5120;
constexpr std::int32_t kCoefficientRows = 1280;

std::size_t capacity(std::int32_t tokens) {
    return static_cast<std::size_t>(kCoefficientRows) * tokens * sizeof(std::uint16_t);
}

// The public capacity query carries no qtype, so it always reserves the larger of the two
// codecs for the requested interval. The q4 route reserves the same amount, which keeps
// peak_used equal to the query for any exact shape.
std::size_t reserved_bytes(std::int32_t width, std::int32_t batch) {
    return std::max(
        bf16_dynamic_grouped_conv_prepare_workspace_capacity_bytes(width, width, batch, batch),
        capacity(width * batch));
}

void execute(const Tensor& residual, const Tensor& norm, float eps, const Tensor& base,
             const Weight& weight, Tensor& prepared, Tensor& finish, WorkspaceArena& workspace,
             cudaStream_t stream) {
    const std::int32_t tokens = residual.ne[1] * residual.ne[2];
    auto scope                = workspace.scope();
    const DeviceSpan storage  = workspace.alloc_bytes(reserved_bytes(residual.ne[1], residual.ne[2]));
    Tensor projected(storage.data, DType::BF16, {kCoefficientRows, tokens});
    // The projection reads the rmsnorm result and writes only the workspace coefficient matrix,
    // so the reduce below may still read prepared in place before it overwrites it.
    rmsnorm(residual, norm, eps, false, prepared, stream);
    select_q4_a16_launch(kCoefficientRows, kHidden, tokens)(prepared.view({kHidden, tokens}),
                                                            weight, projected, stream);
    q4_dynamic_grouped_conv_prepare_reduce_launch(projected, base, prepared, finish, stream);
}
} // namespace

std::size_t q4_dynamic_grouped_conv_prepare_workspace_capacity_bytes(std::int32_t min_width,
                                                                     std::int32_t max_width,
                                                                     std::int32_t min_batch,
                                                                     std::int32_t max_batch) {
    if (min_width < 2 || max_width > 16 || min_width > max_width || min_batch < 1 ||
        max_batch > 8 || min_batch > max_batch)
        throw std::invalid_argument("dynamic grouped conv prepare workspace: invalid W/B interval");
    std::size_t maximum = 0;
    for (std::int32_t w = min_width; w <= max_width; ++w)
        for (std::int32_t b = min_batch; b <= max_batch; ++b)
            maximum = std::max(maximum, capacity(w * b));
    return maximum;
}

void q4_dynamic_grouped_conv_prepare_dispatch(const Tensor& residual, const Tensor& norm,
                                              float eps, const Tensor& base, const Weight& weight,
                                              Tensor& prepared, Tensor& finish,
                                              WorkspaceArena& workspace, cudaStream_t stream) {
    execute(residual, norm, eps, base, weight, prepared, finish, workspace, stream);
}
} // namespace ninfer::ops::detail
