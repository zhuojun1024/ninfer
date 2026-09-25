#include "ops/candidate_selector/plan.h"
#include "ops/candidate_selector/bf16/candidate_selector_path_kernels.h"
#include "ops/candidate_selector/q4/candidate_selector_path_q4_kernels.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

// Rejects a workspace that aliases any live operand of a selector call. The codebook payloads are
// checked separately because a quantized codebook carries two planes.
void require_workspace_disjoint(const SelectorWorkspace& scratch, const Tensor* const* live,
                                std::size_t live_count, const SamplingConfig* configs,
                                std::int32_t batch) {
    if (!scratch.edges.data) return;
    const auto begin = reinterpret_cast<std::uintptr_t>(scratch.edges.data),
               end   = begin + scratch.edges.bytes();
    for (std::size_t i = 0; i < live_count; ++i) {
        const auto tb = reinterpret_cast<std::uintptr_t>(live[i]->data);
        if (begin < tb + live[i]->bytes() && tb < end)
            throw std::invalid_argument("selector workspace overlaps operand");
    }
    const auto cb = reinterpret_cast<std::uintptr_t>(configs);
    if (begin < cb + batch * sizeof(SamplingConfig) && cb < end)
        throw std::invalid_argument("selector workspace overlaps configs");
}

void require_workspace_disjoint_codebook(const SelectorWorkspace& scratch, const Weight& codebook,
                                         const char* label) {
    if (!scratch.edges.data || codebook.payload == nullptr || codebook.payload_bytes == 0) return;
    const auto begin = reinterpret_cast<std::uintptr_t>(scratch.edges.data),
               end   = begin + scratch.edges.bytes();
    const auto wb = reinterpret_cast<std::uintptr_t>(codebook.payload);
    if (begin < wb + codebook.payload_bytes && wb < end)
        throw std::invalid_argument(std::string("selector workspace overlaps ") + label);
}

} // namespace

SelectorRoute candidate_selector_path_route(int steps, int batch) {
    if (steps < 1 || steps > 15 || batch < 1 || batch > 8)
        throw std::invalid_argument("invalid selector K/B");
    return steps <= 4 ? SelectorRoute::Direct : SelectorRoute::Lattice;
}

const char* selector_route_name(SelectorRoute route) {
    return route == SelectorRoute::Direct ? "direct.t512" : "lattice.t512";
}

const char* candidate_selector_path_route_name(int steps, int batch) {
    return selector_route_name(candidate_selector_path_route(steps, batch));
}

void candidate_selector_path_dispatch(const Tensor& candidate_ids, const Tensor& unary_scores,
                                      const Tensor& projected_hidden, const Tensor& anchors,
                                      const Tensor& predecessor_codebook,
                                      const Tensor& successor_codebook,
                                      const Tensor& base_positions, const SamplingConfig* configs,
                                      Tensor& drafts, Tensor& proposal_q, WorkspaceArena& workspace,
                                      cudaStream_t stream) {
    const auto route = candidate_selector_path_route(candidate_ids.ne[1], candidate_ids.ne[2]);
    auto scope       = workspace.scope();
    const auto scratch =
        allocate_selector_workspace(workspace, route, candidate_ids.ne[1], candidate_ids.ne[2]);
    const Tensor* live[]{&candidate_ids,      &unary_scores,   &projected_hidden, &anchors,
                         &predecessor_codebook, &successor_codebook, &base_positions, &drafts,
                         &proposal_q};
    require_workspace_disjoint(scratch, live, 9, configs, candidate_ids.ne[2]);
    candidate_selector_path_launch(route, candidate_ids, unary_scores, projected_hidden, anchors,
                                   predecessor_codebook, successor_codebook, base_positions,
                                   configs, drafts, proposal_q, scratch, stream);
}

void candidate_selector_path_q4_dispatch(const Tensor& candidate_ids, const Tensor& unary_scores,
                                         const Tensor& projected_hidden, const Tensor& anchors,
                                         const Weight& predecessor_codebook,
                                         const Weight& successor_codebook,
                                         const Tensor& base_positions, const SamplingConfig* configs,
                                         Tensor& drafts, Tensor& proposal_q,
                                         WorkspaceArena& workspace, cudaStream_t stream) {
    const auto route = candidate_selector_path_route(candidate_ids.ne[1], candidate_ids.ne[2]);
    auto scope       = workspace.scope();
    const auto scratch =
        allocate_selector_workspace(workspace, route, candidate_ids.ne[1], candidate_ids.ne[2]);
    const Tensor* live[]{&candidate_ids, &unary_scores,   &projected_hidden, &anchors,
                         &base_positions, &drafts,        &proposal_q};
    require_workspace_disjoint(scratch, live, 7, configs, candidate_ids.ne[2]);
    require_workspace_disjoint_codebook(scratch, predecessor_codebook, "predecessor_codebook");
    require_workspace_disjoint_codebook(scratch, successor_codebook, "successor_codebook");
    candidate_selector_path_q4_launch(route, candidate_ids, unary_scores, projected_hidden, anchors,
                                      predecessor_codebook, successor_codebook, base_positions,
                                      configs, drafts, proposal_q, scratch, stream);
}
} // namespace ninfer::ops::detail
