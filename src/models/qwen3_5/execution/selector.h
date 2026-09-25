#pragma once

#include "models/qwen3_5/execution/parameters.h"
#include "ninfer/ops/candidate_selector.h"

namespace ninfer::models::qwen3_5::execution {

// Dispatches the selector call on the artifact's stored codebook representation. Both callers (the
// shard that holds the selector and the TP-2 peer it was moved to) project their own hidden state
// first and hand the projected tensor in.
inline void run_candidate_selector(const SelectorParameters& selector, const Tensor& candidate_ids,
                                   const Tensor& unary_scores, const Tensor& projected_hidden,
                                   const Tensor& anchors, const Tensor& base_positions,
                                   const ops::SamplingConfig* configs, Tensor& drafts,
                                   Tensor& proposal_q, WorkspaceArena& workspace,
                                   cudaStream_t stream) {
    if (selector.predecessor_codebook.quantized) {
        ops::candidate_selector_path(candidate_ids, unary_scores, projected_hidden, anchors,
                                     selector.predecessor_codebook.weight,
                                     selector.successor_codebook.weight, base_positions, configs,
                                     drafts, proposal_q, workspace, stream);
    } else {
        ops::candidate_selector_path(candidate_ids, unary_scores, projected_hidden, anchors,
                                     selector.predecessor_codebook.dense,
                                     selector.successor_codebook.dense, base_positions, configs,
                                     drafts, proposal_q, workspace, stream);
    }
}

} // namespace ninfer::models::qwen3_5::execution
