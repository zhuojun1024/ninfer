#pragma once

#include "models/qwen3_5/execution/parameters.h"

namespace ninfer::models::qwen3_5::execution {

[[nodiscard]] std::size_t ffn_workspace_bytes(const FfnParameters& parameters, std::int32_t first,
                                              std::int32_t last, bool mtp = false);
void ffn(const Tensor& hidden, const FfnParameters& parameters, Tensor& residual,
         const ops::SparseMoeHints& hints, WorkspaceArena& workspace, cudaStream_t stream,
         bool mtp = false);
// Tensor-parallel FFN: writes the row-parallel down projection output (the FFN delta) into
// delta WITHOUT adding it to the residual. The caller all-reduces the delta across shards and
// adds the summed delta to the shared residual once. The hidden input must be identical on every
// shard (the replicated mixer guarantees this).
void ffn_delta(const Tensor& hidden, const FfnParameters& parameters, Tensor& delta,
               const ops::SparseMoeHints& hints, WorkspaceArena& workspace, cudaStream_t stream);

} // namespace ninfer::models::qwen3_5::execution
