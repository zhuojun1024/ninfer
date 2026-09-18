#pragma once

// Builds the TP-2 split spec for the Qwen3.5 27B dense text component (FFN-only sharding).
// The attention and GDN mixers are replicated in full on both GPUs (their fused projection ops
// validate full-model geometry), while the dense FFN is sharded and the lm_head is split:
//   - gate/up: GatherRows (two equal row blocks, each halved) - column-parallel.
//   - FFN down: RowParallel (the intermediate input columns are split).
//   - lm_head: ColumnParallel (vocab rows split).
//   - token_embedding, norms, mixer projections, conv, gating, mixer output projections:
//     Replicated.
// Objects not classified are replicated.

#include "artifact/schema.h"
#include "core/tp/tp_materialize.h"
#include "models/qwen3_5/config.h"

namespace ninfer::models::qwen3_5::loading {

[[nodiscard]] tp::TPSplitSpec build_tp_split_spec(const artifact::Directory& directory,
                                                  const TextConfig& config);

} // namespace ninfer::models::qwen3_5::loading
