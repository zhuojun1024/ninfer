#pragma once

// Builds the TP-2 split spec for the Qwen3.5 27B dense text component (head-parallel mixers).
// The attention and GDN mixers are sharded by head, the dense FFN is sharded, and the output head
// is vocabulary-split when its only consumer is the target logits:
//   - attention/GDN fused input projections: GatherRows; GDN causal conv: GatherCols.
//   - mixer output projections and FFN down: RowParallel; FFN gate/up: GatherRows.
//   - output_head and the reduced proposal/head: ColumnParallel (vocab rows split) when
//     TpSplitOptions asks for it.
//   - token_embedding: RowParallel (hidden columns split; both shards keep every vocabulary row).
//   - norms and GDN gating: Replicated.
//   - mtp/* and vision/*: Replicated but shard-local - the MTP layer is placed on shard 0 alone and
//     the Vision tower on shard 1 alone, because no execution path on the other shard reads them.
// Objects not classified are replicated to both shards.

#include "artifact/schema.h"
#include "core/tp/tp_materialize.h"
#include "models/qwen3_5/config.h"

namespace ninfer::models::qwen3_5::loading {

// Split choices that depend on how the loaded route consumes the artifact.
struct TpSplitOptions {
    // Vocabulary-split the text output head. Valid only when the head has a single consumer, the
    // target logits: selecting an unoptimized (Full) proposal head makes this object weight-tied to
    // the MTP proposal, which runs on shard 0 alone and needs every vocabulary row, so that route
    // keeps it replicated. The engine merges the two halves back into the full [V, T] logits
    // before sampling.
    bool split_output_head = false;
    // Vocabulary-split the optimized (reduced) proposal head. The table is identical on both
    // shards although only shard 0 samples a draft from it; splitting halves the resident copy and
    // the pair merges the draft logits back to the reduced vocabulary before the argmax.
    bool split_proposal_head = false;
};

[[nodiscard]] tp::TPSplitSpec build_tp_split_spec(const artifact::Directory& directory,
                                                  const TextConfig& config,
                                                  const TpSplitOptions& options = {});

} // namespace ninfer::models::qwen3_5::loading
