#pragma once

// TP-2 shard view transform: given the model's pending weight references (logical parameters
// with their object parts) and the split spec, produce per-shard bound weights whose views
// reference the shard's half-parents. Each split kind yields a complete half-parent, so a shard
// view keeps the parent's layout with the split-dimension offsets halved. Replicated objects
// keep their full view. This avoids a full single-GPU materialization (which would not fit).

#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "artifact/schema.h"
#include "core/tp/tp_materialize.h"
#include "models/qwen3_5/load/bindings.h"
#include "models/qwen3_5/weights.h"

#include <span>
#include <vector>

namespace ninfer::models::qwen3_5::loading {

// Transforms the pending weights into shard bound weights for shard (0 or 1). The shard backing
// supplies the half-parents; each part is re-pointed at the shard parent and halved in the split
// dimension per the spec. Objects not in the spec are replicated (view unchanged, parent
// re-pointed). Uses must already be populated on the pending weights.
std::vector<BoundWeight> shard_views(std::span<const PendingWeight> pending,
                                     const artifact::Directory& directory,
                                     const artifact::MaterializedArtifact& shard_backing,
                                     const tp::TPSplitSpec& spec, int shard);

} // namespace ninfer::models::qwen3_5::loading
