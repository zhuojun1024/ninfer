#pragma once

// TP-2 materialization: read each device weight object to host, split it per the spec, and
// upload each shard to its own GPU arena. Produces one MaterializedArtifact per GPU. Objects
// not listed in the spec are replicated (uploaded whole to both GPUs).

#include "artifact/materializer.h"
#include "artifact/schema.h"
#include "core/tp/weight_splitter.h"

#include <cstdint>
#include <memory>
#include <span>
#include <utility>
#include <vector>

namespace ninfer::tp {

struct TPObjectSplit {
    artifact::ObjectHandle object;
    WeightSplitKind kind;
    // Row blocks for GatherRows (fused parents); empty otherwise.
    std::vector<RowPart> parts;
};

struct TPSplitSpec {
    // Objects not present here are replicated to both GPUs.
    std::vector<TPObjectSplit> splits;

    [[nodiscard]] WeightSplitKind kind(artifact::ObjectHandle object) const {
        const auto* s = find(object);
        return s ? s->kind : WeightSplitKind::Replicated;
    }

    [[nodiscard]] const TPObjectSplit* find(artifact::ObjectHandle object) const {
        for (const auto& s : splits) {
            if (s.object == object) { return &s; }
        }
        return nullptr;
    }
};

// Materializes the plan onto two GPUs. device0 receives shard 0, device1 receives shard 1.
// The split spec maps splittable weight objects to ColumnParallel (row slice) or RowParallel
// (column slice); all other device objects are replicated.
std::pair<artifact::MaterializedArtifact, artifact::MaterializedArtifact> materialize_tp2(
    const artifact::Reader& reader, const artifact::MaterializationPlan& plan,
    DeviceContext& device0, DeviceContext& device1, const TPSplitSpec& spec,
    const StartupObserver* startup_observer = nullptr);

} // namespace ninfer::tp
