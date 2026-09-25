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

// Bit per shard: bit 0 = shard 0 (device0), bit 1 = shard 1 (device1). Objects that only one
// shard consumes are placed on that shard alone instead of being duplicated: the MTP layer runs on
// shard 0, the Vision tower on shard 1, and neither is read by the other shard.
inline constexpr std::uint8_t kBothShards = 0x3;
// Named single-shard placements. kLocalShard holds the components that run beside the target on the
// shard that owns the round (MTP and the DFlash2 masked draft); kPeerShard holds the components only
// the peer runs (the Vision tower and the DFlash2 selector).
inline constexpr std::uint8_t kLocalShard = 0x1;
inline constexpr std::uint8_t kPeerShard  = 0x2;

struct TPObjectSplit {
    artifact::ObjectHandle object;
    WeightSplitKind kind;
    // Row blocks for GatherRows (fused parents); empty otherwise.
    std::vector<RowPart> parts;
    // Shards that hold this object's bytes. A shard outside the mask gets no allocation, no upload
    // and an empty device view, so its resident set never pays for the object.
    std::uint8_t shards = kBothShards;

    [[nodiscard]] bool on_shard(int shard) const noexcept {
        return shard >= 0 && shard < 2 && ((shards >> shard) & 1U) != 0;
    }
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

    // Objects without an entry are replicated to both shards.
    [[nodiscard]] bool on_shard(artifact::ObjectHandle object, int shard) const {
        const auto* s = find(object);
        return s == nullptr ? true : s->on_shard(shard);
    }
};

// Materializes the plan onto two GPUs. device0 receives shard 0, device1 receives shard 1.
// The split spec maps splittable weight objects to ColumnParallel (row slice) or RowParallel
// (column slice); all other device objects are replicated. An object whose spec entry lists a
// single shard is placed on that shard only, so each shard arena is sized for exactly what it
// holds.
std::pair<artifact::MaterializedArtifact, artifact::MaterializedArtifact> materialize_tp2(
    const artifact::Reader& reader, const artifact::MaterializationPlan& plan,
    DeviceContext& device0, DeviceContext& device1, const TPSplitSpec& spec,
    const StartupObserver* startup_observer = nullptr);

} // namespace ninfer::tp
