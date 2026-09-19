#include "models/qwen3_5/load/tp_shard_views.h"

#include "core/tp/weight_splitter.h"

#include <stdexcept>

namespace ninfer::models::qwen3_5::loading {
namespace {

using artifact::ObjectHandle;
using tp::RowPart;
using tp::TPSplitSpec;
using tp::WeightSplitKind;

struct ObjectShape {
    std::uint64_t n = 0;
    std::uint64_t k = 0;
};

ObjectShape object_shape(const artifact::Directory& directory, ObjectHandle handle) {
    const auto& object = directory.object(handle);
    const auto* tensor = std::get_if<artifact::TensorObject>(&object);
    if (tensor == nullptr || tensor->shape.size() != 2) {
        throw std::invalid_argument("shard view: weight object is not a matrix");
    }
    return {tensor->shape[0], tensor->shape[1]};
}

} // namespace

std::vector<BoundWeight> shard_views(std::span<const PendingWeight> pending,
                                     const artifact::Directory& directory,
                                     const artifact::MaterializedArtifact& shard_backing,
                                     const TPSplitSpec& spec, int shard) {
    if (shard != 0 && shard != 1) { throw std::invalid_argument("shard must be 0 or 1"); }
    std::vector<BoundWeight> out;
    out.reserve(pending.size());
    for (const auto& item : pending) {
        if (item.reference.residency != artifact::Residency::Device) {
            throw std::invalid_argument("shard view: only device-resident weights are supported");
        }
        BoundWeight bw;
        bw.name           = item.reference.name;
        bw.source_objects = item.source_objects;
        bw.uses           = item.uses;
        WeightView view;
        view.shape = item.reference.shape;
        for (const auto& part : item.reference.binding.parts) {
            const auto* split = spec.find(part.object);
            const WeightSplitKind kind = split ? split->kind : WeightSplitKind::Replicated;
            WeightRegion shard_part;
            shard_part.parent = &shard_backing.device_parent(part.object);
            if (kind == WeightSplitKind::Replicated) {
                // Vectors (norms, biases) and full matrices are replicated verbatim; the shard
                // keeps the parent's begin/end and the reference shape, so no matrix shape query
                // is needed (1-D objects would otherwise fail the matrix check).
                shard_part.begin = part.begin;
                shard_part.end   = part.end;
            } else {
                const auto [n, k] = object_shape(directory, part.object);
                if (kind == WeightSplitKind::GatherRows) {
                if (!split || split->parts.empty()) {
                    throw std::invalid_argument("shard view: GatherRows split has no parts");
                }
                // Identify the row block this part addresses: begin == row_begin * k.
                const std::uint64_t r0 = part.begin / k;
                const std::uint64_t r1 = part.end / k;
                if (part.begin % k != 0 || part.end % k != 0 || r0 >= r1) {
                    throw std::invalid_argument("shard view: GatherRows part is not row-aligned");
                }
                std::uint64_t cum = 0;
                bool found = false;
                std::uint64_t block = 0;
                for (const auto& p : split->parts) {
                    const std::uint64_t pb = static_cast<std::uint64_t>(p.row_begin);
                    const std::uint64_t pc = static_cast<std::uint64_t>(p.row_count);
                    if (pb == r0 && pb + pc == r1) {
                        block = pc;
                        found = true;
                        break;
                    }
                    if (pb >= r0) { break; }
                    cum += pc / 2;
                }
                if (!found) {
                    throw std::invalid_argument("shard view: GatherRows part not in split");
                }
                // The shard payload concatenates, per part, this shard's half of the part's rows
                // (the gather already selected the shard's rows), so the block's offset within the
                // shard's row space is just the cumulative half-rows of the prior parts - identical
                // for both shards. Adding shard * (block / 2) would push the later block past the
                // shard's row count.
                const std::uint64_t row_begin = cum;
                const std::uint64_t row_count = block / 2;
                shard_part.begin = row_begin * k;
                shard_part.end   = (row_begin + row_count) * k;
                view.shape[0]    = row_count;
                view.shape[1]    = k;
            } else if (kind == WeightSplitKind::RowParallel) {
                if (part.begin != 0 || part.end != n * k) {
                    throw std::invalid_argument(
                        "shard view: RowParallel requires a whole-object part");
                }
                shard_part.begin = 0;
                shard_part.end   = n * (k / 2);
                view.shape[0]    = n;
                view.shape[1]    = k / 2;
            } else if (kind == WeightSplitKind::GatherCols) {
                // Column gather (GDN causal conv channels): the parent's columns are tiled into
                // channel blocks, each halved and concatenated, so the shard has n rows and
                // k/2 columns. The conv is a single whole-object part, so the shard region is the
                // first n * (k / 2) BF16 elements (row-major).
                if (part.begin != 0 || part.end != n * k) {
                    throw std::invalid_argument(
                        "shard view: GatherCols requires a whole-object part");
                }
                shard_part.begin = 0;
                shard_part.end   = n * (k / 2);
                view.shape[0]    = n;
                view.shape[1]    = k / 2;
            } else {
                // ColumnParallel: whole-object row split into two equal halves. The materialized
                // shard payload already holds exactly this shard's rows, so its view covers that
                // payload from the start - the same shard-local convention as GatherRows/GatherCols
                // (the parent geometry here is the shard's own n/2 x k slice).
                if (part.begin != 0 || part.end != n * k) {
                    throw std::invalid_argument(
                        "shard view: ColumnParallel requires a whole-object part");
                }
                shard_part.begin = 0;
                shard_part.end   = (n / 2) * k;
                view.shape[0]    = n / 2;
                view.shape[1]    = k;
            }
            }
            view.parts.push_back(shard_part);
        }
        bw.view = std::move(view);
        out.push_back(std::move(bw));
    }
    return out;
}

} // namespace ninfer::models::qwen3_5::loading
