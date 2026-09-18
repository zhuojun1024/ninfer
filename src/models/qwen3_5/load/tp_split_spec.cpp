#include "models/qwen3_5/load/tp_split_spec.h"

#include <algorithm>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>

namespace ninfer::models::qwen3_5::loading {
namespace {

using artifact::ObjectHandle;
using artifact::TensorObject;
using tp::RowPart;
using tp::TPObjectSplit;
using tp::TPSplitSpec;
using tp::WeightSplitKind;

RowPart part(std::int32_t begin, std::int32_t count) {
    RowPart p;
    p.row_begin = begin;
    p.row_count = count;
    return p;
}

std::int32_t as_i32(std::uint64_t value) {
    if (value > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error("tp split spec: dimension exceeds int32");
    }
    return static_cast<std::int32_t>(value);
}

} // namespace

// TP-2 sharding with head-parallel mixers. The dense FFN is sharded (gate/up column-parallel,
// down row-parallel) and the lm_head is column-parallel (vocab split). The attention and GDN
// mixers are sharded by head: the fused input projections are row-gathered into per-shard halves
// (attention q/k/gate/v, GDN q/k/v/z), the GDN causal conv is column-gathered over its q/k/v
// channel blocks, and the mixer output projections are row-parallel (input columns split) with an
// all-reduce after each mixer. The GDN gating projections (a/b), a_log and dt_bias stay
// replicated: each shard runs the full 48-head gating GEMM and slices g/beta to its local value
// heads, so no 1-D fp32 split or gating-op change is needed. Embeddings and norms are replicated.
TPSplitSpec build_tp_split_spec(const artifact::Directory& directory, const TextConfig& config) {
    // Reverse map: physical object index -> text-component parameter names referencing it. Every
    // binding carries at least one part, including a whole-object binding (which stores a single
    // part spanning that object), so indexing by the part's object covers both forms. The
    // whole_object flag is a boolean marker, not an object index, and must not key this map.
    std::map<std::size_t, std::set<std::string>> names_by_object;
    for (const auto& [name, binding] : directory.bindings) {
        if (name.rfind("text/", 0) != 0 && name.rfind("mtp/", 0) != 0) { continue; }
        for (const auto& p : binding.parts) { names_by_object[p.object.index].insert(name); }
    }

    const auto& dense = std::get<DenseConfig>(config.ffn);
    const std::int32_t hidden     = as_i32(config.hidden_size);
    const std::int32_t intermediate = as_i32(dense.intermediate_size);

    // Mixer geometry (full model) used to build the per-part row/column blocks.
    const std::int32_t attn_q = config.attention ? as_i32(config.attention->query_width()) : 0;
    const std::int32_t attn_k = config.attention ? as_i32(config.attention->key_width()) : 0;
    const std::int32_t gdn_k  = config.gdn ? as_i32(config.gdn->key_width()) : 0;
    const std::int32_t gdn_v  = config.gdn ? as_i32(config.gdn->value_width()) : 0;

    TPSplitSpec spec;
    for (std::size_t idx = 0; idx < directory.objects.size(); ++idx) {
        const auto* tensor = std::get_if<TensorObject>(&directory.objects[idx]);
        if (tensor == nullptr || tensor->shape.size() < 2) { continue; }
        const std::int32_t n = as_i32(tensor->shape[0]);
        const std::int32_t k = as_i32(tensor->shape[1]);

        const auto names_it = names_by_object.find(idx);
        const auto& names   = names_it != names_by_object.end() ? names_it->second
                                                                 : std::set<std::string>{};
        const auto has_name = [&](std::string_view suffix) {
            for (const auto& nm : names) {
                if (nm.size() >= suffix.size() &&
                    nm.compare(nm.size() - suffix.size(), suffix.size(), suffix) == 0) {
                    return true;
                }
            }
            return false;
        };

        const auto has_mtp_name = [&] {
            for (const auto& nm : names) {
                if (nm.rfind("mtp/", 0) == 0) { return true; }
            }
            return false;
        };

        TPObjectSplit split;
        split.object.index = idx;

        if (has_mtp_name()) {
            // The MTP proposal runs entirely on shard 0: its weights are tiny (one layer, ~0.2 GiB)
            // and the proposal's input is bit-identical on both shards (the residual stream is
            // all-reduced, the embedding replicated), so splitting them would buy sub-millisecond
            // savings at the cost of a second lockstep all-reduce path. Keep them whole everywhere.
            split.kind = WeightSplitKind::Replicated;
        } else if (has_name("token_embedding")) {
            split.kind = WeightSplitKind::Replicated;
        } else if (has_name("output_head")) {
            // The lm_head is weight-tied to the token embedding and each shard computes the
            // full-vocabulary logits independently (the final hidden state is identical on both
            // shards), so the head stays replicated rather than column-parallel.
            split.kind = WeightSplitKind::Replicated;
        } else if (n == 2 * attn_q + 2 * attn_k && k == hidden) {
            // Attention fused input projection, physical row order [q | k | gate | v]. Each block
            // is halved across the two GPUs and the halves concatenated in block order.
            split.kind = WeightSplitKind::GatherRows;
            split.parts = {part(0, attn_q), part(attn_q, attn_k), part(attn_q + attn_k, attn_q),
                           part(2 * attn_q + attn_k, attn_k)};
        } else if (n == 2 * gdn_k + 2 * gdn_v && k == hidden) {
            // GDN fused input projection, physical row order [q | k | v | z]. Each block is halved
            // across the two GPUs and the halves concatenated in block order.
            split.kind = WeightSplitKind::GatherRows;
            split.parts = {part(0, gdn_k), part(gdn_k, gdn_k), part(2 * gdn_k, gdn_v),
                           part(2 * gdn_k + gdn_v, gdn_v)};
        } else if (n == 4 && k == 2 * gdn_k + gdn_v) {
            // GDN causal conv [taps, channels]; channels are [q | k | v] and are column-gathered
            // into per-shard halves (the conv runs over the sharded qkv projection).
            split.kind = WeightSplitKind::GatherCols;
            split.parts = {part(0, gdn_k), part(gdn_k, gdn_k), part(2 * gdn_k, gdn_v)};
        } else if (n == hidden && k == attn_q && has_name("attention/output")) {
            // Attention output projection (row-parallel; the query-width input columns are split).
            split.kind = WeightSplitKind::RowParallel;
        } else if (n == hidden && k == gdn_v && has_name("gdn/output")) {
            // GDN output projection (row-parallel; the value-width input columns are split).
            split.kind = WeightSplitKind::RowParallel;
        } else if (n == 2 * intermediate && k == hidden) {
            // gate/up (column-parallel; the fused object is split into gate and up row blocks,
            // each halved across the two GPUs).
            split.kind = WeightSplitKind::GatherRows;
            split.parts = {part(0, intermediate), part(intermediate, intermediate)};
        } else if (n == hidden && k == intermediate) {
            // FFN down (row-parallel; the intermediate input columns are split).
            split.kind = WeightSplitKind::RowParallel;
        } else {
            split.kind = WeightSplitKind::Replicated;
        }

        spec.splits.push_back(std::move(split));
    }
    return spec;
}

} // namespace ninfer::models::qwen3_5::loading
