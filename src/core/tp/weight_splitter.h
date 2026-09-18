#pragma once

// TP weight splitting: produce per-GPU shard payloads from a complete encoded weight.
// Column-parallel (row slice) and row-parallel (column slice) for NVFP4
// BlockScaleK16M128x4 and BF16 Contiguous parents. The shards are host byte buffers the
// loader uploads to each GPU; the Op layer then sees a registered half-size shape.

#include "core/weight.h"
#include "core/weight_view.h"

#include <cstdint>
#include <span>
#include <vector>

namespace ninfer::tp {

enum class WeightSplitKind : std::uint8_t {
    ColumnParallel, // slice N (output rows); each shard is a contiguous row block.
    RowParallel,    // slice K (input columns); each shard is a column block.
    Replicated,     // upload the whole weight to every GPU.
    GatherRows,     // per-part row gather (fused q/k/v/z or gate/up parents); each part is
                    // halved independently and the halves are concatenated in part order.
    GatherCols,     // per-part column gather (GDN causal conv channels); each part is halved
                    // independently along K and the halves are concatenated in part order.
};

struct WeightShard {
    std::vector<std::uint8_t> payload;
    Weight weight{};
};

// A logical row block within a fused parent (e.g. one of q/k/v/z in a GDN input projection).
struct RowPart {
    std::int32_t row_begin = 0;
    std::int32_t row_count = 0;
};

// Splits a complete NVFP4, FP8 row-scale, or BF16 weight into two shards.
//   ColumnParallel: shard0 = rows [0, n/2), shard1 = rows [n/2, n). Requires n % 256 == 0.
//   RowParallel:    shard0 = cols [0, k/2), shard1 = cols [k/2, k). Requires k % 128 == 0.
// The shard payload carries the same planes (code/scale/divisor) as the parent, sliced.
std::vector<WeightShard> split_weight(std::span<const std::uint8_t> full_payload,
                                      const Weight& full_weight, WeightSplitKind kind);

// Per-part row gather for fused parents. For each part, takes the `shard`-th half of its rows
// (0 = first half, 1 = second half) and concatenates them, producing a shard with the parent's
// plane structure and n = sum(row_count/2) rows. NVFP4 requires each part's half to be a multiple
// of 128 rows; FP8 row-scale and BF16 have no such constraint. The parts must tile the parent's
// rows in order (this is the logical q/k/v/z or gate/up layout).
WeightShard gather_weight_rows(std::span<const std::uint8_t> full_payload,
                               const Weight& full_weight, std::span<const RowPart> parts,
                               int shard);

// Per-part column gather (K dimension). The parts tile the parent's columns in order (e.g. the
// q/k/v channel blocks of a GDN causal conv). For each part, takes the `shard`-th half of its
// columns and concatenates them, producing a shard with n rows and k = sum(row_count / 2) columns.
// Supported for BF16 Contiguous (the GDN conv); other formats throw.
WeightShard gather_weight_cols(std::span<const std::uint8_t> full_payload,
                               const Weight& full_weight, std::span<const RowPart> parts,
                               int shard);

} // namespace ninfer::tp
