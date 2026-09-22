#pragma once

#include "core/weight.h"
#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninfer::ops {

/**
 * Returns the caller-owned transient capacity required by linear_topk for every column count in the
 * inclusive `[min_columns,max_columns]` interval. The registered profile is identified by
 * its weight format and exact `[head_rows,input_rows]` geometry. Column counts are positive.
 */
[[nodiscard]] std::size_t linear_topk_workspace_capacity_bytes(QType qtype, std::int32_t head_rows,
                                                               std::int32_t input_rows,
                                                               std::int32_t min_columns,
                                                               std::int32_t max_columns);

/**
 * @brief Projects independent matrix columns through a full vocabulary head and returns
 * the stable top sixteen scores and global token ids per column.
 *
 * @details For any positive column count `U`, `hidden` is contiguous BF16 `[5120,U]`, `head` is
 * either Q8_G32_FP16 or FP8_E4M3FN_ROW_BF16 `[248320,5120]`, and `valid_rows` is 248077. For every
 * `t in [0,U)` and valid vocabulary row `v`, the ideal score is
 *
 * @f[
 *   s_{v,t}=\sum_{k=0}^{5119}\mathrm{FP32Dequant}(head)_{v,k}
 *                              \mathrm{FP32}(hidden_{k,t}).
 * @f]
 *
 * `candidate_scores` is contiguous FP32 `[16,U]` and `candidate_ids` is contiguous I32
 * `[16,U]`. Rank is stored fastest; a caller may view the output as `[16,K,B]` when `U=K*B`.
 * Candidates are ordered by descending computed score, with exact score ties resolved by lower
 * global token id. Physical rows `[valid_rows,248320)` never participate. Projection uses the
 * registered A16 arithmetic profile; candidate scores are returned directly in FP32 and no dense
 * vocabulary-logit tensor is an observable intermediate.
 *
 * All tensors and every weight plane are preserved except that both output tensors are completely
 * overwritten. Inputs, outputs, weight planes, and workspace must not overlap. The Op has no
 * persistent state and performs no internal device allocation.
 */
void linear_topk(const Tensor& hidden, const Weight& head, std::int32_t valid_rows,
                 Tensor& candidate_ids, Tensor& candidate_scores, WorkspaceArena& workspace,
                 cudaStream_t stream);

/**
 * @brief Projects through the reduced proposal head (whole or one shard's half) and returns stable
 * top sixteen scores with shortlist rows mapped to global token ids.
 *
 * @details The tensor contract is the same as the full-head overload except that `head` is
 * Q4_G64_FP16 with either all 131072 reduced rows or the 65536 rows one shard of a vocabulary
 * split materializes, and every row of `head` participates. `row_to_global_ids` is contiguous I32
 * with one entry per head row; it contains distinct ids in `[0,248077)` and maps each local head row
 * to the id used for output and tie-breaking. Artifact binding establishes the map's range and
 * uniqueness. Ranking is over the supplied rows only, so the halves of a split table rank exactly
 * the candidates the whole table ranks; `merge_topk_candidates` then restores the whole table's
 * stable top sixteen from the two halves.
 */
void linear_topk(const Tensor& hidden, const Weight& head, const Tensor& row_to_global_ids,
                 Tensor& candidate_ids, Tensor& candidate_scores, WorkspaceArena& workspace,
                 cudaStream_t stream);

/**
 * @brief Merges two stable top-sixteen candidate lists into the stable top sixteen of their union.
 *
 * @details For any positive column count `U`, the four inputs are contiguous `[16,U]` tensors:
 * `candidate_ids_*` are I32 global token ids in `[0,INT_MAX)` and `candidate_scores_*` are finite
 * FP32 scores. `candidate_ids` and `candidate_scores` are contiguous `[16,U]` outputs. Every list
 * uses `linear_topk`'s payload layout, which keeps a column's sixteen candidates contiguous: the
 * element at rank r of column c is at `c * 16 + r`. Rows are
 * ranked by descending score with exact score ties resolved by lower global token id - the same
 * total order `linear_topk` returns - so merging the top sixteen of each half of a
 * vocabulary-split head reproduces the whole head's stable top sixteen bit for bit. A reserved key
 * sentinel orders below every valid candidate, so an input id of INT_MAX or a non-finite score is
 * not a candidate; a slot the inputs cannot fill reports id INT_MAX and score 0.
 *
 * All tensors are preserved except that both outputs are completely overwritten, inputs and outputs
 * must not overlap, and every tensor must be 16-byte aligned. The Op has no workspace, no internal
 * device allocation and no persistent state.
 */
void merge_topk_candidates(const Tensor& candidate_ids_a, const Tensor& candidate_scores_a,
                           const Tensor& candidate_ids_b, const Tensor& candidate_scores_b,
                           Tensor& candidate_ids, Tensor& candidate_scores, cudaStream_t stream);

} // namespace ninfer::ops
