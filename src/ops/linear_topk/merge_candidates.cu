#include "core/device.h"
#include "core/tensor.h"
#include "ninfer/ops/linear_topk.h"

#include "ops/common/score_id_order.cuh"
#include "ops/linear_topk/linear_topk_workspace.h"

#include <climits>
#include <cstdint>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>

namespace ninfer::ops {
namespace {

constexpr int kMergedTopK = static_cast<int>(detail::kLinearTopK);

bool aligned_to(const void* pointer, std::uintptr_t alignment) {
    return pointer != nullptr && (reinterpret_cast<std::uintptr_t>(pointer) & (alignment - 1)) == 0;
}

void require_list(const Tensor& ids, const Tensor& scores, std::int32_t columns,
                  const char* label) {
    const bool ids_ok = ids.dtype == DType::I32 && ids.ne[0] == kMergedTopK &&
                        ids.ne[1] == columns && ids.ne[2] == 1 && ids.ne[3] == 1 &&
                        ids.is_contiguous() && aligned_to(ids.data, 16);
    const bool scores_ok = scores.dtype == DType::FP32 && scores.ne[0] == kMergedTopK &&
                           scores.ne[1] == columns && scores.ne[2] == 1 && scores.ne[3] == 1 &&
                           scores.is_contiguous() && aligned_to(scores.data, 16);
    if (!ids_ok || !scores_ok) {
        throw std::invalid_argument(std::string("merge_topk_candidates: invalid ") + label);
    }
}

bool overlaps(const Tensor& lhs, const Tensor& rhs) {
    if (lhs.data == nullptr || rhs.data == nullptr || lhs.bytes() == 0 || rhs.bytes() == 0) {
        return false;
    }
    const auto lhs_begin = reinterpret_cast<std::uintptr_t>(lhs.data);
    const auto rhs_begin = reinterpret_cast<std::uintptr_t>(rhs.data);
    return lhs_begin < rhs_begin + rhs.bytes() && rhs_begin < lhs_begin + lhs.bytes();
}

// One thread per column. Slot r holds the r-th best key so far, so a candidate is inserted at the
// first slot it beats and the tail shifts down; key zero is the reserved sentinel, and the strict
// comparison keeps equally ranked candidates in the order they were offered.
//
// `linear_topk` keeps a column's sixteen candidates contiguous (merge.cu writes
// `column * kLinearTopK + rank`), so element (rank, column) lives at `column * kLinearTopK + rank`
// here too: the two Ops have to agree on the payload layout, not only on the ranking rule.
__global__ void merge_topk_kernel(const std::int32_t* __restrict__ ids_a,
                                  const float* __restrict__ scores_a,
                                  const std::int32_t* __restrict__ ids_b,
                                  const float* __restrict__ scores_b,
                                  std::int32_t* __restrict__ out_ids,
                                  float* __restrict__ out_scores, std::int32_t columns) {
    const std::int32_t column = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (column >= columns) { return; }
    std::uint64_t best[kMergedTopK];
#pragma unroll
    for (int rank = 0; rank < kMergedTopK; ++rank) { best[rank] = 0; }
    const auto insert = [&](std::uint64_t key) {
        int slot = kMergedTopK;
#pragma unroll
        for (int rank = 0; rank < kMergedTopK; ++rank) {
            if (key > best[rank]) {
                slot = rank;
                break;
            }
        }
        if (slot == kMergedTopK) { return; }
#pragma unroll
        for (int rank = kMergedTopK - 1; rank > slot; --rank) { best[rank] = best[rank - 1]; }
        best[slot] = key;
    };
#pragma unroll
    for (int rank = 0; rank < kMergedTopK; ++rank) {
        const std::int64_t at = static_cast<std::int64_t>(column) * kMergedTopK + rank;
        insert(score_id_order_key(scores_a[at], ids_a[at]));
        insert(score_id_order_key(scores_b[at], ids_b[at]));
    }
#pragma unroll
    for (int rank = 0; rank < kMergedTopK; ++rank) {
        const std::int64_t at = static_cast<std::int64_t>(column) * kMergedTopK + rank;
        out_ids[at]    = best[rank] == 0 ? INT_MAX : id_from_order_key(best[rank]);
        out_scores[at] = best[rank] == 0 ? 0.0f : score_from_order_key(best[rank]);
    }
}

} // namespace

void merge_topk_candidates(const Tensor& candidate_ids_a, const Tensor& candidate_scores_a,
                           const Tensor& candidate_ids_b, const Tensor& candidate_scores_b,
                           Tensor& candidate_ids, Tensor& candidate_scores, cudaStream_t stream) {
    if (candidate_ids_a.ne[1] <= 0 || candidate_ids_a.ne[1] != candidate_ids_b.ne[1] ||
        candidate_ids_a.ne[1] != candidate_ids.ne[1]) {
        throw std::invalid_argument(
            "merge_topk_candidates: the candidate lists must share a positive column count");
    }
    const std::int32_t columns = candidate_ids_a.ne[1];
    require_list(candidate_ids_a, candidate_scores_a, columns, "first candidate list");
    require_list(candidate_ids_b, candidate_scores_b, columns, "second candidate list");
    require_list(candidate_ids, candidate_scores, columns, "merged candidate list");
    const Tensor* inputs[]{&candidate_ids_a, &candidate_scores_a, &candidate_ids_b,
                           &candidate_scores_b};
    for (const Tensor* input : inputs) {
        if (overlaps(*input, candidate_ids) || overlaps(*input, candidate_scores)) {
            throw std::invalid_argument("merge_topk_candidates: inputs and outputs must not overlap");
        }
    }
    constexpr int kThreads = 128;
    merge_topk_kernel<<<(columns + kThreads - 1) / kThreads, kThreads, 0, stream>>>(
        static_cast<const std::int32_t*>(candidate_ids_a.data),
        static_cast<const float*>(candidate_scores_a.data),
        static_cast<const std::int32_t*>(candidate_ids_b.data),
        static_cast<const float*>(candidate_scores_b.data),
        static_cast<std::int32_t*>(candidate_ids.data),
        static_cast<float*>(candidate_scores.data), columns);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops
