// A head whose product materializes dense BF16 logits per chunk reaches the same grouped K-split
// reduction the fused producers use: one producer CTA per 128 rows keeps the stable top sixteen of
// every column through the warp merge sort, keyed by global token id, and the shared merge kernel
// finishes the reduction. GGUF block formats take this route because their product quantizes the
// activation to q8_1 rather than reading the hidden state directly.

#include "ops/linear_topk/linear_topk_launch.h"

#include "core/device.h"
#include "ops/linear_topk/grouped_ksplit_topk.cuh"

#include <cuda_bf16.h>

#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr int kWarps   = 8;
constexpr int kThreads = kWarps * 32;
constexpr int kColumns = kLinearTopKMaterializedChunkColumns;

struct Storage {
    GroupedKSplitTopKStorage<kColumns, kWarps> topk;
    float tile[kLinearTopK * kColumns];
};

__global__ __launch_bounds__(kThreads) void materialized_logits_topk_kernel(
    const __nv_bfloat16* __restrict__ logits, std::int32_t rows, std::int32_t valid_rows,
    std::int32_t columns, std::uint64_t* __restrict__ partial_keys, std::int32_t producer_groups,
    const std::int32_t* __restrict__ row_ids) {
    __shared__ Storage storage;
    grouped_ksplit_topk_initialize(storage.topk);

    const int tid      = static_cast<int>(threadIdx.x);
    const int cta_row0 = static_cast<int>(blockIdx.x) * kLinearTopKGroupedRows;
    for (int slice = 0; slice < kLinearTopKGroupedRows; slice += kLinearTopK) {
        const int row_begin = cta_row0 + slice;
        for (int p = tid; p < kLinearTopK * columns; p += kThreads) {
            const int column = p / kLinearTopK;
            const int r      = p - column * kLinearTopK;
            const int row    = row_begin + r;
            float value      = 0.0f;
            if (row < rows) {
                value = __bfloat162float(logits[static_cast<std::int64_t>(column) * rows + row]);
            }
            storage.tile[r * kColumns + column] = value;
        }
        __syncthreads();
        grouped_ksplit_topk_consume<kColumns, kColumns, kWarps>(
            storage.tile, storage.topk, row_begin, valid_rows, columns, row_ids);
    }
    grouped_ksplit_topk_publish(storage.topk, partial_keys, producer_groups, columns);
}

} // namespace

void linear_topk_logits_launch(const Tensor& logits, std::int32_t valid_rows,
                               const Tensor* row_to_global_ids,
                               const LinearTopKWorkspace& workspace, cudaStream_t stream) {
    const std::int32_t columns = logits.ne[1];
    if (columns <= 0 || columns > kColumns ||
        workspace.rows_per_producer != kLinearTopKGroupedRows || workspace.tile_columns != 0 ||
        logits.dtype != DType::BF16) {
        throw std::invalid_argument("linear_topk: invalid materialized-logits launch");
    }
    materialized_logits_topk_kernel<<<workspace.producer_groups, kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(logits.data), logits.ne[0], valid_rows, columns,
        static_cast<std::uint64_t*>(workspace.partial_keys.data), workspace.producer_groups,
        row_to_global_ids != nullptr ? static_cast<const std::int32_t*>(row_to_global_ids->data)
                                     : nullptr);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::ops::detail
