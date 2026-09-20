#pragma once

#include "core/tensor.h"

#include <cuda_runtime.h>

namespace ninfer::ops {

/**
 * Op: apply_token_mask
 *
 * Math / indexing:
 *   out[v,c] = logits[v,c]   when mask[v,c] != 0
 *              -infinity     otherwise
 *   for 0 <= v < rows and 0 <= c < columns, with element (v,c) stored at v + c*rows.
 *
 * Logical shapes:
 *   logits is contiguous BF16 [rows,columns]; mask is contiguous U8 [rows,columns] with the same
 *   shape; rows >= 1 and columns >= 1. mask is read-only and logits is updated in place.
 *
 * Numeric:
 *   A zero mask element writes negative infinity in BF16, which no finite logit exceeds, so every
 *   downstream argmax, filter, or softmax over the masked logits excludes that vocabulary row.
 *
 * Effects:
 *   Updates logits in place. mask and every other argument remain unchanged.
 *
 * Workspace:
 *   None. There is no state side effect beyond the stated in-place update.
 */
void apply_token_mask(Tensor& logits, const Tensor& mask, cudaStream_t stream);

} // namespace ninfer::ops
