// ninfer::ops - apply_token_mask wrapper: public api validation and launcher dispatch.
#include "ninfer/ops/token_mask.h"

#include "ops/launcher/token_mask.h"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace ninfer::ops {

namespace {

void require_rank2(const Tensor& tensor, DType dtype, const char* name) {
    if (tensor.dtype != dtype) {
        throw std::invalid_argument(std::string("apply_token_mask: ") + name +
                                    " has the wrong dtype");
    }
    if (tensor.ne[2] != 1 || tensor.ne[3] != 1) {
        throw std::invalid_argument(std::string("apply_token_mask: ") + name +
                                    " must be rank-2");
    }
    if (tensor.ne[0] <= 0 || tensor.ne[1] <= 0) {
        throw std::invalid_argument(std::string("apply_token_mask: ") + name +
                                    " dimensions must be positive");
    }
    if (!tensor.is_contiguous()) {
        throw std::invalid_argument(std::string("apply_token_mask: ") + name +
                                    " must be contiguous");
    }
    if (tensor.data == nullptr) {
        throw std::invalid_argument(std::string("apply_token_mask: ") + name +
                                    " data must be non-null");
    }
}

} // namespace

void apply_token_mask(Tensor& logits, const Tensor& mask, cudaStream_t stream) {
    require_rank2(logits, DType::BF16, "logits");
    require_rank2(mask, DType::U8, "mask");
    if (logits.ne[0] != mask.ne[0] || logits.ne[1] != mask.ne[1]) {
        throw std::invalid_argument("apply_token_mask: mask shape must match logits shape");
    }
    if (logits.data == mask.data) {
        throw std::invalid_argument("apply_token_mask: mask must not alias logits");
    }
    detail::apply_token_mask_launch(logits, mask, stream);
}

} // namespace ninfer::ops
