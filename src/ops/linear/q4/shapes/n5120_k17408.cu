#include "ops/linear/q4/q4_shapes.h"
#include "ops/linear/q4/q4_ksplit_launch.cuh"

namespace ninfer::ops::detail {

// The DFlash2 draft MLP down projection [5120,17408], consumed by the fused
// linear_dynamic_grouped_conv_add. One draft round calls it once per layer with T = W*B.
Q4Launch select_q4_n5120_k17408(std::int32_t tokens) {
    if (tokens == 1) return launch_q4_simt_r8_c4;
    if (tokens <= 8) return launch_q4_ksplit<5120, 17408, 8>;
    if (tokens <= 24) return launch_q4_simt_r8_c8;
    return launch_q4_mma_r64_c128;
}

} // namespace ninfer::ops::detail
