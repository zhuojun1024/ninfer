#include "ops/linear/q4/q4_shapes.h"
#include "ops/linear/q4/q4_ksplit_launch.cuh"

namespace ninfer::ops::detail {

// The DFlash2 draft dynamic-convolution kernel projection [1280,5120], consumed by
// rmsnorm_dynamic_grouped_conv_prepare. One draft round calls it once per layer with
// T = W*B in [2,128].
Q4Launch select_q4_n1280_k5120(std::int32_t tokens) {
    if (tokens == 1) return launch_q4_simt_r8_c4;
    if (tokens <= 8) return launch_q4_ksplit<1280, 5120, 8>;
    if (tokens <= 24) return launch_q4_simt_r8_c8;
    return launch_q4_mma_r64_c128;
}

} // namespace ninfer::ops::detail
