#include "ops/linear/q4/q4_shapes.h"
#include "ops/linear/q4/q4_ksplit_launch.cuh"

namespace ninfer::ops::detail {

// The DFlash2 draft feature projection [5120,25600]: five target-layer hidden states are
// concatenated and projected back to the hidden width. Every TP-2 shard holds the whole
// matrix; one masked-draft round calls it once with T = min(window, 2048) * batch.
Q4Launch select_q4_n5120_k25600(std::int32_t tokens) {
    if (tokens == 1) return launch_q4_simt_r8_c4;
    if (tokens <= 8) return launch_q4_ksplit<5120, 25600, 8>;
    if (tokens <= 24) return launch_q4_simt_r8_c8;
    return launch_q4_mma_r64_c128;
}

} // namespace ninfer::ops::detail
