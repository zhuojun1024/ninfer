#pragma once
#include "ops/linear/nvfp4/nvfp4_launch.h"
#include "ops/linear/nvfp4/nvfp4_w4a4_plan.h"

namespace ninfer::ops::detail {
struct Nvfp4LinearShape {
    std::int32_t n, k;
    Nvfp4Launch a16;
    void (*a4)(const Tensor&, const Weight&, Tensor&, Nvfp4W4a4Workspace, cudaStream_t);
    bool (*uses_a4)(std::int32_t min_tokens, std::int32_t max_tokens);
};

extern const Nvfp4LinearShape kNvfp4N14336K5120;
extern const Nvfp4LinearShape kNvfp4N16384K5120;
extern const Nvfp4LinearShape kNvfp4N34816K5120;
extern const Nvfp4LinearShape kNvfp4N5120K6144;
extern const Nvfp4LinearShape kNvfp4N5120K17408;
// TP-2 half-size shapes (column-parallel: N/2; row-parallel: K/2).
extern const Nvfp4LinearShape kNvfp4N7168K5120;    ///< gdn_input_proj shard (14336/2)
extern const Nvfp4LinearShape kNvfp4N8192K5120;    ///< QKV shard (16384/2)
extern const Nvfp4LinearShape kNvfp4N17408K5120;   ///< gate+up shard (34816/2)
extern const Nvfp4LinearShape kNvfp4N5120K3072;    ///< o_proj shard (6144/2)
extern const Nvfp4LinearShape kNvfp4N5120K8704;    ///< down_proj shard (17408/2)
} // namespace ninfer::ops::detail
