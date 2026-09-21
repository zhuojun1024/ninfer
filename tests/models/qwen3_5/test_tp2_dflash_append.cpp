// TP-2 DFlash2 masked-draft context append (PLAN.md section 3.6, stage B3).
//
// The engine gate keeps --spec dflash2 off the TP-2 route until the masked draft has its own
// proposal/verify wiring (src/runtime/engine/model_instance.cpp:101-104, deliberately kept), so this
// test drives the exact seam the B2b commit added instead of opening that gate: the target prefill
// forward with a DFlashFeatureSink whose consumer runs dflash_append_context on shard 0, assembled
// from the same ExecutionCore/DFlashAppendContext fields the runtime core uses in
// TP2GenerationCore::make_dflash_prefill_sink (src/runtime/engine/tp2_generation_core.cpp:1104-1151).
//
// The single-card draft-only oracle is not constructible: the loader binds the whole text component
// unconditionally (src/models/qwen3_5/load.cpp:56) and Parameters::Parameters prepares the full text
// stack (src/models/qwen3_5/execution/parameters.cpp:260-270), which does not fit one 16 GiB card
// (the smallest local 27B artifact is 17.4 GB). Accepting the TP-2 route therefore takes its
// evidence on both cards, and this test establishes, on the real artifact:
//
//   1. the sink does not perturb the target: the last-column logits of an identical prefill are
//      bit-identical with and without the sink installed (both shards);
//   2. the append chain completes: the consumer runs exactly once and the sink captured every
//      configured feature layer and its positions;
//   3. the draft ring is finite, non-zero and structurally sane: K BF16 / V FP16 with the configured
//      geometry, the addressed prefix [0, chunk) written for all five layers, the untouched capacity
//      remainder exactly zero, and the ring bit-identical to the same features appended directly;
//   4. the append is chunk-width independent: one width=chunk append and two width=chunk/2 appends
//      over the same captured features produce a bit-identical ring;
//   5. the same input, prefilled twice from a zeroed state, reproduces the ring bit for bit;
//   6. the chunk=1024 append fits the shipped 192 MiB workspace (tp2_generation_core.cpp:148), with
//      the peak reported against that budget (the PLAN's unmeasured item).
//
// What it does not establish: the ring values are not checked against an independent host oracle
// that decodes the stored q8_g32_fp16 feature/context weights with their scales, so a wrong-but-
// deterministic fused result would pass. See the report for the exact gap.
//
// The artifact is selected with NINFER_TEST_ARTIFACT; two identical sm_120a devices are required.
// Without either the test skips with exit code 77.

#include "artifact/reader.h"
#include "core/arena.h"
#include "core/cyclic_kv_cache.h"
#include "core/device.h"
#include "core/layout.h"
#include "core/linear_attention_state.h"
#include "core/tp/device_pair.h"
#include "models/qwen3_5/execution/parameters.h"
#include "models/qwen3_5/execution/text.h"
#include "models/qwen3_5/load.h"
#include "models/qwen3_5/program/context.h"
#include "models/qwen3_5/program/planning/startup.h"
#include "models/qwen3_5/program/storage/draft_context.h"
#include "models/qwen3_5/state/decoder_state.h"
#include "ninfer/ops/position.h"
#include "ninfer/ops/scalar.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::models;
namespace qwen = ninfer::models::qwen3_5;

// Mirrors the shipped runtime budget (tp2_generation_core.cpp:148). The point of the test is to run
// the chunk=1024 sink prefill inside exactly the arena the product route gives it, not a test-sized
// one.
constexpr std::size_t kWorkspaceBytes = 192ULL << 20;
// Mirrors the DFlash context alignment (tp2_generation_core.cpp:151).
constexpr std::size_t kDflashContextAlignment = 256;
constexpr std::uint32_t kTestCacheTokens      = 2048;
constexpr std::uint32_t kTestCachePages       = kTestCacheTokens / 64;

void require(bool condition, const std::string& message) {
    if (!condition) { throw std::runtime_error(message); }
}

std::pair<int, int> pick_devices() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count < 2) { return {-1, -1}; }
    // Only sm_120a devices can run the TP-2 kernels; a foreign device in the enumeration must not
    // mask the pair we can actually use.
    std::vector<int> candidates;
    std::vector<std::string> names(static_cast<std::size_t>(count));
    for (int index = 0; index < count; ++index) {
        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, index) != cudaSuccess) { return {-1, -1}; }
        if (prop.major != 12) { continue; }
        candidates.push_back(index);
        names[static_cast<std::size_t>(index)] = prop.name;
    }
    for (std::size_t a = 0; a < candidates.size(); ++a) {
        for (std::size_t b = a + 1; b < candidates.size(); ++b) {
            const std::size_t left  = static_cast<std::size_t>(candidates[a]);
            const std::size_t right = static_cast<std::size_t>(candidates[b]);
            if (names[left] == names[right]) { return {candidates[a], candidates[b]}; }
        }
    }
    return {-1, -1};
}

// Minimal per-shard execution state, mirroring the runtime core's build_shard: a paged KV cache
// large enough for the test chunk, a zeroed GDN state pool, and a workspace arena of exactly the
// shipped size.
struct ShardState {
    std::unique_ptr<DeviceArena> kv_arena;
    std::unique_ptr<qwen::DecoderState> decoder;
    std::unique_ptr<DeviceArena> state_arena;
    std::unique_ptr<LinearAttentionStatePool> state;
    DeviceSpan state_backing;
    std::vector<DeviceKVPageLease> kv_pages;
    std::vector<DeviceKVPageHandle> kv_handles;
    KVExecutionRowLease kv_row;
    std::unique_ptr<DeviceArena> workspace;
    qwen::RoundState io;
    Tensor prefill_hidden;
};

ShardState build_shard_state(DeviceContext& device, const qwen::TextConfig& config) {
    ShardState shard;
    LayoutBuilder kv_builder;
    const qwen::DecoderStateLayout kv_layout = qwen::plan_decoder_state(
        kv_builder, qwen::DecoderStateSpec{
                        .full_attention_layers = config.full_attention_layers,
                        .mtp_layers            = 1,
                        .capacity              = kTestCacheTokens,
                        .kv_heads = qwen::execution::dimension(config.attention->num_key_value_heads),
                        .attention_head_dim = qwen::execution::dimension(config.attention->head_dim),
                        .kv_storage         = KvCacheStorage::BFloat16,
                        .enable_mtp         = false,
                        .kv_table_rows      = 1,
                        .text_physical_page_groups = kTestCachePages,
                        .mtp_physical_page_groups  = 0,
                    });
    const std::size_t kv_bytes = kv_builder.finish(256);
    device.bind_to_current_thread();
    shard.kv_arena = std::make_unique<DeviceArena>(kv_bytes);
    shard.decoder  = std::make_unique<qwen::DecoderState>(shard.kv_arena->alloc_bytes(kv_bytes, 256),
                                                          kv_layout);

    const LinearAttentionStatePoolSpec gdn_spec{
        .layers         = config.linear_attention_layers,
        .conv_channels  = (config.gdn ? qwen::execution::dimension(config.gdn->conv_channels()) : 0),
        .conv_width     = (config.gdn ? qwen::execution::dimension(config.gdn->linear_conv_kernel_dim - 1) : 0),
        .value_heads    = (config.gdn ? qwen::execution::dimension(config.gdn->linear_num_value_heads) : 0),
        .value_head_dim = (config.gdn ? qwen::execution::dimension(config.gdn->linear_value_head_dim) : 0),
        .key_head_dim   = (config.gdn ? qwen::execution::dimension(config.gdn->linear_key_head_dim) : 0),
        .slot_count     = 1,
        .conv_dtype     = DType::BF16,
    };
    LayoutBuilder state_builder;
    const LinearAttentionStatePoolLayout state_layout =
        plan_linear_attention_state_pool(state_builder, gdn_spec);
    const std::size_t state_bytes = state_builder.finish(256);
    device.bind_to_current_thread();
    shard.state_arena   = std::make_unique<DeviceArena>(state_bytes);
    shard.state_backing = shard.state_arena->alloc_bytes(state_bytes, 256);
    CUDA_CHECK(cudaMemset(shard.state_backing.data, 0, state_bytes));
    shard.state = std::make_unique<LinearAttentionStatePool>(shard.state_backing, state_layout);

    device.bind_to_current_thread();
    shard.workspace = std::make_unique<DeviceArena>(kWorkspaceBytes);
    shard.prefill_hidden =
        shard.workspace->alloc(DType::BF16, {2 * qwen::execution::dimension(config.hidden_size), 1});
    return shard;
}

// The DFlash draft's persistent context on shard 0, allocated exactly as build_shard does
// (tp2_generation_core.cpp:647-695): one prefill chunk of target features, the positions, the
// pending staging buffer, and the zeroed local cyclic K/V ring for the five sliding layers.
struct DFlashContext {
    DeviceSpan backing;
    std::size_t bytes = 0;
    std::unique_ptr<DeviceArena> arena;
    std::unique_ptr<CyclicKVCache> ring;
    std::unique_ptr<qwen::detail::DFlashPersistentState> state;
    Tensor features;
    Tensor positions;
    Tensor pending;
};

DFlashContext build_dflash_context(DeviceContext& device, const qwen::DraftConfig& draft,
                                   const qwen::TextConfig& target, std::int32_t columns,
                                   std::int32_t lanes) {
    const std::int32_t target_features =
        qwen::execution::dimension(target.hidden_size * draft.target_layer_ids.size());
    LayoutBuilder builder;
    qwen::detail::DFlashPersistentLayout layout;
    layout.prefill_features = builder.add_tensor(DType::BF16, {target_features, columns},
                                                 kDflashContextAlignment,
                                                 "test DFlash prefill target features");
    layout.prefill_positions = builder.add_tensor(DType::I32, {columns}, kDflashContextAlignment,
                                                  "test DFlash prefill target positions");
    layout.pending_features = builder.add_tensor(DType::BF16, {target_features, lanes, 1},
                                                 kDflashContextAlignment,
                                                 "test DFlash pending target features");
    const CyclicKVCacheLayout ring_layout = plan_cyclic_kv_cache(
        builder, draft.local_layer_count(), draft.sliding_window.value_or(0),
        qwen::execution::dimension(draft.attention.num_key_value_heads),
        qwen::execution::dimension(draft.attention.head_dim), 1);
    DFlashContext context;
    context.bytes = builder.finish(kDflashContextAlignment, "test DFlash context");
    device.bind_to_current_thread();
    context.arena   = std::make_unique<DeviceArena>(context.bytes);
    context.backing = context.arena->alloc_bytes(context.bytes, kDflashContextAlignment);
    CUDA_CHECK(cudaMemset(context.backing.data, 0, context.bytes));
    context.ring  = std::make_unique<CyclicKVCache>(context.backing, ring_layout);
    context.state = std::make_unique<qwen::detail::DFlashPersistentState>(context.backing, layout,
                                                                          *context.ring);
    context.features  = context.state->prefill_features;
    context.positions = context.state->prefill_positions;
    context.pending   = context.state->pending_features;
    return context;
}

void zero_dflash(DeviceContext& device, DFlashContext& context) {
    device.bind_to_current_thread();
    CUDA_CHECK(cudaMemsetAsync(context.backing.data, 0, context.bytes, device.stream));
}

std::vector<std::uint16_t> read_bf16(DeviceContext& device, const Tensor& tensor) {
    device.bind_to_current_thread();
    CUDA_CHECK(cudaStreamSynchronize(device.stream));
    std::vector<std::uint16_t> out(static_cast<std::size_t>(tensor.bytes() / sizeof(std::uint16_t)));
    CUDA_CHECK(cudaMemcpy(out.data(), tensor.data, tensor.bytes(), cudaMemcpyDeviceToHost));
    return out;
}

struct RingSnapshot {
    std::vector<std::vector<std::byte>> k;
    std::vector<std::vector<std::byte>> v;
    std::uint64_t hash = 0;
    bool operator==(const RingSnapshot& other) const {
        return k == other.k && v == other.v;
    }
};

std::uint64_t fnv1a(std::span<const std::byte> data) {
    std::uint64_t hash = 1469598103934665603ULL;
    for (const std::byte byte : data) {
        hash ^= static_cast<std::uint64_t>(std::to_integer<unsigned char>(byte));
        hash *= 1099511628211ULL;
    }
    return hash;
}

RingSnapshot read_ring(DeviceContext& device, const CyclicKVCache& ring) {
    RingSnapshot out;
    device.bind_to_current_thread();
    CUDA_CHECK(cudaStreamSynchronize(device.stream));
    std::uint64_t hash = 1469598103934665603ULL;
    const auto feed      = [&hash](const std::vector<std::byte>& bytes) {
        hash ^= fnv1a(bytes);
        hash *= 1099511628211ULL;
    };
    for (std::uint32_t layer = 0; layer < ring.layer_count(); ++layer) {
        const CyclicKVCacheLayerView view = ring.layer_view(layer);
        std::vector<std::byte> k(view.k.bytes());
        std::vector<std::byte> v(view.v.bytes());
        CUDA_CHECK(cudaMemcpy(k.data(), view.k.data, k.size(), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(v.data(), view.v.data, v.size(), cudaMemcpyDeviceToHost));
        feed(k);
        feed(v);
        out.k.push_back(std::move(k));
        out.v.push_back(std::move(v));
    }
    out.hash = hash;
    return out;
}

float bf16_at(const std::vector<std::byte>& bytes, std::size_t element) {
    std::uint16_t raw = 0;
    std::memcpy(&raw, bytes.data() + element * 2, 2);
    const __nv_bfloat16 value = *reinterpret_cast<const __nv_bfloat16*>(&raw);
    return __bfloat162float(value);
}

float fp16_at(const std::vector<std::byte>& bytes, std::size_t element) {
    std::uint16_t raw = 0;
    std::memcpy(&raw, bytes.data() + element * 2, 2);
    const __half value = *reinterpret_cast<const __half*>(&raw);
    return __half2float(value);
}

struct RingLayerStats {
    std::size_t live_elements   = 0;
    std::size_t live_nonzero    = 0;
    std::size_t live_nonfinite  = 0;
    std::size_t dead_nonzero_kv = 0;
    float max_abs               = 0.0F;
    double mean_abs             = 0.0;
};

// The ring stores dim0 = head_dim, dim1 = padded capacity, dim2 = kv heads, dim3 = lane, with dim0
// contiguous. Addresses [0, chunk) are live after a from-scratch append of chunk positions; the
// untouched remainder must still be zero.
RingLayerStats ring_layer_stats(const std::vector<std::byte>& bytes, bool is_fp16,
                                std::int32_t head_dim, std::int32_t padded_capacity,
                                std::int32_t kv_heads, std::int32_t lane_capacity, std::int32_t chunk,
                                bool* dead_ok) {
    RingLayerStats stats;
    for (std::int32_t p = 0; p < padded_capacity; ++p) {
        for (std::int32_t d = 0; d < head_dim; ++d) {
            for (std::int32_t h = 0; h < kv_heads; ++h) {
                for (std::int32_t lane = 0; lane < lane_capacity; ++lane) {
                    // Tensor ne[0] is the contiguous axis, so the element offset is
                    // d + D * (p + P * (h + H * lane)) for ne = [D, P, H, lane].
                    const std::size_t element =
                        static_cast<std::size_t>(d) +
                        static_cast<std::size_t>(head_dim) *
                            (static_cast<std::size_t>(p) +
                             static_cast<std::size_t>(padded_capacity) *
                                 (static_cast<std::size_t>(h) +
                                  static_cast<std::size_t>(kv_heads) * lane));
                    const float value = is_fp16 ? fp16_at(bytes, element) : bf16_at(bytes, element);
                    if (p < chunk) {
                        ++stats.live_elements;
                        if (value != 0.0F) { ++stats.live_nonzero; }
                        if (!std::isfinite(value)) { ++stats.live_nonfinite; }
                        const float magnitude = std::fabs(value);
                        stats.max_abs         = std::max(stats.max_abs, magnitude);
                        stats.mean_abs += static_cast<double>(magnitude);
                    } else if (value != 0.0F) {
                        ++stats.dead_nonzero_kv;
                    }
                }
            }
        }
    }
    if (stats.live_elements != 0) {
        stats.mean_abs /= static_cast<double>(stats.live_elements);
    }
    *dead_ok = *dead_ok && stats.dead_nonzero_kv == 0;
    return stats;
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::filesystem::path path;
        std::int32_t chunk = 1024;
        for (int i = 1; i < argc; ++i) {
            const std::string arg(argv[i]);
            const auto value = [&]() -> std::string {
                if (++i >= argc) { throw std::invalid_argument("missing value for " + arg); }
                return argv[i];
            };
            if (arg == "--artifact") {
                path = value();
            } else if (arg == "--chunk") {
                chunk = std::stoi(value());
            } else if (arg == "--help") {
                std::cout << "--artifact PATH [--chunk N]\n";
                return 0;
            } else {
                throw std::invalid_argument("unknown argument " + arg);
            }
        }
        if (path.empty()) {
            if (const char* env = std::getenv("NINFER_TEST_ARTIFACT")) { path = env; }
        }
        if (path.empty()) {
            std::cout << "SKIP: supply an explicit --artifact or set NINFER_TEST_ARTIFACT\n";
            return 77;
        }
        if (chunk < 64 || chunk > 1024 || chunk % 64 != 0) {
            throw std::invalid_argument("--chunk must be a multiple of 64 in [64,1024]");
        }
        const auto [dev0, dev1] = pick_devices();
        if (dev0 < 0) {
            std::cout << "SKIP: need two identical sm_120a devices\n";
            return 77;
        }
        std::unique_ptr<DeviceContext> device0(new DeviceContext(dev0));
        std::unique_ptr<DeviceContext> device1(new DeviceContext(dev1));

        artifact::Reader reader(path);
        LoadOptions options;
        options.vision      = false;
        options.speculative = SpeculativeBackend::DFlash2;
        auto plan = qwen::plan_load(reader, options);
        auto [model0, model1] =
            qwen::materialize_model_tp2(std::move(plan), *device0, *device1, nullptr);
        require(model0->weights().draft.has_value(), "shard 0 did not materialize the DFlash2 draft");
        require(!model1->has_weight(model0->weights().draft->feature_projection),
                "shard 1 materialized the shard-local DFlash2 draft");
        qwen::execution::Parameters parameters0(*model0);
        qwen::execution::Parameters parameters1(*model1);
        require(parameters0.draft.has_value(), "shard 0 DFlash draft parameters are missing");
        require(!parameters1.draft.has_value(),
                "shard 1 built DFlash draft parameters without the weights");

        const auto& target = model0->config().text;
        const auto& draft  = *model0->config().draft;
        require(draft.dflash2.has_value(), "artifact draft is not DFlash2");
        require(draft.local_layer_count() == draft.num_hidden_layers,
                "test assumes an all-sliding DFlash2 draft");
        const std::int32_t hidden = qwen::execution::dimension(target.hidden_size);
        const std::int32_t vocab  = qwen::execution::dimension(target.vocab_size);
        const std::int32_t kv_heads =
            qwen::execution::dimension(draft.attention.num_key_value_heads);
        const std::int32_t head_dim = qwen::execution::dimension(draft.attention.head_dim);
        const std::int32_t layers   = static_cast<std::int32_t>(draft.local_layer_count());
        std::cout << path.filename().string() << ": TP-2 DFlash2 append | devices=" << dev0 << ","
                  << dev1 << " chunk=" << chunk << " target_layers=" << draft.target_layer_ids.size()
                  << " hidden=" << hidden << " ring_layers=" << layers
                  << " kv_heads=" << kv_heads << " head_dim=" << head_dim
                  << " window=" << draft.sliding_window.value_or(0) << "\n";

        // Per-shard config: the mixer head counts are halved, as the runtime core does, so the KV
        // cache and GDN pool are sized for the per-shard geometry while the replicated components
        // keep the full-model config.
        auto shard_cfg = [](const qwen::TextConfig& full) {
            auto cfg = std::make_unique<qwen::TextConfig>(full);
            if (cfg->attention) {
                cfg->attention->num_attention_heads /= 2;
                cfg->attention->num_key_value_heads /= 2;
            }
            if (cfg->gdn) {
                cfg->gdn->linear_num_key_heads /= 2;
                cfg->gdn->linear_num_value_heads /= 2;
            }
            return cfg;
        };
        auto shard_cfg0 = shard_cfg(target);
        auto shard_cfg1 = shard_cfg(target);
        const qwen::TextConfig& cfg0 = *shard_cfg0;
        const qwen::TextConfig& cfg1 = *shard_cfg1;

        ShardState shard0 = build_shard_state(*device0, cfg0);
        ShardState shard1 = build_shard_state(*device1, cfg1);
        tp::DevicePair pair(dev0, dev1);

        qwen::execution::TextContext card0(*device0, parameters0, *shard0.workspace, {}, *shard0.state,
                                           shard0.io, shard0.prefill_hidden, chunk, 0, {},
                                           &shard0.decoder->text_kv);
        qwen::execution::TextContext card1(*device1, parameters1, *shard1.workspace, {}, *shard1.state,
                                           shard1.io, shard1.prefill_hidden, chunk, 0, {},
                                           &shard1.decoder->text_kv);
        card0.set_shard_config(&cfg0, 0);
        card1.set_shard_config(&cfg1, 1);

        // Publish one fixed physical KV row per shard, as the runtime core does once at startup.
        auto publish_kv_rows = [](DeviceContext* device, ShardState* shard) {
            device->bind_to_current_thread();
            auto& pool   = shard->decoder->text_kv.page_pool();
            auto& tables = shard->decoder->text_kv.execution_tables();
            auto reserved = pool.reserve(kTestCachePages);
            if (!reserved.has_value()) { throw std::logic_error("KV page reservation failed"); }
            DeviceKVPageReservation reservation = std::move(*reserved);
            shard->kv_pages.reserve(kTestCachePages);
            pool.materialize(reservation, kTestCachePages, shard->kv_pages);
            shard->kv_handles.clear();
            for (const auto& lease : shard->kv_pages) {
                shard->kv_handles.push_back(lease.handle());
            }
            shard->kv_row = tables.acquire(0);
            tables.publish(shard->kv_row.handle(), 0, shard->kv_handles, device->stream);
        };
        publish_kv_rows(device0.get(), &shard0);
        publish_kv_rows(device1.get(), &shard1);

        DFlashContext dflash =
            build_dflash_context(*device0, draft, target, chunk, /*lanes=*/1);
        const double ring_bytes = static_cast<double>(layers) * head_dim *
                                  static_cast<double>(dflash.ring->padded_capacity()) * kv_heads *
                                  (2.0 + 2.0);
        std::printf("[mem] shard 0 DFlash2 context arena %.1f MiB (ring %.1f MiB)\n",
                    static_cast<double>(dflash.bytes) / 1048576.0, ring_bytes / 1048576.0);

        std::vector<int> ids(static_cast<std::size_t>(chunk));
        for (std::int32_t i = 0; i < chunk; ++i) { ids[static_cast<std::size_t>(i)] = 1000 + i % 997; }

        auto reset_state = [&]() {
            device0->bind_to_current_thread();
            CUDA_CHECK(cudaMemsetAsync(shard0.state_backing.data, 0, shard0.state_backing.bytes,
                                       device0->stream));
            device1->bind_to_current_thread();
            CUDA_CHECK(cudaMemsetAsync(shard1.state_backing.data, 0, shard1.state_backing.bytes,
                                       device1->stream));
            zero_dflash(*device0, dflash);
        };

        struct RunResult {
            std::vector<std::uint16_t> logits0;
            std::vector<std::uint16_t> logits1;
            std::size_t peak0 = 0;
            std::size_t peak1 = 0;
        };
        auto run_prefill = [&](qwen::execution::DFlashFeatureSink* sink) {
            reset_state();
            shard0.workspace->reset_peak();
            shard1.workspace->reset_peak();
            auto scope0 = shard0.workspace->scope();
            auto scope1 = shard1.workspace->scope();
            device0->bind_to_current_thread();
            Tensor logits0 = shard0.workspace->alloc(DType::BF16, {vocab, 1});
            device1->bind_to_current_thread();
            Tensor logits1 = shard1.workspace->alloc(DType::BF16, {vocab, 1});
            card0.forward_tp2_prefill(card1, pair, ids, 0, &logits0, &logits1, nullptr, nullptr,
                                      nullptr, qwen::TextPhase::Prefill, nullptr, sink);
            RunResult result;
            result.logits0 = read_bf16(*device0, logits0);
            result.logits1 = read_bf16(*device1, logits1);
            result.peak0   = shard0.workspace->peak_used();
            result.peak1   = shard1.workspace->peak_used();
            return result;
        };

        // Direct (sink-free) append driver: the same call the runtime consumer makes, on an explicit
        // feature window, so the append itself can be exercised independently of the capture path.
        std::uint32_t consumed = 0;
        auto append_into = [&](qwen::detail::DFlashPersistentState& state, const Tensor& features,
                               const Tensor& positions, std::uint32_t exact) {
            auto scratch = shard0.workspace->scope();
            device0->bind_to_current_thread();
            Tensor counts = shard0.workspace->alloc(DType::I32, {1});
            Tensor lanes  = shard0.workspace->alloc(DType::I32, {1});
            ops::set_i32_scalar(counts, static_cast<std::int32_t>(exact), device0->stream);
            ops::set_i32_scalar(lanes, 0, device0->stream);
            qwen::execution::DFlashAppendContext append{
                .execution =
                    qwen::execution::ExecutionCore{
                        .device           = *device0,
                        .parameters       = parameters0,
                        .work             = *shard0.workspace,
                        .linear_attention = *shard0.state,
                        .replay_records   = nullptr,
                        .io               = shard0.io,
                        .prefill_hidden   = shard0.prefill_hidden,
                        .prefill_chunk    = static_cast<std::uint32_t>(chunk),
                        .proposal_head    = ProposalHead::Full,
                    },
                .dflash = state,
            };
            qwen::execution::dflash_append_context(
                append, features, positions, counts, lanes, counts, {exact, exact});
        };

        // 1. Target prefill without the sink.
        const RunResult baseline = run_prefill(nullptr);
        std::cout << "  no-sink prefill: workspace peak A=" << (baseline.peak0 >> 20)
                  << " MiB B=" << (baseline.peak1 >> 20) << " MiB" << std::endl;

        // 2. Target prefill with the sink; the consumer runs the B2b append.
        qwen::execution::DFlashFeatureSink sink{
            .features  = &dflash.features,
            .positions = &dflash.positions,
            .layers    = std::span<const std::uint32_t>(draft.target_layer_ids),
            .consume_prefill =
                [&](const Tensor& features, const Tensor& positions, bool /*rewrite*/) {
                    ++consumed;
                    append_into(*dflash.state, features, positions,
                                static_cast<std::uint32_t>(features.ne[1]));
                }};
        const RunResult with_sink = run_prefill(&sink);
        std::cout << "  sink prefill: workspace peak A=" << (with_sink.peak0 >> 20)
                  << " MiB B=" << (with_sink.peak1 >> 20) << " MiB | consumer calls=" << consumed
                  << " captured_mask=0x" << std::hex << sink.captured_mask << std::dec
                  << " active_tokens=" << sink.active_tokens << std::endl;

        const std::uint32_t complete_mask =
            draft.target_layer_ids.size() == 32 ? ~0U
                                                : ((1U << draft.target_layer_ids.size()) - 1U);
        const std::uint32_t first_consumed = consumed;
        require(first_consumed == 1, "the sink consumer did not run exactly once");
        require(sink.captured_mask == complete_mask, "the sink did not capture every feature layer");
        require(sink.active_tokens == chunk, "the sink captured the wrong chunk width");
        require(with_sink.peak0 <= kWorkspaceBytes, "shard 0 workspace exceeded its budget");
        require(with_sink.logits0 == baseline.logits0,
                "shard 0 target logits changed when the sink was installed");
        require(with_sink.logits1 == baseline.logits1,
                "shard 1 target logits changed when the sink was installed");
        require(with_sink.logits0 == with_sink.logits1,
                "the two shards' target logits disagree under the sink");
        std::cout << "  target logits bit-identical with and without the sink; both shards agree"
                  << std::endl;

        // The sink captures the chunk's absolute cache positions; those drive the ring slots.
        std::vector<std::int32_t> captured(static_cast<std::size_t>(chunk), -1);
        device0->bind_to_current_thread();
        CUDA_CHECK(cudaMemcpy(captured.data(), dflash.positions.data,
                              sizeof(std::int32_t) * static_cast<std::size_t>(chunk),
                              cudaMemcpyDeviceToHost));
        for (std::int32_t i = 0; i < chunk; ++i) {
            require(captured[static_cast<std::size_t>(i)] == i,
                    "the sink captured an unexpected absolute position");
        }
        std::cout << "  sink captured positions [0," << chunk << ")" << std::endl;

        // 3. Ring structure.
        const RingSnapshot ring1 = read_ring(*device0, *dflash.ring);
        require(dflash.ring->layer_count() == static_cast<std::uint32_t>(layers),
                "ring layer count is wrong");
        require(dflash.ring->capacity() == draft.sliding_window.value_or(0),
                "ring capacity is not the draft window");
        require(dflash.ring->num_kv_heads() == kv_heads && dflash.ring->head_dim() == head_dim,
                "ring head geometry is wrong");
        require(dflash.ring->lane_capacity() == 1, "ring lane capacity is not one");
        bool dead_ok = true;
        for (std::int32_t layer = 0; layer < layers; ++layer) {
            const RingLayerStats k = ring_layer_stats(
                ring1.k[static_cast<std::size_t>(layer)], false, head_dim,
                static_cast<std::int32_t>(dflash.ring->padded_capacity()), kv_heads, 1, chunk,
                &dead_ok);
            const RingLayerStats v = ring_layer_stats(
                ring1.v[static_cast<std::size_t>(layer)], true, head_dim,
                static_cast<std::int32_t>(dflash.ring->padded_capacity()), kv_heads, 1, chunk,
                &dead_ok);
            std::printf("  ring layer %d: K live_nonzero=%zu/%zu nonfinite=%zu max_abs=%.4g "
                        "mean_abs=%.4g | V live_nonzero=%zu/%zu nonfinite=%zu max_abs=%.4g\n",
                        layer, k.live_nonzero, k.live_elements, k.live_nonfinite,
                        static_cast<double>(k.max_abs), k.mean_abs, v.live_nonzero, v.live_elements,
                        v.live_nonfinite, static_cast<double>(v.max_abs));
            require(k.live_nonfinite == 0 && v.live_nonfinite == 0,
                    "the ring holds a non-finite K/V value");
            require(k.live_nonzero > 0 && v.live_nonzero > 0, "the ring layer is entirely zero");
        }
        require(dead_ok, "the ring wrote past the addressed prefix");
        std::cout << "  ring: geometry and coverage sane, capacity remainder zero" << std::endl;

        // 4. The sink path equals the same features appended directly (one width=chunk call).
        DFlashContext direct = build_dflash_context(*device0, draft, target, chunk, 1);
        std::unique_ptr<DeviceArena> direct_arena(
            new DeviceArena(static_cast<std::size_t>(target.hidden_size) *
                                draft.target_layer_ids.size() * chunk * 2 +
                            4096));
        device0->bind_to_current_thread();
        Tensor features_copy = direct_arena->alloc(
            DType::BF16,
            {qwen::execution::dimension(target.hidden_size * draft.target_layer_ids.size()), chunk});
        Tensor direct_positions = direct_arena->alloc(DType::I32, {chunk});
        CUDA_CHECK(cudaMemcpyAsync(features_copy.data, dflash.features.data, features_copy.bytes(),
                                   cudaMemcpyDeviceToDevice, device0->stream));
        ops::fill_i32_positions(direct_positions, 0, device0->stream);
        append_into(*direct.state, features_copy, direct_positions, static_cast<std::uint32_t>(chunk));
        const RingSnapshot direct1 = read_ring(*device0, *direct.ring);
        require(direct1 == ring1, "the direct append differs from the sink-driven append");
        std::cout << "  direct append over the captured features reproduces the sink ring "
                     "bit-for-bit"
                  << std::endl;

        // 5. Chunk-width independence: two width=chunk/2 appends at absolute positions.
        const std::int32_t half = chunk / 2;
        zero_dflash(*device0, direct);
        Tensor half_features = features_copy.slice(1, 0, half);
        Tensor half_positions = direct_positions.slice(0, 0, half);
        append_into(*direct.state, half_features, half_positions, static_cast<std::uint32_t>(half));
        Tensor tail_features = features_copy.slice(1, half, half);
        Tensor tail_positions = direct_positions.slice(0, half, half);
        append_into(*direct.state, tail_features, tail_positions, static_cast<std::uint32_t>(half));
        const RingSnapshot direct2 = read_ring(*device0, *direct.ring);
        require(direct2 == direct1,
                "two half-width appends differ from one full-width append");
        std::cout << "  " << chunk << " vs 2x" << half
                  << " append: ring bit-identical (absolute positions honoured)" << std::endl;

        // 6. Reproducibility across runs from a zeroed state.
        const RunResult repeat = run_prefill(&sink);
        const RingSnapshot ring2 = read_ring(*device0, *dflash.ring);
        require(repeat.logits0 == with_sink.logits0, "the repeat run changed shard 0 logits");
        require(ring2 == ring1, "the repeat run changed the ring");
        std::printf("  repeat run: ring bit-identical (fnv1a=0x%016llx)\n",
                    static_cast<unsigned long long>(ring1.hash));

        std::printf(
            "TP-2 DFlash2 append passed: chunk=%d sink logits bit-identical, consumer=%u/sink "
            "prefill, ring sane+reproducible, direct/split append bit-identical, workspace peak "
            "A=%.1f MiB (%.0f%% of %zu MiB) B=%.1f MiB\n",
            chunk, first_consumed, static_cast<double>(with_sink.peak0) / 1048576.0,
            100.0 * static_cast<double>(with_sink.peak0) / static_cast<double>(kWorkspaceBytes),
            kWorkspaceBytes >> 20, static_cast<double>(with_sink.peak1) / 1048576.0);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
