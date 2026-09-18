#include "artifact/reader.h"
#include "core/arena.h"
#include "core/device.h"
#include "core/layout.h"
#include "core/linear_attention_state.h"
#include "core/tp/device_pair.h"
#include "models/qwen3_5/execution/parameters.h"
#include "models/qwen3_5/execution/text.h"
#include "models/qwen3_5/load.h"
#include "models/qwen3_5/state/decoder_state.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::models;
namespace qwen = ninfer::models::qwen3_5;

std::pair<int, int> pick_devices() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count < 2) { return {-1, -1}; }
    std::vector<std::string> names(count);
    for (int i = 0; i < count; ++i) {
        cudaDeviceProp prop{};
        cudaGetDeviceProperties(&prop, i);
        names[i] = prop.name;
    }
    for (int a = 0; a < count; ++a) {
        for (int b = a + 1; b < count; ++b) {
            if (names[a] == names[b]) { return {a, b}; }
        }
    }
    return {0, 1};
}

// Logical cache positions the KV pool is sized for; the batched-prefill comparison below walks a
// 300-token prompt, which spans five pages, and the autoregressive decode test covers 32.
constexpr std::uint32_t kTestCacheTokens   = 2048;
constexpr std::uint32_t kTestCachePages    = kTestCacheTokens / 64;

struct ShardState {
    std::unique_ptr<DeviceArena> kv_arena;
    std::unique_ptr<qwen::DecoderState> decoder;
    std::unique_ptr<DeviceArena> state_arena;
    std::unique_ptr<LinearAttentionStatePool> state;
    DeviceSpan state_backing;
    std::vector<DeviceKVPageLease> kv_pages;      // the physical pages KV execution row 0 maps
    std::vector<DeviceKVPageHandle> kv_handles;
    KVExecutionRowLease kv_row;
    std::unique_ptr<DeviceArena> workspace;
    qwen::RoundState io;
    Tensor prefill_hidden;
};

// Build the minimal per-shard execution state: a paged KV cache large enough to prefill a short
// prompt, a zeroed GDN state pool, and a workspace arena large enough for one layer's activations.
ShardState build_shard_state(DeviceContext& device, const qwen::TextConfig& config) {
    ShardState shard;

    // Paged KV cache: full-attention layers only, one execution-table row.
    LayoutBuilder kv_builder;
    const qwen::DecoderStateLayout kv_layout = qwen::plan_decoder_state(
        kv_builder, qwen::DecoderStateSpec{
                     .full_attention_layers     = config.full_attention_layers,
                     .mtp_layers                = 1,
                     .capacity                  = kTestCacheTokens,
                     .kv_heads                  = qwen::execution::dimension(config.attention->num_key_value_heads),
                     .attention_head_dim        = qwen::execution::dimension(config.attention->head_dim),
                     .kv_storage                = KvCacheStorage::BFloat16,
                     .enable_mtp                = false,
                     .kv_table_rows             = 1,
                     .text_physical_page_groups = kTestCachePages,
                     .mtp_physical_page_groups  = 0,
                 });
    const std::size_t kv_bytes = kv_builder.finish(256);
    device.bind_to_current_thread(); // DeviceArena's cudaMalloc lands on the current device
    shard.kv_arena = std::make_unique<DeviceArena>(kv_bytes);
    shard.decoder  = std::make_unique<qwen::DecoderState>(shard.kv_arena->alloc_bytes(kv_bytes, 256),
                                                          kv_layout);

    // GDN state pool: linear-attention layers, one slot, zeroed.
    const LinearAttentionStatePoolSpec gdn_spec{
        .layers        = config.linear_attention_layers,
        .conv_channels = (config.gdn ? qwen::execution::dimension(config.gdn->conv_channels()) : 0),
        .conv_width    = (config.gdn ? qwen::execution::dimension(config.gdn->linear_conv_kernel_dim - 1) : 0),
        .value_heads   = (config.gdn ? qwen::execution::dimension(config.gdn->linear_num_value_heads) : 0),
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
    shard.state_arena = std::make_unique<DeviceArena>(state_bytes);
    const DeviceSpan state_backing = shard.state_arena->alloc_bytes(state_bytes, 256);
    CUDA_CHECK(cudaMemset(state_backing.data, 0, state_bytes));
    shard.state        = std::make_unique<LinearAttentionStatePool>(state_backing, state_layout);
    shard.state_backing = state_backing;

    // Workspace arena: the single-token forward peaks at ~600 KB (the full-vocab logits buffer
    // plus a handful of [N,1] activation tensors), so 256 MiB is ample. The 1 GiB default left
    // no headroom on a 16 GB card once the ~14 GiB replicated-mixer weights arena was resident.
    const std::size_t workspace_bytes = 512ULL << 20;
    device.bind_to_current_thread();
    shard.workspace = std::make_unique<DeviceArena>(workspace_bytes);
    shard.prefill_hidden = shard.workspace->alloc(DType::BF16,
                                                  {2 * qwen::execution::dimension(config.hidden_size), 1});
    std::cout << "  shard state: kv=" << (kv_bytes / 1024 / 1024) << " MiB gdn="
              << (state_bytes / 1024 / 1024) << " MiB workspace="
              << (workspace_bytes / 1024 / 1024) << " MiB\n";
    return shard;
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::filesystem::path path;
        std::int32_t token = 151643; // a common Qwen BOS-ish token; any valid id works
        LoadOptions options;
        options.vision    = false;
        options.speculative = SpeculativeBackend::None;
        for (int i = 1; i < argc; ++i) {
            const std::string arg(argv[i]);
            const auto value = [&]() -> std::string {
                if (++i >= argc) { throw std::invalid_argument("missing value for " + arg); }
                return argv[i];
            };
            if (arg == "--artifact") {
                path = value();
            } else if (arg == "--token") {
                token = std::stoi(value());
            } else if (arg == "--help") {
                std::cout << "--artifact PATH [--token ID]\n";
                return 0;
            } else {
                throw std::invalid_argument("unknown argument " + arg);
            }
        }
        if (path.empty()) {
            std::cout << "SKIP: supply an explicit --artifact\n";
            return 77;
        }
        const auto [dev0, dev1] = pick_devices();
        if (dev0 < 0) {
            std::cout << "SKIP: need two GPUs\n";
            return 77;
        }
        std::unique_ptr<DeviceContext> device0(new DeviceContext(dev0));
        std::unique_ptr<DeviceContext> device1(new DeviceContext(dev1));
        artifact::Reader reader(path);
        auto plan = qwen::plan_load(reader, options);
        auto [model0, model1] =
            qwen::materialize_model_tp2(std::move(plan), *device0, *device1, nullptr);
        qwen::execution::Parameters parameters0(*model0);
        qwen::execution::Parameters parameters1(*model1);

        // Per-shard configs: the mixer head counts are halved (each shard owns half the
        // attention/GDN heads), so the KV cache and GDN state are sized for the per-shard
        // geometry. The execution contexts are pointed at these via set_shard_config.
        auto shard_cfg = [](const qwen::TextConfig& full) {
            auto cfg = std::make_unique<qwen::TextConfig>(full);
            if (cfg->attention) {
                cfg->attention->num_attention_heads /= 2;
                cfg->attention->num_key_value_heads /= 2;
            }
            if (cfg->gdn) {
                cfg->gdn->linear_num_key_heads   /= 2;
                cfg->gdn->linear_num_value_heads /= 2;
            }
            return cfg;
        };
        auto shard_cfg0 = shard_cfg(model0->config().text);
        auto shard_cfg1 = shard_cfg(model1->config().text);
        const qwen::TextConfig& cfg0 = *shard_cfg0;
        const qwen::TextConfig& cfg1 = *shard_cfg1;

        ShardState shard0 = build_shard_state(*device0, cfg0);
        ShardState shard1 = build_shard_state(*device1, cfg1);

        tp::DevicePair pair(dev0, dev1);

        qwen::execution::TextContext card0(
            *device0, parameters0, *shard0.workspace, {}, *shard0.state, shard0.io,
            shard0.prefill_hidden, 1, 0, {}, &shard0.decoder->text_kv);
        qwen::execution::TextContext card1(
            *device1, parameters1, *shard1.workspace, {}, *shard1.state, shard1.io,
            shard1.prefill_hidden, 1, 0, {}, &shard1.decoder->text_kv);
        card0.set_shard_config(&cfg0, 0);
        card1.set_shard_config(&cfg1, 1);

        // Verify the shard-consistency invariant across a representative set of tokens: with the
        // replicated mixers, all-reduced FFN deltas, and gathered logits, both shards must produce
        // identical argmax for every input. Each run starts from a fresh (zeroed) state, so the
        // tokens are independent single-token forwards.
        const std::vector<std::int32_t> probe_tokens = {token, 0, 1, 15, 256, 1000, 10000, 50000,
                                                        100000, 200000, 248319};
        std::cout << path.filename().string() << ": TP-2 single-token forward "
                  << "devices=" << dev0 << "," << dev1 << " probes=" << probe_tokens.size() << "\n";
        // The GDN linear-attention state is updated in place by every forward, so the two
        // per-probe forwards (card0-local then card1-local) must each start from a zeroed state
        // for their argmax to be comparable. Re-zero both shards' state pools before each run.
        auto reset_state = [&]() {
            device0->bind_to_current_thread();
            CUDA_CHECK(cudaMemset(shard0.state_backing.data, 0, shard0.state_backing.bytes));
            device1->bind_to_current_thread();
            CUDA_CHECK(cudaMemset(shard1.state_backing.data, 0, shard1.state_backing.bytes));
        };
        for (const std::int32_t probe : probe_tokens) {
            reset_state();
            const std::int32_t token0 = card0.forward_tp2_token(card1, pair, probe);
            reset_state();
            const std::int32_t token1 = card1.forward_tp2_token(card0, pair, probe);
            std::cout << "  input_token=" << probe << " argmax_shard0=" << token0
                      << " argmax_shard1=" << token1 << "\n";
            if (token0 != token1) {
                std::cerr << "FAIL: shard argmax mismatch for input " << probe << " (" << token0
                          << " vs " << token1 << ")\n";
                return 1;
            }
        }
        // Stateful autoregressive decode: both shards hold replicated paged KV caches and GDN
        // state pools. Publish KV execution row 0 on each shard to point at one physical page
        // (the 64-token capacity covers the 32 decode positions), zero the GDN state once, then
        // run 32 autoregressive steps. Each step binds the same position/KV row/state slot on
        // both shards (mirroring the Program ordinary-decode bindings), so the replicated
        // mixers keep the residual and the evolving KV/GDN state bit-identical; one
        // forward_tp2_token call per step drives both shards in lockstep and throws if their
        // argmax diverges.
        // Materialize every logical page once and publish the whole block table on execution row 0.
        // The pool is a fixed physical allocation reused in place (one request at a time), so the
        // mapping is identical for every walk below and only the GDN state has to be re-zeroed
        // between them - the attention kernel reads exactly the positions the current walk wrote.
        auto publish_kv_rows = [](DeviceContext* device, ShardState* shard) {
            device->bind_to_current_thread();
            auto& pool   = shard->decoder->text_kv.page_pool();
            auto& tables = shard->decoder->text_kv.execution_tables();
            std::optional<DeviceKVPageReservation> reserved = pool.reserve(kTestCachePages);
            if (!reserved.has_value()) { throw std::logic_error("KV page reservation failed"); }
            DeviceKVPageReservation reservation = std::move(*reserved);
            shard->kv_pages.reserve(kTestCachePages);
            pool.materialize(reservation, kTestCachePages, shard->kv_pages);
            shard->kv_handles.clear();
            shard->kv_handles.reserve(shard->kv_pages.size());
            for (const auto& lease : shard->kv_pages) {
                shard->kv_handles.push_back(lease.handle());
            }
            shard->kv_row = tables.acquire(0);
            // Order the block-table H2D copy on the device stream: the attention kernels run on
            // the non-blocking ctx_.stream, which does not implicitly synchronize with the legacy
            // default stream that a default-argument publish would use.
            tables.publish(shard->kv_row.handle(), 0, shard->kv_handles, device->stream);
            CUDA_CHECK(cudaMemsetAsync(shard->state_backing.data, 0, shard->state_backing.bytes,
                                       device->stream));
        };
        publish_kv_rows(device0.get(), &shard0);
        publish_kv_rows(device1.get(), &shard1);
        const std::int32_t decode_start = 151643;
        std::int32_t decode_token       = decode_start;
        const int decode_steps          = 32;
        std::cout << "  TP-2 autoregressive decode: start_token=" << decode_start
                  << " steps=" << decode_steps << "\n";
        for (int step = 0; step < decode_steps; ++step) {
            decode_token = card0.forward_tp2_token(card1, pair, decode_token, step);
            std::cout << "    step=" << step << " position=" << step << " token=" << decode_token
                      << "\n";
        }
        // Batched-prefill oracle, three comparisons over the same weights and the same published
        // KV row:
        //
        // 1. Route parity at A16 precision (T=7, below the A4 activation cutoff of eight tokens):
        //    the chunked Phase::Prefill forward against the sequential single-token Phase::Verify
        //    forward. Both use A16 projector routes here, so the last-column logits must agree
        //    closely; what remains is the attention and GDN route difference (prompt versus small-T,
        //    chunked versus recurrent).
        // 2. Chunking equivalence at production width (T=300, A4): one chunk against the 256+44
        //    split the serving path uses. Both take the same A4 activation route, so the criterion
        //    is tight. This is what makes a chunked prefill equal a single-shot one - the second
        //    chunk must continue the first chunk's KV and GDN state.
        // 3. Informational: the 300-token batched walk against the sequential walk over the same
        //    tokens. The activation route differs (A4 versus the A16 decode kernel), so this only
        //    reports how far the two agree; it is not a pass criterion.
        const std::int32_t vocab = qwen::execution::dimension(model0->config().text.vocab_size);
        // Two routes that differ only in reduction order still diverge on the leading logits:
        // every residual add rounds a bf16 stream, so 64 layers of reordered reductions accumulate
        // a few percent of relative error, and the A4 activation route amplifies it further. The
        // observable that must hold is the sampled token, so the criterion is an identical argmax
        // with bounded leading-logit drift; a lost KV mapping or a broken GDN chunk hand-off moves
        // the logits by whole units and changes the argmax.
        constexpr float kLeadingLogitTolerance = 1.0F;
        // Build the prompt from the model's own greedy continuation. A repetitive, in-distribution
        // sequence drives a confident next-token distribution, so the sampled argmax is robust to
        // the small numerical differences between routes; a uniform random prompt produces an
        // almost flat distribution where the argmax is pure noise and cannot be an oracle.
        std::vector<int> prompt_ids;
        prompt_ids.reserve(300);
        reset_state();
        {
            std::int32_t next = token;
            for (int i = 0; i < 300; ++i) {
                prompt_ids.push_back(next);
                next = card0.forward_tp2_token(card1, pair, next, i);
            }
        }
        std::cout << "    prompt head:";
        for (int i = 0; i < 12 && i < static_cast<int>(prompt_ids.size()); ++i) {
            std::cout << " " << prompt_ids[static_cast<std::size_t>(i)];
        }
        std::cout << "\n";

        auto host_logits = [&](DeviceContext* device, const Tensor& device_logits) {
            std::vector<__nv_bfloat16> raw(static_cast<std::size_t>(vocab));
            std::vector<float> values(static_cast<std::size_t>(vocab));
            device->bind_to_current_thread();
            CUDA_CHECK(cudaStreamSynchronize(device->stream));
            CUDA_CHECK(cudaMemcpy(raw.data(), device_logits.data, device_logits.bytes(),
                                  cudaMemcpyDeviceToHost));
            for (std::int32_t v = 0; v < vocab; ++v) {
                values[static_cast<std::size_t>(v)] = __bfloat162float(raw[static_cast<std::size_t>(v)]);
            }
            return values;
        };
        auto argmax_of = [&](const std::vector<float>& values) {
            std::int32_t best = 0;
            for (std::int32_t v = 1; v < vocab; ++v) {
                if (values[static_cast<std::size_t>(v)] >
                    values[static_cast<std::size_t>(best)]) {
                    best = v;
                }
            }
            return best;
        };
        auto top_five = [&](const std::vector<float>& values) {
            std::vector<std::int32_t> best(5, -1);
            for (std::int32_t v = 0; v < vocab; ++v) {
                for (int slot = 0; slot < 5; ++slot) {
                    if (best[static_cast<std::size_t>(slot)] < 0 ||
                        values[static_cast<std::size_t>(v)] >
                            values[static_cast<std::size_t>(best[static_cast<std::size_t>(slot)])]) {
                        for (int move = 4; move > slot; --move) {
                            best[static_cast<std::size_t>(move)] =
                                best[static_cast<std::size_t>(move - 1)];
                        }
                        best[static_cast<std::size_t>(slot)] = v;
                        break;
                    }
                }
            }
            return best;
        };
        auto print_top = [&](const char* label, const std::vector<float>& values) {
            const auto best = top_five(values);
            std::cout << "    " << label << " top5:";
            for (const std::int32_t v : best) { std::cout << " " << v << "=" << values[v]; }
            std::cout << "\n";
        };
        auto max_abs_diff = [&](const std::vector<float>& a, const std::vector<float>& b) {
            float worst = 0.0F;
            for (std::size_t i = 0; i < a.size(); ++i) {
                worst = std::max(worst, std::fabs(a[i] - b[i]));
            }
            return worst;
        };
        // Ranked top-5 values. Two routes that differ only in reduction order or in private
        // activation precision produce nearly identical leading logits while the 248k-way tail can
        // drift by a few tenths, so the sampled distribution is compared on its leading values
        // rather than on the raw elementwise maximum.
        auto top_five_values = [&](const std::vector<float>& values) {
            const auto best = top_five(values);
            std::vector<float> ranked;
            ranked.reserve(5);
            for (const std::int32_t v : best) { ranked.push_back(values[v]); }
            return ranked;
        };
        auto top_five_gap = [&](const std::vector<float>& a, const std::vector<float>& b) {
            const auto ra = top_five_values(a), rb = top_five_values(b);
            float worst = 0.0F;
            for (std::size_t i = 0; i < ra.size(); ++i) {
                worst = std::max(worst, std::fabs(ra[i] - rb[i]));
            }
            return worst;
        };

        // One walk per call, returning this shard's last-column logits device buffer. The caller
        // must convert it to host values before the next walk reuses the arena. Both walks start
        // from a zeroed GDN state and read the same published KV row.
        auto run_sequential = [&](const std::vector<int>& sequence) {
            reset_state();
            device0->bind_to_current_thread();
            Tensor out = shard0.workspace->alloc(DType::BF16, {vocab, 1});
            for (std::size_t i = 0; i < sequence.size(); ++i) {
                auto scope0 = shard0.workspace->scope();
                auto scope1 = shard1.workspace->scope();
                Tensor last_a = shard0.workspace->alloc(DType::BF16, {vocab, 1});
                Tensor last_b = shard1.workspace->alloc(DType::BF16, {vocab, 1});
                card0.forward_tp2(card1, pair, sequence[i], static_cast<std::int32_t>(i), last_a,
                                  last_b);
                if (i + 1 == sequence.size()) {
                    device0->bind_to_current_thread();
                    CUDA_CHECK(cudaMemcpyAsync(out.data, last_a.data, out.bytes(),
                                               cudaMemcpyDeviceToDevice, device0->stream));
                }
            }
            return out;
        };
        auto run_prefill = [&](const std::vector<int>& sequence,
                               const std::vector<std::int32_t>& cuts) {
            reset_state();
            device0->bind_to_current_thread();
            Tensor out = shard0.workspace->alloc(DType::BF16, {vocab, 1});
            device1->bind_to_current_thread();
            Tensor out_peer = shard1.workspace->alloc(DType::BF16, {vocab, 1});
            for (std::size_t chunk = 0; chunk + 1 < cuts.size(); ++chunk) {
                const std::int32_t begin = cuts[chunk];
                const std::int32_t width = cuts[chunk + 1] - begin;
                auto scope0 = shard0.workspace->scope();
                auto scope1 = shard1.workspace->scope();
                Tensor last_a = shard0.workspace->alloc(DType::BF16, {vocab, 1});
                Tensor last_b = shard1.workspace->alloc(DType::BF16, {vocab, 1});
                card0.forward_tp2_prefill(card1, pair,
                                          std::span<const int>(sequence.data() + begin, width),
                                          begin, &last_a, &last_b);
                if (chunk + 2 == cuts.size()) {
                    device0->bind_to_current_thread();
                    CUDA_CHECK(cudaMemcpyAsync(out.data, last_a.data, out.bytes(),
                                               cudaMemcpyDeviceToDevice, device0->stream));
                    device1->bind_to_current_thread();
                    CUDA_CHECK(cudaMemcpyAsync(out_peer.data, last_b.data, out_peer.bytes(),
                                               cudaMemcpyDeviceToDevice, device1->stream));
                }
            }
            // The lm_head is replicated in full on both shards, so their logits must be identical.
            const auto peer_a = host_logits(device0.get(), out);
            const auto peer_b = host_logits(device1.get(), out_peer);
            for (std::int32_t v = 0; v < vocab; ++v) {
                if (peer_a[static_cast<std::size_t>(v)] != peer_b[static_cast<std::size_t>(v)]) {
                    throw std::runtime_error("batched prefill shard logits disagree");
                }
            }
            return out;
        };

        // 1. Route parity at A16 precision.
        const std::int32_t short_tokens = 7;
        const std::vector<int> short_ids(prompt_ids.begin(), prompt_ids.begin() + short_tokens);
        const auto seq_short = host_logits(device0.get(), run_sequential(short_ids));
        const auto batch_short =
            host_logits(device0.get(), run_prefill(short_ids, {0, short_tokens}));
        const float short_gap = top_five_gap(seq_short, batch_short);
        std::cout << "  batched prefill route parity: T=" << short_tokens
                  << " top5_gap=" << short_gap
                  << " max_logit_diff=" << max_abs_diff(seq_short, batch_short)
                  << " argmax_sequential=" << argmax_of(seq_short)
                  << " argmax_batched=" << argmax_of(batch_short) << "\n";
        print_top("sequential", seq_short);
        print_top("batched   ", batch_short);
        if (argmax_of(seq_short) != argmax_of(batch_short)) {
            std::cerr << "FAIL: batched-prefill argmax differs from the sequential walk\n";
            return 1;
        }
        if (short_gap > kLeadingLogitTolerance) {
            std::cerr << "FAIL: batched-prefill leading logits diverge at A16 (top5 gap "
                      << short_gap << ")\n";
            return 1;
        }

        // 2. Chunking equivalence at production width. Every split is compared against the same
        // single-chunk walk, so a width-specific route problem shows up as one split standing out.
        const auto whole_long = host_logits(device0.get(), run_prefill(prompt_ids, {0, 300}));
        std::cout << "  batched prefill single chunk: argmax=" << argmax_of(whole_long) << "\n";
        print_top("single", whole_long);
        // A split whose chunks all stay in the same activation-chunk schedule class as the
        // single-chunk walk must reproduce it bit for bit: every row is fed the same operands, so a
        // byte-exact result is the strongest available evidence that the KV pages and the GDN
        // convolution/recurrent state continue across the chunk boundary. A remainder of 44 tokens
        // is a genuinely different problem shape (narrower activation MMA schedule, and its first
        // three columns take the convolution state branch instead of the input taps), so it is held
        // to the sampled token plus bounded leading-logit drift.
        struct SplitCase {
            const char* label;
            std::vector<std::int32_t> cuts;
            bool require_exact;
        };
        const std::vector<SplitCase> split_cases{
            {"300 -> 256+44", {0, 256, 300}, false},
            {"300 -> 128+172", {0, 128, 300}, true},
            {"300 -> 64+236", {0, 64, 300}, true},
        };
        for (const auto& test_case : split_cases) {
            const auto split = host_logits(device0.get(), run_prefill(prompt_ids, test_case.cuts));
            const float gap = top_five_gap(split, whole_long);
            const float drift = max_abs_diff(split, whole_long);
            std::cout << "    chunking " << test_case.label << ": top5_gap=" << gap
                      << " max_logit_diff=" << drift << " argmax=" << argmax_of(split) << "\n";
            print_top("  split ", split);
            if (argmax_of(split) != argmax_of(whole_long)) {
                std::cerr << "FAIL: chunked prefill argmax differs from the single-chunk walk ("
                          << test_case.label << ")\n";
                return 1;
            }
            if (test_case.require_exact ? (drift != 0.0F) : (gap > kLeadingLogitTolerance)) {
                std::cerr << "FAIL: chunked prefill diverges from the single-chunk walk ("
                          << test_case.label << ", top5 gap " << gap << ", max diff " << drift
                          << ")\n";
                return 1;
            }
        }

        // 3. Widest chunk: the engine default is a 1024-token prefill chunk, whose width selects
        //    different attention, convolution, GDN-chunk and activation-schedule routes than the
        //    256-token chunks above. The prompt is the 300-token greedy walk cycled up to 1024
        //    tokens; the reference is the same prompt as four 256-token chunks.
        std::vector<int> wide_ids;
        wide_ids.reserve(1024);
        while (wide_ids.size() < 1024) {
            for (const int id : prompt_ids) {
                if (wide_ids.size() == 1024) { break; }
                wide_ids.push_back(id);
            }
        }
        // Report the arena peak so the engine's workspace bound can be sized from a measurement,
        // not an estimate.
        shard0.workspace->reset_peak();
        const auto wide_split =
            host_logits(device0.get(), run_prefill(wide_ids, {0, 256, 512, 768, 1024}));
        const std::size_t split_peak = shard0.workspace->peak_used();
        shard0.workspace->reset_peak();
        const auto wide_single = host_logits(device0.get(), run_prefill(wide_ids, {0, 1024}));
        const std::size_t single_peak = shard0.workspace->peak_used();
        std::cout << "    prefill workspace peak: 4x256=" << (split_peak >> 20)
                  << " MiB  1x1024=" << (single_peak >> 20) << " MiB\n";
        const float wide_gap = top_five_gap(wide_split, wide_single);
        std::cout << "  batched prefill widest chunk: T=1024 one chunk vs 4x256 top5_gap=" << wide_gap
                  << " max_logit_diff=" << max_abs_diff(wide_split, wide_single)
                  << " argmax_single=" << argmax_of(wide_single)
                  << " argmax_split=" << argmax_of(wide_split) << "\n";
        print_top("single", wide_single);
        print_top("split ", wide_split);
        if (argmax_of(wide_split) != argmax_of(wide_single)) {
            std::cerr << "FAIL: 1024-token chunk argmax differs from the 4x256 split\n";
            return 1;
        }
        if (wide_gap > kLeadingLogitTolerance) {
            std::cerr << "FAIL: 1024-token chunk leading logits diverge (top5 gap " << wide_gap
                      << ")\n";
            return 1;
        }

        // 4. Informational: A4 prefill against the A16 sequential walk.
        const auto seq_long = host_logits(device0.get(), run_sequential(prompt_ids));
        std::cout << "  batched prefill vs sequential walk (A4 vs A16): max_logit_diff="
                  << max_abs_diff(whole_long, seq_long)
                  << " argmax_sequential=" << argmax_of(seq_long)
                  << " argmax_batched=" << argmax_of(whole_long) << "\n";
        print_top("sequential", seq_long);

        std::cout << "TP-2 single-token forward passed: all " << probe_tokens.size()
                  << " probes shard-consistent; " << decode_steps
                  << "-step autoregressive decode shard-consistent; batched prefill consistent with"
                     " the sequential walk and chunk-split invariant\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
