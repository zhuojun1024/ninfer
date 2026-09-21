#include "runtime/engine/tp2_generation_core.h"

#include "artifact/formats.h"
#include "artifact/reader.h"
#include "core/cyclic_kv_cache.h"
#include "core/layout.h"
#include "models/registry.h"
#include "models/qwen3_5/frontend/prepared_prompt.h"
#include "models/qwen3_5/frontend/tool_call_constraint.h"
#include "models/qwen3_5/load.h"
#include "models/qwen3_5/program/context.h"
#include "models/qwen3_5/program/dflash_round.h"
#include "models/qwen3_5/program/planning/graph_profiles.h"
#include "ninfer/ops/argmax.h"
#include "ninfer/ops/position.h"
#include "ninfer/ops/sampling.h"
#include "ninfer/ops/scalar.h"
#include "ninfer/ops/speculative_round.h"
#include "ninfer/ops/token_mask.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::runtime {
namespace {

using namespace ninfer;
namespace qwen = ninfer::models::qwen3_5;
using Clock = std::chrono::steady_clock;

// Phase timing for the TP-2 prefill/decode walk, enabled with NINFER_TP2_TIMING=1. It exists to
// attribute a decode round to its phases while the round is being optimised (see PLAN.md Round 11);
// with the flag off every call is one branch on a cached bool. The marks are recorded on shard A's
// stream and read back after that round's synchronization, so every event has completed and
// cudaEventElapsedTime cannot report a partial interval.
struct Tp2RoundTiming {
    bool enabled = false;
    bool events_ready = false;
    bool created = false;
    cudaEvent_t mark[7]{};
    double mtp_ms     = 0.0;
    double verify_ms  = 0.0;
    double accept_ms  = 0.0;
    double copy_ms    = 0.0;
    double sync_ms    = 0.0;
    double fold_ms    = 0.0;
    double round_ms   = 0.0;
    double prefill_ms = 0.0;
    std::uint64_t rounds    = 0;
    std::uint64_t committed = 0;

    // The events are recorded on shard A's stream only, so they must be created with shard A's
    // device current: an event from another device makes cudaEventRecord fail with
    // cudaErrorInvalidResourceHandle, and because these paths do not check CUDA errors that latched
    // failure would be reported by the next unrelated CUDA_CHECK in the round.
    void init() {
        if (created) { return; }
        created = true;
        const char* env = std::getenv("NINFER_TP2_TIMING");
        enabled         = env != nullptr && env[0] == '1';
        if (!enabled) { return; }
        for (cudaEvent_t& event : mark) {
            if (cudaEventCreateWithFlags(&event, cudaEventDefault) != cudaSuccess) {
                (void)cudaGetLastError();
                enabled = false;
                return;
            }
        }
        events_ready = true;
    }

    // Drop a latched failure from a timing call itself: this facility must never change what the
    // engine reports, and every diagnostic call failure means "report nothing" for that interval.
    static void discard_error() { (void)cudaGetLastError(); }

    void reset() {
        mtp_ms = verify_ms = accept_ms = copy_ms = sync_ms = fold_ms = round_ms = prefill_ms = 0.0;
        rounds    = 0;
        committed = 0;
    }

    void record(int index, cudaStream_t stream) {
        if (!events_ready) { return; }
        if (cudaEventRecord(mark[index], stream) != cudaSuccess) {
            discard_error();
            events_ready = false;
        }
    }

    [[nodiscard]] double elapsed(int begin, int end) const {
        if (!events_ready) { return 0.0; }
        float ms = 0.0F;
        if (cudaEventElapsedTime(&ms, mark[begin], mark[end]) != cudaSuccess) {
            discard_error();
            return 0.0;
        }
        return ms;
    }

    // Called once per round after that round's synchronization: every mark up to the licenced-token
    // copy has completed, so the intervals are final. The fold runs after the synchronization and is
    // accounted separately by the caller.
    void close_round(double sync_wait_ms) {
        if (!events_ready) { return; }
        mtp_ms += elapsed(0, 1);
        verify_ms += elapsed(1, 2);
        accept_ms += elapsed(2, 3);
        copy_ms += elapsed(3, 4);
        sync_ms += sync_wait_ms;
    }

    void report(const char* label) const {
        if (!enabled || rounds == 0) { return; }
        const double n = static_cast<double>(rounds);
        std::fprintf(stderr,
                     "[tp2-time] %s rounds=%llu committed=%llu avg_round=%.2fms mtp=%.2f verify=%.2f "
                     "accept=%.2f copy=%.2f sync_wait=%.2f fold=%.2f prefill=%.1fms\n",
                     label, static_cast<unsigned long long>(rounds),
                     static_cast<unsigned long long>(committed), round_ms / n, mtp_ms / n,
                     verify_ms / n, accept_ms / n, copy_ms / n, sync_ms / n, fold_ms / n,
                     prefill_ms);
    }
};

Tp2RoundTiming& tp2_timing() {
    static Tp2RoundTiming timing;
    return timing;
}


// Workspace arena per shard. The single-token forward_tp2 peaks at ~600 KB (full-vocab logits
// plus a handful of [N,1] activations), but a batched prefill chunk holds every intermediate of the
// widest layer at once: at the maximum chunk width the fused FFN gate/up [17408, T] BF16 alone is
// ~36 MiB and the per-layer peak is ~140 MiB. 192 MiB covers the largest chunk with headroom while
// keeping the per-shard steady state (weights 10.7 GiB + KV + GDN state + snapshots + workspace)
// under the 16 GB card limit. The context ceiling is set by the KV pool, so this budget is spent
// exactly: at 262,144 tokens the 128 MiB a 384 MiB arena would hold is the difference between
// fitting the card and not, and the oversized arena overflow is a reported error, not corruption.
constexpr std::size_t kWorkspaceBytes = 192ULL << 20;
// Smallest Vision item ceiling the route will fall back to when the full envelope does not fit
// beside the KV pool: below this an ordinary photo would no longer fit in one item, so the route
// reports the failure instead of silently admitting only thumbnails.
constexpr std::uint32_t kVisionItemTokenFloor = 2048;

// TP-2 keeps one resident conversation plus this many host-resident ones unless
// --max-private-continuations overrides it. The host KV budget is the real bound on how many fit.
constexpr std::uint32_t kTp2DefaultSessions = 6;

// Upper bound on the batched TP-2 prefill chunk width. One chunk reads every weight once, so the
// per-token weight traffic that dominates the single-token walk is amortized over the whole chunk;
// the cross-device allreduce bytes per token and the tensor-core work per token do not shrink with
// the chunk, so widening the chunk trades only the weight term against the per-chunk activation
// peak. The engine option (--prefill-chunk) picks the actual width.
constexpr std::uint32_t kPrefillChunkMaximum = 1024;
// Finest spacing between prefix-reuse checkpoints in pinned host memory, in prompt tokens. It
// matches the checkpoint step llama.cpp uses by default. A ring of N slots widens it to
// ceil(max_context / N) so that N * stride always covers the whole context; a prompt that diverges
// inside the previous prompt then restarts at most one stride behind its shared prefix.
constexpr std::uint32_t kReuseCheckpointStride = 8192;
// The last few chunk ends of a walk are checkpointed on top of the stride grid. A chat client
// re-renders the assistant turn it is about to continue, so the divergence a turn boundary shows
// sits just behind the previous prompt's end - observed gaps run from ~150 tokens to tens of
// thousands. The grid alone only guarantees a checkpoint within one stride of it; this window
// makes the common small gap land within one chunk. llama.cpp calls the same idea a "near prompt
// end" checkpoint (its checkpoint-min-step does not apply there either).
constexpr std::uint32_t kReuseTailWindow = 8192;
// Host ring slots reserved for those tail anchors. Without the reservation a long session fills the
// ring with them (each prefill leaves up to one per chunk) and evicts the position grid, which is
// exactly the coverage a divergence deep inside the history needs.
constexpr std::uint32_t kReuseTailCheckpointCount = 8;
// Host ring slots reserved for the divergence anchor: the position where the last prefill's prompt
// stopped matching the lineage it inherited. A new session - and a context compression - shares only
// the client's stable system prompt and tool definitions with the conversation before it, so its
// divergence lands well inside the previous prompt. Neither other group reaches there: the position
// grid's first checkpoint sits a whole stride in, and a small ring widens that stride to tens of
// thousands of tokens; the tail anchors sit at the opposite end. One slot is enough because a client
// renders its stable prefix once, and because the ring prunes on a single lineage: a checkpoint past
// the newest shared prefix is dead for good, so a second slot could never hold a rival prefix.
constexpr std::uint32_t kReuseDivergenceCheckpointCount = 1;
// How far before the observed divergence the anchor is placed. A shared prefix is only reusable
// while every later prompt agrees on every token before it, and the index where two prompts first
// differ is where the tokenizer stopped emitting the same ids - the token that straddles the stable
// block's end absorbs a different amount of what follows it in each pair. The next request that
// renders the same block can therefore report a shared prefix one or two tokens shorter than the one
// that froze the anchor, and the prune above would then discard it. The margin trades single-digit
// tokens of recompute for keeping the whole prefix.
constexpr std::uint32_t kReuseDivergenceMargin = 8;

std::uint32_t pages_for_tokens(std::uint32_t tokens) noexcept {
    return tokens == 0 ? 0U : 1U + (tokens - 1U) / static_cast<std::uint32_t>(kPagedKVPageSize);
}

// WSL2 re-enumerates GPUs on every VM restart, so the indices passed via --devices can
// silently point at the wrong cards (e.g. the Tesla T10 instead of a second RTX 5060 Ti).
// The engine is sm_120a-only, so a mismatched card fails deep inside the first kernel launch
// with a cryptic cudaErrorSymbolNotFound. Fail fast at construction with an actionable message.
// '--devices' takes CUDA runtime indices. The default CUDA_DEVICE_ORDER is fastest-first, which
// need not match the PCI-bus order that 'nvidia-smi -L' prints, so the actionable form of the hint
// is the CUDA-visible device list itself.
std::string cuda_visible_devices() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) { return {}; }
    std::string out;
    for (int i = 0; i < count; ++i) {
        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, i) != cudaSuccess) { continue; }
        if (!out.empty()) { out += ", "; }
        out += std::to_string(i) + "=" + prop.name + " (sm_" +
               std::to_string(prop.major * 10 + prop.minor) + ")";
    }
    return out;
}

void validate_tp2_devices(const DeviceContext& a, const DeviceContext& b) {
    const int cap_a = a.compute_capability();
    const int cap_b = b.compute_capability();
    if (cap_a != 120 || cap_b != 120) {
        throw std::invalid_argument(
            "TP-2 requires two sm_120a devices (compute capability 12.0); device " +
            std::to_string(a.device) + " (" + a.props.name + ") is sm_" + std::to_string(cap_a) +
            " and device " + std::to_string(b.device) + " (" + b.props.name + ") is sm_" +
            std::to_string(cap_b) +
            ". WSL2 re-enumerates GPUs across VM restarts - pass the two RTX 5060 Ti indices via "
            "--devices. CUDA-visible devices: " +
            cuda_visible_devices());
    }
    if (std::strcmp(a.props.name, b.props.name) != 0) {
        throw std::invalid_argument(
            "TP-2 requires identical devices; device " + std::to_string(a.device) + " is " +
            a.props.name + " and device " + std::to_string(b.device) + " is " + b.props.name +
            ". Pass two identical devices via --devices. CUDA-visible devices: " +
            cuda_visible_devices());
    }
}

ops::SamplingConfig make_sampling_config(const ResolvedSamplingParameters& source) {
    ops::SamplingConfig out;
    out.temperature       = source.temperature;
    out.top_k             = source.top_k;
    out.top_p             = source.top_p;
    out.min_p             = source.min_p;
    out.presence_penalty  = source.presence_penalty;
    out.frequency_penalty = source.frequency_penalty;
    out.seed              = source.seed;
    out.token_counts      = nullptr;
    return out;
}

// Moves a workspace arena to an exact offset. A verify graph bakes the addresses of the round's
// scratch, so a replay must reproduce them rather than hope two rounds happened to allocate the
// same amount. The arena never moves below `floor`, the watermark the current request keeps alive.
void position_arena(WorkspaceArena& arena, std::size_t floor, std::size_t target) {
    if (target < floor) {
        throw std::logic_error("TP-2 verify CUDA Graph was captured below this request's workspace");
    }
    const std::size_t used = arena.used();
    if (target < used) {
        arena.rewind(target);
    } else if (target > used) {
        (void)arena.alloc_bytes(target - used);
    }
}

} // namespace

TP2GenerationCore::TP2GenerationCore(const EngineOptions& options, int device_a, int device_b)
    : options_(options), pair_(device_a, device_b) {
    const Clock::time_point load_start = Clock::now();
    shard_a_.device = DeviceContext(device_a);
    shard_b_.device = DeviceContext(device_b);
    validate_tp2_devices(shard_a_.device, shard_b_.device);

    models::LoadOptions load;
    load.vision        = options.enable_vision;
    load.speculative   = options.speculative.backend;
    load.proposal_head = options.speculative.proposal_head;

    mtp_enabled_ = options.speculative.backend == SpeculativeBackend::Mtp;
    if (mtp_enabled_) {
        mtp_drafts_ = std::clamp<std::uint32_t>(options.speculative.draft_tokens, 1U, 5U);
    }
    dflash2_enabled_ = options.speculative.backend == SpeculativeBackend::DFlash2;
    if (dflash2_enabled_) {
        dflash_drafts_ = std::clamp<std::uint32_t>(options.speculative.draft_tokens, 1U,
                                                   qwen::kDFlashDecodeMaximumDrafts);
    }

    artifact::Reader reader(options.artifact_path);
    auto plan = qwen::plan_load(reader, load);
    auto [model_a, model_b] =
        qwen::materialize_model_tp2(std::move(plan), shard_a_.device, shard_b_.device,
                                    &options.startup_observer);
    shard_a_.model = std::move(model_a);
    shard_b_.model = std::move(model_b);

    shard_a_.parameters = std::make_unique<qwen::execution::Parameters>(*shard_a_.model);
    shard_b_.parameters = std::make_unique<qwen::execution::Parameters>(*shard_b_.model);

    // Prefix-reuse checkpoints in pinned host memory: the same budget the single-device context
    // cache uses for its complete host state images. Each checkpoint is one state image per shard
    // (~73 MiB), so the ring is enough to cover the whole context at kReuseCheckpointStride or
    // coarser; a zero budget keeps the ring disabled, which is what --no-prefix-reuse selects.
    {
        const std::uint32_t host_slots = options_.context_cache.host_state_slots;
        if (host_slots != 0) {
            host_checkpoint_tail_slots_ =
                std::max(1U, std::min(kReuseTailCheckpointCount, host_slots / 2U));
            host_checkpoint_divergence_slots_ = kReuseDivergenceCheckpointCount;
            const std::uint32_t grid_slots =
                std::max(1U, host_slots - host_checkpoint_tail_slots_);
            const std::uint32_t per_slot = (options_.max_context + grid_slots - 1U) / grid_slots;
            const std::uint32_t rounded  = (per_slot + 127U) / 128U * 128U;
            host_checkpoint_stride_      = std::max(kReuseCheckpointStride, rounded);
        }
        // The masked-draft route cannot reuse a prefix yet: the draft's local context ring is not
        // part of the device snapshot, the host checkpoint ring or the session catalog (PLAN.md
        // section 3.6, stage B6). Until it is, a reused prefix would leave the ring describing the
        // wrong positions, so the route forfeits reuse rather than publish a draft context it cannot
        // vouch for. The ring is disabled too, since nothing can consume it.
        if (dflash2_enabled_) { host_checkpoint_stride_ = 0; }
    }

    // Cross-session retention budget. --host-kv-mib is the whole host KV budget, split evenly
    // between the two shards; the arenas that back it are built on the first eviction, so a workload
    // that never leaves one conversation never allocates it. The backing is pageable unless
    // --host-kv-pinned asks for it. A zero budget keeps the catalog empty, which is exactly the
    // pre-retention behaviour.
    {
        const std::size_t host_kv_bytes = options_.context_cache.host_kv_capacity_bytes;
        const std::uint32_t sessions =
            options_.context_cache.max_private_continuations.value_or(kTp2DefaultSessions);
        // One entry is the resident conversation; retention needs room for at least one more.
        // DFlash2 disables cross-session retention for the same reason it disables reuse: a
        // recalled session would restore the target KV and GDN state but not the draft ring.
        if (host_kv_bytes / 2 != 0 && sessions > 1 && !dflash2_enabled_) {
            host_kv_shard_bytes_ = host_kv_bytes / 2;
            session_capacity_    = sessions;
        }
        std::fprintf(stderr,
                     "[mem] host sessions capacity %zu | host KV %.1f MiB/shard | retention %s\n",
                     session_capacity_, static_cast<double>(host_kv_shard_bytes_) / 1048576.0,
                     session_capacity_ == 0 ? "disabled" : "enabled");
    }

    build_shard(shard_a_, 0);
    build_shard(shard_b_, 1);
    if (session_capacity_ != 0) {
        // Report the budget in the unit an operator sizes against: full-context conversations per
        // shard. The MTP layer's own slab is not counted, so shard 0 holds slightly fewer.
        const HostKVPageLayout layout =
            plan_host_kv_page_layout(shard_a_.decoder->text_kv.page_pool().geometry());
        const std::size_t full_session =
            layout.page_stride * pages_for_tokens(options_.max_context);
        const std::size_t fits = full_session == 0 ? 0 : host_kv_shard_bytes_ / full_session;
        std::fprintf(stderr,
                     "[tp2-session] host budget %.1f MiB/shard holds %zu of %zu full-context "
                     "sessions (%.1f MiB each)\n",
                     static_cast<double>(host_kv_shard_bytes_) / 1048576.0, fits,
                     session_capacity_ - 1,
                     static_cast<double>(full_session) / 1048576.0);
    }
    // Column-split weights (the token embedding) are consumed by operations that run on one shard
    // alone, so both contexts learn their peer and the pair once both shards exist.
    shard_a_.context->set_tp_peer(shard_b_.context.get(), &pair_);
    shard_b_.context->set_tp_peer(shard_a_.context.get(), &pair_);

    if (mtp_enabled_ && pair_.in_kernel_allreduce()) {
        const char* env        = std::getenv("NINFER_TP2_VERIFY_GRAPH");
        verify_graph_enabled_  = env == nullptr || env[0] != '0';
    }
    if (mtp_enabled_ || dflash2_enabled_) {
        // The verify window is assembled in this portable pinned buffer every round: the capture
        // path reads it through a memcpy node, and the eager path reads it directly. MTP assembles
        // it from the host proposal chain; DFlash2 reads the verify ids/positions the proposal
        // published back from the frame. Both use the same [ids(W), positions(W)] layout.
        const std::uint32_t width = (mtp_enabled_ ? mtp_drafts_ : dflash_drafts_) + 1U;
        verify_window_host_ = std::make_unique<PinnedHostBuffer>(
            static_cast<std::size_t>(2) * width * sizeof(std::int32_t), true);
    }
    if (mtp_enabled_) {
        // The envelope buckets are the single-GPU MTP decode graph's split-policy boundaries, so a
        // window always covers the same attention route and launch geometry. The bucket is keyed by
        // the window's widest visible extent, which is what the envelope carries; both the capture
        // and the eager reference use them, so the two differ only in how they are launched.
        const std::uint32_t window_width = mtp_drafts_ + 1U;
        for (const auto& profile :
             qwen::detail::mtp_graph_profiles(options_.max_context, mtp_drafts_)) {
            WindowGraph graph;
            graph.visible_begin = profile.min + 1U;
            graph.visible_end = std::min(options_.max_context, profile.max + window_width);
            verify_graphs_.push_back(std::move(graph));
        }
    }

    // The plain (non-speculative) decode step is the same kind of object as the verify window: one
    // token, the same launch sequence per round, and its only per-round inputs are the decoded token,
    // its absolute position and the attention envelope. Its buckets are therefore the ordinary
    // (one-token visible window) split-policy boundaries, exactly the set the single-GPU ordinary
    // decode graph uses. The step contains the pair's allreduce, so it may only be captured when
    // that transport is the in-kernel one: the P2P and host-staging transports synchronize the
    // caller's streams, which a captured sequence cannot express.
    {
        const char* env = std::getenv("NINFER_TP2_DECODE_GRAPH");
        if (env != nullptr && std::strcmp(env, "0") == 0) {
            decode_step_mode_ = DecodeStepMode::EagerBucket;
        } else if (env != nullptr && std::strcmp(env, "exact") == 0) {
            decode_step_mode_ = DecodeStepMode::EagerExact;
        }
        if (!pair_.in_kernel_allreduce()) { decode_step_mode_ = DecodeStepMode::EagerExact; }
        if (decode_step_mode_ != DecodeStepMode::EagerExact) {
            for (const auto& profile : qwen::detail::ordinary_graph_profiles(options_.max_context)) {
                WindowGraph graph;
                graph.visible_begin = profile.min + 1U;
                graph.visible_end   = std::min(options_.max_context, profile.max + 1U);
                decode_graphs_.push_back(std::move(graph));
            }
        }
        decode_window_host_ = std::make_unique<PinnedHostBuffer>(
            static_cast<std::size_t>(2) * sizeof(std::int32_t), true);
        std::fprintf(stderr, "[tp2-graph] plain decode step: %s | verify step: %s\n",
                     decode_step_mode_ == DecodeStepMode::Graph        ? "graph"
                     : decode_step_mode_ == DecodeStepMode::EagerBucket ? "eager(bucket)"
                                                                        : "eager(exact)",
                     verify_graph_enabled_ ? "graph" : "eager");
    }

    frontend_ = std::make_unique<qwen::Frontend>(
        qwen::make_frontend(shard_a_.model->resources(),
                            {.architecture           = shard_a_.model->config().text.architecture,
                             .vision_enabled         = options.enable_vision,
                             .max_context            = options.max_context,
                             .media_cache_bytes      = options.media_cache_bytes,
                             .media_live_bytes       = options.media_live_bytes,
                             .media_preprocess_threads = options.media_preprocess_threads}));
    load_seconds_ = std::chrono::duration<double>(Clock::now() - load_start).count();
}

TP2GenerationCore::~TP2GenerationCore() {
    // The catalog's host KV slabs are suballocations of the host arenas, so the entries have to be
    // released before the arenas they were taken from.
    sessions_.clear();
}

void TP2GenerationCore::build_shard(Shard& shard, int shard_index) {
    const auto& config = shard.model->config().text;
    const std::uint32_t capacity = options_.max_context;
    // Multi-token prediction runs on shard 0 (see the class comment): only that shard plans and
    // materializes the MTP layer's own KV cache.
    const bool mtp_shard = mtp_enabled_ && shard_index == 0;

    // Per-shard config: the mixer head counts are halved (each shard owns half the attention/GDN
    // heads), so the KV cache and GDN state are sized for the per-shard geometry while the
    // replicated components (embeddings, norms, lm_head) keep the full-model config.
    auto shard_config = std::make_unique<models::qwen3_5::TextConfig>(config);
    if (shard_config->attention) {
        shard_config->attention->num_attention_heads /= 2;
        shard_config->attention->num_key_value_heads /= 2;
    }
    if (shard_config->gdn) {
        shard_config->gdn->linear_num_key_heads   /= 2;
        shard_config->gdn->linear_num_value_heads /= 2;
    }
    shard.shard_config = std::move(shard_config);
    const auto& scfg = *shard.shard_config;

    // Paged KV cache: full-attention layers only, one execution-table row (single request).
    // The MTP layer appends its own K/V up to `mtp_drafts_` positions past the committed frontier
    // (the speculative window), so its pool carries that many tokens of extra physical pages; its
    // execution row only maps the logical capacity.
    // Everything resident before the pool: the CUDA context and this shard's weight share.
    double resident_bytes = 0.0;
    const std::uint32_t mtp_physical_pages =
        mtp_shard
            ? pages_for_tokens(capacity) +
                  (mtp_drafts_ == 0 ? 0U : 1U + (mtp_drafts_ - 1U) / kPagedKVPageSize)
            : 0U;
    LayoutBuilder kv_builder;
    const qwen::DecoderStateLayout kv_layout = qwen::plan_decoder_state(
        kv_builder,
        qwen::DecoderStateSpec{
            .full_attention_layers = scfg.full_attention_layers,
            .mtp_layers            = 1,
            .capacity              = capacity,
            .kv_heads              = qwen::execution::dimension(scfg.attention->num_key_value_heads),
            // The MTP layer is replicated (not head-split), so its cache holds the full head count.
            .mtp_kv_heads =
                qwen::execution::dimension(config.attention->num_key_value_heads),
            .attention_head_dim    = qwen::execution::dimension(scfg.attention->head_dim),
            .kv_storage            = options_.kv_cache,
            .enable_mtp            = mtp_shard,
            .kv_table_rows         = 1,
            .text_physical_page_groups = pages_for_tokens(capacity),
            .mtp_physical_page_groups  = mtp_physical_pages,
        });
    const std::size_t kv_bytes = kv_builder.finish(256);
    shard.device.bind_to_current_thread();
    {
        std::size_t free_bytes = 0, total_bytes = 0;
        cudaMemGetInfo(&free_bytes, &total_bytes);
        resident_bytes = static_cast<double>(total_bytes - free_bytes) / 1048576.0;
    }
    shard.kv_arena = std::make_unique<DeviceArena>(kv_bytes);
    shard.decoder  = std::make_unique<qwen::DecoderState>(shard.kv_arena->alloc_bytes(kv_bytes, 256),
                                                          kv_layout);

    // GDN state pool: linear-attention layers, one slot, zeroed. Sized for the per-shard
    // geometry (half the value heads and conv channels); each shard owns its local heads.
    const LinearAttentionStatePoolSpec gdn_spec{
        .layers         = scfg.linear_attention_layers,
        .conv_channels  = (scfg.gdn ? qwen::execution::dimension(scfg.gdn->conv_channels()) : 0),
        .conv_width     = (scfg.gdn ? qwen::execution::dimension(scfg.gdn->linear_conv_kernel_dim - 1) : 0),
        .value_heads    = (scfg.gdn ? qwen::execution::dimension(scfg.gdn->linear_num_value_heads) : 0),
        .value_head_dim = (scfg.gdn ? qwen::execution::dimension(scfg.gdn->linear_value_head_dim) : 0),
        .key_head_dim   = (scfg.gdn ? qwen::execution::dimension(scfg.gdn->linear_key_head_dim) : 0),
        .slot_count     = 1,
        .conv_dtype     = DType::BF16,
    };
    LayoutBuilder state_builder;
    const LinearAttentionStatePoolLayout state_layout =
        plan_linear_attention_state_pool(state_builder, gdn_spec);
    const std::size_t state_bytes = state_builder.finish(256);
    shard.device.bind_to_current_thread();
    // Live linear-attention pool (two buffers), one prefix-reuse snapshot per slot, and the MTP
    // round scratch (~77 MiB per plane).
    shard.state_arena   = std::make_unique<DeviceArena>((2 + kReuseSnapshotCount) * state_bytes);
    shard.state_backing = shard.state_arena->alloc_bytes(state_bytes, 256);
    for (auto& snapshot : shard.state_snapshots) {
        snapshot = shard.state_arena->alloc_bytes(state_bytes, 256);
    }
    CUDA_CHECK(cudaMemset(shard.state_backing.data, 0, state_bytes));
    shard.state = std::make_unique<LinearAttentionStatePool>(shard.state_backing, state_layout);
    if (host_checkpoint_stride_ != 0) {
        // Portable pinned memory, like the MTP verify window: every checkpoint is read and written
        // through this shard's own device, but a portable allocation keeps that true if the
        // execution context ever binds the peer first.
        const std::uint32_t slots = options_.context_cache.host_state_slots +
                                    host_checkpoint_divergence_slots_;

        shard.host_checkpoint_grid_slots = options_.context_cache.host_state_slots -
                                           host_checkpoint_tail_slots_;
        shard.host_checkpoints.reserve(slots);
        for (std::uint32_t index = 0; index < slots; ++index) {
            Shard::HostCheckpoint checkpoint;
            checkpoint.buffer = std::make_unique<PinnedHostBuffer>(shard.state_backing.bytes, true);
            shard.host_checkpoints.push_back(std::move(checkpoint));
        }
    }

    std::size_t record_bytes = 0;
    std::size_t round_bytes  = 0;
    if (mtp_enabled_ || dflash2_enabled_) {
        // ReplaySSM records for one verify window wide, one physical row, per-shard GDN geometry.
        // Both shards verify the window, so both need their own records and fold plan.
        // The verify forward records here instead of advancing the live state, and the fold replays
        // the accepted prefix back into it (in place: with a single row the Op allows it, so the
        // state pool still needs one slot). MTP and DFlash2 use the same mechanism; only the window
        // width differs.
        LayoutBuilder record_builder;
        const GdnReplayRecordLayout record_layout = plan_gdn_replay_records(
            record_builder,
            GdnReplayRecordSpec{
                .layers          = qwen::execution::dimension(scfg.linear_attention_layers),
                .record_capacity = 1,
                .width           = static_cast<std::int32_t>(mtp_enabled_ ? mtp_drafts_ : dflash_drafts_) + 1,
                .conv_channels =
                    (scfg.gdn ? qwen::execution::dimension(scfg.gdn->conv_channels()) : 0),
                .qk_heads =
                    (scfg.gdn ? qwen::execution::dimension(scfg.gdn->linear_num_key_heads) : 0),
                .value_heads =
                    (scfg.gdn ? qwen::execution::dimension(scfg.gdn->linear_num_value_heads) : 0),
                .key_dim =
                    (scfg.gdn ? qwen::execution::dimension(scfg.gdn->linear_key_head_dim) : 0),
                .value_dim =
                    (scfg.gdn ? qwen::execution::dimension(scfg.gdn->linear_value_head_dim) : 0),
            });
        record_bytes = record_builder.finish(256);
        shard.record_arena = std::make_unique<DeviceArena>(record_bytes);
        shard.device.bind_to_current_thread();
        DeviceSpan record_backing = shard.record_arena->alloc_bytes(record_bytes, 256);
        shard.records             = GdnReplayRecords(record_backing, record_layout);
        shard.replay_fold =
            std::make_unique<ops::GdnReplayFoldPlan>(shard.records, shard.state->all_layers_view());
    }

    // Workspace arena: ample for the single-token forward (full-vocab logits plus a handful of
    // [N,1] activations) and the sampling workspace.
    shard.device.bind_to_current_thread();
    shard.workspace = std::make_unique<DeviceArena>(kWorkspaceBytes);
    shard.prefill_hidden = shard.workspace->alloc(DType::BF16,
                                                  {2 * qwen::execution::dimension(config.hidden_size), 1});
    if (mtp_shard) {
        // Survives every round scope: the first round's MTP bridge consumes the last prompt column's
        // final-norm hidden, captured during prefill priming.
        shard.mtp_anchor_hidden =
            shard.workspace->alloc(DType::BF16, {qwen::execution::dimension(config.hidden_size), 1});
    }
    // Vision (--vision) runs on the shard that materialized the Vision component: the static split
    // that keeps the MTP layer on shard 0 puts the Vision tower and its encode/handoff workspace on
    // shard 1. The workspace is planned by the same planner the single-device route uses, with this
    // shard's text workspace capacity as the general extent so the item handoff region lands above
    // every text activation, and it is owned by a dedicated arena that outlives every request.
    std::size_t vision_bytes = 0;
    if (options_.enable_vision && shard_index == 1) {
        if (!shard.model->config().vision || !shard.parameters->vision) {
            throw std::invalid_argument("TP-2 vision shard has no Vision component parameters");
        }
        // The Vision workspace is the one resident block whose extent follows the largest image the
        // route admits, and it is what decides whether the tower fits beside the 262,144-token KV
        // pool. Its item ceiling is therefore selected by what this device actually accepts: the plan
        // is rebuilt at a halved ceiling until cudaMalloc succeeds. The driver's own free-memory
        // reading is not usable here (on this WSL2 host cudaMemGetInfo under-reports free bytes by
        // about a gigabyte), so a refused allocation is the only honest signal. The ceiling that
        // survives is reported below and bounds what a request may ask for.
        const auto& vision_config     = *shard.model->config().vision;
        const auto& vision_parameters = *shard.parameters->vision;
        std::uint32_t max_item        = static_cast<std::uint32_t>(
            std::min<std::uint64_t>(capacity, qwen::kMaximumVisionItemTokens));
        for (;;) {
            vision_workspace_ = qwen::execution::VisionContext::plan_workspace(
                vision_config, vision_parameters, max_item, kWorkspaceBytes);
            shard.device.bind_to_current_thread();
            try {
                vision_arena_ = std::make_unique<DeviceArena>(vision_workspace_->capacity_bytes);
                break;
            } catch (const std::runtime_error&) {
                vision_arena_.reset();
                vision_workspace_.reset();
                // A refused cudaMalloc leaves cudaErrorMemoryAllocation latched in the runtime's
                // last-error slot, where the next kernel launch's check would report it as that
                // launch's own failure. Read it once here so the retry starts from a clean state.
                (void)cudaGetLastError();
                if (max_item <= kVisionItemTokenFloor) { throw; }
                max_item = std::max<std::uint32_t>(kVisionItemTokenFloor, max_item / 2);
            }
        }
        vision_bytes = vision_workspace_->capacity_bytes;
        std::fprintf(stderr,
                     "[mem] vision shard %d item ceiling %u | encode peak %.1f | handoff %.1f | "
                     "arena %.1f MiB\n",
                     shard_index, max_item,
                     static_cast<double>(vision_workspace_->encode_peak_bytes) / 1048576.0,
                     static_cast<double>(vision_workspace_->handoff_capacity_bytes) / 1048576.0,
                     static_cast<double>(vision_bytes) / 1048576.0);
    }
    // DFlash2 masked-draft round: the draft component is materialized whole on shard 0 (the split
    // spec keeps dflash2/* shard-local), so only that shard pays for the target-feature staging, the
    // draft's own local K/V and the round's proposal workspace. The round owns all of it
    // (program/dflash_round.h) so the production assembly is exactly the one the loader-only test
    // drives. One prefill chunk of features is enough because TP-2 clamps its chunk to
    // kPrefillChunkMaximum (1024) while the draft's local window is 2048, so no prefill ever stages
    // more than one window's worth of target residual.
    if (shard.parameters->draft.has_value()) {
        const auto& draft = *shard.model->config().draft;
        if (!draft.dflash2.has_value()) {
            // The single-device DFlash v1 draft materializes its context through the same state,
            // but through the per-layer path (and a full KV layer) that this core does not own.
            throw std::logic_error("TP-2 masked draft requires the DFlash2 context layout");
        }
        const std::uint32_t feature_columns = std::min<std::uint32_t>(
            std::max<std::uint32_t>(options_.prefill_chunk, 64U), kPrefillChunkMaximum);
        const std::uint32_t draft_window = options_.speculative.draft_tokens;
        // The proposal gets an arena of its own, sized by the planner the single-device route uses.
        // It is deliberately not carved out of the shard workspace: propose_dflash2_batch resets the
        // arena it is handed, and that workspace holds the resident prefill hidden state.
        const std::size_t proposal_bytes = qwen::execution::dflash2_proposal_workspace_bytes(
            *shard.parameters, config, capacity, ProposalHead::Full,
            static_cast<std::int32_t>(draft_window) + 1, 1);
        const qwen::execution::DFlash2RoundSpec dflash_spec = qwen::execution::plan_dflash2_round(
            draft, config, feature_columns, draft_window, proposal_bytes);
        shard.device.bind_to_current_thread();
        shard.dflash_round =
            std::make_unique<qwen::execution::DFlash2Round>(shard.device, dflash_spec);
        std::fprintf(stderr,
                     "[mem] shard %d DFlash2 round | targets %d x %d | ring %.1f | context %.1f | "
                     "frame %.1f | proposal %.1f MiB\n",
                     shard_index, dflash_spec.target_features, dflash_spec.feature_columns,
                     static_cast<double>(shard.dflash_round->ring_payload_bytes()) / 1048576.0,
                     static_cast<double>(shard.dflash_round->context_bytes()) / 1048576.0,
                     static_cast<double>(shard.dflash_round->frame_bytes()) / 1048576.0,
                     static_cast<double>(proposal_bytes) / 1048576.0);
    }
    // One startup ledger line per shard: every resident block is allocated before the first
    // request, so this is the whole device budget at the requested context ceiling.
    {
        std::size_t free_bytes = 0, total_bytes = 0;
        cudaMemGetInfo(&free_bytes, &total_bytes);
        std::fprintf(stderr,
                     "[mem] shard %d capacity %u | weights+ctx %.1f | kv %.1f | state %.1f | "
                     "record %.1f | round %.1f | workspace %u.0 | vision %.1f | free %.1f of %.1f "
                     "MiB\n",
                     shard_index, capacity, resident_bytes,
                     static_cast<double>(kv_bytes) / 1048576.0,
                     static_cast<double>((2 + kReuseSnapshotCount) * state_bytes) / 1048576.0,
                     static_cast<double>(record_bytes) / 1048576.0,
                     static_cast<double>(round_bytes) / 1048576.0,
                     static_cast<unsigned>(kWorkspaceBytes >> 20),
                     static_cast<double>(vision_bytes) / 1048576.0,
                     static_cast<double>(free_bytes) / 1048576.0,
                     static_cast<double>(total_bytes) / 1048576.0);
        if (!shard.host_checkpoints.empty()) {
            std::fprintf(stderr,
                         "[mem] host-checkpoints shard %d slots %zu (grid %zu + tail %u + "
                         "divergence %u) x %.1f MiB | stride %u tok | pinned %.1f MiB\n",
                         shard_index, shard.host_checkpoints.size(),
                         shard.host_checkpoint_grid_slots, host_checkpoint_tail_slots_,
                         host_checkpoint_divergence_slots_,
                         static_cast<double>(shard.state_backing.bytes) / 1048576.0,
                         host_checkpoint_stride_,
                         static_cast<double>(shard.state_backing.bytes *
                                             shard.host_checkpoints.size()) / 1048576.0);
        }
    }

    // The execution context is constructed at the end of this function: the MTP layer needs its KV
    // execution view, which only exists after the page list above is materialized.

    // Materialize the full KV page list once at startup. The pool is a fixed physical allocation
    // reused in place across requests (max_concurrency 1, single execution row), so pinning the
    // page leases and execution row 0 here means each request only re-zeros the GDN state; there
    // is no per-request reserve/allocate churn (which would leak the pool's capacity after the
    // first request). The block-table mapping is identical every request, so publishing once
    // suffices.
    {
        auto& pool   = shard.decoder->text_kv.page_pool();
        auto& tables = shard.decoder->text_kv.execution_tables();
        const std::uint32_t pages = pages_for_tokens(capacity);
        shard.device.bind_to_current_thread();
        auto reserved = pool.reserve(pages);
        if (!reserved.has_value()) { throw std::logic_error("TP-2 KV page reservation failed"); }
        DeviceKVPageReservation reservation = std::move(*reserved);
        shard.kv_pages.reserve(pages);
        pool.materialize(reservation, pages, shard.kv_pages);
        shard.kv_page_handles.clear();
        shard.kv_page_handles.reserve(shard.kv_pages.size());
        for (const auto& lease : shard.kv_pages) { shard.kv_page_handles.push_back(lease.handle()); }
        shard.kv_row = tables.acquire(0);
        tables.publish(shard.kv_row.handle(), 0, shard.kv_page_handles, shard.device.stream);
    }

    // The MTP layer's own attention context. Same fixed-page treatment as the text cache: one
    // physical page list, one execution row, published once.
    if (mtp_shard) {
        auto* cache = shard.decoder->mtp_cache();
        if (cache == nullptr) { throw std::logic_error("TP-2 MTP KV cache was not planned"); }
        auto& pool   = cache->page_pool();
        auto& tables = cache->execution_tables();
        const std::uint32_t logical_pages = pages_for_tokens(capacity);
        shard.device.bind_to_current_thread();
        auto reserved = pool.reserve(mtp_physical_pages);
        if (!reserved.has_value()) {
            throw std::logic_error("TP-2 MTP KV page reservation failed");
        }
        DeviceKVPageReservation reservation = std::move(*reserved);
        shard.mtp_pages.reserve(mtp_physical_pages);
        pool.materialize(reservation, mtp_physical_pages, shard.mtp_pages);
        shard.mtp_page_handles.clear();
        shard.mtp_page_handles.reserve(logical_pages);
        for (std::uint32_t i = 0; i < logical_pages; ++i) {
            shard.mtp_page_handles.push_back(shard.mtp_pages[i].handle());
        }
        shard.mtp_row = tables.acquire(0);
        tables.publish(shard.mtp_row.handle(), 0, shard.mtp_page_handles, shard.device.stream);
    }

    // MTP proposal scratch (shard 0): the round state carries the step token/position, the RoPE
    // delta, the MTP KV table row and the MTP prefill/autoregressive buffers for K drafts.
    qwen::PagedKVCacheView mtp_view;
    const qwen::PagedKVCache* batch_mtp = nullptr;
    if (mtp_shard) {
        LayoutBuilder round_builder;
        qwen::RoundStateLayout round_layout = qwen::begin_round_state_layout(
            round_builder,
            qwen::RoundStateSpec{
                .hidden         = qwen::execution::dimension(config.hidden_size),
                .output_rows    = qwen::execution::dimension(config.vocab_size),
                .batch_capacity = 1,
                .draft_window   = mtp_drafts_,
                .backend        = SpeculativeBackend::Mtp,
            });
        qwen::complete_round_state_layout(round_builder, round_layout);
        round_bytes = round_builder.finish(256);
        shard.round_arena = std::make_unique<DeviceArena>(round_bytes);
        shard.device.bind_to_current_thread();
        DeviceSpan backing = shard.round_arena->alloc_bytes(round_bytes, 256);
        CUDA_CHECK(cudaMemset(backing.data, 0, round_bytes));
        shard.io = qwen::RoundState(backing, round_layout);
        ops::set_i32_scalar(shard.io.backend_kv_table_row, 0, shard.device.stream);
        mtp_view  = shard.decoder->mtp_cache()->execution_view(shard.mtp_row);
        batch_mtp = shard.decoder->mtp_cache();
    }

    {
        // The text KV is bound through the batch cache (the TP-2 path never uses a per-request
        // execution view); the MTP layer binds both its execution view (prefill chunk) and its
        // batch cache (proposal steps).
        qwen::PagedKVCacheView empty_kv;
        const qwen::PagedKVCache* batch_text = &shard.decoder->text_kv;
        shard.context = std::make_unique<qwen::execution::TextContext>(
            shard.device, *shard.parameters, *shard.workspace, empty_kv, *shard.state, shard.io,
            shard.prefill_hidden, options_.prefill_chunk, 0, mtp_view, batch_text, batch_mtp);
        shard.context->set_shard_config(&scfg, shard_index);
        if (mtp_shard) { shard.context->set_mtp_proposal_extent(mtp_drafts_); }
    }
}

void TP2GenerationCore::mtp_prefill_priming(Shard& shard, const int* ids, std::uint32_t length,
                                            std::uint32_t first_position, Tensor& mtp_input,
                                            const Tensor* last_token, bool final_chunk) {
    const auto& config = shard.model->config().text;
    const std::int32_t hidden = qwen::execution::dimension(config.hidden_size);
    const std::int32_t vocab  = qwen::execution::dimension(config.vocab_size);
    auto& ws = *shard.workspace;
    shard.device.bind_to_current_thread();
    cudaStream_t stream = shard.device.stream;
    // The MTP layer's embedding column i is the token at position i+1: its input is shifted by one
    // relative to the hidden columns. Only the final chunk's last column is unknown here (the token
    // that follows the prompt), so it is overwritten on the device with the sampled token.
    std::vector<int> shifted(length, 0);
    const std::uint32_t known_columns = last_token != nullptr ? length - 1 : length;
    for (std::uint32_t i = 0; i < known_columns; ++i) { shifted[i] = ids[i + 1]; }
    Tensor ids_t = ws.alloc(DType::I32, {static_cast<std::int32_t>(length)});
    CUDA_CHECK(cudaMemcpyAsync(ids_t.data, shifted.data(), sizeof(int) * length,
                               cudaMemcpyHostToDevice, stream));
    if (last_token != nullptr) {
        // Element offset, not a byte offset: Tensor::data is void*, so plain pointer arithmetic
        // would land on the wrong column and corrupt two adjacent token ids.
        CUDA_CHECK(cudaMemcpyAsync(static_cast<std::int32_t*>(ids_t.data) + (length - 1),
                                   last_token->data, sizeof(std::int32_t),
                                   cudaMemcpyDeviceToDevice, stream));
    }
    Tensor positions = ws.alloc(DType::I32, {static_cast<std::int32_t>(length)});
    ops::fill_i32_positions(positions, static_cast<std::int32_t>(first_position), stream);
    const std::uint32_t visible = first_position + length;
    const ops::CausalAttentionExecutionEnvelope envelope{visible, visible};
    // The MTP layer's own K/V appends at the same absolute positions as the text layers, so the
    // cache positions and the RoPE positions coincide (the TP-2 path runs without a RoPE delta).
    // No proposal here: the round loop's first bridge re-runs this last column and proposes the
    // window itself, so priming only has to leave the MTP K/V complete.
    shard.context->mtp_prefill_chunk(ids_t, mtp_input, nullptr, positions, positions, envelope,
                                     false, nullptr, nullptr, nullptr);
    if (final_chunk) {
        // This chunk's last column is the prompt's last position: its hidden feeds the first bridge.
        CUDA_CHECK(cudaMemcpyAsync(shard.mtp_anchor_hidden.data,
                                   mtp_input.slice(1, static_cast<std::int32_t>(length) - 1, 1).data,
                                   shard.mtp_anchor_hidden.bytes(), cudaMemcpyDeviceToDevice,
                                   stream));
    }
}

TP2GenerationCore::WindowGraph*
TP2GenerationCore::select_window_graph(std::vector<WindowGraph>& graphs, std::uint32_t visible_end) {
    for (WindowGraph& graph : graphs) {
        if (graph.visible_begin <= visible_end && visible_end <= graph.visible_end) { return &graph; }
    }
    return nullptr;
}

TP2GenerationCore::WindowGraph*
TP2GenerationCore::reusable_window_graph(std::vector<WindowGraph>& graphs,
                                         std::uint32_t visible_end) {
    WindowGraph* graph = select_window_graph(graphs, visible_end);
    if (graph == nullptr || !graph->captured || graph->round_base[0] < shard_a_.round_base ||
        graph->round_base[1] < shard_b_.round_base) {
        return nullptr;
    }
    return graph;
}

// Launches one captured window on both devices. Nothing here may synchronize the host: a replay is
// enqueued back to back with the round's other work, and the two devices order themselves through
// the in-kernel allreduce's arrival tokens.
void TP2GenerationCore::launch_window_graph(WindowGraph& graph) {
    shard_a_.device.bind_to_current_thread();
    graph.executable[0].launch(shard_a_.device.stream);
    shard_b_.device.bind_to_current_thread();
    graph.executable[1].launch(shard_b_.device.stream);
    shard_a_.device.bind_to_current_thread();
}

void TP2GenerationCore::capture_verify_graph(WindowGraph& graph, const std::int32_t* ids,
                                             const std::int32_t* positions, Tensor& logits_columns,
                                             Tensor& hidden_columns) {
    Shard& shard_a = shard_a_;
    Shard& shard_b = shard_b_;
    const ops::CausalAttentionExecutionEnvelope envelope{graph.visible_begin, graph.visible_end};
    shard_a.device.bind_to_current_thread();
    graph.round_base[0]  = shard_a.round_base;
    graph.round_base[1]  = shard_b.round_base;
    graph.arena_begin[0] = shard_a.workspace->used();
    graph.arena_begin[1] = shard_b.workspace->used();
    // The verify's state transitions are recorded, not applied: the fold replays the committed
    // columns from the pre-round snapshot. The action is a launch-time choice, so the capture must
    // run with it set and a replay picks it up from the recorded kernels.
    shard_a.context->set_gdn_state_action(qwen::execution::GdnStateAction::RecordForReplay,
                                          &shard_a.records);
    shard_b.context->set_gdn_state_action(qwen::execution::GdnStateAction::RecordForReplay,
                                          &shard_b.records);
    DecodeGraphDefinition* definitions[2] = {&graph.definition[0], &graph.definition[1]};
    cudaStream_t streams[2] = {shard_a.device.stream, shard_b.device.stream};
    DecodeGraphDefinition::capture_group(definitions, streams, [&] {
        shard_a.context->forward_tp2_window(*shard_b.context, pair_, ids, positions, envelope,
                                           logits_columns, &hidden_columns);
    });
    graph.arena_bytes[0] = shard_a.workspace->used() - graph.arena_begin[0];
    graph.arena_bytes[1] = shard_b.workspace->used() - graph.arena_begin[1];
    shard_a.device.bind_to_current_thread();
    graph.executable[0].instantiate(graph.definition[0]);
    shard_b.device.bind_to_current_thread();
    graph.executable[1].instantiate(graph.definition[1]);
    shard_a.device.bind_to_current_thread();
    graph.captured = true;
}

void TP2GenerationCore::run_verify_window(const std::int32_t* ids, std::int32_t first_position,
                                          Tensor& logits_columns, Tensor& hidden_columns,
                                          qwen::execution::DFlashFeatureSink* sink) {
    Shard& shard_a = shard_a_;
    Shard& shard_b = shard_b_;
    const auto width    = static_cast<std::size_t>(logits_columns.ne[1]);
    const auto* positions = ids + width;
    const std::uint32_t visible_end =
        static_cast<std::uint32_t>(first_position) + static_cast<std::uint32_t>(width);
    if (!verify_graph_enabled_) {
        // The eager route runs the same window and the same envelope; only the launch differs. That
        // keeps NINFER_TP2_VERIFY_GRAPH=0 a true A/B of the captured sequence rather than a
        // comparison of two different attention routes.
        // The masked-draft route builds no verify-graph buckets (it has no capture path), so its
        // window runs under the exact envelope the bucket would have covered. MTP keeps the bucket
        // the eager arm is compared against.
        const ops::CausalAttentionExecutionEnvelope envelope =
            verify_graphs_.empty()
                ? ops::CausalAttentionExecutionEnvelope{
                      static_cast<std::uint32_t>(first_position) + 1U, visible_end}
                : [&] {
                      const WindowGraph* bucket = select_window_graph(verify_graphs_, visible_end);
                      if (bucket == nullptr) {
                          throw std::logic_error("TP-2 verify window coverage is incomplete");
                      }
                      return ops::CausalAttentionExecutionEnvelope{bucket->visible_begin,
                                                                   bucket->visible_end};
                  }();
        shard_a.context->set_gdn_state_action(qwen::execution::GdnStateAction::RecordForReplay,
                                              &shard_a.records);
        shard_b.context->set_gdn_state_action(qwen::execution::GdnStateAction::RecordForReplay,
                                              &shard_b.records);
        shard_a.context->forward_tp2_window(*shard_b.context, pair_, ids, positions, envelope,
                                            logits_columns, &hidden_columns, sink);
    } else {
        if (sink != nullptr) {
            // A captured window bakes the feature sink's scatter addresses but not the host-side
            // capture calls, so the masked-draft verify always runs the eager route
            // (verify_graph_enabled_ is only set for MTP).
            throw std::logic_error("TP-2 masked-draft verify cannot use a captured window");
        }
        WindowGraph* graph = select_window_graph(verify_graphs_, visible_end);
        if (graph == nullptr) {
            throw std::logic_error("TP-2 verify CUDA Graph coverage is incomplete");
        }
        if (reusable_window_graph(verify_graphs_, visible_end) == nullptr) {
            // Capturing records the sequence without executing it, so the capture round still has to
            // run the window it just captured (with the operands the capture itself allocated).
            capture_verify_graph(*graph, ids, positions, logits_columns, hidden_columns);
            launch_window_graph(*graph);
        } else {
            if (shard_a.workspace->used() != graph->arena_begin[0] ||
                shard_b.workspace->used() != graph->arena_begin[1]) {
                throw std::logic_error(
                    "TP-2 verify CUDA Graph replay found a different workspace layout");
            }
            launch_window_graph(*graph);
            // The captured body allocated its operands during capture; a replay does not run that
            // host code, so advance both arenas by what the capture consumed.
            (void)shard_a.workspace->alloc_bytes(graph->arena_bytes[0]);
            (void)shard_b.workspace->alloc_bytes(graph->arena_bytes[1]);
        }
    }
    shard_a.context->set_gdn_state_action(qwen::execution::GdnStateAction::UpdateInPlace, nullptr);
    shard_b.context->set_gdn_state_action(qwen::execution::GdnStateAction::UpdateInPlace, nullptr);
}

void TP2GenerationCore::capture_decode_graph(WindowGraph& graph, const std::int32_t* token,
                                             const std::int32_t* position, Tensor& logits) {
    Shard& shard_a = shard_a_;
    Shard& shard_b = shard_b_;
    const ops::CausalAttentionExecutionEnvelope envelope{graph.visible_begin, graph.visible_end};
    shard_a.device.bind_to_current_thread();
    graph.round_base[0]  = shard_a.round_base;
    graph.round_base[1]  = shard_b.round_base;
    graph.arena_begin[0] = shard_a.workspace->used();
    graph.arena_begin[1] = shard_b.workspace->used();
    // The step's state transitions are already applied in place: a plain decode is not speculative,
    // so unlike the verify window it neither records nor replays anything.
    DecodeGraphDefinition* definitions[2] = {&graph.definition[0], &graph.definition[1]};
    cudaStream_t streams[2] = {shard_a.device.stream, shard_b.device.stream};
    DecodeGraphDefinition::capture_group(definitions, streams, [&] {
        shard_a.context->forward_tp2_decode_window(*shard_b.context, pair_, token, position,
                                                  envelope, logits);
    });
    graph.arena_bytes[0] = shard_a.workspace->used() - graph.arena_begin[0];
    graph.arena_bytes[1] = shard_b.workspace->used() - graph.arena_begin[1];
    shard_a.device.bind_to_current_thread();
    graph.executable[0].instantiate(graph.definition[0]);
    shard_b.device.bind_to_current_thread();
    graph.executable[1].instantiate(graph.definition[1]);
    shard_a.device.bind_to_current_thread();
    graph.captured = true;
}

Tensor TP2GenerationCore::run_plain_decode_step(std::int32_t token, std::uint32_t position) {
    Shard& shard_a = shard_a_;
    Shard& shard_b = shard_b_;
    const std::int32_t vocab = qwen::execution::dimension(shard_a.model->config().text.vocab_size);
    if (decode_step_mode_ == DecodeStepMode::EagerExact) {
        // The pre-graph route: the exact visible extent, with the token and the position baked into
        // the forward's own scalar kernels. Kept as the reference arm of the A/B.
        Tensor logits_a = shard_a.workspace->alloc(DType::BF16, {vocab, 1});
        Tensor logits_b = shard_b.workspace->alloc(DType::BF16, {vocab, 1});
        shard_a.context->forward_tp2(*shard_b.context, pair_, token,
                                     static_cast<std::int32_t>(position), logits_a, logits_b);
        return logits_a;
    }
    const std::uint32_t visible_end = position + 1U;
    WindowGraph* graph = select_window_graph(decode_graphs_, visible_end);
    if (graph == nullptr) { throw std::logic_error("TP-2 decode step coverage is incomplete"); }
    const bool captured = decode_step_mode_ == DecodeStepMode::Graph;
    WindowGraph* reusable =
        captured ? reusable_window_graph(decode_graphs_, visible_end) : nullptr;
    // A replay must reproduce the addresses the capture baked into its kernel arguments, so both
    // arenas first go back to the captured watermark: the capture ran its host-side allocations, a
    // replay does not.
    position_arena(*shard_a.workspace, shard_a.round_base,
                   reusable != nullptr ? reusable->round_base[0] : shard_a.round_base);
    position_arena(*shard_b.workspace, shard_b.round_base,
                   reusable != nullptr ? reusable->round_base[1] : shard_b.round_base);
    Tensor logits = shard_a.workspace->alloc(DType::BF16, {vocab, 1});
    auto* window  = static_cast<std::int32_t*>(decode_window_host_->data());
    window[0]     = token;
    window[1]     = static_cast<std::int32_t>(position);
    const ops::CausalAttentionExecutionEnvelope envelope{graph->visible_begin, graph->visible_end};
    if (reusable == nullptr) {
        if (captured) {
            // Capturing records the sequence without executing it, so this round still has to run the
            // step it just captured, with the operands the capture itself allocated.
            capture_decode_graph(*graph, window, window + 1, logits);
            launch_window_graph(*graph);
        } else {
            // The eager arm runs the same sequence with the same bucket envelope, so the A/B compares
            // the launch mechanism alone.
            shard_a.context->forward_tp2_decode_window(*shard_b.context, pair_, window, window + 1,
                                                       envelope, logits);
        }
        return logits;
    }
    if (shard_a.workspace->used() != graph->arena_begin[0] ||
        shard_b.workspace->used() != graph->arena_begin[1]) {
        throw std::logic_error("TP-2 decode CUDA Graph replay found a different workspace layout");
    }
    launch_window_graph(*graph);
    // The captured body allocated its operands during capture; a replay does not run that host code,
    // so advance both arenas by what the capture consumed.
    (void)shard_a.workspace->alloc_bytes(graph->arena_bytes[0]);
    (void)shard_b.workspace->alloc_bytes(graph->arena_bytes[1]);
    return logits;
}

std::vector<TokenId> TP2GenerationCore::mtp_propose_window(Shard& shard, Tensor& mtp_input,
                                                           const Tensor& anchor,
                                                           std::uint32_t position,
                                                           DeviceArena& ws) {
    auto& ctx = *shard.context;
    const auto& config = shard.model->config().text;
    const std::int32_t hidden = qwen::execution::dimension(config.hidden_size);
    const std::int32_t vocab  = qwen::execution::dimension(config.vocab_size);
    shard.device.bind_to_current_thread();
    cudaStream_t stream = shard.device.stream;
    Tensor positions = ws.alloc(DType::I32, {1});
    ops::set_i32_scalar(positions, static_cast<std::int32_t>(position), stream);
    Tensor ar_position = ws.alloc(DType::I32, {1});
    ops::set_i32_scalar(ar_position, static_cast<std::int32_t>(position) + 1, stream);
    Tensor drafts    = ws.alloc(DType::I32, {static_cast<std::int32_t>(mtp_drafts_)});
    Tensor ar_hidden = ws.alloc(DType::BF16, {hidden, 1});
    Tensor logits    = ws.alloc(DType::BF16, {vocab, 1});
    const auto visible = static_cast<std::uint32_t>(position) + 1;
    const ops::CausalAttentionExecutionEnvelope bridge{visible, visible};
    Tensor draft0 = drafts.slice(0, 0, 1);
    ctx.mtp_forward_batch(anchor, mtp_input, positions, bridge, ar_hidden, 0, &logits, &draft0);
    for (std::uint32_t i = 1; i < mtp_drafts_; ++i) {
        Tensor previous    = drafts.slice(0, static_cast<std::int32_t>(i) - 1, 1);
        Tensor next_draft  = drafts.slice(0, static_cast<std::int32_t>(i), 1);
        Tensor step_hidden = ws.alloc(DType::BF16, {hidden, 1});
        const auto step_visible = static_cast<std::uint32_t>(position) + i + 1;
        const ops::CausalAttentionExecutionEnvelope envelope{step_visible, step_visible};
        ctx.mtp_forward_ar_step(previous, ar_hidden, ar_position, envelope, step_hidden, logits,
                                next_draft);
        CUDA_CHECK(cudaMemcpyAsync(ar_hidden.data, step_hidden.data, ar_hidden.bytes(),
                                   cudaMemcpyDeviceToDevice, stream));
        ops::increment_i32_scalar(ar_position, stream);
    }
    std::vector<TokenId> host(mtp_drafts_, 0);
    CUDA_CHECK(cudaMemcpyAsync(host.data(), drafts.data, sizeof(TokenId) * mtp_drafts_,
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return host;
}

std::optional<qwen::execution::DFlashFeatureSink>
TP2GenerationCore::make_dflash_prefill_sink(Shard& shard) {
    if (shard.dflash_round == nullptr) { return std::nullopt; }
    // The chunk width the prefill forward will actually run with. The sink's buffer is sized for the
    // maximum and each chunk slices its own prefix out of it (capture_positions copies the matching
    // positions prefix, consume_prefill_chunk slices the feature prefix). Only the prefill fields are
    // set: the batch (verify-window) fields stay null until the masked draft has its verify wiring.
    // One resident session, so the draft ring's lane is 0 and the whole captured chunk is committed.
    const std::uint32_t chunk = std::min<std::uint32_t>(
        std::max<std::uint32_t>(options_.prefill_chunk, 64U), kPrefillChunkMaximum);
    return shard.dflash_round->make_prefill_sink(qwen::execution::ExecutionCore{
        .device           = shard.device,
        .parameters       = *shard.parameters,
        .work             = *shard.workspace,
        .linear_attention = *shard.state,
        .replay_records   = nullptr,
        .io               = shard.io,
        .prefill_hidden   = shard.prefill_hidden,
        .prefill_chunk    = chunk,
        .proposal_head    = options_.speculative.proposal_head,
    });
}

TP2GenerationCore::Submission TP2GenerationCore::submit(
    qwen::PreparedPrompt prompt, PromptSummary summary, double prepare_seconds,
    ResolvedRequestOptions options, OutputConsumerMode consumer_mode,
    GenerationObservationOptions, std::chrono::steady_clock::time_point) {
    auto output = frontend_->make_output_session(prompt, options.stop, options.output,
                                                 options.execution.thinking);
    const std::uint32_t capacity_output =
        options_.max_context - summary.prompt_tokens + static_cast<std::uint32_t>(1);
    const std::uint32_t effective =
        std::min(options.execution.requested_output_tokens, capacity_output);
    const FinishReason limit_reason =
        options.execution.requested_output_tokens <= capacity_output ? FinishReason::OutputLimit
                                                                     : FinishReason::ContextCapacity;
    try {
        output.validate_generation_capacity(effective);
    } catch (const std::invalid_argument& error) {
        throw RequestError(RequestErrorKind::ThinkingBudgetCapacityInsufficient, error.what());
    }
    GenerationBudget budget(effective, limit_reason);
    return Submission(*this, std::make_unique<Request>(std::move(prompt), std::move(output),
                                                       summary, prepare_seconds, std::move(budget),
                                                       options.execution.sampling,
                                                       consumer_mode));
}

GenerationResult TP2GenerationCore::Submission::wait(OutputSink* sink,
                                                     const CancellationView& cancellation) {
    if (owner_ == nullptr || request_ == nullptr) {
        throw std::logic_error("concurrent submission is empty");
    }
    const bool streaming = request_->consumer_mode == OutputConsumerMode::Streaming;
    if (streaming != (sink != nullptr)) {
        throw std::invalid_argument(
            "GenerationHandle wait sink does not match its submitted consumer mode");
    }
    // Take the single execution slot. A request cancelled while queued still enters execute(),
    // whose first cancellation check is cheap and returns a Cancelled result with the proper
    // preview state.
    std::unique_lock<std::mutex> slot(owner_->execution_mutex_);
    return owner_->execute(*request_, sink, cancellation);
}

// ---------------------------------------------------------------------------------------------
// Cross-session KV retention
//
// The device KV page pool is materialized once at startup and reused in place, so it holds exactly
// one session at a time. Turning a client to another conversation therefore copies the resident
// session's KV and GDN state into host KV slabs and copies the returning session back, which
// costs one PCIe round trip instead of a full prefill. The catalog below owns both the token
// history that describes what the device pools must reproduce and the host slabs that hold it while
// another conversation is resident.
// ---------------------------------------------------------------------------------------------

namespace {

bool session_trace_enabled() {
    const char* env = std::getenv("NINFER_TP2_SESSION_TRACE");
    return env != nullptr && env[0] == '1';
}

} // namespace

void TP2GenerationCore::invalidate_host_checkpoints() {
    Shard* const shards[2] = {&shard_a_, &shard_b_};
    for (Shard* shard : shards) {
        for (auto& checkpoint : shard->host_checkpoints) { checkpoint.valid = false; }
    }
}

void TP2GenerationCore::session_invalidate_active() {
    if (active_session_ != kNoSession) { session_drop(active_session_); }
    active_session_     = kNoSession;
    live_state_valid_   = false;
    cached_state_valid_ = false;
    cached_prompt_tokens_.clear();
    invalidate_host_checkpoints();
    reuse_source_ = ReuseSource::None;
}

void TP2GenerationCore::session_drop(std::size_t index) {
    if (index >= sessions_.size()) { return; }
    sessions_.erase(sessions_.begin() + static_cast<std::ptrdiff_t>(index));
    if (active_session_ == index) {
        active_session_ = kNoSession;
    } else if (active_session_ != kNoSession && active_session_ > index) {
        --active_session_;
    }
}

bool TP2GenerationCore::session_evict_one() {
    std::size_t victim     = kNoSession;
    std::uint64_t oldest   = 0;
    for (std::size_t index = 0; index < sessions_.size(); ++index) {
        if (sessions_[index].device_resident) { continue; }
        if (victim == kNoSession || sessions_[index].lru_clock < oldest) {
            victim = index;
            oldest = sessions_[index].lru_clock;
        }
    }
    if (victim == kNoSession) { return false; }
    if (session_trace_enabled()) {
        std::fprintf(stderr, "[tp2-session] evict frontier=%u tokens=%zu\n",
                     sessions_[victim].frontier, sessions_[victim].tokens.size());
    }
    session_drop(victim);
    ++session_evictions_;
    return true;
}

bool TP2GenerationCore::session_ensure_host_slabs(SessionEntry& entry, std::uint32_t pages) {
    if (pages == 0) { return false; }
    Shard* const shards[2] = {&shard_a_, &shard_b_};
    for (std::size_t index = 0; index < 2; ++index) {
        Shard& shard = *shards[index];
        // The arena pins its whole budget on construction, so a workload that never leaves one
        // conversation never pays for the capability.
        if (host_kv_arena_[index] == nullptr) {
            std::vector<HostKVPageLayout> layouts;
            layouts.push_back(plan_host_kv_page_layout(shard.decoder->text_kv.page_pool().geometry()));
            if (index == 0 && mtp_enabled_) {
                const qwen::PagedKVCache* mtp = shard.decoder->mtp_cache();
                if (mtp == nullptr) { return false; }
                layouts.push_back(plan_host_kv_page_layout(mtp->page_pool().geometry()));
            }
            shard.device.bind_to_current_thread();
            try {
                host_kv_arena_[index] = std::make_unique<HostKVArena>(
                    host_kv_shard_bytes_, layouts,
                    options_.context_cache.host_kv_pinned ? HostPinning::PreferPinned
                                                          : HostPinning::Pageable);
            } catch (const std::exception& error) {
                // A failed allocation can leave cudaErrorMemoryAllocation latched, where the next
                // unrelated CUDA_CHECK would report it as that call's own failure; report both halves
                // here, because a silent failure would degrade every later request to a full prefill
                // with no other trace. A refused pin does not reach this path: the backing store falls
                // back to pageable memory on its own.
                const cudaError_t latched = cudaGetLastError();
                std::fprintf(stderr,
                             "[tp2-session] host KV arena shard %zu refused: %.1f MiB (%s): "
                             "%s\n",
                             index, static_cast<double>(host_kv_shard_bytes_) / 1048576.0,
                             cudaGetErrorString(latched), error.what());
                host_kv_arena_[index].reset();
                return false;
            }
            std::fprintf(stderr,
                         "[tp2-session] host KV arena shard %zu: %.1f MiB %s, %zu layouts\n",
                         index, static_cast<double>(host_kv_shard_bytes_) / 1048576.0,
                         host_kv_arena_[index]->backing_pinned() ? "pinned" : "pageable",
                         layouts.size());
        }
        const HostKVPageLayout* layout =
            host_kv_arena_[index]->layout_for(shard.decoder->text_kv.page_pool().geometry());
        if (layout == nullptr) { return false; }
        // The GDN state image is independent of the KV extent: it is always one full state per
        // shard, and it survives a frontier that a later eviction grows.
        if (entry.host_state[index] == nullptr) {
            shard.device.bind_to_current_thread();
            try {
                entry.host_state[index] =
                    std::make_unique<PinnedHostBuffer>(shard.state_backing.bytes, true);
            } catch (const std::exception& error) {
                const cudaError_t latched = cudaGetLastError();
                std::fprintf(stderr,
                             "[tp2-session] host state image shard %zu refused: %.1f MiB pinned "
                             "(%s): %s\n",
                             index, static_cast<double>(shard.state_backing.bytes) / 1048576.0,
                             cudaGetErrorString(latched), error.what());
                return false;
            }
        }
        // The two extra images are best effort: they make a returning conversation cheaper, but the
        // frontier image above is what makes it recallable at all. A shortage of pinned memory
        // therefore degrades the recall instead of dropping the conversation.
        if (entry.host_prompt_state[index] == nullptr) {
            shard.device.bind_to_current_thread();
            try {
                entry.host_prompt_state[index] =
                    std::make_unique<PinnedHostBuffer>(shard.state_backing.bytes, true);
            } catch (const std::exception& error) {
                const cudaError_t latched = cudaGetLastError();
                std::fprintf(stderr,
                             "[tp2-session] warning: host prompt-end state image shard %zu refused: "
                             "%.1f MiB pinned (%s): %s; this session needs the generated tail "
                             "reproduced to be recalled\n",
                             index, static_cast<double>(shard.state_backing.bytes) / 1048576.0,
                             cudaGetErrorString(latched), error.what());
            }
        }
        if (entry.host_shared_state[index] == nullptr) {
            shard.device.bind_to_current_thread();
            try {
                entry.host_shared_state[index] =
                    std::make_unique<PinnedHostBuffer>(shard.state_backing.bytes, true);
            } catch (const std::exception& error) {
                const cudaError_t latched = cudaGetLastError();
                std::fprintf(stderr,
                             "[tp2-session] warning: host shared-prefix state image shard %zu "
                             "refused: %.1f MiB pinned (%s): %s; this session will not carry a "
                             "stable prefix for the next conversation\n",
                             index, static_cast<double>(shard.state_backing.bytes) / 1048576.0,
                             cudaGetErrorString(latched), error.what());
            }
        }
        if (entry.host_kv[index] != nullptr && entry.host_kv[index]->page_count() >= pages) {
            // A slab that already covers this frontier is reused as it is; only the first `pages`
            // pages carry the session, and the rest is stale.
            continue;
        }
        entry.host_kv[index].reset();
        std::optional<HostKVAllocation> allocation = host_kv_arena_[index]->allocate(*layout, pages);
        if (!allocation.has_value()) { return false; }
        entry.host_kv[index] = std::make_unique<HostKVAllocation>(std::move(*allocation));
    }
    if (mtp_enabled_) {
        const qwen::PagedKVCache* mtp = shard_a_.decoder->mtp_cache();
        const HostKVPageLayout* layout =
            host_kv_arena_[0]->layout_for(mtp->page_pool().geometry());
        if (layout == nullptr) { return false; }
        if (entry.host_mtp_kv == nullptr || entry.host_mtp_kv->page_count() < pages) {
            entry.host_mtp_kv.reset();
            std::optional<HostKVAllocation> allocation =
                host_kv_arena_[0]->allocate(*layout, pages);
            if (!allocation.has_value()) { return false; }
            entry.host_mtp_kv = std::make_unique<HostKVAllocation>(std::move(*allocation));
        }
        entry.host_mtp_pages = pages;
    }
    return true;
}

bool TP2GenerationCore::session_store_active() {
    if (active_session_ == kNoSession) { return true; }
    SessionEntry& entry = sessions_[active_session_];
    if (!entry.device_resident) { return true; }
    const std::uint32_t pages = pages_for_tokens(entry.frontier);
    if (pages == 0) {
        // Nothing to copy, and nothing a recall could stand on either: clear the boundaries so a
        // later request cannot restore a state whose KV the device pools do not hold.
        entry.device_resident = false;
        entry.host_prompt_end = 0;
        entry.host_shared_end = 0;
        return true;
    }
    if (!session_ensure_host_slabs(entry, pages)) { return false; }
    Shard* const shards[2] = {&shard_a_, &shard_b_};
    for (std::size_t index = 0; index < 2; ++index) {
        Shard& shard = *shards[index];
        shard.device.bind_to_current_thread();
        const std::span<const DeviceKVPageHandle> source(shard.kv_page_handles.data(), pages);
        const HostKVAllocationView view =
            host_kv_arena_[index]->writable_view(*entry.host_kv[index]).subview(0, pages);
        shard.decoder->text_kv.page_pool().copy_to_host(source, view, shard.device.stream);
        CUDA_CHECK(cudaMemcpyAsync(entry.host_state[index]->data(), shard.state_backing.data,
                                   shard.state_backing.bytes, cudaMemcpyDeviceToHost,
                                   shard.device.stream));
        // Snapshot slot 0 is the state the last completed prefill froze at this conversation's own
        // prompt end. A client that re-renders the answer it was handed sends a next prompt that
        // contains that whole prompt and then diverges inside the generated tail, so this image is
        // the one its return trip can stand on.
        if (entry.host_prompt_state[index] != nullptr) {
            CUDA_CHECK(cudaMemcpyAsync(entry.host_prompt_state[index]->data(),
                                       shard.state_snapshots[0].data, shard.state_backing.bytes,
                                       cudaMemcpyDeviceToHost, shard.device.stream));
        }
    }
    if (mtp_enabled_) {
        shard_a_.device.bind_to_current_thread();
        const std::span<const DeviceKVPageHandle> source(shard_a_.mtp_page_handles.data(),
                                                         entry.host_mtp_pages);
        const HostKVAllocationView view =
            host_kv_arena_[0]->writable_view(*entry.host_mtp_kv).subview(0, entry.host_mtp_pages);
        shard_a_.decoder->mtp_cache()->page_pool().copy_to_host(source, view, shard_a_.device.stream);
    }
    // A copy still in flight is not state: the entry only becomes host-resident once both devices
    // have finished their D2H, so an error here leaves the session on the device instead.
    for (std::size_t index = 0; index < 2; ++index) {
        shards[index]->device.bind_to_current_thread();
        CUDA_CHECK(cudaStreamSynchronize(shards[index]->device.stream));
    }

    shard_a_.device.bind_to_current_thread();
    entry.device_resident = false;
    entry.host_prompt_end = (entry.host_prompt_state[0] != nullptr &&
                             entry.host_prompt_state[1] != nullptr)
                                ? entry.prompt_end
                                : 0;
    entry.lru_clock       = ++session_lru_clock_;
    ++session_stores_;
    return true;
}

void TP2GenerationCore::session_restore(SessionEntry& entry, std::uint32_t boundary,
                                        RecallState state) {
    const std::uint32_t pages = pages_for_tokens(boundary);
    if (pages == 0) {
        entry.tokens.clear();
        entry.frontier        = 0;
        entry.device_resident = true;
        return;
    }
    Shard* const shards[2] = {&shard_a_, &shard_b_};
    for (std::size_t index = 0; index < 2; ++index) {
        Shard& shard = *shards[index];
        shard.device.bind_to_current_thread();
        const std::span<const DeviceKVPageHandle> destination(shard.kv_page_handles.data(), pages);
        const HostKVAllocationConstView view =
            host_kv_arena_[index]->view(*entry.host_kv[index]).subview(0, pages);
        shard.decoder->text_kv.page_pool().copy_from_host(view, destination, shard.device.stream);

        const PinnedHostBuffer& frozen = state == RecallState::Frontier
                                            ? *entry.host_state[index]
                                        : state == RecallState::PromptEnd
                                            ? *entry.host_prompt_state[index]
                                            : *entry.host_shared_state[index];
        CUDA_CHECK(cudaMemcpyAsync(shard.state_backing.data, frozen.data(),
                                   shard.state_backing.bytes, cudaMemcpyHostToDevice,
                                   shard.device.stream));
    }
    if (mtp_enabled_) {
        shard_a_.device.bind_to_current_thread();
        const std::span<const DeviceKVPageHandle> destination(shard_a_.mtp_page_handles.data(), pages);
        const HostKVAllocationConstView view =
            host_kv_arena_[0]->view(*entry.host_mtp_kv).subview(0, pages);
        shard_a_.decoder->mtp_cache()->page_pool().copy_from_host(view, destination,
                                                                  shard_a_.device.stream);
    }
    for (std::size_t index = 0; index < 2; ++index) {
        shards[index]->device.bind_to_current_thread();
        CUDA_CHECK(cudaStreamSynchronize(shards[index]->device.stream));
    }
    shard_a_.device.bind_to_current_thread();
    // A prompt-end recall leaves the entry standing where that prompt ended: the device pools hold
    // its KV and the state at its end, and the tail the entry had generated past it is gone. A
    // history longer than the restored frontier would describe KV the pools no longer carry.
    if (boundary < entry.frontier) { entry.tokens.resize(boundary); }
    entry.frontier        = boundary;
    entry.device_resident = true;
    entry.lru_clock       = ++session_lru_clock_;
    ++session_recalls_;
}

void TP2GenerationCore::session_capture_shared_state(std::size_t index, std::uint32_t position,
                                                     bool from_device,
                                                     const PinnedHostBuffer* const* frozen) {
    if (index >= sessions_.size() || position == 0) { return; }
    SessionEntry& entry = sessions_[index];
    // Only a host-resident entry has a slab to pair the state with, and only a position its slab
    // covers can be recalled: the KV before the frontier is the most there is.
    if (entry.device_resident || position <= entry.host_shared_end || position > entry.frontier) {
        return;
    }
    Shard* const shards[2] = {&shard_a_, &shard_b_};
    for (std::size_t shard_index = 0; shard_index < 2; ++shard_index) {
        if (entry.host_shared_state[shard_index] == nullptr) { return; }
    }
    for (std::size_t shard_index = 0; shard_index < 2; ++shard_index) {
        Shard& shard = *shards[shard_index];
        if (from_device) {
            shard.device.bind_to_current_thread();
            CUDA_CHECK(cudaMemcpyAsync(entry.host_shared_state[shard_index]->data(),
                                       shard.state_backing.data, shard.state_backing.bytes,
                                       cudaMemcpyDeviceToHost, shard.device.stream));
            continue;
        }
        if (frozen[shard_index] == nullptr) { return; }
        std::memcpy(entry.host_shared_state[shard_index]->data(), frozen[shard_index]->data(),
                    shard.state_backing.bytes);
    }
    entry.host_shared_end = position;
}

void TP2GenerationCore::session_recall(std::span<const TokenId> prompt_tokens) {
    if (session_capacity_ == 0) { return; }
    // The entry this recall moves into its host slabs is the only conversation that can still be
    // told where the incoming prompt stopped matching it, and the running prefill is the last walk
    // that sees the shared state. execute_walk reads this before it advances anything.
    host_stored_session_ = kNoSession;
    // How deep the resident lineage can start this prompt, mirroring what the reuse scan below
    // finds: the live state at its frontier, a device snapshot, or a host checkpoint. Zero means
    // the prompt shares nothing the device can reuse, and keeping the resident session would leave
    // the catalog describing a conversation the device no longer holds.
    std::size_t active_shared = 0;
    if (active_session_ != kNoSession) {
        const SessionEntry& entry = sessions_[active_session_];
        const std::size_t common  = std::min(entry.tokens.size(), prompt_tokens.size());
        while (active_shared < common &&
               entry.tokens[active_shared] == prompt_tokens[active_shared]) {
            ++active_shared;
        }
    }
    std::uint32_t resident_depth = 0;
    if (cached_state_valid_) {
        if (live_state_valid_ && active_session_ != kNoSession) {
            const std::uint32_t frontier = sessions_[active_session_].frontier;
            if (frontier != 0 && frontier <= active_shared) { resident_depth = frontier; }
        }
        for (std::size_t slot = 0; slot < kReuseSnapshotCount; ++slot) {
            const std::uint32_t boundary = cached_boundaries_[slot];
            if (boundary <= active_shared && boundary > resident_depth) {
                resident_depth = boundary;
            }
        }
        for (const auto& checkpoint : shard_a_.host_checkpoints) {
            if (checkpoint.valid && checkpoint.position <= active_shared &&
                checkpoint.position > resident_depth) {
                resident_depth = checkpoint.position;
            }
        }
    }

    // A host-resident entry carries three frozen GDN states: the one its evicted frontier stood on,
    // the one its last completed prefill froze at the entry's own prompt end, and the one frozen
    // where another conversation diverged from it. The frontier is only reachable by a prompt that
    // reproduces the generated tail token for token; a client that re-renders the answer it was
    // handed stops at the first generated token and uses the prompt-end image instead; and a
    // conversation that only shares a stable prefix - a system prompt - uses the divergence image.
    // The deepest reachable boundary that still beats the resident lineage wins; a shallower one
    // would spend a PCIe round trip to save fewer tokens than staying put.
    //
    // Every boundary has to leave at least one token to forward: the first sample's logits come from
    // the column the walk ends on, so a prompt that is exactly the recalled history would restore
    // the whole thing and then prefill it again from zero. Taking that recall also discards the
    // device lineage and the checkpoint ring, which is strictly worse than not taking it.
    std::size_t best_host       = kNoSession;
    std::uint32_t best_frontier = 0;
    RecallState best_state      = RecallState::Frontier;
    for (std::size_t index = 0; index < sessions_.size(); ++index) {
        const SessionEntry& entry = sessions_[index];
        if (index == active_session_ || entry.device_resident || entry.tokens.empty()) { continue; }
        const std::size_t common = std::min(entry.tokens.size(), prompt_tokens.size());
        std::size_t shared       = 0;
        while (shared < common && entry.tokens[shared] == prompt_tokens[shared]) { ++shared; }
        static constexpr const char* kReachName[3] = {"frontier", "prompt-end", "shared"};
        const std::uint32_t offered[3] = {entry.frontier, entry.host_prompt_end, entry.host_shared_end};
        std::uint32_t reach = 0;
        RecallState via     = RecallState::Frontier;
        for (std::size_t kind = 0; kind < 3; ++kind) {
            // The shared-prefix image is capped by the frontier it was evicted at, so the slab
            // covers every page the recall reads.
            const std::uint32_t ceiling =
                kind == 2 ? entry.frontier : std::numeric_limits<std::uint32_t>::max();
            if (offered[kind] == 0 || offered[kind] > shared ||
                offered[kind] >= prompt_tokens.size() || offered[kind] > ceiling) {
                continue;
            }
            if (offered[kind] > reach) {
                reach = offered[kind];
                via   = static_cast<RecallState>(kind);
            }
        }
        if (session_trace_enabled()) {
            std::fprintf(stderr,
                         "[tp2-session]   entry %zu tokens=%zu frontier=%u prompt_end=%u shared_end=%u "
                         "shared=%zu reach=%u via=%s resident=%d\n",
                         index, entry.tokens.size(), entry.frontier, entry.prompt_end,
                         entry.host_shared_end, shared, reach,
                         reach == 0 ? "none" : kReachName[static_cast<std::size_t>(via)],
                         entry.device_resident ? 1 : 0);
        }
        if (reach == 0) { continue; }
        if (best_host == kNoSession || reach > best_frontier) {
            best_host     = index;
            best_frontier = reach;
            best_state    = via;
        }
    }
    if (best_host != kNoSession && best_frontier > resident_depth) {
        const std::size_t previous = active_session_;
        const bool drop_previous   = previous != kNoSession && !session_store_active();
        host_stored_session_       = drop_previous ? kNoSession : previous;
        // Restoring before dropping keeps the recalled entry's index valid: nothing has been erased
        // yet, and the drop below only shifts indices the restore is already done with.
        session_restore(sessions_[best_host], best_frontier, best_state);
        active_session_       = best_host;
        live_state_valid_     = true;
        cached_prompt_tokens_ = sessions_[best_host].tokens;
        // A recalled session has no device state snapshots of its own: those belong to the lineage
        // it displaces. Its frontier state is in the device pool already, which is what LiveState
        // means to the scan below. The checkpoint ring belongs to that lineage too, and its
        // positions can sit inside the recalled history while its states do not.
        cached_boundaries_.fill(0);
        invalidate_host_checkpoints();
        cached_state_valid_ = true;
        reuse_source_       = ReuseSource::None;
        if (drop_previous) { session_drop(previous); }
        if (session_trace_enabled()) {
            std::fprintf(
                stderr,
                "[tp2-session] recall frontier=%u resident_depth=%u tokens=%zu entries=%zu\n",
                sessions_[active_session_].frontier, resident_depth,
                sessions_[active_session_].tokens.size(), sessions_.size());
        }
        return;
    }
    // The resident lineage is only worth keeping while this prompt is a later turn of it, and the
    // last completed prefill's end is what proves that: the resident entry's prompt_end is the token
    // count of the prompt the device pools were built from, so a prompt that still shares those
    // tokens contains that whole prompt. A prompt that only matches a host checkpoint inside the
    // shared system prefix - the shape of a client switching conversations - is a different
    // conversation: the resident entry has to reach its host slabs before the prefill below
    // overwrites the device pools, or the conversation is lost.
    const bool resident_continues = active_session_ != kNoSession &&
                                    sessions_[active_session_].prompt_end != 0 &&
                                    active_shared >= sessions_[active_session_].prompt_end;
    if (resident_depth > 0 && resident_continues) {
        // The resident conversation already serves this prompt better than any host slab; the reuse
        // scan below turns its depth into a prefill start point.
        if (session_trace_enabled()) {
            std::fprintf(stderr,
                         "[tp2-session] continue active_shared=%zu resident_depth=%u frontier=%u "
                         "prompt_end=%u entries=%zu\n",
                         active_shared, resident_depth, sessions_[active_session_].frontier,
                         sessions_[active_session_].prompt_end, sessions_.size());
        }
        return;
    }
    // A different conversation, or one no host entry can serve: preserve the resident session when
    // the budget allows. The prefix-reuse lineage survives the switch - cached_prompt_tokens_ and
    // every checkpoint at or before this prompt's shared prefix still describe tokens this prompt
    // holds - so the scan below keeps whatever they cover instead of re-prefilling the whole system
    // prefix on every turn of a switch.
    const std::size_t previous = active_session_;
    const bool dropped         = previous != kNoSession && !session_store_active();
    host_stored_session_       = dropped ? kNoSession : previous;
    if (dropped) { session_drop(previous); }
    active_session_   = kNoSession;
    live_state_valid_ = false;
    reuse_source_     = ReuseSource::None;
    ++session_full_prefills_;
    if (session_trace_enabled()) {
        std::fprintf(stderr,
                     "[tp2-session] switch active_shared=%zu resident_depth=%u stored=%d "
                     "entries=%zu\n",
                     active_shared, resident_depth, dropped ? 0 : 1, sessions_.size());
    }
}

void TP2GenerationCore::session_publish(const std::vector<TokenId>& history,
                                        std::uint32_t frontier) {
    if (session_capacity_ == 0) { return; }
    if (active_session_ != kNoSession && sessions_[active_session_].device_resident) {
        SessionEntry& entry = sessions_[active_session_];
        entry.tokens.assign(history.begin(), history.end());
        entry.frontier  = frontier;
        entry.lru_clock = ++session_lru_clock_;
        // The reuse scan compares against the resident history, so it has to carry the generated
        // tail too: that is what lets a continued turn start at the committed frontier instead of
        // replaying the previous answer.
        cached_prompt_tokens_ = entry.tokens;
        live_state_valid_     = true;
        return;
    }
    if (sessions_.size() >= session_capacity_ && !session_evict_one()) {
        // The catalog is full of entries the device cannot help with right now. This conversation
        // still runs - the device pools hold it - it just has no recalling entry.
        return;
    }
    SessionEntry entry;
    entry.tokens.assign(history.begin(), history.end());
    entry.frontier        = frontier;
    entry.device_resident = true;
    entry.lru_clock       = ++session_lru_clock_;
    sessions_.push_back(std::move(entry));
    active_session_       = sessions_.size() - 1;
    cached_prompt_tokens_ = sessions_[active_session_].tokens;
    live_state_valid_     = true;
}

GenerationResult TP2GenerationCore::execute(Request& request, OutputSink* sink,
                                            const CancellationView& cancellation) {
    // The live-frontier shortcut is only sound while the device GDN state sits exactly where the
    // catalog says it does. A walk that throws leaves it somewhere the catalog cannot name, so the
    // guard retires both the claim and the resident entry before the next request reads them.
    try {
        return execute_walk(request, sink, cancellation);
    } catch (...) {
        session_invalidate_active();
        throw;
    }
}

GenerationResult TP2GenerationCore::execute_walk(Request& request, OutputSink* sink,
                                                 const CancellationView& cancellation) {
    const bool streaming = request.consumer_mode == OutputConsumerMode::Streaming;
    auto& data = qwen::PreparedPromptAccess::mutable_view(request.prompt);
    const auto& token_ids = data.token_ids;
    // A multimodal request runs its Vision prefill on the Vision shard and then decodes through the
    // same speculative window as a text request. A chat turn that carries an image keeps that image
    // in every later turn's history, so treating "has media" as "no speculation" would cost the whole
    // conversation its MTP rounds; the request-level flag stays a property of the session, not of the
    // prompt.
    const bool media = data.has_media();
    const std::uint32_t prompt_tokens = static_cast<std::uint32_t>(token_ids.size());
    const std::int32_t vocab =
        qwen::execution::dimension(shard_a_.model->config().text.vocab_size);
    const std::int32_t hidden =
        qwen::execution::dimension(shard_a_.model->config().text.hidden_size);
    // The embedding and lm_head are packed to `vocab` rows, but only the first `public_tokens` of them
    // name something the tokenizer can spell; the rows in between are padding. Every selection over
    // these logits is therefore bounded by the public domain, so a padding row can never be sampled,
    // corrected, or licensed.
    const std::int32_t public_tokens =
        static_cast<std::int32_t>(shard_a_.model->resources().public_token_count);

    // Constrained tool-call decoding. The declared-name grammar turns the tool-call region into a
    // logit mask, so a name the request did not declare is unreachable at the sampling layer instead
    // of being rejected after the fact. The mask is applied outside the captured verify graph, to
    // the logits the graph hands back, so the graph itself is unchanged.
    std::shared_ptr<qwen::frontend::ToolCallConstraint> tool_constraint;
    if (data.tool_call_output != nullptr) {
        tool_constraint = frontend_->make_tool_call_constraint(data.tool_call_output);
    }
    const bool tool_constrained = tool_constraint != nullptr;
    // The logits domain is the packed embedding row count, which the artifact contract allows to be
    // wider than the tokenizer public domain the constraint table covers; build_mask() excludes the
    // rows in between. A table wider than the logits domain means the artifact and tokenizer disagree.
    const std::size_t logits_domain = static_cast<std::size_t>(vocab);
    if (tool_constrained && tool_constraint->vocab_size() > logits_domain) {
        throw std::logic_error("TP-2 tool-call constraint vocabulary " +
                               std::to_string(tool_constraint->vocab_size()) +
                               " exceeds the logits domain " + std::to_string(logits_domain));
    }

    GenerationResult result;
    result.prompt = request.summary;
    result.timings.prepare_seconds = request.prepare_seconds;
    const Clock::time_point start = Clock::now();
    // The timing events belong to shard A's device, so bind it before creating them.
    shard_a_.device.bind_to_current_thread();
    Tp2RoundTiming& timing = tp2_timing();
    timing.init();
    timing.reset();
    // Prefill-stage split, reported under NINFER_TP2_TIMING. The stages are separated by device
    // synchronizations so a host-side segment cannot hide device work, which perturbs the absolute
    // numbers a little; this is a diagnostic mode, not a measurement of the production path.
    const bool prefill_trace = [] {
        const char* env = std::getenv("NINFER_TP2_TIMING");
        return env != nullptr && env[0] == '1';
    }();
    std::uint32_t prefill_chunks      = 0;
    std::uint64_t prefill_host_writes = 0;
    Clock::time_point scan_done       = start;
    Clock::time_point state_done      = start;
    Clock::time_point walk_done       = start;

    // Cross-session recall, before the prefix scan reads the device lineage. A prompt that belongs
    // to another conversation makes the resident session swap out here, so by the time the scan
    // below compares against cached_prompt_tokens_ the pools already hold the conversation this
    // prompt continues. A prompt that continues the resident conversation is left untouched.
    session_recall(token_ids);

    // Prompt-prefix reuse. The KV pages hold the K/V of every position the last completed prefill
    // wrote, and the state snapshots hold the matching GDN states, so a prompt that extends the
    // previous prompt can skip the shared prefix and prefill only its suffix. The single-device
    // route gets this from the context cache; TP-2 runs with that cache disabled, so the core keeps
    // the boundaries it already owns. The deepest boundary at or before the shared prefix wins; the
    // final prompt token is always forwarded, because its logits drive the first sample.
    // NINFER_TP2_REUSE_TRACE=1 prints what the boundaries offered and what the prompt actually
    // matched. It is the difference between a client that resends its history verbatim and one that
    // re-renders it, which decides whether a boundary past the previous prompt can ever be used.
    const bool reuse_trace = [] {
        const char* env = std::getenv("NINFER_TP2_REUSE_TRACE");
        return env != nullptr && env[0] == '1';
    }();
    std::uint32_t reuse       = 0;
    std::size_t reuse_slot    = 0;
    std::size_t shared_prefix = 0;
    reuse_source_             = ReuseSource::None;
    // A recalled walk has to chunk its suffix exactly like a from-scratch walk of the same prompt,
    // and those chunks end on multiples of `prefill_chunk` (see the chunk plan below). A boundary
    // inside a chunk - the per-token checkpoints a decode leaves behind, or a shared prefix cut
    // mid-chunk - would make the two walks round differently and produce different logits, so only
    // grid-aligned boundaries qualify. Rounding down costs at most one chunk of re-prefill, and the
    // shared system-prompt boundary a cascade relies on is grid-aligned already.
    const std::uint32_t reuse_grid =
        std::min<std::uint32_t>(std::max<std::uint32_t>(options_.prefill_chunk, 64),
                                kPrefillChunkMaximum);
    if (!dflash2_enabled_ && cached_state_valid_ && !cached_prompt_tokens_.empty()) {
        const std::size_t common = std::min(cached_prompt_tokens_.size(), token_ids.size());
        while (shared_prefix < common &&
               cached_prompt_tokens_[shared_prefix] == token_ids[shared_prefix]) {
            ++shared_prefix;
        }
        // The live GDN state is the deepest boundary a continued conversation can offer: it sits
        // exactly at the resident entry's frontier, and the device KV holds the whole history
        // before it. Reusing it copies nothing at all - but only a prompt that extends it has a
        // column to forward, and the final prompt token must still be forwarded to produce the
        // logits that drive the first sample.
        if (live_state_valid_ && active_session_ != kNoSession) {
            const std::uint32_t frontier = sessions_[active_session_].frontier;
            if (frontier != 0 && frontier % reuse_grid == 0 && frontier <= shared_prefix &&
                frontier < prompt_tokens && frontier > reuse) {
                reuse        = frontier;
                reuse_source_ = ReuseSource::LiveState;
            }
        }
        for (std::size_t slot = 0; slot < kReuseSnapshotCount; ++slot) {
            const std::uint32_t boundary = cached_boundaries_[slot];
            if (boundary % reuse_grid == 0 && boundary <= shared_prefix && boundary < prompt_tokens &&
                boundary > reuse) {
                reuse        = boundary;
                reuse_slot   = slot;
                reuse_source_ = ReuseSource::DeviceSnapshot;
            }
        }
        // A host checkpoint is a state plus the KV before it, and both are only this prompt's while
        // the lineage agrees on the tokens before it: the state was taken over the tokens of the
        // prompt that wrote it, and every prefill since kept the KV consistent only up to the prefix
        // it reused. The shared prefix of the last pair is therefore the bound for every older
        // frontier, and a checkpoint past it can never become usable again - a later prompt would
        // have to match a token this one already diverged from. Prune on that bound, then take the
        // deepest survivor. Older checkpoints matter: a prefill that only walked the last few tokens
        // of a fully reused prompt leaves no checkpoint of its own.
        Shard* const shards[2] = {&shard_a_, &shard_b_};
        for (Shard* shard : shards) {
            for (auto& checkpoint : shard->host_checkpoints) {
                if (checkpoint.position > shared_prefix) { checkpoint.valid = false; }
            }
        }
        for (std::size_t index = 0; index < shard_a_.host_checkpoints.size(); ++index) {
            const auto& checkpoint = shard_a_.host_checkpoints[index];
            if (!checkpoint.valid || checkpoint.position % reuse_grid != 0 ||
                checkpoint.position <= reuse ||
                checkpoint.position > shared_prefix || checkpoint.position >= prompt_tokens) {
                continue;
            }
            reuse         = checkpoint.position;
            reuse_slot    = index;
            reuse_source_ = ReuseSource::HostCheckpoint;
        }
        // A lineage whose every boundary sits inside its first prefill chunk (a conversation far
        // shorter than one chunk) has no grid-aligned candidate at all, and the aligned scan above
        // would drop all the way to a full prefill. Keeping the deepest boundary is better: the
        // recalled state and KV are still this prompt's own, and only the suffix chunking differs
        // from a from-scratch walk. Rescan the same boundaries without the grid restriction.
        if (reuse == 0) {
            auto take = [&](std::uint32_t position, std::size_t slot, ReuseSource source) {
                if (position != 0 && position <= shared_prefix && position < prompt_tokens &&
                    position > reuse) {
                    reuse         = position;
                    reuse_slot    = slot;
                    reuse_source_ = source;
                }
            };
            if (live_state_valid_ && active_session_ != kNoSession) {
                take(sessions_[active_session_].frontier, 0, ReuseSource::LiveState);
            }
            for (std::size_t slot = 0; slot < kReuseSnapshotCount; ++slot) {
                take(cached_boundaries_[slot], slot, ReuseSource::DeviceSnapshot);
            }
            for (std::size_t index = 0; index < shard_a_.host_checkpoints.size(); ++index) {
                const auto& checkpoint = shard_a_.host_checkpoints[index];
                if (checkpoint.valid) { take(checkpoint.position, index, ReuseSource::HostCheckpoint); }
            }
        }
    }
    if (prefill_trace) { scan_done = Clock::now(); }
    if (reuse_trace) {
        std::size_t valid_checkpoints = 0;
        for (const auto& checkpoint : shard_a_.host_checkpoints) {
            valid_checkpoints += checkpoint.valid ? 1U : 0U;
        }
        const char* source = reuse_source_ == ReuseSource::HostCheckpoint ? "host"
                             : reuse_source_ == ReuseSource::LiveState    ? "live"
                             : reuse_source_ == ReuseSource::DeviceSnapshot ? "device"
                                                                            : "none";
        std::fprintf(stderr,
                     "[tp2-reuse] prompt=%u cached=%zu shared=%zu prefill_end=%u rewind=%u "
                     "host=%zu/%zu stride=%u -> reuse=%u slot=%zu src=%s\n",
                     prompt_tokens, cached_prompt_tokens_.size(), shared_prefix,
                     cached_boundaries_[0], cached_boundaries_[1], valid_checkpoints,
                     shard_a_.host_checkpoints.size(), host_checkpoint_stride_, reuse, reuse_slot,
                     source);
    }

    // Tag the checkpoints this walk leaves behind and start the ring at the first stride multiple
    // past the reused boundary: a checkpoint at the boundary itself would only duplicate the device
    // snapshot the walk starts from. The new checkpoints become usable when this prefill completes.
    host_checkpoint_live_id_           = host_checkpoint_next_id_++;
    std::uint32_t next_host_checkpoint = 0;
    if (host_checkpoint_stride_ != 0) {
        next_host_checkpoint = host_checkpoint_stride_;
        while (next_host_checkpoint <= reuse) { next_host_checkpoint += host_checkpoint_stride_; }
    }

    // Multimodal request: one Vision session owns the items this request still has to encode, on top
    // of the startup plan. An item that lies entirely inside a reused prefix is already in the KV -
    // its embeddings were scattered when that prefix was first prefilled - so the session covers only
    // the suffix, and a request whose reused prefix covers every item (the common case for a chat
    // turn that carries an image in its history) runs with no session at all. Its chunks still bind
    // the prompt's 3-axis RoPE table, which is a property of the prompt rather than of the encoding.
    qwen::execution::VisionPrefillPlan vision_plan;
    std::unique_ptr<qwen::execution::VisionPrefillSession> vision_session;
    if (media) {
        if (!vision_workspace_ || vision_arena_ == nullptr) {
            throw std::invalid_argument("multimodal request without a Vision workspace plan");
        }
        const auto& vision_config = shard_b_.model->config().vision.value();
        auto control_plan = std::make_shared<qwen::VisionControlPlan>(
            qwen::plan_vision_control(data, vision_config));
        std::size_t max_merged  = 0;
        std::size_t first_item  = control_plan->items.size();
        vision_plan.uses.reserve(control_plan->items.size());
        for (std::size_t index = 0; index < control_plan->items.size(); ++index) {
            const qwen::VisionItemControlPlan& item = control_plan->items[index];
            if (item.merged_count > vision_workspace_->max_merged_tokens) {
                throw std::invalid_argument(
                    "image needs " + std::to_string(item.merged_count) +
                    " vision tokens but this TP-2 route admitted only " +
                    std::to_string(vision_workspace_->max_merged_tokens) +
                    " at startup (see the [mem] vision ledger line)");
            }
            if (item.token_end <= reuse) {
                // Already encoded by the prefill that owns the reused prefix; the patch payload is
                // not needed again, so drop it instead of holding it for the request's lifetime.
                data.media_payloads[index].reset();
                continue;
            }
            if (first_item == control_plan->items.size()) { first_item = index; }
            vision_plan.uses.push_back(qwen::execution::VisionUseSpan{
                .begin               = item.token_begin,
                .end                 = item.token_end,
                .prepared_item_index = static_cast<std::uint32_t>(index),
                .control_index       = 0,
            });
            max_merged = std::max(max_merged, item.merged_count);
        }
        if (!vision_plan.uses.empty()) {
            const auto first = static_cast<std::uint32_t>(first_item);
            vision_plan.max_merged_count = max_merged;
            vision_plan.control          = std::make_shared<const qwen::VisionControl>(
                qwen::build_vision_control(data, *control_plan, first));
            for (qwen::execution::VisionUseSpan& use : vision_plan.uses) {
                use.control_index = use.prepared_item_index - first;
            }
            shard_b_.device.bind_to_current_thread();
            vision_session = std::make_unique<qwen::execution::VisionPrefillSession>(
                shard_b_.device, *shard_b_.parameters,
                DeviceSpan{vision_arena_->base(), vision_arena_->capacity()}, *vision_workspace_,
                data, vision_plan, vision_handoff_peak_bytes_);
        }
    }

    // Reset per-request state on both shards. The KV pages and execution row 0 are materialized
    // once at startup (build_shard) and reused in place, so a reused prefix needs no KV work at
    // all; only the GDN state must be restored to (or reset at) the prefill frontier.
    auto begin_gdn_state = [&](Shard& shard) {
        shard.device.bind_to_current_thread();
        switch (reuse_source_) {
        case ReuseSource::None:
            CUDA_CHECK(cudaMemsetAsync(shard.state_backing.data, 0, shard.state_backing.bytes,
                                       shard.device.stream));
            return;
        case ReuseSource::LiveState:
            // The device state already sits at this boundary: a restored session put it there, or
            // the conversation that just decoded left it exactly at its frontier. Nothing to copy.
            return;
        case ReuseSource::DeviceSnapshot:
            CUDA_CHECK(cudaMemcpyAsync(shard.state_backing.data,
                                       shard.state_snapshots[reuse_slot].data,
                                       shard.state_backing.bytes, cudaMemcpyDeviceToDevice,
                                       shard.device.stream));
            return;
        case ReuseSource::HostCheckpoint:
            CUDA_CHECK(cudaMemcpyAsync(shard.state_backing.data,
                                       shard.host_checkpoints[reuse_slot].buffer->data(),
                                       shard.state_backing.bytes, cudaMemcpyHostToDevice,
                                       shard.device.stream));
            return;
        }
    };
    begin_gdn_state(shard_a_);
    begin_gdn_state(shard_b_);
    if (prefill_trace) {
        CUDA_CHECK(cudaStreamSynchronize(shard_a_.device.stream));
        CUDA_CHECK(cudaStreamSynchronize(shard_b_.device.stream));
        state_done = Clock::now();
    }

    if (streaming) {
        sink->start(GenerationStart{.prompt = request.summary, .reused_prompt_tokens = reuse});
    }
    result.reused_prompt_tokens = reuse;

    // Sampling config, device-resident, for ops::sample.
    ops::SamplingConfig sampling_config = make_sampling_config(request.sampling);
    auto sampling_scope = shard_a_.workspace->scope();
    auto& ws_a = *shard_a_.workspace;
    auto& ws_b = *shard_b_.workspace;
    // Reserved on the packed row count, which is an upper bound over every token domain ops::sample
    // is called with below.
    const std::size_t sampling_ws = ops::sampling_workspace_capacity_bytes(vocab, 1, 1);
    auto sampling_buf_a = ws_a.alloc_bytes(sizeof(ops::SamplingConfig) + sampling_ws, 256);
    auto sampling_buf_b = ws_b.alloc_bytes(sizeof(ops::SamplingConfig) + sampling_ws, 256);
    // A configured penalty reads the request's committed-token counts through this array, and both
    // ops::sample and the speculative accept kernel add to it as they produce every token. Reset it:
    // the arena hands back the previous request's bytes. With no penalty the op reads no counts at
    // all, so the array is only created when a request actually asks for one.
    if (sampling_config.presence_penalty != 0.0F || sampling_config.frequency_penalty != 0.0F) {
        const Tensor counts = ws_a.alloc(DType::I32, {public_tokens});
        shard_a_.device.bind_to_current_thread();
        CUDA_CHECK(cudaMemsetAsync(counts.data, 0, counts.bytes(), shard_a_.device.stream));
        sampling_config.token_counts = static_cast<std::int32_t*>(counts.data);
    }
    shard_a_.device.bind_to_current_thread();
    CUDA_CHECK(cudaMemcpyAsync(sampling_buf_a.data, &sampling_config,
                               sizeof(ops::SamplingConfig), cudaMemcpyHostToDevice,
                               shard_a_.device.stream));
    shard_b_.device.bind_to_current_thread();
    CUDA_CHECK(cudaMemcpyAsync(sampling_buf_b.data, &sampling_config,
                               sizeof(ops::SamplingConfig), cudaMemcpyHostToDevice,
                               shard_b_.device.stream));
    const auto* sampling_a = static_cast<const ops::SamplingConfig*>(sampling_buf_a.data);
    const auto* sampling_b = static_cast<const ops::SamplingConfig*>(sampling_buf_b.data);
    Tensor logical_pos_a = ws_a.alloc(DType::I32, {1});
    Tensor logical_pos_b = ws_b.alloc(DType::I32, {1});
    (void)sampling_b;
    (void)logical_pos_b;

    // Constrained-decoding scratch. The mask is computed on the host (declared-name trie over the
    // decoded vocabulary pieces) and uploaded once per constrained step. The device buffer is
    // allocated in the request scope, below every round watermark, so it cannot disturb a captured
    // verify graph.
    const std::int32_t constraint_columns =
        dflash2_enabled_ ? static_cast<std::int32_t>(dflash_drafts_) + 1
        : mtp_enabled_  ? static_cast<std::int32_t>(mtp_drafts_) + 1
                        : 1;
    Tensor tool_mask_dev;
    std::vector<std::uint8_t> tool_mask_one;
    std::vector<std::uint8_t> tool_mask_columns;
    std::size_t constraint_fed = 0;
    if (tool_constrained) {
        tool_mask_dev = ws_a.alloc(DType::U8, {vocab, constraint_columns});
        tool_mask_columns.resize(static_cast<std::size_t>(vocab) *
                                 static_cast<std::size_t>(constraint_columns));
    }
    auto constraint_advance = [&]() {
        if (!tool_constrained) { return; }
        const std::string_view raw = request.output.raw_content_text();
        if (raw.size() > constraint_fed) {
            tool_constraint->feed(raw.substr(constraint_fed));
            constraint_fed = raw.size();
        }
    };
    // The mask binds only while the committed output is content: the tool parser reads no
    // reasoning channel, so the grammar must not constrain the thinking phase either.
    auto constraint_live = [&]() {
        return tool_constrained && !request.output.in_reasoning();
    };

    auto& ctx_a = *shard_a_.context;
    auto& ctx_b = *shard_b_.context;

    auto publish_preview = [&](bool terminal) {
        if (terminal) { (void)request.output.preview_terminal(request.budget.limit_reason()); }
        auto published = request.output.commit_preview();
        for (auto& delta : published) {
            (delta.channel == OutputChannel::Reasoning ? result.reasoning : result.content) +=
                delta.text;
            if (streaming) { sink->publish(delta); }
        }
    };

    // Prefix-reuse snapshots are taken on the shard streams, so each one captures the GDN state
    // exactly at its boundary: the enclosing loop has enqueued every earlier token's forward and
    // none of the later ones yet.
    auto snapshot_state = [&](Shard& shard, std::size_t slot) {
        shard.device.bind_to_current_thread();
        CUDA_CHECK(cudaMemcpyAsync(shard.state_snapshots[slot].data, shard.state_backing.data,
                                   shard.state_backing.bytes, cudaMemcpyDeviceToDevice,
                                   shard.device.stream));
    };
    // The same copy into the host ring, tagged with the frontier it captures. It rides the shard
    // stream, so it sees exactly the tokens the enclosing loop has enqueued and none of the later
    // ones, and the ring slot it lands in is only revisited by a later prefill's checkpoint at the
    // same index.
    // Which reserved group of the ring a checkpoint lands in. The grid and the tail anchors rotate
    // through their own slots; the divergence anchor owns a single slot and is rewritten in place,
    // so it advances no cursor and neither rotation can evict it.
    enum class HostRing { Grid, Tail, Divergence };
    auto snapshot_host_checkpoint = [&](Shard& shard, std::uint32_t frontier, HostRing ring) {
        if (shard.host_checkpoints.empty()) { return; }
        ++prefill_host_writes;
        const std::size_t grid = shard.host_checkpoint_grid_slots;
        const std::size_t tail = host_checkpoint_tail_slots_;
        std::size_t begin      = 0;
        std::size_t count      = grid;
        switch (ring) {
            case HostRing::Grid: break;
            case HostRing::Tail: begin = grid; count = tail; break;
            case HostRing::Divergence:
                begin = grid + tail;
                count = host_checkpoint_divergence_slots_;
                break;
        }
        if (count == 0 || begin + count > shard.host_checkpoints.size()) { return; }
        const std::size_t index = ring == HostRing::Tail   ? shard.host_checkpoint_tail_next
                                  : ring == HostRing::Grid ? shard.host_checkpoint_next
                                                           : 0;
        shard.device.bind_to_current_thread();
        Shard::HostCheckpoint& checkpoint = shard.host_checkpoints[begin + index];
        // Overwriting the slot already destroys the checkpoint it held, so retire it before the copy
        // is enqueued: a prefill that throws between here and the publish sweep must not leave a
        // torn buffer behind a flag that still says the last completed prefill wrote it. Only the
        // sweep below, which runs after the walk finished, makes a checkpoint usable again.
        checkpoint.valid      = false;
        checkpoint.position   = frontier;
        checkpoint.prefill_id = host_checkpoint_live_id_;
        CUDA_CHECK(cudaMemcpyAsync(checkpoint.buffer->data(), shard.state_backing.data,
                                   shard.state_backing.bytes, cudaMemcpyDeviceToHost,
                                   shard.device.stream));
        if (ring == HostRing::Tail) {
            shard.host_checkpoint_tail_next = (index + 1) % count;
        } else if (ring == HostRing::Grid) {
            shard.host_checkpoint_next = (index + 1) % count;
        }
    };

    // Prefill: batched forwards over chunk-sized slices of the prompt suffix, accumulating KV and
    // GDN state in place. A reused prefix starts the walk at its boundary; those positions keep
    // their K/V from the earlier prefill and the GDN state restored above. A chunk reads every
    // weight once, so the weight traffic that dominates a per-token walk is paid once per chunk
    // instead of once per token.
    //
    // The next request's rewind snapshots can only be frozen on a chunk boundary, so a chunk whose
    // end would step over one of the target depths is truncated to end exactly on it. Slot 0 is the
    // prefill end, which is always a boundary.
    // Chunk width from the engine option, clamped to what the cross-device allreduce staging buffer
    // (DevicePair) and the per-chunk activation peak (workspace) can carry.
    const std::uint32_t prefill_chunk =
        std::min<std::uint32_t>(std::max<std::uint32_t>(options_.prefill_chunk, 64),
                                kPrefillChunkMaximum);
    // The tail anchors cover the last few chunk ends of the walk. Their count is bounded by the tail
    // sub-ring, so a narrow chunk cannot flood the ring with anchors that all sit within one chunk of
    // the prompt end.
    const std::uint32_t tail_span =
        std::min<std::uint32_t>(kReuseTailWindow,
                                prefill_chunk * std::max(1U, host_checkpoint_tail_slots_));
    std::array<std::uint32_t, kReuseSnapshotCount> snapshot_at{};
    snapshot_at[0] = prompt_tokens;
    // Rewind snapshots sit on the walk's own chunk boundaries, which are fixed by `reuse`,
    // `prompt_tokens` and `prefill_chunk` alone. They used to be derived from the last observed
    // divergence (`rewind_near_`), which shortened whichever chunk covered the target: the walk's
    // chunking then depended on engine history, so the same prompt prefilled with different chunk
    // widths from one walk to the next and produced different logits. The dense tail ring already
    // keeps a rewind cheap, which is not worth history-dependent arithmetic here.
    for (std::size_t slot = 1; slot < kReuseSnapshotCount; ++slot) {
        const std::uint32_t span = prompt_tokens - reuse;
        std::uint32_t last       = span % prefill_chunk;
        if (last == 0) { last = std::min(prefill_chunk, span); }
        snapshot_at[slot] = (span > last) ? prompt_tokens - last : 0;
        if (snapshot_at[slot] != 0 && snapshot_at[slot] == reuse) {
            // The restored state already sits exactly on this boundary, and the walk never revisits
            // its own starting point, so freeze it before the first chunk.
            snapshot_state(shard_a_, slot);
            snapshot_state(shard_b_, slot);
        }
    }
    // Divergence anchor. shared_prefix is where this prompt stopped matching the lineage it
    // inherited, and the next new session - or the next context compression - renders the same stable
    // block and diverges in the same place, so it is the one position a later request is already
    // known to want. Freezing the state there lets that request restart on the boundary instead of at
    // the grid point behind it. The anchor sits kReuseDivergenceMargin tokens *before* the observed
    // divergence rather than on it: the next pair can report a shared prefix a token or two shorter,
    // and an anchor on the exact index would be pruned before it could ever be used. Only a frontier
    // strictly inside this walk is worth freezing: at or below reuse the walk's own starting state is
    // already checkpointed, and at the prompt end the device snapshot holds it. A checkpoint that
    // already sits there stays untouched, which is what keeps a reused stable prefix free of work.
    const auto has_valid_host_checkpoint_at = [](const Shard& shard, std::uint32_t position) {
        for (const auto& checkpoint : shard.host_checkpoints) {
            if (checkpoint.valid && checkpoint.position == position) { return true; }
        }
        return false;
    };
    const std::uint32_t anchor_position = shared_prefix > kReuseDivergenceMargin
                                              ? shared_prefix - kReuseDivergenceMargin
                                              : shared_prefix;
    const bool anchor_divergence = host_checkpoint_stride_ != 0 && anchor_position > reuse &&
                                   anchor_position < prompt_tokens &&
                                   !has_valid_host_checkpoint_at(shard_a_, anchor_position) &&
                                   !has_valid_host_checkpoint_at(shard_b_, anchor_position);
    // The conversation the recall just moved into its host slabs stopped matching this prompt exactly
    // where the walk starts. The device state sitting there right now is its state at that boundary
    // too, so freeze it before the first chunk overwrites it: a later conversation opening with the
    // same stable block can then be recalled onto the entry's own slab instead of prefilling the
    // block again. The device pools still hold it because the recall restored it and nothing has run
    // since.
    if (host_stored_session_ != kNoSession && reuse != 0 && reuse == shared_prefix) {
        session_capture_shared_state(host_stored_session_, reuse, true, nullptr);
    }
    // The masked-draft route restarts the draft context with the request. Its ring is not part of
    // any session or checkpoint image yet (PLAN.md section 3.6, stage B6) and the route forfeited
    // reuse above, so the walk is about to capture the whole prompt again; zeroing first makes the
    // untouched tail explicitly empty rather than a previous request's features.
    dflash_context_frontier_ = 0;
    if (dflash2_enabled_) {
        if (shard_a_.dflash_round == nullptr) {
            throw std::logic_error("TP-2 DFlash2 route has no masked-draft round");
        }
        shard_a_.dflash_round->zero_context();
    }
    for (std::uint32_t t0 = reuse; t0 < prompt_tokens;) {
        if (cancellation.requested()) {
            // Chunks that finished wrote KV for tokens the prompt really has and left the GDN state
            // at the end of the last one, so the catalog can name where the walk stopped. Publishing
            // that much costs nothing and lets the retry of the same prompt continue from there
            // instead of prefilling its whole history again. Before the first chunk nothing moved:
            // the recall's bookkeeping already describes the device pools, so that case leaves the
            // catalog exactly as the recall left it.
            if (t0 > 0) {
                snapshot_state(shard_a_, 0);
                snapshot_state(shard_b_, 0);
                cached_boundaries_[0] = t0;
                cached_boundaries_[1] = 0;
                cached_state_valid_   = true;
                session_publish(
                    std::vector<TokenId>(token_ids.begin(),
                                         token_ids.begin() + static_cast<std::ptrdiff_t>(t0)),
                    t0);
                if (active_session_ != kNoSession) { sessions_[active_session_].prompt_end = t0; }
            }
            (void)request.output.preview_terminal(FinishReason::Cancelled);
            publish_preview(false);
            result.finish_reason = FinishReason::Cancelled;
            result.timings.total_seconds =
                std::chrono::duration<double>(Clock::now() - start).count();
            result.timings.prompt_wall_seconds = result.timings.total_seconds;
            return result;
        }
        std::uint32_t length = std::min(prefill_chunk, prompt_tokens - t0);
        // A multimodal chunk is capped at the boundary of the item it overlaps, so the encoder hands
        // out one item at a time and the scatter below stays one contiguous column range of it. The
        // snapshot clamp still comes last: the boundary a rewind slot must freeze on wins over the
        // Vision cap, and the next chunk simply re-enters the same item.
        qwen::execution::VisionChunk vision_chunk;
        if (vision_session) {
            vision_chunk = vision_session->prepare_chunk(t0, length);
            length       = static_cast<std::uint32_t>(vision_chunk.length);
        }
        for (std::size_t slot = 1; slot < kReuseSnapshotCount; ++slot) {
            const std::uint32_t target = snapshot_at[slot];
            if (target > t0 && target < t0 + length) { length = target - t0; }
        }
        // The grid checkpoints are the only boundaries a prompt that diverges early can still use,
        // and a checkpoint taken at the first chunk end *past* a stride covers more tokens than the
        // stride asks for: a prompt sharing a stable block shorter than that coverage has nothing to
        // restart on and prefills the block again. Ending the chunk on the stride costs one shorter
        // chunk per stride and puts every grid checkpoint exactly on the tokens before it.
        if (next_host_checkpoint > t0 && next_host_checkpoint < t0 + length) {
            length = next_host_checkpoint - t0;
        }
        if (anchor_divergence && anchor_position > t0 && anchor_position < t0 + length) {
            length = anchor_position - t0;
        }
        qwen::execution::Tp2VisionChunk media_chunk;
        const qwen::execution::Tp2VisionChunk* media_ptr = nullptr;
        if (media) {
            media_chunk.control       = vision_chunk.control;
            media_chunk.embeddings    = &vision_chunk.embeddings;
            media_chunk.positions     = data.positions.data();
            media_chunk.prompt_tokens = data.token_ids.size();
            media_ptr                 = &media_chunk;
        }
        // Scope both shards' workspaces so each chunk starts from a clean arena. The batched
        // forward allocates all of its intermediate activations in each shard's workspace; without
        // a scope the next chunk's arena would build on this one and eventually overflow.
        auto scope_a = ws_a.scope();
        auto scope_b = ws_b.scope();
        Tensor logits_a = ws_a.alloc(DType::BF16, {vocab, 1});
        Tensor logits_b = ws_b.alloc(DType::BF16, {vocab, 1});
        // MTP priming consumes the chunk's final-norm hidden, so the forward hands it back.
        Tensor mtp_input_a;
        if (mtp_enabled_) {
            mtp_input_a = ws_a.alloc(DType::BF16, {hidden, static_cast<std::int32_t>(length)});
        }
        // The masked draft taps this shard's prefill residual at its configured block ids; the
        // sink is empty unless this shard materialized the draft component (shard 0 under a
        // DFlash/DFlash2 backend), so every other route keeps the NullTap forward.
        auto dflash_sink = make_dflash_prefill_sink(shard_a_);
        ctx_a.forward_tp2_prefill(ctx_b, pair_, std::span<const int>(token_ids.data() + t0, length),
                                  static_cast<std::int32_t>(t0), &logits_a, &logits_b,
                                  mtp_enabled_ ? &mtp_input_a : nullptr, nullptr, nullptr,
                                  qwen::TextPhase::Prefill, media_ptr,
                                  dflash_sink ? &*dflash_sink : nullptr);
        if (dflash_sink) {
            // The sink's consumer ran synchronously inside the forward and appended this chunk's
            // absolute positions to the draft ring, so the context frontier tracks the prefill.
            dflash_context_frontier_ = t0 + length;
        }
        if (mtp_enabled_ && t0 + length != prompt_tokens) {
            mtp_prefill_priming(shard_a_, token_ids.data() + t0, length, t0, mtp_input_a, nullptr,
                                false);
        }
        for (std::size_t slot = 1; slot < kReuseSnapshotCount; ++slot) {
            if (snapshot_at[slot] == t0 + length) {
                snapshot_state(shard_a_, slot);
                snapshot_state(shard_b_, slot);
            }
        }
        if (host_checkpoint_stride_ != 0) {
            // One checkpoint per stride, tagged with the frontier this chunk actually reached, so a
            // chunk width that does not divide the stride cannot mislabel a state; plus the dense
            // tail window. The end of the prompt is skipped either way: the device snapshot below
            // holds that same state already.
            const std::uint32_t frontier = t0 + length;
            const bool on_grid           = frontier >= next_host_checkpoint;
            const bool in_tail           = frontier != prompt_tokens &&
                                 static_cast<std::uint64_t>(frontier) + tail_span > prompt_tokens;
            if (on_grid) {
                snapshot_host_checkpoint(shard_a_, frontier, HostRing::Grid);
                snapshot_host_checkpoint(shard_b_, frontier, HostRing::Grid);
                next_host_checkpoint =
                    (frontier / host_checkpoint_stride_ + 1U) * host_checkpoint_stride_;
            } else if (in_tail) {
                snapshot_host_checkpoint(shard_a_, frontier, HostRing::Tail);
                snapshot_host_checkpoint(shard_b_, frontier, HostRing::Tail);
            }
            if (anchor_divergence && frontier == anchor_position) {
                snapshot_host_checkpoint(shard_a_, frontier, HostRing::Divergence);
                snapshot_host_checkpoint(shard_b_, frontier, HostRing::Divergence);
            }
        }
        if (t0 + length == prompt_tokens) {
            // First token: sample from the last chunk's last-column logits.
            ops::set_i32_scalar(logical_pos_a, static_cast<std::int32_t>(prompt_tokens),
                                shard_a_.device.stream);

            Tensor sampled_a = ws_a.alloc(DType::I32, {1});
            ops::sample(logits_a, sampled_a, public_tokens, sampling_a, logical_pos_a,
                        ops::kSamplePurposePrefill, ws_a, shard_a_.device.stream);
            std::int32_t first = 0;
            shard_a_.device.bind_to_current_thread();
            CUDA_CHECK(cudaMemcpyAsync(&first, sampled_a.data, sizeof(std::int32_t),
                                       cudaMemcpyDeviceToHost, shard_a_.device.stream));
            CUDA_CHECK(cudaStreamSynchronize(shard_a_.device.stream));
            request.generated.push_back(first);
            request.budget.commit(1);
            if (mtp_enabled_) {
                // The final MTP column embeds the token just sampled, so the MTP layer's own K/V for
                // the prompt is appended only after the first token exists.
                mtp_prefill_priming(shard_a_, token_ids.data() + t0, length, t0, mtp_input_a,
                                    &sampled_a, true);
            }
        }
        ++prefill_chunks;
        t0 += length;
    }
    // The divergence anchor is the state of the block the two conversations still agreed on: the one
    // the recall just moved into its host slabs agrees with this prompt up to the same token, so the
    // anchor is its state there as well. Copying it into that entry's own image pairs the state with
    // the KV its slab already holds, and unlike the ring slot it survives the next evictions. It
    // lands before session_publish, which may erase the entry and shift every later index.
    if (host_stored_session_ != kNoSession && anchor_divergence) {
        const PinnedHostBuffer* frozen[2] = {nullptr, nullptr};
        bool complete                    = true;
        for (std::size_t shard_index = 0; shard_index < 2 && complete; ++shard_index) {
            Shard& shard = shard_index == 0 ? shard_a_ : shard_b_;
            for (const auto& checkpoint : shard.host_checkpoints) {
                if (checkpoint.prefill_id == host_checkpoint_live_id_ &&
                    checkpoint.position == anchor_position) {
                    frozen[shard_index] = checkpoint.buffer.get();
                    break;
                }
            }
            complete = frozen[shard_index] != nullptr;
        }
        if (complete) {
            Shard* const anchor_shards[2] = {&shard_a_, &shard_b_};
            for (Shard* shard : anchor_shards) {
                shard->device.bind_to_current_thread();
                CUDA_CHECK(cudaStreamSynchronize(shard->device.stream));
            }
            session_capture_shared_state(host_stored_session_, anchor_position, false, frozen);
        }
    }
    if (prefill_trace) {
        CUDA_CHECK(cudaStreamSynchronize(shard_a_.device.stream));
        CUDA_CHECK(cudaStreamSynchronize(shard_b_.device.stream));
        walk_done = Clock::now();
    }
    computed_prefill_tokens_ += prompt_tokens - reuse;
    if (vision_session) {
        // Every item the walk overlapped is encoded and its embeddings are in the KV now, so release
        // the host patch payloads and the handoff binding: the decode loop never revisits them.
        vision_session->release_encoded_media_payloads();
        vision_session->retire_handoff();
    }

    // Freeze the GDN state at every boundary this request can offer the next one. Slot 0 is the
    // prefill end the walk just reached; the rewind slots were captured at their chunk boundaries.
    // A slot is published only when this prefill actually reached its boundary, so a request that
    // fails mid-prefill leaves the previous request's boundaries and snapshots intact.
    snapshot_state(shard_a_, 0);
    snapshot_state(shard_b_, 0);
    cached_prompt_tokens_.assign(token_ids.begin(), token_ids.end());
    for (std::size_t slot = 0; slot < kReuseSnapshotCount; ++slot) {
        cached_boundaries_[slot] = snapshot_at[slot];
    }
    cached_state_valid_ = true;
    // The pools now hold exactly this prompt and the live GDN state sits at its end, so the
    // resident catalog entry describes what the walk just wrote. A conversation that had no entry
    // yet gets one here, before decode extends its history.
    session_publish(token_ids, prompt_tokens);
    // The entry now describes the prompt the device pools were built from, which is what a later
    // request compares against to tell a later turn of this conversation from a switch away.
    if (active_session_ != kNoSession) { sessions_[active_session_].prompt_end = prompt_tokens; }
    // Only now are this prefill's checkpoints usable: their state is one the walk reached and their
    // KV prefix is one the walk wrote.
    {
        Shard* const shards[2] = {&shard_a_, &shard_b_};
        for (Shard* shard : shards) {
            for (auto& checkpoint : shard->host_checkpoints) {
                if (checkpoint.prefill_id == host_checkpoint_live_id_) { checkpoint.valid = true; }
            }
        }
    }

    if (prefill_trace) {
        const auto millis = [](Clock::time_point from, Clock::time_point to) {
            return std::chrono::duration<double, std::milli>(to - from).count();
        };
        const Clock::time_point report = Clock::now();
        std::fprintf(stderr,
                     "[tp2-prefill] reuse=%u suffix=%u chunks=%u host_writes=%llu | scan=%.2f "
                     "state=%.2f walk=%.2f post=%.2f total=%.2f ms\n",
                     reuse, prompt_tokens - reuse, prefill_chunks,
                     static_cast<unsigned long long>(prefill_host_writes), millis(start, scan_done),
                     millis(scan_done, state_done), millis(state_done, walk_done),
                     millis(walk_done, report), millis(start, report));
    }

    // Prompt wall time spans prefill and the first-sample step: the first accepted token is
    // produced by the last prefill iteration. That is a stable commit boundary (the sampled token
    // is read back on shard A after both shards finished the forward), matching the single-device
    // Engine's phase accounting. Generation wall time covers the decode rounds.
    const Clock::time_point first_token_time = Clock::now();
    const double prompt_seconds = std::chrono::duration<double>(first_token_time - start).count();
    // The Vision encode is reported separately from the text walk, matching the single-device route;
    // prompt wall time (the response's TTFT) is the whole span either way.
    result.timings.vision_seconds = vision_session ? vision_session->elapsed_seconds() : 0.0;
    result.timings.prefill_seconds =
        std::max(0.0, prompt_seconds - result.timings.vision_seconds);
    result.timings.prompt_wall_seconds = prompt_seconds;

    // Decode: with MTP each round verifies a K-draft window on both shards, records its
    // linear-attention transitions instead of committing them, and folds only the prefix the target
    // and the output policy accept. Without MTP it is a plain one-token autoregressive loop.
    std::int32_t current = request.generated.back();
    std::uint32_t position = prompt_tokens;
    // MTP round state: the anchor token whose transition is still pending, its position, and the
    // final-norm hidden of the column before it (the bridge's hidden input).
    std::int32_t mtp_anchor    = current;
    std::uint32_t mtp_position = position;
    // Per-round scratch starts here: the per-request tensors above (the sampling buffers, the
    // logical positions) must survive every round, and the prefix ends at the request's first
    // round. Every decode round allocates and frees its own scratch above this watermark, so a
    // round that discards its speculative chain by rewinding to it gets the same allocation
    // layout every time - which is what a captured verify graph bakes into its kernel arguments.
    shard_a_.round_base = ws_a.used();
    shard_b_.round_base = ws_b.used();
    bool finished = false;
    while (!finished) {
        if (cancellation.requested()) {
            (void)request.output.preview_terminal(FinishReason::Cancelled);
            publish_preview(false);
            result.finish_reason = FinishReason::Cancelled;
            break;
        }
        if (request.budget.remaining() == 0) {
            (void)request.output.preview_terminal(request.budget.limit_reason());
            publish_preview(false);
            result.finish_reason = request.budget.limit_reason();
            break;
        }
        // Scope both shards' workspaces so each decode round starts from a clean arena (see the
        // prefill loop above).
        auto scope_a = ws_a.scope();
        auto scope_b = ws_b.scope();
        std::vector<TokenId> step;
        Tensor round_hidden;
        // True when the DFlash2 round found no room for a window and fell back to a plain step,
        // which already advanced current/position and updated the target state in place.
        bool dflash_plain_step = false;
        // Phase marks: 0 round start, 1 after the MTP proposal chain, 2 after the verify forward,
        // 3 after the licensing kernels, 4 after the licenced-token readback, 5/6 around the state
        // fold. The plain loop has no proposal chain and no fold, so it collapses 1 onto 0.
        const Clock::time_point round_start = Clock::now();
        timing.record(0, shard_a_.device.stream);
        if (!mtp_enabled_) { timing.record(1, shard_a_.device.stream); }
        if (dflash2_enabled_) {
            // One masked-draft round, mirroring the single-device sequence
            // (execution/draft.cpp dflash_decode_batch_body): commit the pending verify window of
            // the previous round into the draft ring, propose the masked block, verify the window
            // through the target with the feature sink installed, then accept with DFlash2's sparse
            // semantics and fold only the committed columns.
            Shard& shard = shard_a_;
            auto& round  = *shard.dflash_round;
            const std::uint32_t budget_remaining = request.budget.remaining();
            const std::uint32_t max_by_budget =
                budget_remaining > 1 ? budget_remaining - 1U : 0U;
            const std::uint32_t capacity_left =
                position + 1U < options_.max_context ? options_.max_context - position - 1U : 0U;
            const std::uint32_t extent =
                std::min({dflash_drafts_, max_by_budget, capacity_left});
            const std::int32_t width = static_cast<std::int32_t>(dflash_drafts_) + 1;
            if (extent == 0) {
                // The window has no room for a draft (the output budget or the context ends after
                // this token). Forward the anchor as a plain step; the target state advances in
                // place and the request terminates, so the draft context has no next round to feed.
                Tensor logits_a = run_plain_decode_step(current, position);
                timing.record(2, shard_a_.device.stream);
                ops::set_i32_scalar(logical_pos_a, static_cast<std::int32_t>(position + 1),
                                    shard_a_.device.stream);
                if (constraint_live()) {
                    constraint_advance();
                    if (tool_constraint->build_mask(logits_domain, tool_mask_one)) {
                        shard_a_.device.bind_to_current_thread();
                        CUDA_CHECK(cudaMemcpyAsync(tool_mask_dev.data, tool_mask_one.data(),
                                                   tool_mask_one.size(),
                                                   cudaMemcpyHostToDevice,
                                                   shard_a_.device.stream));
                        ops::apply_token_mask(logits_a, tool_mask_dev, shard_a_.device.stream);
                    }
                }
                Tensor sampled_a = ws_a.alloc(DType::I32, {1});
                ops::sample(logits_a, sampled_a, public_tokens, sampling_a, logical_pos_a,
                            ops::kSamplePurposeDecode, ws_a, shard_a_.device.stream);
                timing.record(3, shard_a_.device.stream);
                std::int32_t next = 0;
                shard_a_.device.bind_to_current_thread();
                CUDA_CHECK(cudaMemcpyAsync(&next, sampled_a.data, sizeof(std::int32_t),
                                           cudaMemcpyDeviceToHost, shard_a_.device.stream));
                timing.record(4, shard_a_.device.stream);
                const Clock::time_point sync_start = Clock::now();
                CUDA_CHECK(cudaStreamSynchronize(shard_a_.device.stream));
                timing.close_round(
                    std::chrono::duration<double, std::milli>(Clock::now() - sync_start).count());
                current = next;
                ++position;
                dflash_plain_step = true;
                step.assign(1, static_cast<TokenId>(next));
            } else {
                const qwen::execution::ExecutionCore dflash_execution{
                    .device           = shard.device,
                    .parameters       = *shard.parameters,
                    .work             = *shard.workspace,
                    .linear_attention = *shard.state,
                    .replay_records   = nullptr,
                    .io               = shard.io,
                    .prefill_hidden   = shard.prefill_hidden,
                    .prefill_chunk    = options_.prefill_chunk,
                    .proposal_head    = options_.speculative.proposal_head,
                };
                // Hand off the previous verify window: its columns [C, E) are the target residual
                // the draft context is missing, exactly the prepare_ragged_prefix append the
                // single-device route makes at the head of its round.
                round.append_pending(dflash_execution, dflash_context_frontier_, position);
                dflash_context_frontier_ = position;
                qwen::DFlashDecodeIngress& ingress = round.ingress();
                ingress                            = {};
                ingress.anchors[0]                 = current;
                ingress.execution_frontiers[0]     = static_cast<std::int32_t>(position);
                ingress.context_frontiers[0]       = static_cast<std::int32_t>(position);
                ingress.proposal_extents[0]        = static_cast<std::int32_t>(extent);
                ingress.proposal_valid_columns[0]  = width;
                ingress.target_valid_columns[0]    = static_cast<std::int32_t>(extent) + 1;
                for (std::int32_t column = 0; column < width; ++column) {
                    const std::int32_t offset =
                        std::min(column, static_cast<std::int32_t>(extent));
                    ingress.target_rope_positions[column] =
                        static_cast<std::int32_t>(position) + offset;
                }
                ingress.text_kv_table_rows[0]      = 0;
                ingress.dflash_kv_table_rows[0]    = 0;
                ingress.active_lanes[0]            = 0;
                ingress.state_source_slots[0]      = 0;
                ingress.state_destination_slots[0] = 0;
                ingress.sampling[0]                = sampling_config;
                const qwen::execution::DFlashEnvelopes envelopes{
                    .local  = {0, position},
                    .full   = {0, position},
                    .append = {0, static_cast<std::uint32_t>(width)},
                };
                shard.device.bind_to_current_thread();
                round.propose(dflash_execution, shard.decoder->text_kv, shard.context.get(),
                              dflash_drafts_, envelopes);
                timing.record(1, shard_a_.device.stream);

                qwen::DFlashDecodeState& frame = round.frame();
                Tensor window_logits = frame.target_logits.view({vocab, width});
                Tensor window_hidden = frame.target_hidden.view({hidden, width});
                ops::speculative_prepare_verify_inputs(
                    frame.anchors, frame.draft_tokens, frame.execution_frontiers,
                    frame.proposal_extents, frame.verify_ids, frame.verify_positions,
                    shard_a_.device.stream);
                // The window is assembled in the same pinned buffer the MTP verify uses, so the
                // eager forward reads the verified ids/positions directly.
                auto* window_ids       = static_cast<std::int32_t*>(verify_window_host_->data());
                auto* window_positions = window_ids + width;
                shard_a_.device.bind_to_current_thread();
                CUDA_CHECK(cudaMemcpyAsync(window_ids, frame.verify_ids.data,
                                           sizeof(std::int32_t) * static_cast<std::size_t>(width),
                                           cudaMemcpyDeviceToHost, shard_a_.device.stream));
                CUDA_CHECK(cudaMemcpyAsync(window_positions, frame.verify_positions.data,
                                           sizeof(std::int32_t) * static_cast<std::size_t>(width),
                                           cudaMemcpyDeviceToHost, shard_a_.device.stream));
                // The window is produced on the device, so unlike the MTP path (whose host builds
                // the buffer directly) both copies above are asynchronous into pinned memory.
                // forward_tp2_window then reads that host buffer to build *each* shard's own copy,
                // and shard B's copy runs on a different stream with no ordering against shard A's
                // D2H. Synchronize before the host buffer becomes the forward's input.
                CUDA_CHECK(cudaStreamSynchronize(shard_a_.device.stream));
                if (constraint_live()) {
                    // The mask is built on the host, so the window has to be visible first.
                    CUDA_CHECK(cudaStreamSynchronize(shard_a_.device.stream));
                    constraint_advance();
                    std::fill(tool_mask_columns.begin(), tool_mask_columns.end(), std::uint8_t{1});
                    bool masked = false;
                    std::string drafted_prefix;
                    for (std::int32_t column = 0; column < width; ++column) {
                        if (tool_constraint->build_mask_after(drafted_prefix, logits_domain,
                                                              tool_mask_one)) {
                            const std::size_t base =
                                static_cast<std::size_t>(column) *
                                static_cast<std::size_t>(vocab);
                            std::copy(tool_mask_one.begin(), tool_mask_one.end(),
                                      tool_mask_columns.begin() +
                                          static_cast<std::ptrdiff_t>(base));
                            masked = true;
                        }
                        if (column + 1 < width) {
                            drafted_prefix.append(tool_constraint->piece(
                                static_cast<std::size_t>(window_ids[column + 1])));
                        }
                    }
                    if (masked) {
                        shard_a_.device.bind_to_current_thread();
                        CUDA_CHECK(cudaMemcpyAsync(tool_mask_dev.data, tool_mask_columns.data(),
                                                   tool_mask_columns.size(),
                                                   cudaMemcpyHostToDevice,
                                                   shard_a_.device.stream));
                        ops::apply_token_mask(window_logits, tool_mask_dev,
                                              shard_a_.device.stream);
                    }
                }
                // The target verify: RecordForReplay leaves the live GDN state untouched, so the
                // fold below can replay exactly the committed columns from the pre-round snapshot.
                // The feature sink captures the window's residuals for the next round's append.
                snapshot_state(shard_a_, kRoundScratchSlot);
                snapshot_state(shard_b_, kRoundScratchSlot);
                qwen::execution::DFlashFeatureSink verify_sink = round.make_verify_sink();
                const ops::CausalAttentionExecutionEnvelope target_envelope{
                    position + 1U, position + static_cast<std::uint32_t>(width)};
                run_verify_window(window_ids, static_cast<std::int32_t>(position), window_logits,
                                  window_hidden, &verify_sink);
                timing.record(2, shard_a_.device.stream);
                ops::argmax(window_logits, frame.target_argmax, public_tokens,
                            shard_a_.device.stream);
                // DFlash2 acceptance is the sparse 16-candidate rejection sampler, not MTP's greedy
                // matcher: the draft distribution is the selector's proposal q.
                ops::speculative_accept_sparse_drafts(
                    frame.target_argmax, window_logits, frame.draft_tokens, frame.candidate_ids,
                    frame.proposal_q, frame.proposal_extents, frame.execution_frontiers,
                    frame.anchors, frame.licensed_tokens, frame.licensed_counts,
                    frame.accepted_drafts, public_tokens, sampling_a,
                    ops::SpeculativeAcceptExecutionEnvelope{false}, ws_a, shard_a_.device.stream);
                timing.record(3, shard_a_.device.stream);
                std::vector<TokenId> licensed_host(static_cast<std::size_t>(width), 0);
                std::int32_t licensed_count = 0;
                CUDA_CHECK(cudaMemcpyAsync(licensed_host.data(), frame.licensed_tokens.data,
                                           sizeof(TokenId) * static_cast<std::size_t>(width),
                                           cudaMemcpyDeviceToHost, shard_a_.device.stream));
                CUDA_CHECK(cudaMemcpyAsync(&licensed_count, frame.licensed_counts.data,
                                           sizeof(std::int32_t), cudaMemcpyDeviceToHost,
                                           shard_a_.device.stream));
                timing.record(4, shard_a_.device.stream);
                const Clock::time_point sync_start = Clock::now();
                CUDA_CHECK(cudaStreamSynchronize(shard_a_.device.stream));
                shard_b_.device.bind_to_current_thread();
                CUDA_CHECK(cudaStreamSynchronize(shard_b_.device.stream));
                shard_a_.device.bind_to_current_thread();
                timing.close_round(
                    std::chrono::duration<double, std::milli>(Clock::now() - sync_start).count());
                if (licensed_count < 1 || licensed_count > width) {
                    throw std::logic_error("TP-2 DFlash2 round produced an invalid licensed prefix");
                }
                step.assign(licensed_host.begin(),
                            licensed_host.begin() + static_cast<std::ptrdiff_t>(licensed_count));
            }
        } else if (!mtp_enabled_) {
            Tensor logits_a = run_plain_decode_step(current, position);
            timing.record(2, shard_a_.device.stream);
            ops::set_i32_scalar(logical_pos_a, static_cast<std::int32_t>(position + 1),
                                shard_a_.device.stream);
            if (constraint_live()) {
                constraint_advance();
                if (tool_constraint->build_mask(logits_domain, tool_mask_one)) {
                    shard_a_.device.bind_to_current_thread();
                    CUDA_CHECK(cudaMemcpyAsync(tool_mask_dev.data, tool_mask_one.data(),
                                               tool_mask_one.size(), cudaMemcpyHostToDevice,
                                               shard_a_.device.stream));
                    ops::apply_token_mask(logits_a, tool_mask_dev, shard_a_.device.stream);
                }
            }

            Tensor sampled_a = ws_a.alloc(DType::I32, {1});
            ops::sample(logits_a, sampled_a, public_tokens, sampling_a, logical_pos_a,
                        ops::kSamplePurposeDecode, ws_a, shard_a_.device.stream);
            timing.record(3, shard_a_.device.stream);
            std::int32_t next = 0;
            shard_a_.device.bind_to_current_thread();
            CUDA_CHECK(cudaMemcpyAsync(&next, sampled_a.data, sizeof(std::int32_t),
                                       cudaMemcpyDeviceToHost, shard_a_.device.stream));
            timing.record(4, shard_a_.device.stream);
            const Clock::time_point sync_start = Clock::now();
            CUDA_CHECK(cudaStreamSynchronize(shard_a_.device.stream));
            timing.close_round(
                std::chrono::duration<double, std::milli>(Clock::now() - sync_start).count());
            step.assign(1, static_cast<TokenId>(next));
        } else {
            // One window: [anchor, d0, ..., d_{K-1}] at consecutive positions from the anchor's. The
            // MTP layer's column at (anchor - 1) embeds the anchor token and predicts the token after
            // it, so its drafts become the window's columns 1..K.
            Tensor anchor_dev = ws_a.alloc(DType::I32, {1});
            ops::set_i32_scalar(anchor_dev, mtp_anchor, shard_a_.device.stream);
            const std::vector<TokenId> drafts =
                mtp_propose_window(shard_a_, shard_a_.mtp_anchor_hidden, anchor_dev,
                                   mtp_position - 1, ws_a);
            timing.record(1, shard_a_.device.stream);
            const std::int32_t width = static_cast<std::int32_t>(mtp_drafts_) + 1;
            // Every per-round input to the verify goes through the pinned window: the capture path
            // reads it with a memcpy node, and the eager path passes it straight to the forward.
            auto* window_ids       = static_cast<std::int32_t*>(verify_window_host_->data());
            auto* window_positions = window_ids + width;
            window_ids[0]          = static_cast<std::int32_t>(mtp_anchor);
            window_positions[0]    = static_cast<std::int32_t>(mtp_position);
            for (std::int32_t i = 1; i < width; ++i) {
                window_ids[i] =
                    static_cast<std::int32_t>(drafts[static_cast<std::size_t>(i - 1)]);
                window_positions[i] = static_cast<std::int32_t>(mtp_position) + i;
            }
            // The proposal chain's operands are dead now - the window is assembled on the host and
            // the chain's own arena footprint varies with the round's attention route - so both
            // workspaces go back to where this round's fixed scratch belongs. A captured window
            // fixes that place: everything the verify and the accept path allocate is then a
            // function of the round's fixed sequence alone, which is what the graph bakes into its
            // kernel arguments.
            const std::uint32_t visible_end =
                static_cast<std::uint32_t>(mtp_position) + static_cast<std::uint32_t>(width);
            const WindowGraph* reusable = reusable_window_graph(verify_graphs_, visible_end);
            position_arena(ws_a, shard_a_.round_base,
                           reusable != nullptr ? reusable->round_base[0] : shard_a_.round_base);
            position_arena(ws_b, shard_b_.round_base,
                           reusable != nullptr ? reusable->round_base[1] : shard_b_.round_base);
            Tensor window_logits   = ws_a.alloc(DType::BF16, {vocab, width});
            round_hidden           = ws_a.alloc(DType::BF16, {hidden, width});
            Tensor window_drafts =
                ws_a.alloc(DType::I32, {static_cast<std::int32_t>(mtp_drafts_), 1});
            Tensor target_tokens   = ws_a.alloc(DType::I32, {width, 1});
            Tensor current_extents = ws_a.alloc(DType::I32, {1});
            Tensor round_lengths   = ws_a.alloc(DType::I32, {1});
            Tensor round_anchors   = ws_a.alloc(DType::I32, {1});
            Tensor licensed        = ws_a.alloc(DType::I32, {width, 1});
            Tensor licensed_counts = ws_a.alloc(DType::I32, {1});
            Tensor accepted        = ws_a.alloc(DType::I32, {1});
            shard_a_.device.bind_to_current_thread();
            // From the pinned window, not from the proposal chain's arena: that region has been
            // reused by the allocations above.
            CUDA_CHECK(cudaMemcpyAsync(window_drafts.data, window_ids + 1,
                                       sizeof(TokenId) * mtp_drafts_, cudaMemcpyHostToDevice,
                                       shard_a_.device.stream));
            ops::set_i32_scalar(current_extents, static_cast<std::int32_t>(mtp_drafts_),
                                shard_a_.device.stream);
            ops::set_i32_scalar(round_lengths, static_cast<std::int32_t>(mtp_position),
                                shard_a_.device.stream);
            ops::set_i32_scalar(round_anchors, mtp_anchor, shard_a_.device.stream);
            // RecordForReplay records the window's transitions but also advances the live state, so the
            // pre-verify state is snapshotted and the fold replays only the committed columns from it.
            snapshot_state(shard_a_, kRoundScratchSlot);
            snapshot_state(shard_b_, kRoundScratchSlot);
            // The verify window runs Phase::Prefill through the capture-safe window path: the
            // Phase::Verify (decode-equivalent) variant is a known-incomplete TP-2 path (batched
            // GDN/attention faults), so MTP stays approximate. run_verify_window owns the GDN
            // record/replay action around the launch, and replays a captured graph of the whole
            // window when one covers this round's extent.
            run_verify_window(window_ids, static_cast<std::int32_t>(mtp_position), window_logits,
                              round_hidden);
            timing.record(2, shard_a_.device.stream);
            if (constraint_live()) {
                constraint_advance();
                // Column c is drawn after the draft columns before it, and those are exactly the
                // columns a round commits before its first mismatch, so the position the mask is
                // built from is the position that column is consumed at.
                std::fill(tool_mask_columns.begin(), tool_mask_columns.end(), std::uint8_t{1});
                bool masked = false;
                std::string drafted_prefix;
                for (std::int32_t column = 0; column < width; ++column) {
                    if (tool_constraint->build_mask_after(drafted_prefix, logits_domain,
                                                       tool_mask_one)) {
                        const std::size_t base =
                            static_cast<std::size_t>(column) * static_cast<std::size_t>(vocab);
                        std::copy(tool_mask_one.begin(), tool_mask_one.end(),
                                  tool_mask_columns.begin() +
                                      static_cast<std::ptrdiff_t>(base));
                        masked = true;
                    }
                    if (column + 1 < width) {
                        drafted_prefix.append(tool_constraint->piece(
                            static_cast<std::size_t>(window_ids[column + 1])));
                    }
                }
                if (masked) {
                    shard_a_.device.bind_to_current_thread();
                    CUDA_CHECK(cudaMemcpyAsync(tool_mask_dev.data, tool_mask_columns.data(),
                                               tool_mask_columns.size(), cudaMemcpyHostToDevice,
                                               shard_a_.device.stream));
                    ops::apply_token_mask(window_logits, tool_mask_dev, shard_a_.device.stream);
                }
            }
            ops::argmax(window_logits, target_tokens, public_tokens, shard_a_.device.stream);
            ops::speculative_accept_greedy_drafts(
                target_tokens, window_logits, window_drafts, current_extents, round_lengths,
                round_anchors, licensed, licensed_counts, accepted, public_tokens, sampling_a,
                ws_a, shard_a_.device.stream);
            timing.record(3, shard_a_.device.stream);
            std::vector<TokenId> licensed_host(static_cast<std::size_t>(width), 0);
            std::int32_t licensed_count = 0;
            CUDA_CHECK(cudaMemcpyAsync(licensed_host.data(), licensed.data,
                                       sizeof(TokenId) * width, cudaMemcpyDeviceToHost,
                                       shard_a_.device.stream));
            CUDA_CHECK(cudaMemcpyAsync(&licensed_count, licensed_counts.data, sizeof(std::int32_t),
                                       cudaMemcpyDeviceToHost, shard_a_.device.stream));
            timing.record(4, shard_a_.device.stream);
            const Clock::time_point sync_start = Clock::now();
            CUDA_CHECK(cudaStreamSynchronize(shard_a_.device.stream));
            // Both devices must be idle before the host moves the round on: the fold, the state
            // restore and the next round's window all rewrite buffers the peer's verify may still be
            // reading, and the pair's allreduce is the only thing that orders the two devices'
            // kernels against each other.
            shard_b_.device.bind_to_current_thread();
            CUDA_CHECK(cudaStreamSynchronize(shard_b_.device.stream));
            shard_a_.device.bind_to_current_thread();
            timing.close_round(
                std::chrono::duration<double, std::milli>(Clock::now() - sync_start).count());
            if (licensed_count < 1 || licensed_count > width) {
                throw std::logic_error("TP-2 MTP round produced an invalid licensed prefix");
            }
            step.assign(licensed_host.begin(),
                        licensed_host.begin() + static_cast<std::ptrdiff_t>(licensed_count));
        }
        // A speculative round can license more tokens than the request still has budget for; the
        // output policy only ever commits a prefix of what it is shown.
        const std::uint32_t remaining = request.budget.remaining();
        if (step.size() > remaining) { step.resize(remaining); }
        const OutputDecision decision =
            request.output.preview_model(step, remaining, request.budget.limit_reason());
        if (decision.accepted_tokens == 0 || decision.accepted_tokens > step.size()) {
            throw std::logic_error("TP-2 output policy returned an invalid licensed prefix");
        }
        request.generated.insert(request.generated.end(), step.begin(),
                                 step.begin() + static_cast<std::ptrdiff_t>(decision.accepted_tokens));
        request.budget.commit(decision.accepted_tokens);
        committed_decode_tokens_ += decision.accepted_tokens;
        ++decode_rounds_;
        // preview_model already established the (possibly terminal) preview; commit it as-is.
        publish_preview(false);
        if (dflash2_enabled_) {
            if (!dflash_plain_step) {
                // The output policy licenses a prefix of the round's tokens. The verify recorded
                // the window's GDN transitions and advanced the live state; undo that and replay
                // exactly the committed columns, as the single-device resolve does.
                const std::uint32_t committed = decision.accepted_tokens;
                timing.record(5, shard_a_.device.stream);
                for (Shard* fold_shard : {&shard_a_, &shard_b_}) {
                    fold_shard->device.bind_to_current_thread();
                    CUDA_CHECK(cudaMemcpyAsync(
                        fold_shard->state_backing.data,
                        fold_shard->state_snapshots[kRoundScratchSlot].data,
                        fold_shard->state_backing.bytes, cudaMemcpyDeviceToDevice,
                        fold_shard->device.stream));
                }
                const ops::GdnReplayFoldRow fold_row{
                    .source_state_slot      = 0,
                    .destination_state_slot = 0,
                    .commit_columns         = static_cast<std::int32_t>(committed)};
                const std::span<const ops::GdnReplayFoldRow> fold_rows(&fold_row, 1);
                shard_a_.device.bind_to_current_thread();
                shard_a_.replay_fold->execute(fold_rows, shard_a_.device.stream);
                shard_b_.device.bind_to_current_thread();
                shard_b_.replay_fold->execute(fold_rows, shard_b_.device.stream);
                // The target frontier advances by exactly the committed columns. The draft ring
                // stays one verify window behind, except when this round ends the request: then it
                // is caught up to the final frontier, which is the single-device rule.
                const std::uint32_t base         = position;
                position                         = base + committed;
                const bool dflash_finished       = decision.finished();
                dflash_context_frontier_         = dflash_finished ? position : base;
                if (dflash_finished && position > base) {
                    const qwen::execution::ExecutionCore dflash_execution{
                        .device           = shard_a_.device,
                        .parameters       = *shard_a_.parameters,
                        .work             = *shard_a_.workspace,
                        .linear_attention = *shard_a_.state,
                        .replay_records   = nullptr,
                        .io               = shard_a_.io,
                        .prefill_hidden   = shard_a_.prefill_hidden,
                        .prefill_chunk    = options_.prefill_chunk,
                        .proposal_head    = options_.speculative.proposal_head,
                    };
                    shard_a_.dflash_round->append_pending(dflash_execution, base, position);
                }
                current = static_cast<std::int32_t>(step[committed - 1]);
                timing.record(6, shard_a_.device.stream);
                timing.fold_ms += timing.elapsed(5, 6);
            }
        } else if (!mtp_enabled_) {
            current = step.back();
            ++position;
        } else {
            // The output policy licenses a prefix of the round's tokens. The state advances by exactly
            // those columns: their transitions were recorded, not applied.
            const std::uint32_t committed = decision.accepted_tokens;
            timing.record(5, shard_a_.device.stream);
            // Undo the verify's advance: restore the pre-round state, then fold the committed columns.
            for (Shard* shard : {&shard_a_, &shard_b_}) {
                shard->device.bind_to_current_thread();
                CUDA_CHECK(cudaMemcpyAsync(shard->state_backing.data,
                                           shard->state_snapshots[kRoundScratchSlot].data,
                                           shard->state_backing.bytes, cudaMemcpyDeviceToDevice,
                                           shard->device.stream));
            }
            const ops::GdnReplayFoldRow fold_row{.source_state_slot      = 0,
                                                 .destination_state_slot = 0,
                                                 .commit_columns = static_cast<std::int32_t>(committed)};
            const std::span<const ops::GdnReplayFoldRow> rows(&fold_row, 1);
            shard_a_.device.bind_to_current_thread();
            shard_a_.replay_fold->execute(rows, shard_a_.device.stream);
            shard_b_.device.bind_to_current_thread();
            shard_b_.replay_fold->execute(rows, shard_b_.device.stream);
            // The next anchor is the last committed token; its own transition stays pending, so the
            // next bridge reads the final-norm hidden of the column before it.
            mtp_anchor   = static_cast<std::int32_t>(step[committed - 1]);
            mtp_position = mtp_position + committed;
            const std::int32_t source_column = static_cast<std::int32_t>(committed) - 1;
            shard_a_.device.bind_to_current_thread();
            CUDA_CHECK(cudaMemcpyAsync(
                shard_a_.mtp_anchor_hidden.data,
                static_cast<const char*>(round_hidden.data) +
                    static_cast<std::size_t>(source_column) *
                        static_cast<std::size_t>(hidden) * sizeof(std::uint16_t),
                shard_a_.mtp_anchor_hidden.bytes(), cudaMemcpyDeviceToDevice,
                shard_a_.device.stream));
            timing.record(6, shard_a_.device.stream);
            timing.fold_ms += timing.elapsed(5, 6);
        }
        finished = decision.finished();
        if (finished) { result.finish_reason = decision.finish_reason; }
        timing.committed += decision.accepted_tokens;
        ++timing.rounds;
        timing.round_ms +=
            std::chrono::duration<double, std::milli>(Clock::now() - round_start).count();
    }

    // Publish the finished conversation: the pools hold prompt plus committed output and the live
    // GDN state sits at that frontier, which is exactly what a returning turn extends and what an
    // eviction copies out.
    {
        std::vector<TokenId> history = token_ids;
        history.insert(history.end(), request.generated.begin(), request.generated.end());
        // The last sampled token is not forwarded yet, so the device state stops one token short
        // of the history the next turn will send back. A walk that produced no token at all is
        // already a logic error elsewhere; publishing the prompt end keeps the catalog honest.
        const std::uint32_t sampled = static_cast<std::uint32_t>(request.generated.size());
        const std::uint32_t frontier =
            sampled == 0 ? prompt_tokens : prompt_tokens + sampled - 1U;
        session_publish(history, frontier);
    }
    result.generated_token_ids = std::move(request.generated);
    result.tool_calls          = request.output.take_tool_calls();
    result.tool_call_parse     = request.output.tool_call_parse_diagnostics();
    result.reasoning_tokens    = request.output.reasoning_tokens();
    result.matched_stop_string = request.output.matched_stop_string();
    result.thinking            = request.output.thinking_stats();
    const Clock::time_point finished_at = Clock::now();
    result.timings.decode_seconds =
        std::chrono::duration<double>(finished_at - first_token_time).count();
    result.timings.generation_wall_seconds = result.timings.decode_seconds;
    result.timings.first_token_seconds =
        result.timings.prepare_seconds + result.timings.prompt_wall_seconds;
    result.timings.total_seconds = std::chrono::duration<double>(finished_at - start).count();
    timing.prefill_ms = result.timings.prefill_seconds * 1000.0;
    timing.report("decode");
    return result;
}

LoadSummary TP2GenerationCore::load_summary() const {
    LoadSummary summary;
    const auto& model = *shard_a_.model;
    summary.architecture =
        std::string(models::architecture_name(model.config().text.architecture));
    summary.model_name   = model.info().name;
    std::set<std::string> formats;
    for (const auto& weight : model.weight_data()) {
        for (const auto& part : weight.view.parts) {
            formats.emplace(artifact::format_name(part.parent->geometry.format));
        }
    }
    summary.weight_formats.assign(formats.begin(), formats.end());
    const auto& stats = model.storage_stats();
    summary.load_seconds         = load_seconds_;
    summary.upload_seconds       = stats.upload_seconds;
    summary.artifact_bytes_read  = stats.read_bytes;
    summary.host_to_device_bytes = stats.h2d_bytes;
    summary.peak_staging_bytes   = stats.peak_staging_bytes;
    summary.device_object_count  = stats.device_object_count;
    summary.host_object_count    = stats.host_object_count;
    return summary;
}

MemorySummary TP2GenerationCore::memory_summary() const {
    MemorySummary summary;
    summary.device        = options_.device;
    summary.max_context   = options_.max_context;
    summary.kv_capacity   = options_.max_context;
    summary.kv_cache      = options_.kv_cache;
    summary.workspace     = {.capacity_bytes = kWorkspaceBytes, .used_bytes = 0,
                             .peak_used_bytes = shard_a_.workspace->peak_used()};
    return summary;
}

RuntimeStats TP2GenerationCore::runtime_stats() const {
    RuntimeStats stats;
    stats.computed_prefill_tokens = computed_prefill_tokens_;
    stats.committed_decode_tokens = committed_decode_tokens_;
    stats.decode_rounds           = decode_rounds_;
    stats.decode_row_rounds       = decode_rounds_;
    return stats;
}

void TP2GenerationCore::reset_memory_peaks() noexcept {
    shard_a_.workspace->reset_peak();
    shard_b_.workspace->reset_peak();
}

} // namespace ninfer::runtime
