#include "runtime/engine/tp2_generation_core.h"

#include "artifact/formats.h"
#include "artifact/reader.h"
#include "core/layout.h"
#include "models/registry.h"
#include "models/qwen3_5/frontend/prepared_prompt.h"
#include "models/qwen3_5/load.h"
#include "ninfer/ops/argmax.h"
#include "ninfer/ops/position.h"
#include "ninfer/ops/sampling.h"
#include "ninfer/ops/scalar.h"
#include "ninfer/ops/speculative_round.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
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

// Bounds on the adaptive rewind depth (see rewind_near_).
constexpr std::uint32_t kReuseRewindMinimum = 4;
constexpr std::uint32_t kReuseRewindMaximum = 4096;

// Upper bound on the batched TP-2 prefill chunk width. One chunk reads every weight once, so the
// per-token weight traffic that dominates the single-token walk is amortized over the whole chunk;
// the cross-device allreduce bytes per token and the tensor-core work per token do not shrink with
// the chunk, so widening the chunk trades only the weight term against the per-chunk activation
// peak. The engine option (--prefill-chunk) picks the actual width.
constexpr std::uint32_t kPrefillChunkMaximum = 1024;

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

    artifact::Reader reader(options.artifact_path);
    auto plan = qwen::plan_load(reader, load);
    auto [model_a, model_b] =
        qwen::materialize_model_tp2(std::move(plan), shard_a_.device, shard_b_.device,
                                    &options.startup_observer);
    shard_a_.model = std::move(model_a);
    shard_b_.model = std::move(model_b);

    shard_a_.parameters = std::make_unique<qwen::execution::Parameters>(*shard_a_.model);
    shard_b_.parameters = std::make_unique<qwen::execution::Parameters>(*shard_b_.model);

    build_shard(shard_a_, 0);
    build_shard(shard_b_, 1);
    // Column-split weights (the token embedding) are consumed by operations that run on one shard
    // alone, so both contexts learn their peer and the pair once both shards exist.
    shard_a_.context->set_tp_peer(shard_b_.context.get(), &pair_);
    shard_b_.context->set_tp_peer(shard_a_.context.get(), &pair_);

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

TP2GenerationCore::~TP2GenerationCore() = default;

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

    std::size_t record_bytes = 0;
    std::size_t round_bytes  = 0;
    if (mtp_enabled_) {
        // ReplaySSM records for one verify window wide, one physical row, per-shard GDN geometry.
        // Both shards verify the window, so both need their own records and fold plan.
        // The verify forward records here instead of advancing the live state, and the fold replays
        // the accepted prefix back into it (in place: with a single row the Op allows it, so the
        // state pool still needs one slot).
        LayoutBuilder record_builder;
        const GdnReplayRecordLayout record_layout = plan_gdn_replay_records(
            record_builder,
            GdnReplayRecordSpec{
                .layers          = qwen::execution::dimension(scfg.linear_attention_layers),
                .record_capacity = 1,
                .width           = static_cast<std::int32_t>(mtp_drafts_) + 1,
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

GenerationResult TP2GenerationCore::execute(Request& request, OutputSink* sink,
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

    GenerationResult result;
    result.prompt = request.summary;
    result.timings.prepare_seconds = request.prepare_seconds;
    const Clock::time_point start = Clock::now();

    // Prompt-prefix reuse. The KV pages hold the K/V of every position the last completed prefill
    // wrote, and the state snapshots hold the matching GDN states, so a prompt that extends the
    // previous prompt can skip the shared prefix and prefill only its suffix. The single-device
    // route gets this from the context cache; TP-2 runs with that cache disabled, so the core keeps
    // the boundaries it already owns. The deepest boundary at or before the shared prefix wins; the
    // final prompt token is always forwarded, because its logits drive the first sample.
    std::uint32_t reuse    = 0;
    std::size_t reuse_slot = 0;
    if (cached_state_valid_ && !cached_prompt_tokens_.empty()) {
        std::size_t lcp = 0;
        const std::size_t common = std::min(cached_prompt_tokens_.size(), token_ids.size());
        while (lcp < common && cached_prompt_tokens_[lcp] == token_ids[lcp]) { ++lcp; }
        for (std::size_t slot = 0; slot < kReuseSnapshotCount; ++slot) {
            const std::uint32_t boundary = cached_boundaries_[slot];
            if (boundary <= lcp && boundary < prompt_tokens && boundary > reuse) {
                reuse      = boundary;
                reuse_slot = slot;
            }
        }
        // Predict the next prefill's rewind depths from the gap this pair of prompts showed: the
        // snapshot that captures the next shared prefix should sit just inside it.
        const std::uint32_t gap = static_cast<std::uint32_t>(cached_prompt_tokens_.size() - lcp);
        if (gap != 0) {
            rewind_near_ = std::clamp(gap + 2, kReuseRewindMinimum, kReuseRewindMaximum);
        }
    }
    const std::array<std::uint32_t, kReuseSnapshotCount> rewind_depths{0, rewind_near_};

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
        if (reuse != 0) {
            CUDA_CHECK(cudaMemcpyAsync(shard.state_backing.data,
                                       shard.state_snapshots[reuse_slot].data,
                                       shard.state_backing.bytes, cudaMemcpyDeviceToDevice,
                                       shard.device.stream));
            return;
        }
        CUDA_CHECK(cudaMemsetAsync(shard.state_backing.data, 0, shard.state_backing.bytes,
                                   shard.device.stream));
    };
    begin_gdn_state(shard_a_);
    begin_gdn_state(shard_b_);

    if (streaming) {
        sink->start(GenerationStart{.prompt = request.summary, .reused_prompt_tokens = reuse});
    }
    result.reused_prompt_tokens = reuse;

    // Sampling config, device-resident, for ops::sample.
    const ops::SamplingConfig sampling_config = make_sampling_config(request.sampling);
    auto sampling_scope = shard_a_.workspace->scope();
    auto& ws_a = *shard_a_.workspace;
    auto& ws_b = *shard_b_.workspace;
    const std::size_t sampling_ws = ops::sampling_workspace_capacity_bytes(vocab, 1, 1);
    auto sampling_buf_a = ws_a.alloc_bytes(sizeof(ops::SamplingConfig) + sampling_ws, 256);
    auto sampling_buf_b = ws_b.alloc_bytes(sizeof(ops::SamplingConfig) + sampling_ws, 256);
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
    std::array<std::uint32_t, kReuseSnapshotCount> snapshot_at{};
    snapshot_at[0] = prompt_tokens;
    for (std::size_t slot = 1; slot < kReuseSnapshotCount; ++slot) {
        const std::uint32_t depth = rewind_depths[slot];
        snapshot_at[slot] = (prompt_tokens > depth && prompt_tokens - depth >= reuse)
                                ? prompt_tokens - depth
                                : 0;
        if (snapshot_at[slot] != 0 && snapshot_at[slot] == reuse) {
            // The restored state already sits exactly on this boundary, and the walk never revisits
            // its own starting point, so freeze it before the first chunk.
            snapshot_state(shard_a_, slot);
            snapshot_state(shard_b_, slot);
        }
    }
    for (std::uint32_t t0 = reuse; t0 < prompt_tokens;) {
        if (cancellation.requested()) {
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
        ctx_a.forward_tp2_prefill(ctx_b, pair_, std::span<const int>(token_ids.data() + t0, length),
                                  static_cast<std::int32_t>(t0), &logits_a, &logits_b,
                                  mtp_enabled_ ? &mtp_input_a : nullptr, nullptr, nullptr,
                                  qwen::TextPhase::Prefill, media_ptr);
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
        if (t0 + length == prompt_tokens) {
            // First token: sample from the last chunk's last-column logits.
            ops::set_i32_scalar(logical_pos_a, static_cast<std::int32_t>(prompt_tokens),
                                shard_a_.device.stream);
            Tensor sampled_a = ws_a.alloc(DType::I32, {1});
            ops::sample(logits_a, sampled_a, vocab, sampling_a, logical_pos_a,
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
        t0 += length;
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
    bool finished = false;
    const std::int32_t public_tokens =
        static_cast<std::int32_t>(shard_a_.model->resources().public_token_count);
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
        if (!mtp_enabled_) {
            Tensor logits_a = ws_a.alloc(DType::BF16, {vocab, 1});
            Tensor logits_b = ws_b.alloc(DType::BF16, {vocab, 1});
            ctx_a.forward_tp2(ctx_b, pair_, current, static_cast<std::int32_t>(position), logits_a,
                              logits_b);
            ops::set_i32_scalar(logical_pos_a, static_cast<std::int32_t>(position + 1),
                                shard_a_.device.stream);
            Tensor sampled_a = ws_a.alloc(DType::I32, {1});
            ops::sample(logits_a, sampled_a, vocab, sampling_a, logical_pos_a,
                        ops::kSamplePurposeDecode, ws_a, shard_a_.device.stream);
            std::int32_t next = 0;
            shard_a_.device.bind_to_current_thread();
            CUDA_CHECK(cudaMemcpyAsync(&next, sampled_a.data, sizeof(std::int32_t),
                                       cudaMemcpyDeviceToHost, shard_a_.device.stream));
            CUDA_CHECK(cudaStreamSynchronize(shard_a_.device.stream));
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
            const std::int32_t width = static_cast<std::int32_t>(mtp_drafts_) + 1;
            std::vector<int> window_host(static_cast<std::size_t>(width), mtp_anchor);
            for (std::int32_t i = 1; i < width; ++i) {
                window_host[static_cast<std::size_t>(i)] =
                    static_cast<int>(drafts[static_cast<std::size_t>(i - 1)]);
            }
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
            CUDA_CHECK(cudaMemcpyAsync(window_drafts.data, drafts.data(),
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
            shard_a_.context->set_gdn_state_action(qwen::execution::GdnStateAction::RecordForReplay,
                                                  &shard_a_.records);
            shard_b_.context->set_gdn_state_action(qwen::execution::GdnStateAction::RecordForReplay,
                                                  &shard_b_.records);
            ctx_a.forward_tp2_prefill(
                ctx_b, pair_,
                std::span<const int>(window_host.data(), window_host.size()),
                static_cast<std::int32_t>(mtp_position), nullptr, nullptr, nullptr,
                // The verify window runs Phase::Prefill: the Phase::Verify (decode-equivalent) variant is a
                // known-incomplete TP-2 path (batched GDN/attention faults), so MTP stays approximate.
                &window_logits, &round_hidden);
            shard_a_.context->set_gdn_state_action(qwen::execution::GdnStateAction::UpdateInPlace,
                                                  nullptr);
            shard_b_.context->set_gdn_state_action(qwen::execution::GdnStateAction::UpdateInPlace,
                                                  nullptr);
            ops::argmax(window_logits, target_tokens, vocab, shard_a_.device.stream);
            ops::speculative_accept_greedy_drafts(
                target_tokens, window_logits, window_drafts, current_extents, round_lengths,
                round_anchors, licensed, licensed_counts, accepted, vocab, sampling_a, ws_a,
                shard_a_.device.stream);
            std::vector<TokenId> licensed_host(static_cast<std::size_t>(width), 0);
            std::int32_t licensed_count = 0;
            CUDA_CHECK(cudaMemcpyAsync(licensed_host.data(), licensed.data,
                                       sizeof(TokenId) * width, cudaMemcpyDeviceToHost,
                                       shard_a_.device.stream));
            CUDA_CHECK(cudaMemcpyAsync(&licensed_count, licensed_counts.data, sizeof(std::int32_t),
                                       cudaMemcpyDeviceToHost, shard_a_.device.stream));
            CUDA_CHECK(cudaStreamSynchronize(shard_a_.device.stream));
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
        if (!mtp_enabled_) {
            current = step.back();
            ++position;
        } else {
            // The output policy licenses a prefix of the round's tokens. The state advances by exactly
            // those columns: their transitions were recorded, not applied.
            const std::uint32_t committed = decision.accepted_tokens;
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
            mtp_draft_checked_ += mtp_drafts_;
            mtp_draft_hit_ += committed > 1 ? committed - 1 : 0;
            std::fprintf(stderr,
                         "[mtp] round pos=%u anchors=%d accepted=%u rate=%llu/%llu\n",
                         mtp_position, mtp_anchor, committed,
                         static_cast<unsigned long long>(mtp_draft_hit_),
                         static_cast<unsigned long long>(mtp_draft_checked_));
        }
        finished = decision.finished();
        if (finished) { result.finish_reason = decision.finish_reason; }
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
