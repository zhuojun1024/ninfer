#include "models/qwen3_5/program/dflash_round.h"

#include "models/qwen3_5/execution/workspace.h"

#include "ninfer/ops/candidate_selector.h"
#include "ninfer/ops/dynamic_grouped_conv.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/linear_topk.h"
#include "ninfer/ops/prepare_ragged_prefix.h"
#include "ninfer/ops/scalar.h"
#include "ninfer/ops/sliding_window_attention.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace ninfer::models::qwen3_5::execution {
namespace {

// A dead scratch allocation: the layout builder only tracks the peak, so a scoped allocation that
// is never read is exactly what sizes one branch of the proposal forward.
void add_scratch(WorkspaceLayoutBuilder& layout, std::size_t bytes) {
    if (bytes == 0) { return; }
    auto scope = layout.scope();
    (void)layout.alloc_bytes(bytes);
}

void add_linear_scratch(WorkspaceLayoutBuilder& layout, const LinearParameters& parameter,
                        std::int32_t first, std::int32_t last) {
    add_scratch(layout, ops::linear_workspace_capacity_bytes(parameter.weight.qtype,
                                                             parameter.weight.n, parameter.weight.k,
                                                             parameter.policy, first, last));
}

} // namespace

DFlash2RoundSpec plan_dflash2_round(const DraftConfig& draft, const TextConfig& target,
                                    std::uint32_t feature_columns, std::uint32_t draft_window,
                                    std::size_t proposal_workspace_bytes) {
    if (!draft.dflash2.has_value()) {
        throw std::invalid_argument("a TP-2 masked-draft round requires the DFlash2 layout");
    }
    DFlash2RoundSpec spec;
    spec.hidden          = dimension(target.hidden_size);
    spec.output_rows     = dimension(target.vocab_size);
    spec.target_features =
        dimension(target.hidden_size * std::uint64_t(draft.target_layer_ids.size()));
    spec.feature_columns = static_cast<std::int32_t>(feature_columns);
    spec.feature_lanes   = static_cast<std::int32_t>(std::max<std::uint32_t>(draft_window + 1U, 1U));
    spec.local_layers    = draft.local_layer_count();
    spec.local_window    = draft.sliding_window.value_or(0);
    spec.local_kv_heads  = dimension(draft.attention.num_key_value_heads);
    spec.local_head_dim  = dimension(draft.attention.head_dim);
    spec.batch_capacity  = 1;
    spec.draft_window    = draft_window;
    spec.proposal_workspace_bytes = proposal_workspace_bytes;
    spec.target_layer_ids         = draft.target_layer_ids;
    return spec;
}

std::size_t dflash2_proposal_workspace_bytes(const Parameters& parameters, const TextConfig& target,
                                             std::uint32_t capacity,
                                             ProposalHead proposal_head, std::int32_t width,
                                             std::int32_t batch) {
    if (width < 2 || batch < 1 || capacity == 0) {
        throw std::invalid_argument("DFlash2 proposal workspace geometry is invalid");
    }
    if (!parameters.draft.has_value() || !parameters.draft->selector.has_value()) {
        throw std::invalid_argument("DFlash2 proposal workspace requires the selector weights");
    }
    if (!parameters.model.config().draft.has_value() ||
        !parameters.model.config().draft->dflash2.has_value()) {
        throw std::invalid_argument("DFlash2 proposal workspace requires a DFlash2 draft");
    }
    const DraftConfig& draft = *parameters.model.config().draft;
    const std::int32_t columns   = width * batch;
    const std::int32_t drafts    = width - 1;
    const std::int32_t mask_cols = drafts * batch;
    WorkspaceLayoutBuilder layout;
    // The proposal block's own residual, allocated by propose_dflash2_batch before the layer loop.
    (void)layout.alloc(DType::BF16, {dimension(target.hidden_size), columns});
    const auto prepare = [&] {
        (void)workspace::dflash2_branch(layout, target, draft, width, batch);
        add_scratch(layout, ops::rmsnorm_dynamic_grouped_conv_prepare_workspace_capacity_bytes(
                                width, width, batch, batch));
    };
    {
        // One attention branch: the dynamic convolution, Q/K/V, the sliding-window attention over
        // the draft ring, and the output projection back onto the residual.
        auto attention = layout.scope();
        prepare();
        (void)layout.alloc(DType::BF16, {dimension(draft.attention.query_width()), columns});
        (void)layout.alloc(DType::BF16, {dimension(draft.attention.key_width()), columns});
        (void)layout.alloc(DType::BF16, {dimension(draft.attention.key_width()), columns});
        (void)layout.alloc(DType::BF16, {dimension(draft.attention.query_width()), columns});
        add_scratch(layout,
                    ops::sliding_window_attention_workspace_capacity_bytes(
                        {dimension(draft.attention.head_dim),
                         dimension(draft.attention.num_attention_heads),
                         dimension(draft.attention.num_key_value_heads)},
                        dimension(draft.sliding_window.value_or(0)), {0, capacity}, width, width,
                        batch));
        add_scratch(layout,
                    ops::linear_dynamic_grouped_conv_add_workspace_capacity_bytes(
                        dimension(draft.attention.query_width()), width, width, batch, batch));
    }
    {
        // One MLP branch: the second dynamic convolution and the SwiGLU projection.
        auto mlp = layout.scope();
        prepare();
        (void)layout.alloc(DType::BF16, {dimension(draft.intermediate_size), columns});
        for (const auto& block : parameters.draft->layers) {
            const auto& projection = block.mlp.gate_up;
            add_scratch(layout,
                        ops::linear_swiglu_workspace_capacity_bytes(
                            projection.weight.qtype, projection.weight.n, projection.weight.k,
                            projection.policy, columns, columns));
        }
        add_scratch(layout,
                    ops::linear_dynamic_grouped_conv_add_workspace_capacity_bytes(
                        dimension(draft.intermediate_size), width, width, batch, batch));
    }
    // Tail: the packed final-norm hidden, the reduced head's top-k ids and scores, the selector
    // projection and the sparse proposal.
    (void)layout.alloc(DType::BF16, {dimension(target.hidden_size), mask_cols});
    (void)layout.alloc(DType::FP32, {dimension(draft.dflash2->selector_top_k), mask_cols});
    const LinearParameters& head = proposal_head == ProposalHead::Optimized
                                      ? parameters.proposal->head
                                      : parameters.draft->output_head;
    add_scratch(layout, ops::linear_topk_workspace_capacity_bytes(
                            head.weight.qtype, head.weight.n, head.weight.k, mask_cols,
                            mask_cols));
    (void)layout.alloc(DType::BF16, {dimension(draft.dflash2->selector_rank), mask_cols});
    add_linear_scratch(layout, parameters.draft->selector->hidden_projection, mask_cols, mask_cols);
    add_scratch(layout, ops::candidate_selector_path_workspace_capacity_bytes(drafts, drafts,
                                                                              batch, batch));
    return layout.peak_bytes(1);
}

DFlash2Round::DFlash2Round(DeviceContext& device, const DFlash2RoundSpec& spec)
    : device_(&device), spec_(spec) {
    if (spec_.hidden <= 0 || spec_.output_rows <= 0 || spec_.target_features <= 0 ||
        spec_.feature_columns <= 0 || spec_.feature_lanes <= 0 || spec_.local_layers == 0 ||
        spec_.local_window == 0 || spec_.local_kv_heads <= 0 || spec_.local_head_dim <= 0 ||
        spec_.batch_capacity == 0 || spec_.draft_window == 0 ||
        spec_.draft_window > kDFlashDecodeMaximumDrafts || spec_.target_layer_ids.empty() ||
        spec_.proposal_workspace_bytes == 0) {
        throw std::invalid_argument("DFlash2 round spec is outside the supported domain");
    }
    device_->bind_to_current_thread();

    // The shard-local draft context: the target-feature staging the prefill sink fills, and the
    // draft's own K/V ring for its sliding layers.
    {
        LayoutBuilder builder;
        detail::DFlashPersistentLayout layout;
        layout.prefill_features =
            builder.add_tensor(DType::BF16, {spec_.target_features, spec_.feature_columns},
                               kDFlash2RoundAlignment, "DFlash2 round prefill target features");
        layout.prefill_positions =
            builder.add_tensor(DType::I32, {spec_.feature_columns}, kDFlash2RoundAlignment,
                               "DFlash2 round prefill target positions");
        layout.pending_features =
            builder.add_tensor(DType::BF16, {spec_.target_features, spec_.feature_lanes, 1},
                               kDFlash2RoundAlignment, "DFlash2 round pending target features");
        const CyclicKVCacheLayout ring_layout =
            plan_cyclic_kv_cache(builder, spec_.local_layers, spec_.local_window,
                                 spec_.local_kv_heads, spec_.local_head_dim, 1);
        context_bytes_      = builder.finish(kDFlash2RoundAlignment, "DFlash2 round context");
        ring_payload_bytes_ = ring_layout.payload_bytes();
        context_arena_      = std::make_unique<DeviceArena>(context_bytes_);
        const DeviceSpan backing =
            context_arena_->alloc_bytes(context_bytes_, kDFlash2RoundAlignment);
        CUDA_CHECK(cudaMemset(backing.data, 0, context_bytes_));
        ring_  = std::make_unique<CyclicKVCache>(backing, ring_layout);
        state_ = std::make_unique<detail::DFlashPersistentState>(backing, layout, *ring_);
    }

    // The exact-B decode frame, planned by the shared round planner rather than by a sequence plan.
    {
        LayoutBuilder builder;
        qwen3_5::RoundStateLayout layout = qwen3_5::begin_round_state_layout(
            builder, qwen3_5::RoundStateSpec{
                         .hidden         = spec_.hidden,
                         .output_rows    = spec_.output_rows,
                         .batch_capacity = spec_.batch_capacity,
                         .draft_window   = spec_.draft_window,
                         .backend        = SpeculativeBackend::DFlash2,
                     });
        qwen3_5::complete_round_state_layout(builder, layout);
        frame_bytes_ = builder.finish(kDFlash2RoundAlignment, "DFlash2 round frame");
        frame_arena_ = std::make_unique<DeviceArena>(frame_bytes_);
        const DeviceSpan backing = frame_arena_->alloc_bytes(frame_bytes_, kDFlash2RoundAlignment);
        CUDA_CHECK(cudaMemset(backing.data, 0, frame_bytes_));
        io_    = std::make_unique<qwen3_5::RoundState>(backing, layout);
        frame_ = &*io_->dflash_decode;
        proposal_ = DFlash2Proposal{
            .drafts        = frame_->draft_tokens,
            .candidate_ids = frame_->candidate_ids,
            .scores        = frame_->proposal_q,
            .query_ids     = frame_->proposal_ids,
            .positions     = frame_->proposal_positions,
        };
    }

    // The proposal workspace, deliberately not the shard's: propose_dflash2_batch resets the arena
    // it is handed at its first line, and the shard workspace holds resident buffers.
    proposal_workspace_ = std::make_unique<DeviceArena>(spec_.proposal_workspace_bytes);

    // Scratch that outlives the frame: the continuation hidden the verify stage stores.
    const std::size_t continuation_bytes =
        static_cast<std::size_t>(spec_.hidden) * spec_.batch_capacity * sizeof(std::uint16_t);
    scratch_arena_ = std::make_unique<DeviceArena>(continuation_bytes + kDFlash2RoundAlignment);
    continuation_hidden_ =
        scratch_arena_->alloc(DType::BF16,
                              {spec_.hidden, static_cast<std::int32_t>(spec_.batch_capacity)},
                              kDFlash2RoundAlignment);
}

DFlash2Round::~DFlash2Round() = default;

std::size_t DFlash2Round::proposal_workspace_capacity() const noexcept {
    return proposal_workspace_->capacity();
}

std::size_t DFlash2Round::proposal_workspace_peak() const noexcept {
    return proposal_workspace_->peak_used();
}

void DFlash2Round::reset_proposal_workspace_peak() noexcept { proposal_workspace_->reset_peak(); }

void DFlash2Round::zero_context() {
    device_->bind_to_current_thread();
    CUDA_CHECK(cudaMemsetAsync(context_arena_->base(), 0, context_bytes_, device_->stream));
}

namespace {

// One draft context image: every layer's K rows followed by every layer's V rows, packed at the
// layer extent. The ring's layers are separated by an alignment pitch that is not part of the image,
// which is why this walks the layers instead of copying the arena. The same packing is used for a
// host image and for a device snapshot, so a checkpoint, a device snapshot and a session slab can be
// swapped for one another.
void transfer_ring_image(const CyclicKVCacheSlotView& view, std::byte* image, bool to_image,
                         cudaMemcpyKind kind, cudaStream_t stream) {
    const std::size_t k_total = static_cast<std::size_t>(view.layers) * view.k_layer_bytes;
    for (std::uint32_t layer = 0; layer < view.layers; ++layer) {
        void* k = static_cast<std::byte*>(view.k_layer0.data) +
                  static_cast<std::ptrdiff_t>(layer) * view.k_layer_pitch_bytes;
        void* v = static_cast<std::byte*>(view.v_layer0.data) +
                  static_cast<std::ptrdiff_t>(layer) * view.v_layer_pitch_bytes;
        std::byte* image_k = image + static_cast<std::size_t>(layer) * view.k_layer_bytes;
        std::byte* image_v =
            image + k_total + static_cast<std::size_t>(layer) * view.v_layer_bytes;
        CUDA_CHECK(cudaMemcpyAsync(to_image ? static_cast<void*>(image_k) : k,
                                   to_image ? k : static_cast<const void*>(image_k),
                                   view.k_layer_bytes, kind, stream));
        CUDA_CHECK(cudaMemcpyAsync(to_image ? static_cast<void*>(image_v) : v,
                                   to_image ? v : static_cast<const void*>(image_v),
                                   view.v_layer_bytes, kind, stream));
    }
}

} // namespace

void DFlash2Round::copy_context_to_host(std::byte* destination, cudaStream_t stream) const {
    device_->bind_to_current_thread();
    transfer_ring_image(ring_->slot_view(0), destination, true, cudaMemcpyDeviceToHost, stream);
}

void DFlash2Round::copy_context_from_host(const std::byte* source, cudaStream_t stream) {
    device_->bind_to_current_thread();
    transfer_ring_image(ring_->slot_view(0), const_cast<std::byte*>(source), false,
                        cudaMemcpyHostToDevice, stream);
}

void DFlash2Round::copy_context_to_device(DeviceSpan destination, cudaStream_t stream) const {
    if (destination.bytes < ring_payload_bytes_) {
        throw std::invalid_argument("DFlash2 context image is smaller than the draft ring");
    }
    device_->bind_to_current_thread();
    transfer_ring_image(ring_->slot_view(0), static_cast<std::byte*>(destination.data), true,
                        cudaMemcpyDeviceToDevice, stream);
}

void DFlash2Round::copy_context_from_device(DeviceSpan source, cudaStream_t stream) {
    if (source.bytes < ring_payload_bytes_) {
        throw std::invalid_argument("DFlash2 context image is smaller than the draft ring");
    }
    device_->bind_to_current_thread();
    transfer_ring_image(ring_->slot_view(0), static_cast<std::byte*>(source.data), false,
                        cudaMemcpyDeviceToDevice, stream);
}

ExecutionCore DFlash2Round::round_execution(const ExecutionCore& source, DeviceArena& work) const {
    return ExecutionCore{
        .device           = source.device,
        .parameters       = source.parameters,
        .work             = work,
        .linear_attention = source.linear_attention,
        .replay_records   = source.replay_records,
        .io               = *io_,
        .prefill_hidden   = source.prefill_hidden,
        .prefill_chunk    = source.prefill_chunk,
        .proposal_head    = source.proposal_head,
    };
}

DFlashFeatureSink DFlash2Round::make_prefill_sink(ExecutionCore execution) {
    return DFlashFeatureSink{
        .features  = &state_->prefill_features,
        .positions = &state_->prefill_positions,
        .layers    = std::span<const std::uint32_t>(spec_.target_layer_ids),
        .consume_prefill =
            [this, execution](const Tensor& features, const Tensor& positions, bool /*rewrite*/) {
                append(execution, features, positions,
                       static_cast<std::uint32_t>(features.ne[1]), 0);
            }};
}

void DFlash2Round::append(ExecutionCore execution, const Tensor& features, const Tensor& positions,
                          std::uint32_t exact, std::int32_t lane) {
    device_->bind_to_current_thread();
    // The materialization scratch is dead when this returns; holding it would only raise the
    // prefill forward's peak against the caller's fixed arena.
    auto scratch  = execution.work.scope();
    Tensor counts = execution.work.alloc(DType::I32, {1});
    Tensor lanes  = execution.work.alloc(DType::I32, {1});
    ops::set_i32_scalar(counts, static_cast<std::int32_t>(exact), device_->stream);
    ops::set_i32_scalar(lanes, lane, device_->stream);
    DFlashAppendContext append_state{
        .execution = round_execution(execution, execution.work),
        .dflash    = *state_,
    };
    dflash_append_context(append_state, features, positions, counts, lanes, counts, {exact, exact});
}

DFlashFeatureSink DFlash2Round::make_verify_sink() {
    // The window's residuals land in the pending staging buffer one column per absolute position;
    // the destination lane is the row's frame lane, and valid_columns limits the write to the
    // columns the target actually forwarded (the physical tail holds the last valid column).
    return DFlashFeatureSink{
        .batch_features      = &state_->pending_features,
        .batch_lanes         = &frame_->active_lanes,
        .batch_valid_columns = &frame_->target_valid_columns,
        .batch_width         = spec_.feature_lanes,
        .batch_size          = static_cast<std::int32_t>(spec_.batch_capacity),
        .layers              = std::span<const std::uint32_t>(spec_.target_layer_ids),
    };
}

void DFlash2Round::append_pending(ExecutionCore execution, std::uint32_t start, std::uint32_t end) {
    if (end < start) {
        throw std::invalid_argument("DFlash2 pending append has an inverted frontier");
    }
    const std::uint32_t count = end - start;
    if (count == 0) { return; }
    if (count > static_cast<std::uint32_t>(spec_.feature_lanes)) {
        throw std::invalid_argument("DFlash2 pending append exceeds the staging window");
    }
    device_->bind_to_current_thread();
    auto scratch = execution.work.scope();
    // Compact the committed prefix of the pending window into the physical width the context
    // materialization consumes. prepare_ragged_prefix is the same op the single-device route uses
    // for this hand-off: it copies columns [start, end) and publishes the absolute positions and the
    // per-lane count that drive both the context projection and the ring write.
    Tensor compact = execution.work.alloc(
        DType::BF16, {spec_.target_features, spec_.feature_lanes, 1});
    Tensor positions = execution.work.alloc(DType::I32, {spec_.feature_lanes, 1});
    Tensor counts    = execution.work.alloc(DType::I32, {1});
    Tensor lanes     = execution.work.alloc(DType::I32, {1});
    Tensor starts    = execution.work.alloc(DType::I32, {1});
    Tensor ends      = execution.work.alloc(DType::I32, {1});
    ops::set_i32_scalar(lanes, 0, device_->stream);
    ops::set_i32_scalar(starts, static_cast<std::int32_t>(start), device_->stream);
    ops::set_i32_scalar(ends, static_cast<std::int32_t>(end), device_->stream);
    ops::prepare_ragged_prefix(state_->pending_features, lanes, starts, ends, compact, positions,
                               counts, device_->stream);
    DFlashAppendContext append_state{
        .execution = round_execution(execution, execution.work),
        .dflash    = *state_,
    };
    dflash_append_context(append_state, compact, positions, counts, lanes, counts, {count, count});
}

void DFlash2Round::propose(ExecutionCore execution, const qwen3_5::PagedKVCache& text_cache,
                           TextContext* card, std::uint32_t k, DFlashEnvelopes envelopes) {
    if (k == 0 || k > kDFlashDecodeMaximumDrafts) {
        throw std::invalid_argument("DFlash2 round proposal window is outside [1,15]");
    }
    if (k != spec_.draft_window) {
        throw std::invalid_argument("DFlash2 round proposal width disagrees with its frame");
    }
    device_->bind_to_current_thread();
    CUDA_CHECK(cudaMemcpyAsync(frame_->ingress.data, &ingress_, sizeof(ingress_),
                               cudaMemcpyHostToDevice, device_->stream));
    DFlashBatchContext context{
        .execution                 = round_execution(execution, *proposal_workspace_),
        .text_cache                = text_cache,
        .dflash                    = *state_,
        .frame                     = *frame_,
        .host_ingress              = ingress_,
        .host_egress               = egress_,
        .continuation_hidden_store = continuation_hidden_,
        .tp_card                   = card,
    };
    dflash_propose_batch(context, static_cast<std::int32_t>(spec_.batch_capacity), k, envelopes);
}

} // namespace ninfer::models::qwen3_5::execution
