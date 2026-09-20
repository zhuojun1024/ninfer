#pragma once

#include "core/arena.h"
#include "core/decode_graph.h"
#include "core/device.h"
#include "core/gdn_replay_records.h"
#include "core/linear_attention_state.h"
#include "ninfer/ops/gdn_replay.h"
#include "core/tp/device_pair.h"
#include "models/qwen3_5/execution/parameters.h"
#include "models/qwen3_5/execution/text.h"
#include "models/qwen3_5/execution/vision.h"
#include "models/qwen3_5/frontend/frontend.h"
#include "models/qwen3_5/model.h"
#include "models/qwen3_5/program/runtime_types.h"
#include "models/qwen3_5/state/decoder_state.h"
#include "runtime/contract/request.h"
#include "runtime/engine/generation_budget.h"
#include "ninfer/types.h"

#include <array>
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

namespace ninfer::runtime {

// Dedicated tensor-parallel (TP-2) generation core. It owns two model shards on two devices and
// drives a lockstep single-request prefill/decode, reusing the validated forward_tp2 execution
// path. It bypasses the full EngineCore scheduler and CUDA Graphs: exactly one request runs at a
// time. Weights are head/row-split (mixer projections per head, MLP gate/up column-split, o_proj/
// down_proj row-split), so every mixer and FFN output delta is all-reduced per layer. The lm_head is
// weight-tied to the embedding (replicated), so each shard computes the full-vocabulary logits
// independently and the two agree exactly.
class TP2GenerationCore {
public:
    // Prefix-reuse device state snapshots per shard: slot 0 is the prefill end, the others are
    // rewinds behind it. Chat templates render the previous assistant turn and the generation tail
    // differently, so two consecutive prompts share everything up to a point a little before the
    // earlier prompt's end. One rewind behind the prefill end covers that gap; every extra slot is
    // another 73 MiB of resident state per shard, which this context ceiling cannot spare. A prompt
    // that diverges *deeper* inside the previous prompt - a client that re-renders a shorter
    // history at a turn boundary - is served by the host checkpoint ring instead, which holds the
    // same states in pinned host memory and therefore costs no device memory.
    static constexpr std::size_t kReuseSnapshotCount = 2;
    // Extra state slot (beyond the reuse snapshots) holding the pre-verify state of the current MTP
    // round: RecordForReplay advances the live state by the whole window, so the fold must replay the
    // committed columns from this snapshot instead of stacking on an already advanced state.
    static constexpr std::size_t kRoundScratchSlot = kReuseSnapshotCount;
    TP2GenerationCore(const EngineOptions& options, int device_a, int device_b);
    ~TP2GenerationCore();

    TP2GenerationCore(const TP2GenerationCore&)            = delete;
    TP2GenerationCore& operator=(const TP2GenerationCore&) = delete;

    class Request {
    public:
        Request(models::qwen3_5::PreparedPrompt prompt, models::qwen3_5::OutputSession output,
                PromptSummary summary, double prepare_seconds, GenerationBudget budget,
                ResolvedSamplingParameters sampling, OutputConsumerMode consumer_mode)
            : prompt(std::move(prompt)), output(std::move(output)), summary(summary),
              prepare_seconds(prepare_seconds), budget(std::move(budget)), sampling(sampling),
              consumer_mode(consumer_mode) {}

        models::qwen3_5::PreparedPrompt prompt;
        models::qwen3_5::OutputSession output;
        PromptSummary summary;
        double prepare_seconds = 0.0;
        GenerationBudget budget;
        ResolvedSamplingParameters sampling;
        std::vector<TokenId> generated;
        OutputConsumerMode consumer_mode = OutputConsumerMode::Aggregate;
    };

    class Submission {
    public:
        Submission() noexcept = default;
        ~Submission() = default;
        Submission(Submission&&) noexcept            = default;
        Submission& operator=(Submission&&) noexcept = default;
        Submission(const Submission&)            = delete;
        Submission& operator=(const Submission&) = delete;

        GenerationResult wait(OutputSink* sink, const CancellationView& cancellation);

    private:
        Submission(TP2GenerationCore& owner, std::unique_ptr<Request> request) noexcept
            : owner_(&owner), request_(std::move(request)) {}

        TP2GenerationCore* owner_ = nullptr;
        std::unique_ptr<Request> request_;

        friend class TP2GenerationCore;
    };

    Submission submit(models::qwen3_5::PreparedPrompt prompt, PromptSummary summary,
                      double prepare_seconds, ResolvedRequestOptions options,
                      OutputConsumerMode consumer_mode, GenerationObservationOptions observation,
                      std::chrono::steady_clock::time_point pending_deadline);

    [[nodiscard]] bool is_available() const noexcept { return true; }
    // The core owns the tokenizer/chat-template Frontend built from its shard model; the Engine
    // routes its request-preparation methods through this so TP-2 does not load a third model.
    [[nodiscard]] const models::qwen3_5::Frontend& frontend() const noexcept { return *frontend_; }
    [[nodiscard]] LoadSummary load_summary() const;
    [[nodiscard]] MemorySummary memory_summary() const;
    [[nodiscard]] RuntimeStats runtime_stats() const;
    void reset_memory_peaks() noexcept;

private:
    struct Shard {
        DeviceContext device;
        std::unique_ptr<models::qwen3_5::Model> model;
        std::unique_ptr<models::qwen3_5::execution::Parameters> parameters;
        // Per-shard TextConfig (head counts halved) used to size the KV cache and GDN state and
        // to drive the mixer ops. The full-model config (from the model) is used for the
        // replicated components (embeddings, norms, lm_head) and for binding.
        std::unique_ptr<models::qwen3_5::TextConfig> shard_config;
        std::unique_ptr<DeviceArena> kv_arena;
        std::unique_ptr<models::qwen3_5::DecoderState> decoder;
        std::unique_ptr<DeviceArena> state_arena;
        std::unique_ptr<LinearAttentionStatePool> state;
        DeviceSpan state_backing;
        // GDN state at reuse boundaries of the last completed prefill (see kReuseSnapshot* in the
        // implementation), used to skip a shared prompt prefix on the next request.
        std::array<DeviceSpan, kReuseSnapshotCount + 1> state_snapshots{};
        // Prefix-reuse checkpoints in pinned host memory, one ring per shard. They carry the state
        // of the frontier they were taken at, so a prompt whose shared prefix ends *inside* the
        // previous prompt restarts at the deepest checkpoint at or before that prefix instead of
        // recomputing from zero. The ring is sized from the host state-image budget
        // (--host-state-slots) and costs no device memory.
        struct HostCheckpoint {
            std::unique_ptr<PinnedHostBuffer> buffer;
            std::uint32_t position   = 0;
            // The prefill that wrote this checkpoint. A prefill that never completed (cancelled)
            // leaves its slots invalid instead of usable state.
            std::uint64_t prefill_id = 0;
            // Set when that prefill completes and cleared as soon as a later request's shared prefix
            // stops covering this frontier: the state is only the state of *this* prompt's prefix
            // while the whole lineage agrees on the tokens before it.
            bool valid = false;
        };
        // The ring is split in two: [0, grid_slots) holds the position grid, which keeps the whole
        // context covered and is never evicted by the tail anchors; [grid_slots, size) holds the
        // tail anchors, refreshed every prefill, which land within one chunk of a prompt end.
        std::vector<HostCheckpoint> host_checkpoints;
        std::size_t host_checkpoint_grid_slots = 0;
        std::size_t host_checkpoint_next       = 0;
        std::size_t host_checkpoint_tail_next  = 0;
        std::unique_ptr<DeviceArena> workspace;
        Tensor prefill_hidden;
        // Workspace offset after the tensors that must survive every round scope (prefill_hidden and
        // the MTP bridge hidden). A decode round discards its speculative proposal chain by rewinding
        // to this watermark, so the verify's allocation layout is the same in every round.
        std::size_t round_base = 0;
        models::qwen3_5::RoundState io;
        std::unique_ptr<models::qwen3_5::execution::TextContext> context;
        std::vector<DeviceKVPageLease> kv_pages;
        std::vector<DeviceKVPageHandle> kv_page_handles;
        KVExecutionRowLease kv_row;
        // MTP layer KV (shard 0 only, when --spec mtp): its own page list and execution row.
        std::vector<DeviceKVPageLease> mtp_pages;
        std::vector<DeviceKVPageHandle> mtp_page_handles;
        KVExecutionRowLease mtp_row;
        // Round scratch for the MTP proposal (step token/position, RoPE delta, backend KV table row
        // and the MTP prefill/autoregressive buffers). Its backing must outlive the context.
        std::unique_ptr<DeviceArena> round_arena;
        // ReplaySSM records for one speculative verify window and the plan that folds the accepted
        // record prefix back into the live linear-attention state (shard 0 only). The verify forward
        // runs with RecordForReplay, so the live state only advances when the fold says so.
        std::unique_ptr<DeviceArena> record_arena;
        GdnReplayRecords records;
        std::unique_ptr<ops::GdnReplayFoldPlan> replay_fold;
        // Final-norm hidden at the last prompt position: the first round's MTP bridge input.
        Tensor mtp_anchor_hidden;
    };

    [[nodiscard]] GenerationResult execute(Request& request, OutputSink* sink,
                                           const CancellationView& cancellation);

    void build_shard(Shard& shard, int shard_index);

    // MTP prefill priming on shard 0: runs the MTP layer over one prefill chunk, appending its own
    // K/V from the chunk's final-norm hidden. last_token is the token sampled from the final chunk's
    // logits, which the MTP layer's last prompt column embeds; that column's hidden becomes the first
    // round's bridge input.
    void mtp_prefill_priming(Shard& shard, const int* ids, std::uint32_t length,
                             std::uint32_t first_position, Tensor& mtp_input,
                             const Tensor* last_token, bool final_chunk);

    // One MTP proposal window on shard 0: the layer's column at this position embeds the anchor
    // token (the one just sampled) and predicts the token after it; the rest of the window follows
    // autoregressively on the MTP layer. Returns mtp_drafts_ draft token ids.
    std::vector<TokenId> mtp_propose_window(Shard& shard, Tensor& mtp_input, const Tensor& anchor,
                                            std::uint32_t position, DeviceArena& ws);

    // The Engine routes TP-2 submissions from the calling (HTTP) thread, so this core owns
    // serialization: the shard state, the startup-materialized KV pages and the DevicePair belong
    // to exactly one in-flight request, and the workspace arenas are not thread-safe. Requests
    // queue in arrival order, matching the single-device EngineCore contract (TP-2 normalizes
    // max_concurrency to one).
    std::mutex execution_mutex_;

    EngineOptions options_;
    std::unique_ptr<models::qwen3_5::Frontend> frontend_;
    Shard shard_a_;
    Shard shard_b_;
    tp::DevicePair pair_;

    // Captured single-step windows, one graph per device per envelope bucket. The speculative verify
    // forward and the plain one-token decode step are the same launch sequence every round - about a
    // thousand kernels in lockstep on both devices - and their only per-round inputs are the window
    // tokens, their absolute positions and the attention envelope, so capturing the sequence removes
    // the whole step's launch cost from the round (the round was enqueue-limited: the device stream
    // idled for roughly the last few percent of every kernel). See run_verify_window and
    // run_plain_decode_step.
    struct WindowGraph {
        // Inclusive range of visible key extents this bucket covers. The captured envelope is the
        // bucket's widest extent; the per-round positions still come from device memory, exactly as
        // the single-GPU MTP graph relies on.
        std::uint32_t visible_begin = 0;
        std::uint32_t visible_end   = 0;
        bool captured               = false;
        DecodeGraphDefinition definition[2];
        DecodeGraphExecutable executable[2];
        // Workspace watermark the capture ran at, per shard, and the arena offsets it recorded. A
        // replay positions the arenas back at that watermark before the round allocates its fixed
        // scratch, so the scratch lands on the addresses the graph baked; it then advances the
        // arenas by the captured amount, because the captured body ran its host-side arena
        // allocations during capture but a replay does not.
        std::size_t round_base[2]  = {0, 0};
        std::size_t arena_begin[2] = {0, 0};
        std::size_t arena_bytes[2] = {0, 0};
    };
    // The captured graph covering an envelope bucket, or nullptr when no bucket does.
    [[nodiscard]] static WindowGraph* select_window_graph(std::vector<WindowGraph>& graphs,
                                                          std::uint32_t visible_end);
    // The captured graph covering this window that the current request can still use, or nullptr
    // when the window has to be captured (again). A capture bakes the workspace watermark it ran at,
    // so a request whose own watermark is higher has to capture afresh: the graphs therefore track
    // the highest watermark seen rather than the first one.
    [[nodiscard]] WindowGraph* reusable_window_graph(std::vector<WindowGraph>& graphs,
                                                     std::uint32_t visible_end);
    // Launches one captured window on both devices, leaving shard A's device current.
    void launch_window_graph(WindowGraph& graph);
    void capture_verify_graph(WindowGraph& graph, const std::int32_t* ids,
                              const std::int32_t* positions, Tensor& logits_columns,
                              Tensor& hidden_columns);
    void run_verify_window(const std::int32_t* ids, std::int32_t first_position,
                           Tensor& logits_columns, Tensor& hidden_columns);

    // One plain (non-speculative) decode step at the given position. The launch mechanism is the
    // decode step mode: a captured graph by default, the eager forward for either A/B partner.
    // Returns this shard's [V,1] logits, allocated where the captured layout expects it.
    Tensor run_plain_decode_step(std::int32_t token, std::uint32_t position);
    void capture_decode_graph(WindowGraph& graph, const std::int32_t* token,
                              const std::int32_t* position, Tensor& logits);

    std::vector<WindowGraph> verify_graphs_;
    std::unique_ptr<PinnedHostBuffer> verify_window_host_;
    bool verify_graph_enabled_ = false;

    // Plain decode steps. A captured step bakes a bucket-wide attention envelope, so the eager
    // partners stay reachable: EagerBucket isolates the launch mechanism from the envelope, and
    // EagerExact reproduces the pre-graph behaviour of an exact visible extent.
    enum class DecodeStepMode { Graph, EagerBucket, EagerExact };
    std::vector<WindowGraph> decode_graphs_;
    std::unique_ptr<PinnedHostBuffer> decode_window_host_;
    DecodeStepMode decode_step_mode_ = DecodeStepMode::Graph;

    // Total seconds spent loading and materializing both shards (for LoadSummary).
    double load_seconds_ = 0.0;

    // Prefix-reuse bookkeeping: the token ids of the last completed prefill, the absolute token
    // positions its state snapshots correspond to, and whether those snapshots are usable.
    std::vector<TokenId> cached_prompt_tokens_;
    std::array<std::uint32_t, kReuseSnapshotCount> cached_boundaries_{};
    // Rewind depth used for the snapshots of the next prefill, predicted from the previous pair of
    // prompts' shared-prefix gap.
    std::uint32_t rewind_near_ = 9;
    bool cached_state_valid_ = false;

    // Host checkpoint ring: the token stride between checkpoints (0 disables the ring, which is
    // what a zero host state-image budget selects), the id the running prefill tags its new
    // checkpoints with, and the next id to hand out. Which checkpoints are usable is decided per
    // request by Shard::HostCheckpoint::valid, not by the id.
    std::uint32_t host_checkpoint_stride_     = 0;
    std::uint32_t host_checkpoint_tail_slots_ = 0;
    std::uint64_t host_checkpoint_live_id_    = 0;
    std::uint64_t host_checkpoint_next_id_    = 1;
    // Slots appended to the end of every shard's ring for the divergence anchor. They sit outside
    // the configured --host-state-slots budget: the anchor answers a different question than the
    // position grid, and carving them out of the grid would coarsen the stride that covers the
    // whole context. They cost pinned host memory only.
    std::uint32_t host_checkpoint_divergence_slots_ = 0;
    // Set by the reuse scan when the winning boundary is a host checkpoint rather than a device
    // snapshot, so the state restore knows which memory it copies from.
    bool reuse_from_host_ = false;

    // Multi-token prediction (--spec mtp): whether proposals are enabled, and the proposal window
    // width (how many drafts the MTP layer proposes per decode round).
    // Vision (--vision) runs on the shard that materialized the Vision component: the same static
    // split that keeps the MTP layer on shard 0 puts the Vision tower and its encode/handoff arena
    // on shard 1. The plan and the arena are built once at startup; one session per multimodal
    // request owns the encoding of that request's items on top of them.
    std::optional<models::qwen3_5::execution::VisionWorkspacePlan> vision_workspace_;
    std::unique_ptr<DeviceArena> vision_arena_;
    std::size_t vision_handoff_peak_bytes_ = 0;

    bool mtp_enabled_         = false;
    std::uint32_t mtp_drafts_ = 0;
    // The first draft of the previous decode step, and whether there is one to compare. A draft
    // proposed at position p predicts the token at p+2, so the target's argmax at the next step is
    // exactly the acceptance oracle for it.
    std::int32_t mtp_previous_draft_       = -1;
    bool mtp_have_previous_                = false;

    // Monotonic counters for runtime_stats().
    std::uint64_t computed_prefill_tokens_ = 0;
    std::uint64_t committed_decode_tokens_ = 0;
    std::uint64_t decode_rounds_           = 0;
};

} // namespace ninfer::runtime
