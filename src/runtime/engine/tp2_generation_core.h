#pragma once

#include "core/arena.h"
#include "core/decode_graph.h"
#include "core/device.h"
#include "core/gdn_replay_records.h"
#include "core/host_kv_arena.h"
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
#include <span>
#include <vector>

namespace ninfer {
namespace models::qwen3_5::execution {
class DFlash2Round;
}
} // namespace ninfer

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
            // The masked draft's context at the same frontier, on the shard that owns the draft
            // (shard 0). The draft cannot be recomputed from the target state, so a checkpoint that
            // carries only the GDN image would leave the draft context describing the wrong tokens.
            std::unique_ptr<PinnedHostBuffer> dflash_buffer;
            std::uint32_t dflash_frontier = 0;
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
        // DFlash2 masked-draft round, owned by the shard that materialized the draft component
        // (shard 0 alone; tp_split_spec places dflash2/* whole there). The round owns the draft's
        // persistent context - the prefill target-feature staging, the pending verify-window
        // staging and the draft's own sliding-window K/V ring - its exact-B decode frame and a
        // proposal workspace of its own (program/dflash_round.h). Shard 1 holds no draft weights
        // and builds none of this.
        std::unique_ptr<models::qwen3_5::execution::DFlash2Round> dflash_round;
        // The draft context at the reuse boundaries the GDN snapshots freeze: slot 0 is the prefill
        // end and slot 1 the rewind behind it (the round-scratch slot 2 is not paired - the verify
        // never advances the draft ring). The draft cannot be recomputed from the target state, so a
        // boundary that cannot restore these bytes cannot be offered for reuse.
        std::array<DeviceSpan, kReuseSnapshotCount> dflash_snapshots{};
        std::unique_ptr<DeviceArena> dflash_snapshot_arena;
    };

    // Cross-session KV retention. Exactly one session's KV and GDN state live in the device pools,
    // so a request that belongs to another conversation evicts the resident session to pinned host
    // memory and pulls the returning one back instead of prefilling it again. An entry owns the
    // token history the device pools must reproduce, the frontier that history reaches, and - once
    // evicted - its host slabs. Entries are the replacement for the single-lineage reuse trio:
    // the resident entry *is* the device lineage, and its tokens are what a prefix scan compares
    // against.
    struct SessionEntry {
        // Prompt plus every committed generated token, in order. The device KV at [0, frontier)
        // holds exactly this prefix, which is what lets a returning prompt prefill only its suffix.
        std::vector<TokenId> tokens;
        std::uint32_t frontier = 0;
        // Token count of the prompt the last completed prefill for this conversation walked. A later
        // prompt that still shares this many tokens contains that whole prompt, which is what makes
        // it a later turn of the same conversation instead of a client switching away from it. It
        // stays put while decode extends `tokens`, and survives an eviction and a recall unchanged.
        std::uint32_t prompt_end = 0;
        // True while this entry's KV and GDN state are the ones in the device pools.
        bool device_resident = false;
        // Host copies, one KV slab per shard (shard B carries no MTP slab) and two GDN state images
        // per shard: the state the evicted frontier sat on, and the state the last completed prefill
        // froze at this conversation's own prompt end. A client that re-renders the answer it was
        // handed stops matching at the first generated token, so for it only the second image is
        // reachable. A slab is sized to the frontier that was evicted, never to max_context.
        std::array<std::unique_ptr<HostKVAllocation>, 2> host_kv;
        std::array<std::unique_ptr<PinnedHostBuffer>, 2> host_state;
        // The masked draft's context image paired with each of the three target state images above,
        // on the shard that owns the draft (shard 0; shard 1 stays empty). A recall restores the
        // draft ring together with the target state it belongs to, because the skipped prefix
        // produces no target residual and nothing else can rebuild it. The counterpart frontier is
        // the boundary the recall restores to, so no separate field is needed.
        std::array<std::unique_ptr<PinnedHostBuffer>, 2> host_dflash;
        std::array<std::unique_ptr<PinnedHostBuffer>, 2> host_dflash_prompt;
        std::array<std::unique_ptr<PinnedHostBuffer>, 2> host_dflash_shared;
        std::array<std::unique_ptr<PinnedHostBuffer>, 2> host_prompt_state;
        // The state at the deepest position another conversation was seen to diverge from this one.
        // A position's state is a function of the tokens before it and this entry's slab carries the
        // KV before that position, so a later conversation sharing this much history can be recalled
        // onto it. That is what keeps a stable block - a system prompt every conversation opens with
        // - reusable after a small unrelated request has taken over the device pools.
        std::array<std::unique_ptr<PinnedHostBuffer>, 2> host_shared_state;
        std::unique_ptr<HostKVAllocation> host_mtp_kv;
        std::uint32_t host_mtp_pages = 0;
        // Token count the prompt-end state image corresponds to. Zero means the entry was evicted
        // before its first prompt completed, and only the frontier image can be recalled.
        std::uint32_t host_prompt_end = 0;
        // Token count the shared-prefix image corresponds to, never deeper than the frontier it was
        // evicted at. Zero means no other conversation has diverged from this one yet.
        std::uint32_t host_shared_end = 0;
        std::uint64_t lru_clock       = 0;
    };
    static constexpr std::size_t kNoSession = static_cast<std::size_t>(-1);

    [[nodiscard]] GenerationResult execute(Request& request, OutputSink* sink,
                                           const CancellationView& cancellation);
    // The walk itself. execute wraps it so that an exception cannot leave the catalog claiming a
    // live frontier the device no longer holds.
    [[nodiscard]] GenerationResult execute_walk(Request& request, OutputSink* sink,
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
    std::vector<TokenId> mtp_propose_window(Shard& shard, Tensor& mtp_input, std::int32_t anchor,
                                            std::uint32_t position, DeviceArena& ws);

    // The prefill feature sink for the shard that owns the masked draft: it captures the target
    // residual at the draft's configured block ids and materializes the draft's local context
    // through dflash_append_context. Empty on a shard that materialized no draft (shard 1, and
    // every backend other than DFlash/DFlash2); the prefill call site then runs NullTap exactly as
    // before.
    [[nodiscard]] std::optional<models::qwen3_5::execution::DFlashFeatureSink>
    make_dflash_prefill_sink(Shard& shard);

    // The masked draft's context image, on the shard that owns the draft; a no-op on shard 1 and on
    // every backend other than DFlash2. The device slot pairs with state_snapshots[slot] and the
    // host image with a checkpoint or session slab, so the two are always copied and restored
    // together with the target state at the same absolute frontier.
    void store_dflash_image(Shard& shard, PinnedHostBuffer& image);
    void load_dflash_image(Shard& shard, const PinnedHostBuffer& image);
    void snapshot_dflash_state(Shard& shard, std::size_t slot);

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
        // Rendezvous id channel for this graph (see DevicePair::create_ar_channel): the host publishes
        // a fresh id block before every replay, and the graph's memcpy node carries it in.
        tp::DevicePair::ArChannel ar_channel = tp::DevicePair::kNoArChannel;
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
                              Tensor& hidden_columns,
                              models::qwen3_5::execution::DFlashFeatureSink* sink,
                              const std::int32_t* valid_columns);
    // The speculative verify window. sink, when non-null, is the masked-draft feature sink the
    // target residual blocks are tapped into; its device scatters travel with a captured window
    // (stable addresses, per-round lane/column tensors re-read by the kernels) while its host
    // bookkeeping ran on the capture's own execution. valid_columns is the number of leading window
    // columns that own their cache slot (the masked-draft budget clamp duplicates the last valid
    // column's position in the trailing columns); 0 appends every column.
    void run_verify_window(const std::int32_t* ids, std::int32_t first_position,
                           Tensor& logits_columns, Tensor& hidden_columns,
                           models::qwen3_5::execution::DFlashFeatureSink* sink = nullptr,
                           std::int32_t valid_columns = 0);

    // One plain (non-speculative) decode step at the given position. The launch mechanism is the
    // decode step mode: a captured graph by default, the eager forward for either A/B partner.
    // Returns this shard's [V,1] logits, allocated where the captured layout expects it.
    // The whole per-round MTP draft chain on shard 0: the anchor and its two position scalars
    // copied in from the pinned buffer, the batch forward, the autoregressive steps, and the drafts
    // copied back to the pinned buffer. Both launch mechanisms run exactly this body - a capture
    // records it without executing, so the capture round then launches the graph it captured.
    void mtp_chain_body(Shard& shard, Tensor& mtp_input, const std::int32_t* pins,
                        std::int32_t* host_drafts, const WindowGraph& bucket, DeviceArena& ws);
    void capture_mtp_chain_graph(WindowGraph& graph, Tensor& mtp_input, const std::int32_t* pins,
                                 std::int32_t* host_drafts, DeviceArena& ws);
    Tensor run_plain_decode_step(std::int32_t token, std::uint32_t position);
    void capture_decode_graph(WindowGraph& graph, const std::int32_t* token,
                              const std::int32_t* position, Tensor& logits);

    // A captured step bakes a bucket-wide attention envelope, so the eager partners stay reachable:
    // EagerBucket isolates the launch mechanism from the envelope, and EagerExact reproduces the
    // pre-graph behaviour of an exact visible extent. The plain decode step and the MTP draft chain
    // are the same kind of object and share this switch.
    enum class StepLaunchMode { Graph, EagerBucket, EagerExact };

    std::vector<WindowGraph> verify_graphs_;
    std::unique_ptr<PinnedHostBuffer> verify_window_host_;
    bool verify_graph_enabled_ = false;

    // The MTP draft chain, captured per envelope bucket like the verify window. One pinned
    // [anchor, position, position+1, drafts(K)] buffer carries every per-round input and output.
    // NINFER_TP2_MTP_CHAIN_GRAPH selects the launch mechanism like NINFER_TP2_DECODE_GRAPH.
    std::vector<WindowGraph> mtp_chain_graphs_;
    std::unique_ptr<PinnedHostBuffer> mtp_chain_host_;
    StepLaunchMode mtp_chain_mode_ = StepLaunchMode::Graph;

    // Plain decode steps.
    std::vector<WindowGraph> decode_graphs_;
    std::unique_ptr<PinnedHostBuffer> decode_window_host_;
    StepLaunchMode decode_step_mode_ = StepLaunchMode::Graph;

    // Total seconds spent loading and materializing both shards (for LoadSummary).
    double load_seconds_ = 0.0;

    // Session retention. session_recall runs at the head of execute, before the prefix scan
    // decides how deep this prompt can start: it displaces the resident session when the incoming
    // prompt belongs to another conversation, so the device pools hold the recalled session by the
    // time the scan reads them.
    // Which frozen state a recall restores: the frontier the entry was evicted at, the end of the
    // prompt its last prefill walked, or the boundary another conversation diverged at.
    enum class RecallState : std::uint8_t { Frontier, PromptEnd, Shared };
    void session_recall(std::span<const TokenId> prompt_tokens);
    // Freezes the state at 'position' into an evicted entry's shared-prefix image. 'from_device'
    // takes it from the device pools, which still hold the state the walk is about to advance;
    // otherwise the two pointers are pinned host images to copy from.
    // 'dflash_frozen', when non-null, is the masked draft's context image at the same boundary, on
    // the shard that owns the draft; 'from_device' takes the live ring instead.
    void session_capture_shared_state(std::size_t index, std::uint32_t position, bool from_device,
                                      const PinnedHostBuffer* const* frozen,
                                      const PinnedHostBuffer* dflash_frozen);
    // Copies the resident session into its host slabs. Returns false when the host budget cannot
    // hold it, in which case the entry is dropped instead: the next prefill overwrites the device
    // pools, and an entry must never claim state that no longer exists.
    bool session_store_active();
    // Copies an entry's host slabs back into the device pools and makes it the resident session:
    // the KV before 'boundary' plus the frozen GDN state 'state' names.
    void session_restore(SessionEntry& entry, std::uint32_t boundary, RecallState state);
    // Frees an entry's host slabs, drops it from the catalog, and keeps the active index valid.
    void session_drop(std::size_t index);
    // Gives entry host KV slabs of at least 'pages' pages per shard, reusing larger existing ones.
    [[nodiscard]] bool session_ensure_host_slabs(SessionEntry& entry, std::uint32_t pages);
    // Evicts the least recently used non-resident entry; false when only the resident one remains.
    bool session_evict_one();
    // Publishes a walk's full history as the resident catalog entry. `frontier` is how far the
    // device KV and GDN state actually reach: the sampled token that ends the prompt is not
    // forwarded until the first decode round, so a finished response's frontier is one token short
    // of its history.
    void session_publish(const std::vector<TokenId>& history, std::uint32_t frontier);
    // The in-kernel transport gives up on its deadline instead of waiting forever (see DevicePair),
    // so a stalled rendezvous no longer freezes the process: it surfaces here as a round whose data
    // is a local partial sum rather than an allreduce result. Fails the request with the retryable
    // engine status and discards every reusable prefix, because the stalled round may have
    // half-written KV and GDN state; the next request prefills from scratch.
    void abort_if_ar_stalled();
    // Retires the prefix-reuse checkpoint ring. A checkpoint is only usable while every prompt that
    // followed the prefill that wrote it agreed on the tokens before its position; once the device
    // pools hold another session, that chain is broken even though the ring's positions may still
    // sit inside the recalled history.
    void invalidate_host_checkpoints();
    // Drops only the resident entry and invalidates the device lineage: the paths that abort a
    // walk leave the device pools holding a prefix no catalog entry describes, while every
    // host-resident entry stays valid.
    void session_invalidate_active();
    // Prefix-reuse bookkeeping: the token ids of the last completed prefill, the absolute token
    // positions its state snapshots correspond to, and whether those snapshots are usable.
    std::vector<TokenId> cached_prompt_tokens_;
    std::array<std::uint32_t, kReuseSnapshotCount> cached_boundaries_{};
    bool cached_state_valid_ = false;
    // Session catalog. sessions_ holds the resident entry plus the host-resident ones; the
    // resident entry is the device lineage, so cached_prompt_tokens_ mirrors its history while
    // the GDN state sits at its frontier (live_state_valid_).
    std::vector<SessionEntry> sessions_;
    std::size_t active_session_      = kNoSession;
    std::uint64_t session_lru_clock_ = 0;
    // The entry the last recall moved into its host slabs, while the running prefill is the only
    // thing that can still name the boundary the two conversations diverged at. kNoSession when the
    // recall found nothing to store. Every prefill reads it once, before the walk advances the state.
    std::size_t host_stored_session_ = kNoSession;
    // Entries the catalog accepts, resident one included. Zero disables session retention, which
    // is what a zero host KV budget selects.
    std::size_t session_capacity_    = 0;
    std::size_t host_kv_shard_bytes_ = 0;
    // One pinned host KV arena per shard, built lazily on the first eviction so a single-session
    // workload never pins the budget. Shard A's arena also carries the MTP layer geometry.
    std::array<std::unique_ptr<HostKVArena>, 2> host_kv_arena_;
    // Set while the device GDN state sits exactly at the resident entry's frontier, so a prompt
    // that extends that history can reuse it in place with no state copy at all. Every path that
    // aborts a walk clears it before the catalog can be read again.
    bool live_state_valid_ = false;
    // Diagnostics counters, reported by NINFER_TP2_SESSION_TRACE and the runtime ledger.
    std::uint64_t session_recalls_       = 0;
    std::uint64_t session_stores_        = 0;
    std::uint64_t session_evictions_     = 0;
    std::uint64_t session_full_prefills_ = 0;

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
    // Which memory the prefix scan's winning boundary restores its GDN state from. LiveState means
    // the device state already sits at that boundary - the resident session's frontier restored by
    // a recall, or the tail of a conversation that just decoded - so nothing is copied at all.
    enum class ReuseSource : std::uint8_t { None, DeviceSnapshot, HostCheckpoint, LiveState };
    ReuseSource reuse_source_ = ReuseSource::None;

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
    // DFlash2 masked-draft route (--spec dflash2): the draft is materialized whole on shard 0 and
    // this core drives its proposal/verify/accept/fold round itself. The two backends are mutually
    // exclusive (the option normalizer admits one), and DFlash2 keeps its own proposal width.
    bool dflash2_enabled_         = false;
    std::uint32_t dflash_drafts_  = 0;
    // How far the draft's local ring has been materialized, in absolute target tokens. The prefill
    // sink advances it by each chunk; every decode round advances it by the committed prefix of the
    // previous round's verify window (append_pending). It never exceeds the target execution
    // frontier and lags it by at most one verify window.
    std::uint32_t dflash_context_frontier_ = 0;
    // Set when this request reused a target prefix whose boundary is not on the prefill grid, so the
    // recalled draft ring belongs to a differently chunked walk and cannot drive the verify window a
    // from-scratch walk would produce. The request then runs target-only: the target prefix reuse
    // stays, the masked draft does not. See GenerationResult::draft_context_declined.
    bool dflash_draft_declined_ = false;
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
