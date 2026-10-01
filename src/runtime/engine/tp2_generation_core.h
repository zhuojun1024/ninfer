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
#include "models/qwen3_5/frontend/prepared_prompt.h"
#include "models/qwen3_5/model.h"
#include "models/qwen3_5/program/runtime_types.h"
#include "models/qwen3_5/state/decoder_state.h"
#include "runtime/contract/request.h"
#include "runtime/engine/generation_budget.h"
#include "ninfer/tp2_capacity.h"
#include "ninfer/types.h"

#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
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
    // One lane's GDN image inside a whole-pool state buffer (the live pool or a device snapshot).
    // The pool keeps the slot as its outermost dimension, so a lane's image is one block per layer;
    // the pinned image is those blocks back to back with no arena padding.
    struct LaneStateGeometry {
        std::size_t image_bytes        = 0;
        std::size_t conv_bytes         = 0;
        std::size_t recurrent_bytes    = 0;
        std::ptrdiff_t conv_base       = 0;
        std::ptrdiff_t recurrent_base  = 0;
        std::ptrdiff_t conv_pitch      = 0;
        std::ptrdiff_t recurrent_pitch = 0;
        std::uint32_t layers           = 0;
    };
    // Which reserved group of the host checkpoint ring a write lands in. The grid and the tail
    // anchors rotate through their own slots; the divergence anchor and the block anchor own a fixed
    // slot each and are rewritten in place, so they advance no cursor and neither rotation can evict
    // them.
    enum class HostRing { Grid, Tail, Divergence, Block };

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
        // One lane's GDN image inside any whole-pool state buffer of this shard (the live pool or a
        // device snapshot), computed once the pool exists.
        LaneStateGeometry lane_state_geometry;
        // GDN state at reuse boundaries of the last completed prefill (see kReuseSnapshot* in the
        // implementation), used to skip a shared prompt prefix on the next request.
        std::array<DeviceSpan, kReuseSnapshotCount + 1> state_snapshots{};
        // Prefix-reuse checkpoints in pinned host memory, one ring per shard. They carry the state
        // of the frontier they were taken at, so a prompt whose shared prefix ends *inside* the
        // previous prompt restarts at the deepest checkpoint at or before that prefix instead of
        // recomputing from zero. The ring is sized from the host state-image budget
        // (--host-state-slots) and costs no device memory.
        //
        // A slot holds one compact single-lane image and belongs to the lane whose slice of the ring
        // it is, so the pinned budget is the lanes=1 budget whatever the lane count. The per-frontier
        // fields are indexed by lane to keep a stale claim from a re-partitioned ring readable.
        struct HostCheckpoint {
            std::unique_ptr<PinnedHostBuffer> buffer;
            // The masked draft's context at the same frontier, on the shard that owns the draft
            // (shard 0). The draft cannot be recomputed from the target state, so a checkpoint that
            // carries only the GDN image would leave the draft context describing the wrong tokens.
            // Like buffer, it holds one compact ring image per lane.
            std::unique_ptr<PinnedHostBuffer> dflash_buffer;
            std::array<std::uint32_t, kTp2GenerationMaxConcurrency> dflash_frontier{};
            std::array<std::uint32_t, kTp2GenerationMaxConcurrency> position{};
            // The prefill that wrote this lane's checkpoint. A prefill that never completed
            // (cancelled) leaves its slots invalid instead of usable state.
            std::array<std::uint64_t, kTp2GenerationMaxConcurrency> prefill_id{};
            // Set when that prefill completes and cleared as soon as a later request's shared prefix
            // stops covering this frontier: the state is only the state of *this* prompt's prefix
            // while the whole lineage agrees on the tokens before it.
            std::array<bool, kTp2GenerationMaxConcurrency> valid{};
        };
        // Within a lane's slice the ring is split in two: [0, grid_slots) holds the position grid,
        // which keeps the whole context covered and is never evicted by the tail anchors;
        // [grid_slots, size) holds the tail anchors, refreshed every prefill, which land within one
        // chunk of a prompt end.
        std::vector<HostCheckpoint> host_checkpoints;
        std::size_t host_checkpoint_grid_slots = 0;
        // One round-robin cursor per lane: a lane's coverage belongs to its own conversation, so a
        // store on one lane must not advance where another lane writes next.
        std::array<std::size_t, kTp2GenerationMaxConcurrency> host_checkpoint_next{};
        std::array<std::size_t, kTp2GenerationMaxConcurrency> host_checkpoint_tail_next{};
        std::unique_ptr<DeviceArena> workspace;
        Tensor prefill_hidden;
        // Workspace offset after the tensors that must survive every round scope (prefill_hidden and
        // the MTP bridge hidden). A decode round discards its speculative proposal chain by rewinding
        // to this watermark, so the verify's allocation layout is the same in every round.
        std::size_t round_base = 0;
        models::qwen3_5::RoundState io;
        std::unique_ptr<models::qwen3_5::execution::TextContext> context;
        // One KV execution row per lane (P1.2c). lanes == 1 reproduces the original single row.
        // Every row covers its own disjoint physical page group, published once at startup and
        // reused in place across requests. Cross-session retention is per lane as well (P2.3), so
        // the export and import paths address this lane's own handle list.
        std::vector<std::vector<DeviceKVPageLease>> kv_lane_pages;
        std::vector<std::vector<DeviceKVPageHandle>> kv_lane_handles;
        std::vector<KVExecutionRowLease> kv_rows;
        // MTP layer KV (shard 0 only, when --spec mtp): its own page list and one execution row per
        // lane, split the same way as the text cache (P2.1b). `mtp_page_handles` stays the flat
        // physical page list a host checkpoint exports; `mtp_lane_handles`/`mtp_rows`/`mtp_views` are
        // indexed by lane. The MTP layer's extra page groups are dead capacity - the single-GPU
        // route never publishes them either - so only the `logical` slice is mapped.
        std::vector<DeviceKVPageLease> mtp_pages;
        std::vector<DeviceKVPageHandle> mtp_page_handles;
        std::vector<std::vector<DeviceKVPageHandle>> mtp_lane_handles;
        std::vector<KVExecutionRowLease> mtp_rows;
        std::vector<models::qwen3_5::PagedKVCacheView> mtp_views;
        // Round scratch for the MTP proposal (step token/position, RoPE delta, backend KV table row
        // and the MTP prefill/autoregressive buffers). Its backing must outlive the context.
        std::unique_ptr<DeviceArena> round_arena;
        // ReplaySSM records for one speculative verify window and the plan that folds the accepted
        // record prefix back into the live linear-attention state (shard 0 only). The verify forward
        // runs with RecordForReplay, so the live state only advances when the fold says so.
        //
        // The window carries one record row per active lane, and the record op requires each layer's
        // slice of the outer extent to be contiguous - which holds only when the row count equals
        // record_capacity. A round that drives fewer lanes than the route's capacity therefore needs
        // its own narrower geometry, not a slice of the widest one. The backing is sized once for the
        // full lane count and carries one bound view per batch size; a round picks the view matching
        // its own lane count, and the record path and the fold must agree on it.
        std::unique_ptr<DeviceArena> record_arena;
        struct ReplayViews {
            GdnReplayRecords records;
            std::unique_ptr<ops::GdnReplayFoldPlan> fold;
        };
        std::vector<ReplayViews> replay_views;  // index batch - 1
        [[nodiscard]] const GdnReplayRecords& records_for(std::int32_t batch) const {
            if (batch < 1 || static_cast<std::size_t>(batch) > replay_views.size()) {
                throw std::out_of_range("TP-2 replay record batch is out of range");
            }
            return replay_views[static_cast<std::size_t>(batch - 1)].records;
        }
        [[nodiscard]] ops::GdnReplayFoldPlan& fold_for(std::int32_t batch) const {
            if (batch < 1 || static_cast<std::size_t>(batch) > replay_views.size()) {
                throw std::out_of_range("TP-2 replay fold batch is out of range");
            }
            return *replay_views[static_cast<std::size_t>(batch - 1)].fold;
        }
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

    // Identity of one Vision consumer inside a prepared prompt, in prompt-token coordinates. Every
    // image merges to the same placeholder token ids, so two prompts showing different pictures
    // compare equal on tokens alone; the digest and the span it covers are what tell them apart, and
    // a reuse boundary that would keep KV from another picture has to be pulled back to this item's
    // begin. `collect_media_spans` builds one of these per Vision item.
    struct MediaSpan {
        std::uint32_t begin = 0;
        std::uint32_t end   = 0;
        models::qwen3_5::VisionItem item;
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
        // The history survives a recall that comes back shallower than it: the frontier says how far
        // the device returned, while the tokens still say which conversation this is, and the prefix
        // scan needs the longer of the two to find where the next prompt diverges from this one.
        std::vector<TokenId> tokens;
        // The Vision items that history carries, in the same coordinates as the tokens above. The
        // catalog entry is what a recall compares a prompt against, so it has to carry the media
        // identity its token ids cannot see.
        std::vector<MediaSpan> media;
        std::uint32_t frontier = 0;
        // Token count of the prompt the last completed prefill for this conversation walked. A later
        // prompt that still shares this many tokens contains that whole prompt, which is what makes
        // it a later turn of the same conversation instead of a client switching away from it. It
        // stays put while decode extends `tokens`, and survives an eviction and a recall unchanged.
        std::uint32_t prompt_end = 0;
        // The lane whose device pools hold this entry's KV and GDN state, or -1 when it is evicted.
        // Each lane carries one resident conversation, so residency is an (entry, lane) pair rather
        // than a single flag.
        std::int32_t device_lane = -1;
        // The conversation the client named for this entry, when it named one. Naming a session is
        // how a client says it will come back, so a named entry outranks an anonymous continuation
        // when the catalog has to shed one. Only the newest entry of a conversation carries its
        // session: publishing a later turn demotes the entry that held it, exactly as the single-GPU
        // context cache re-binds its session index.
        std::optional<models::qwen3_5::PreparedSessionKey> session;
        // Eviction order between the non-resident entries. Mirrors the single-GPU context cache's
        // private_retention_weight: a named live session (16) outranks an anonymous continuation
        // (4), which outranks a disposable one (1).
        std::uint32_t retention_weight = 4;
        // Host copies, one KV slab per shard (shard B carries no MTP slab) and two GDN state images
        // per shard: the state the evicted frontier sat on, and the state the last completed prefill
        // froze at this conversation's own prompt end. A client that re-renders the answer it was
        // handed normally loses only the framing the template rewrites - the leading whitespace its
        // trim removes - and the walk adopts the span this lineage generated (adopt_generated_turn),
        // so its frontier stays reachable; only a replay that describes a different turn stops at the
        // first generated token, and for that one the second image is the deepest boundary. A slab is
        // sized to the frontier that was evicted, never to max_context.
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
        // Token count the shared-prefix image corresponds to, never deeper than the KV the slabs
        // hold. Zero means no other conversation has diverged from this one yet.
        std::uint32_t host_shared_end = 0;
        // Token count the host KV slabs carry. A store fills them to the evicted frontier; a recall
        // restores the device to a boundary that may be shallower without shortening them, so this
        // is the extent that anything paired with the slabs - a recall, or a shared-prefix capture -
        // has to stay inside. Zero means no slab holds this entry.
        std::uint32_t host_kv_end = 0;
        std::uint64_t lru_clock   = 0;
    };
    static constexpr std::size_t kNoSession = static_cast<std::size_t>(-1);

    [[nodiscard]] GenerationResult execute(Request& request, OutputSink* sink,
                                           const CancellationView& cancellation);
    // The walk itself. execute wraps it so that an exception cannot leave the catalog claiming a
    // live frontier the device no longer holds.
    [[nodiscard]] GenerationResult execute_walk(Request& request, OutputSink* sink,
                                                const CancellationView& cancellation);

    // Multi-lane admission (PLAN-tp2-concurrency.md P1.4). Only reachable when lanes_ > 1, which the
    // constructor keeps for every backend but DFlash v1 (that one collapses to one lane). A submission
    // never touches the device itself: it appends a PendingRequest to the FIFO and then either waits
    // for the batch that picks it up, or - when no driver is active - becomes the driver and forms
    // batches of at most lanes_ from the queue head. The driver holds `execution_mutex_`, so
    // "exactly one thread drives the devices" survives the split, and admission order is FIFO.
    struct PendingRequest {
        std::unique_ptr<Request> request;
        OutputSink* sink = nullptr;
        CancellationView cancellation;
        GenerationResult result;
        std::exception_ptr failure;
        bool complete = false;
    };
    [[nodiscard]] GenerationResult wait_lanes(std::unique_ptr<Request> request, OutputSink* sink,
                                              const CancellationView& cancellation);
    void drive_lane_queue();
    [[nodiscard]] GenerationResult execute_lane(PendingRequest& pending, std::uint32_t lane);
    // P1.4c: one shared decode round for the whole batch, which is the only path that turns
    // concurrency into throughput. Prefill stays serial per lane, each on its own KV row and GDN
    // slot; only the decode rounds are shared. Reachable only for a plain, unconstrained, text-only
    // group, and the constructor keeps lanes_ > 1 to the plain route already.
    void execute_plain_batch(const std::vector<std::shared_ptr<PendingRequest>>& batch);
    // P2.1c: the same shared-round shape as execute_plain_batch, but each round runs one masked-draft
    // proposal over the whole batch and one batched target verify, then accepts and folds per lane.
    // Reachable only for a DFlash2 group with no tool grammar and no media; the plain route keeps the
    // function above, and MTP keeps the serial walk (its draft chain is still single-row).
    void execute_spec_batch(const std::vector<std::shared_ptr<PendingRequest>>& batch);

    // The turn this lineage generated, adopted when the client's rendering of it is the same turn.
    // The template trims the reasoning and the content and writes its own separators between them,
    // so a replay can differ from the generated bytes in exactly those runs and still describe the
    // same turn; replacing the replay with the generated tokens lets the walk restart at this
    // lineage's own frontier instead of re-prefilling the whole answer. Anything else outside those
    // runs - a reordered argument object, an added default, an edited line - keeps the replay.
    struct TurnAdoption {
        bool adopted                = false;
        std::uint32_t prompt_tokens = 0;
        // Longest common prefix of the resident history and this prompt, measured before any
        // replacement, so a trace can still report where the client's rendering parted from the
        // generated bytes after the replay itself has been replaced.
        std::size_t divergence = 0;
    };
    [[nodiscard]] TurnAdoption adopt_generated_turn(models::qwen3_5::PreparedPromptData& data,
                                                    std::uint32_t prompt_tokens, bool trace,
                                                    std::uint32_t lane) const;

    // Names the boundary the prefix scan accepted in the checkpoint vocabulary the Engine already
    // publishes, so a serve log can tell an exact endpoint apart from a fall-back behind the answer
    // this lineage generated.
    [[nodiscard]] PrefixReusePath reuse_path(std::uint32_t reuse,
                                             std::uint32_t block_frontier,
                                             std::uint32_t lane) const noexcept;

    void build_shard(Shard& shard, int shard_index);

    // Width one prefill chunk runs with: the engine option, bounded by the model's chunk maximum and
    // by what the cross-device all-reduce staging buffer carries in one payload (see the definition).
    [[nodiscard]] std::uint32_t prefill_chunk_width(const models::qwen3_5::TextConfig& config) const;

    // The context ceiling a lane may actually write at. `lane_token_capacity_` is the KV budget a
    // lane owns after the static page split, but a lane that reached that position would write on
    // the next lane's first page (and the last lane past the pool), so the admitted window stops one
    // token short of it. A batched MTP route also runs its draft chain a few columns past the
    // longest lane's position, so it keeps that write tail inside the lane's own pages too (P2.1b).
    // The single-lane route keeps advertising `options_.max_context` unchanged.
    [[nodiscard]] std::uint32_t lane_context_window() const noexcept {
        if (lanes_ == 1U) { return options_.max_context; }
        const std::uint32_t margin = mtp_enabled_ ? mtp_drafts_ + 2U : 1U;
        return lane_token_capacity_ > margin ? lane_token_capacity_ - margin : 1U;
    }

    // MTP prefill priming on shard 0: runs the MTP layer over one prefill chunk, appending its own
    // K/V from the chunk's final-norm hidden. last_token is the token sampled from the final chunk's
    // logits, which the MTP layer's last prompt column embeds; that column's hidden becomes the first
    // round's bridge input. Priming re-points the MTP prefill view and the round state's scalar KV
    // row at `lane`'s own execution row and writes the anchor hidden into `lane`'s column (P2.1b).
    void mtp_prefill_priming(Shard& shard, const int* ids, std::uint32_t length,
                             std::uint32_t first_position, Tensor& mtp_input,
                             const Tensor* last_token, bool final_chunk, std::int32_t lane = 0);

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
    make_dflash_prefill_sink(Shard& shard, std::int32_t lane = 0);

    // The masked draft's context image, on the shard that owns the draft; a no-op on shard 1 and on
    // every backend other than DFlash2. The device slot pairs with state_snapshots[slot] and the
    // host image with a checkpoint or session slab, so the two are always copied and restored
    // together with the target state at the same absolute frontier. Every image carries one compact
    // ring image per lane and the lane index selects the slice, exactly like the target state image.
    void store_dflash_image(Shard& shard, PinnedHostBuffer& image, std::uint32_t lane);
    void load_dflash_image(Shard& shard, const PinnedHostBuffer& image, std::uint32_t lane);
    void snapshot_dflash_state(Shard& shard, std::size_t slot, std::uint32_t lane);

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
        // Lane shape this capture belongs to. 'batch' is the number of active lanes the graph was
        // captured for; 'lane' is the state slot a single-column capture baked into the scalar
        // decode path (-1 when the capture binds every slot from device memory, which is the case
        // from two lanes up). One graph per (bucket, batch, lane) is therefore reusable by any
        // assignment of lanes that has that shape.
        std::int32_t batch = 1;
        std::int32_t lane  = -1;
        bool captured      = false;
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
                                                          std::uint32_t visible_end,
                                                          std::int32_t batch = 1,
                                                          std::int32_t lane  = -1);
    // The captured graph covering this window that the current request can still use, or nullptr
    // when the window has to be captured (again). A capture bakes the workspace watermark it ran at,
    // so a request whose own watermark is higher has to capture afresh: the graphs therefore track
    // the highest watermark seen rather than the first one.
    [[nodiscard]] WindowGraph* reusable_window_graph(std::vector<WindowGraph>& graphs,
                                                     std::uint32_t visible_end,
                                                     std::int32_t batch = 1,
                                                     std::int32_t lane  = -1);
    // Launches one captured window on both devices, leaving shard A's device current.
    void launch_window_graph(WindowGraph& graph);
    void capture_verify_graph(WindowGraph& graph, const std::int32_t* ids,
                              const std::int32_t* positions, Tensor& logits_columns,
                              Tensor& hidden_columns,
                              models::qwen3_5::execution::DFlashFeatureSink* sink,
                              const std::int32_t* valid_columns);
    // The batched form of the verify window (P2.2c): the aggregate window is width * batch columns
    // and every per-lane binding is read from the caller's pinned arrays by memcpy nodes, so one
    // capture covers every assignment of a given lane count. A one-column batch still runs the
    // scalar GDN path of the window, whose slot the capture bakes (see WindowGraph::lane).
    void capture_verify_batch_graph(WindowGraph& graph, const std::int32_t* ids,
                                    const std::int32_t* positions,
                                    const std::int32_t* kv_table_rows,
                                    const std::int32_t* state_slots, std::int32_t width,
                                    std::int32_t batch, Tensor& logits_columns,
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

    // The batched form of the verify window (P2.1a): the aggregate window is width * batch columns,
    // lane by lane, and every per-lane binding is a [batch] host array - the paged-KV execution row,
    // the linear-attention state slot, and optionally the clamp extent. first_position must be the
    // farthest lane's first column, because the attention envelope has to cover every lane. It
    // records the window's transitions one physical row per lane, so the fold visits every active
    // lane. On the multi-lane route the window runs through the batch captures below (P2.2c); its
    // single-lane caller stays eager.
    void run_verify_window_batch(const std::int32_t* ids, const std::int32_t* positions,
                                 const std::int32_t* kv_table_rows, const std::int32_t* state_slots,
                                 std::int32_t width, std::int32_t batch, std::int32_t first_position,
                                 Tensor& logits_columns, Tensor& hidden_columns,
                                 models::qwen3_5::execution::DFlashFeatureSink* sink,
                                 const std::int32_t* valid_columns);
    // Replays each active lane's recorded verify-window prefix into its own state slot. state_slots
    // and commit_columns are [batch] host arrays naming the slot a lane recorded into and how many
    // leading columns that lane commits; rows are one per lane, in lane order. The caller restores
    // the round snapshot first, exactly as the single-lane routes do.
    void fold_verify_window(const std::int32_t* state_slots, const std::int32_t* commit_columns,
                            std::int32_t batch);

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

    // The batched form of the plain decode step (P2.2b): one token per active lane in one capture.
    // The four per-lane operands arrive as pinned host arrays and become a small device copy per
    // shard inside the captured sequence, so the graph carries no lane identity of its own except
    // the one the scalar single-column path bakes (see WindowGraph::lane). The round's own scratch
    // and logits are allocated by the caller before this call: the batch stores set the watermark at
    // a startup constant, so a captured layout is reproduced by every later round of that width.
    void capture_decode_batch_graph(WindowGraph& graph, const std::int32_t* tokens,
                                    const std::int32_t* positions, const std::int32_t* kv_table_rows,
                                    const std::int32_t* state_slots, std::int32_t columns,
                                    Tensor& logits);
    void run_plain_decode_step_batch(const std::int32_t* tokens, const std::int32_t* positions,
                                     const std::int32_t* kv_table_rows,
                                     const std::int32_t* state_slots, std::int32_t columns,
                                     const ops::CausalAttentionExecutionEnvelope& exact_envelope,
                                     Tensor& logits);

    // A captured step bakes a bucket-wide attention envelope, so the eager partners stay reachable:
    // EagerBucket isolates the launch mechanism from the envelope, and EagerExact reproduces the
    // pre-graph behaviour of an exact visible extent. The plain decode step and the MTP draft chain
    // are the same kind of object and share this switch.
    enum class StepLaunchMode { Graph, EagerBucket, EagerExact };

    std::vector<WindowGraph> verify_graphs_;
    std::unique_ptr<PinnedHostBuffer> verify_window_host_;
    bool verify_graph_enabled_ = false;

    // Batched verify windows, one graph per (bucket, lane count, and the slot a one-column capture
    // bakes), captured on the multi-lane speculative route. The single-lane route keeps
    // verify_graphs_. NINFER_TP2_VERIFY_BATCH_GRAPH selects the launch mechanism like the other
    // graph switches.
    std::vector<WindowGraph> verify_batch_graphs_;
    StepLaunchMode verify_batch_mode_ = StepLaunchMode::EagerExact;

    // Per-lane operands of a batched verify window, in the same layout as the look-ahead decode
    // round's: width columns of tokens, then the same of positions, then one entry per lane for the
    // clamp extent, the paged-KV execution row and the linear-attention state slot. A captured
    // window re-reads them through memcpy nodes, so their addresses are baked and the buffer is a
    // member whose sections are laid out with the startup lane count, not the round's live one.
    // The block is laid out as [ids: width * lanes_][positions: width * lanes_][valid: lanes_]
    // [KV rows: lanes_][state slots: lanes_], so the width-scaled sections come first and the
    // per-lane ones follow.
    std::unique_ptr<PinnedHostBuffer> batch_window_host_;
    [[nodiscard]] std::int32_t* batch_window_base() noexcept {
        return static_cast<std::int32_t*>(batch_window_host_->data());
    }

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

    // Batched plain decode steps, one graph per (bucket, lane count, and the slot a one-lane
    // capture bakes). Captured on the multi-lane route only; the single-lane route keeps
    // decode_graphs_. NINFER_TP2_DECODE_BATCH_GRAPH selects the launch mechanism like the other
    // graph switches.
    std::vector<WindowGraph> decode_batch_graphs_;
    StepLaunchMode decode_batch_mode_ = StepLaunchMode::EagerExact;

    // Per-lane operands of a batched decode round: tokens, absolute positions, paged-KV execution
    // rows and linear-attention state slots, one lane per entry, with the lane index as the stride
    // in every section. A captured batch step reads these through memcpy nodes, so the addresses
    // are baked into the graph: the buffer has to outlive the round and keep its address, which is
    // why it is a member rather than the round's stack. The fifth section carries the per-round
    // logical positions, whose copy stays eager and does not have to be pinned for a replay.
    std::unique_ptr<PinnedHostBuffer> batch_lane_host_;
    [[nodiscard]] std::int32_t* batch_lane_host_section(std::size_t section) noexcept {
        auto* base = static_cast<std::int32_t*>(batch_lane_host_->data());
        return base + section * static_cast<std::size_t>(lanes_);
    }

    // Total seconds spent loading and materializing both shards (for LoadSummary).
    double load_seconds_ = 0.0;

    // Session retention. session_recall runs at the head of execute, before the prefix scan
    // decides how deep this prompt can start: it displaces the resident session when the incoming
    // prompt belongs to another conversation, so the device pools hold the recalled session by the
    // time the scan reads them.
    // Which frozen state a recall restores: the frontier the entry was evicted at, the end of the
    // prompt its last prefill walked, or the boundary another conversation diverged at.
    enum class RecallState : std::uint8_t { Frontier, PromptEnd, Shared };
    [[nodiscard]] static LaneStateGeometry make_lane_state_geometry(const Shard& shard);
    // Copies the GDN state this lane's device slot holds right now into the ring slot the group
    // owns, tagged with the frontier it captures. A prefill writes these at its chunk boundaries and
    // the sweep that follows a completed prefill is what makes them usable.
    void snapshot_host_checkpoint(Shard& shard, std::uint32_t frontier, HostRing ring,
                                  std::uint32_t lane);
    // Copies one lane's image between a whole-pool device buffer and its compact pinned image, or
    // between two whole-pool device buffers. `other` is the compact pinned image for a host copy
    // and the source pool for a device-to-device one.
    static void copy_lane_state(const LaneStateGeometry& geometry, const void* device_base,
                                std::int32_t lane, void* other, cudaMemcpyKind kind,
                                cudaStream_t stream);
    static void zero_lane_state(const LaneStateGeometry& geometry, void* device_base,
                                std::int32_t lane, cudaStream_t stream);
    // `media` is the incoming prompt's Vision identity. The token ids alone cannot license a
    // boundary: every image merges to the same placeholder ids, so a comparison that stops at the
    // tokens would recall another conversation's KV for a prompt showing a different picture.
    void session_recall(std::uint32_t lane, std::span<const TokenId> prompt_tokens,
                        std::span<const MediaSpan> media);
    // Freezes the state at 'position' into an evicted entry's shared-prefix image. 'from_device'
    // takes it from the device pools, which still hold the state the walk is about to advance;
    // otherwise the two pointers are pinned host images to copy from.
    // 'dflash_frozen', when non-null, is the masked draft's context image at the same boundary, on
    // the shard that owns the draft; 'from_device' takes the live ring instead.
    void session_capture_shared_state(std::size_t index, std::uint32_t position, bool from_device,
                                      const PinnedHostBuffer* const* frozen,
                                      const PinnedHostBuffer* dflash_frozen, std::uint32_t lane);
    // Copies the resident session into its host slabs. Returns false when the host budget cannot
    // hold it, in which case the entry is dropped instead: the next prefill overwrites the device
    // pools, and an entry must never claim state that no longer exists.
    bool session_store_active(std::uint32_t lane);
    // Copies an entry's host slabs back into the device pools and makes it the resident session:
    // the KV before 'boundary' plus the frozen GDN state 'state' names.
    void session_restore(SessionEntry& entry, std::uint32_t boundary, RecallState state,
                         std::uint32_t lane);
    // Frees an entry's host slabs, drops it from the catalog, and keeps the active index valid.
    void session_drop(std::size_t index);
    // Gives entry host KV slabs of at least 'pages' pages per shard, reusing larger existing ones.
    [[nodiscard]] bool session_ensure_host_slabs(SessionEntry& entry, std::uint32_t pages);
    // Evicts the non-resident entry with the lowest retention weight, breaking ties by least recent
    // use, so a client-named session outlives the anonymous continuations around it; false when only
    // the resident one remains.
    bool session_evict_one();
    // Records the conversation and the retention class a request's cache hints assign to an entry,
    // demoting whichever entry held the same session before it: only the newest entry of a
    // conversation keeps the session weight, because that is the one the next turn extends.
    void bind_entry_session(SessionEntry& entry,
                            const models::qwen3_5::PreparedContextCache& cache_hints);
    // Drops a session binding from every entry that still holds it, leaving them anonymous
    // continuations.
    void demote_session_owners(const models::qwen3_5::PreparedSessionKey& key);
    [[nodiscard]] static std::uint32_t private_retention_weight(RetentionClass retention) noexcept;
    // Publishes a walk's full history as the resident catalog entry. `frontier` is how far the
    // device KV and GDN state actually reach: the sampled token that ends the prompt is not
    // forwarded until the first decode round, so a finished response's frontier is one token short
    // of its history.
    void session_publish(const std::vector<TokenId>& history, std::uint32_t frontier,
                         const models::qwen3_5::PreparedContextCache& cache_hints,
                         std::span<const MediaSpan> media, std::uint32_t lane);
    // The in-kernel transport gives up on its deadline instead of waiting forever (see DevicePair),
    // so a stalled rendezvous no longer freezes the process: it surfaces here as a round whose data
    // is a local partial sum rather than an allreduce result. Fails the request with the retryable
    // engine status and discards every reusable prefix, because the stalled round may have
    // half-written KV and GDN state; the next request prefills from scratch.
    void abort_if_ar_stalled();
    // Retires the prefix-reuse checkpoint ring. A checkpoint is only usable while every prompt that
    // followed the prefill that wrote it agreed on the tokens before its position; once the device
    // pools hold another session, that chain is broken even though the ring's positions may still
    // sit inside the recalled history. The lane-scoped overload retires one lane's slice of the
    // ring, which is what a single lane's own recall or abort must do; the global one is for a
    // failure that leaves the device pools describing no lineage at all.
    void invalidate_host_checkpoints();
    void invalidate_host_checkpoints(std::uint32_t lane);
    // Drops only the resident entry and invalidates the device lineage: the paths that abort a
    // walk leave the device pools holding a prefix no catalog entry describes, while every
    // host-resident entry stays valid.
    void session_invalidate_active(std::uint32_t lane);
    // Retires every lane's resident lineage, for the paths that cannot name one lane.
    void session_invalidate_all();
    // The boundary a batched lane accepted in the prefix scan, plus the ring cursor the prefill that
    // follows it starts from. Both batched executors run the same scan the single-lane walk runs, on
    // the lane they are prefilling, so the three share these steps instead of a copy each.
    struct LaneReuse {
        std::uint32_t tokens               = 0;
        std::uint32_t slot                 = 0;
        std::uint32_t next_host_checkpoint = 0;
    };
    // Scans this lane's own lineage for the deepest boundary at or before the shared prefix, tags
    // the checkpoints the coming prefill will write, and records the source in the lane's retention
    // state. Everything it reads is per lane, so two lanes in one batch never see each other's state.
    [[nodiscard]] LaneReuse scan_lane_reuse(std::uint32_t lane, std::span<const TokenId> token_ids,
                                            std::span<const MediaSpan> media,
                                            std::uint32_t prompt_tokens, std::size_t replay_split,
                                            bool adopted, bool trace);
    // The Vision identity of one prepared prompt: one span per item, in prompt-token coordinates,
    // ordered by `begin`. Empty for a text-only prompt.
    [[nodiscard]] static std::vector<MediaSpan> collect_media_spans(
        const models::qwen3_5::PreparedPromptData& data, std::size_t limit);
    // Caps a token-id shared prefix at the first Vision item the two prompts do not share, so a
    // boundary never keeps KV from a picture the incoming prompt does not show. Both lists are
    // ordered by `begin` and describe their own prompt, so the walk is a merge; an item that starts
    // at or beyond the boundary constrains nothing, because the boundary does not reuse it.
    [[nodiscard]] static std::size_t media_prefix_cap(std::span<const MediaSpan> cached,
                                                      std::span<const MediaSpan> incoming,
                                                      std::size_t shared_prefix);
    // Brings this lane's GDN state to the scanned boundary on both shards.
    void restore_lane_gdn(std::uint32_t lane, const LaneReuse& reuse);
    // Brings this lane's masked-draft ring to the same boundary, on the shard that owns the draft.
    // 'zero_when_none' clears the whole ring when there is no boundary: the single-lane walk needs
    // that because it may leave the ring at another lineage's frontier, while a batched lane rebuilds
    // the whole ring from its own prefill sink.
    void restore_lane_dflash(std::uint32_t lane, const LaneReuse& reuse, bool zero_when_none);
    // Publishes a completed lane prefill: the catalog entry, the checkpoint sweep that makes this
    // prefill's ring slots usable, and this lane's slice of snapshot plane 0 (GDN state and, on a
    // masked-draft route, the draft ring).
    void publish_lane_prefill(std::uint32_t lane, std::uint32_t prompt_tokens,
                              const std::vector<TokenId>& tokens,
                              std::span<const MediaSpan> media,
                              const models::qwen3_5::PreparedContextCache& cache_hints);
    // Retires this lane's lineage after a torn prefill, so the next request cannot stand on a state
    // this one left half written.
    void invalidate_lane_prefill(std::uint32_t lane);
    // Opens this request's Vision prefill session on top of the startup plan (P2.4). Returns null
    // when the request carries no media, or when every item it carries lies entirely inside the
    // reused prefix and is therefore already in the KV. The caller keeps `plan` alive while the
    // session lives, because the session binds it by reference.
    [[nodiscard]] std::unique_ptr<models::qwen3_5::execution::VisionPrefillSession>
    open_vision_session(models::qwen3_5::PreparedPromptData& data, std::uint32_t reuse,
                        models::qwen3_5::execution::VisionPrefillPlan& plan);
    // Which memory the prefix scan's winning boundary restores its GDN state from. LiveState means
    // the device state already sits at that boundary - the resident session's frontier restored by
    // a recall, or the tail of a conversation that just decoded - so nothing is copied at all.
    enum class ReuseSource : std::uint8_t { None, DeviceSnapshot, HostCheckpoint, LiveState };

    // Per-lane retention state (PLAN-tp2-concurrency.md P2.3). The single-lane walk drives slot 0;
    // the batched executors drive one slot per active lane. Everything a lane needs in order to know
    // what prefix it may reuse lives here, because a lane's device state is its own slot in the
    // shared pools and two lanes must never share a cached lineage.
    struct RetentionState {
        // Prefix-reuse bookkeeping: the token ids of the last completed prefill, the absolute token
        // positions its state snapshots correspond to, and whether those snapshots are usable.
        std::vector<TokenId> cached_prompt_tokens;
        // The Vision items `cached_prompt_tokens` carries, mirroring the catalog entry's. A reuse
        // scan compares the incoming prompt's items against these: the merged placeholder ids are
        // identical across pictures, so only this can say whether the KV a boundary would keep
        // belongs to the picture the new prompt shows.
        std::vector<MediaSpan> cached_media;
        std::array<std::uint32_t, kReuseSnapshotCount> cached_boundaries{};
        bool cached_state_valid = false;
        // The catalog entry this lane's device pools hold, if any.
        std::size_t active_session = kNoSession;
        // The catalog entry whose history the running prefill compares this prompt against, and
        // therefore the one a divergence state at or before the shared prefix belongs to: after a
        // recall, the entry that recall restored, whose own history is what the scan reads; after a
        // switch, the session the switch just moved into its host slabs. kNoSession when the prompt
        // is compared against no catalog entry, or against one the host budget could not keep. Every
        // prefill reads it once, before the walk advances the state.
        std::size_t anchor_session = kNoSession;
        // Set while this lane's device GDN state sits exactly at the resident entry's frontier, so a
        // prompt that extends that history can reuse it in place with no state copy at all. Every
        // path that aborts a walk clears it before the catalog can be read again.
        bool live_state_valid = false;
        // The block boundary the last prefill's own prompt named, and the ring id that froze its
        // state. A conversation keeps it across its turns and hands it to its slabs when it is
        // evicted, so the next conversation that opens with the same block is recalled on the
        // boundary instead of prefilling the block again. Zero means the running lineage has none.
        std::uint32_t block_anchor_position   = 0;
        std::uint64_t block_anchor_prefill_id = 0;
        ReuseSource reuse_source              = ReuseSource::None;
        // The ring id the running prefill tags its new checkpoints with.
        std::uint64_t host_checkpoint_live_id = 0;
        // DFlash2: how far the draft's local ring has been materialized, in absolute target tokens.
        // The prefill sink advances it by each chunk; every decode round advances it by the
        // committed prefix of the previous round's verify window (append_pending). It never exceeds
        // the target execution frontier and lags it by at most one verify window.
        std::uint32_t dflash_context_frontier = 0;
        // MTP: the first draft of the previous decode step, and whether there is one to compare. A
        // draft proposed at position p predicts the token at p+2, so the target's argmax at the next
        // step is exactly the acceptance oracle for it.
        std::int32_t mtp_previous_draft = -1;
        bool mtp_have_previous          = false;
    };
    // One slot per lane; a route with fewer lanes leaves the tail unused.
    std::array<RetentionState, kTp2GenerationMaxConcurrency> lane_retention_{};
    [[nodiscard]] RetentionState& retention(std::uint32_t lane) noexcept {
        return lane_retention_[lane];
    }
    [[nodiscard]] const RetentionState& retention(std::uint32_t lane) const noexcept {
        return lane_retention_[lane];
    }
    // Session catalog. sessions_ holds the resident entries plus the host-resident ones; a resident
    // entry is the device lineage of exactly one lane, so a lane's cached_prompt_tokens mirrors its
    // history while its GDN state sits at that entry's frontier (live_state_valid).
    std::vector<SessionEntry> sessions_;
    std::uint64_t session_lru_clock_ = 0;
    // Entries the catalog accepts, resident ones included. Zero disables session retention, which
    // is what a zero host KV budget selects.
    std::size_t session_capacity_    = 0;
    std::size_t host_kv_shard_bytes_ = 0;
    // One pinned host KV arena per shard, built lazily on the first eviction so a single-session
    // workload never pins the budget. Shard A's arena also carries the MTP layer geometry.
    std::array<std::unique_ptr<HostKVArena>, 2> host_kv_arena_;
    // Diagnostics counters, reported by NINFER_TP2_SESSION_TRACE and the runtime ledger.
    std::uint64_t session_recalls_       = 0;
    std::uint64_t session_stores_        = 0;
    std::uint64_t session_evictions_     = 0;
    std::uint64_t session_full_prefills_ = 0;

    // Host checkpoint ring: the token stride between checkpoints (0 disables the ring, which is
    // what a zero host state-image budget selects), and the next id to hand out. Which checkpoints
    // are usable is decided per request by Shard::HostCheckpoint::valid, not by the id.
    // Decode width this core was provisioned for (PLAN-tp2-concurrency.md P1.2).
    std::uint32_t lanes_                      = 1;
    std::uint32_t host_checkpoint_stride_     = 0;
    std::uint32_t host_checkpoint_tail_slots_ = 0;
    // Grid slots in one lane's slice of the ring (the whole ring at lanes=1).
    std::uint32_t host_checkpoint_grid_slots_ = 0;
    // Each lane owns this many ring slots, and a slot holds one compact single-lane state image, so
    // the pinned footprint is the lanes=1 budget whatever the lane count
    // (PLAN-tp2-concurrency.md 12.6).
    std::uint32_t host_checkpoint_slots_per_lane_ = 0;
    std::uint64_t host_checkpoint_next_id_    = 1;
    // Host-ring writes of the request being served, reported by the NINFER_TP2_TIMING trace.
    std::uint64_t prefill_host_writes_        = 0;
    // Slots appended to the end of every shard's ring for the divergence anchor and for the stable
    // block's own anchor. They sit outside the configured --host-state-slots budget: the anchors
    // answer a different question than the position grid, and carving them out of the grid would
    // coarsen the stride that covers the whole context. They cost pinned host memory only.
    std::uint32_t host_checkpoint_divergence_slots_ = 0;
    std::uint32_t host_checkpoint_block_slots_      = 0;
    // Committed history below this many tokens is not worth a catalog slot; 0 retains everything.
    std::uint32_t session_retention_floor_tokens_ = 0;

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

    // Multi-lane admission state (P1.4). `lane_token_capacity_` is the KV budget every lane owns
    // after the static page split, which is the context ceiling submit() must clamp against once
    // lanes_ > 1: the pool is shared, so no lane may be offered the whole options_.max_context.
    std::deque<std::shared_ptr<PendingRequest>> lane_queue_;
    std::mutex lane_queue_mutex_;
    std::condition_variable lane_queue_cv_;
    bool lane_driver_active_ = false;
    std::uint32_t lane_token_capacity_ = 0;
    // The lane the batch member currently driving is bound to. It only reaches the non-batch windows,
    // which take it as an argument; the driver holds `execution_mutex_`, so there is exactly one
    // writer and one reader at a time. It stays 0 on the single-lane route.
    std::int32_t active_lane_ = 0;

    // Monotonic counters for runtime_stats().
    std::uint64_t computed_prefill_tokens_ = 0;
    std::uint64_t committed_decode_tokens_ = 0;
    std::uint64_t decode_rounds_           = 0;
};

} // namespace ninfer::runtime
