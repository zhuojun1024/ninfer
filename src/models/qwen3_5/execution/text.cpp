#include "models/qwen3_5/program/internal.h"
#include "models/qwen3_5/execution/text.h"
#include "models/qwen3_5/execution/attention.h"
#include "models/qwen3_5/execution/gdn.h"
#include "models/qwen3_5/execution/ffn.h"
#include "models/qwen3_5/execution/mtp.h"
#include "models/qwen3_5/execution/workspace.h"

#include "core/nvtx.h"
#include "models/qwen3_5/execution/visual_scatter.h"
#include "models/qwen3_5/execution/vision.h"
#include "models/qwen3_5/program/vision_control.h"
#include "ninfer/ops/argmax.h"
#include "ninfer/ops/attn_input_proj.h"
#include "ninfer/ops/causal_conv1d_silu.h"
#include "ninfer/ops/embedding.h"
#include "ninfer/ops/gated_delta_net.h"
#include "ninfer/ops/gated_rmsnorm.h"
#include "ninfer/ops/gdn_gating.h"
#include "ninfer/ops/gdn_gating_proj.h"
#include "ninfer/ops/gdn_input_proj.h"
#include "ninfer/ops/linear.h"
#include "ninfer/ops/kv_cache_append.h"
#include "ninfer/ops/speculative_round.h"
#include "ninfer/ops/linear_add.h"
#include "ninfer/ops/linear_pair.h"
#include "ninfer/ops/linear_swiglu.h"
#include "ninfer/ops/mtp_pack.h"
#include "ninfer/ops/position.h"
#include "ninfer/ops/residual_add.h"
#include "ninfer/ops/rmsnorm.h"
#include "ninfer/ops/rope.h"
#include "ninfer/ops/sparse_moe.h"
#include "ninfer/ops/scatter.h"
#include "ninfer/ops/scalar.h"
#include "ninfer/ops/sigmoid_mul.h"
#include "ninfer/ops/silu_mul.h"
#include "ninfer/ops/softmax_attention.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::models::qwen3_5::execution {
namespace {

void project(const Tensor& x, const LinearParameters& parameters, Tensor& out,
             WorkspaceArena& workspace, cudaStream_t stream) {
    ops::linear(x, parameters.weight, out, parameters.policy, workspace, stream);
}

void copy_i32(const std::int32_t* source, Tensor& destination, cudaStream_t stream) {
    if (source == nullptr || destination.dtype != DType::I32 || !destination.is_contiguous() ||
        destination.data == nullptr) {
        throw std::invalid_argument("copy_i32: invalid host source or I32 destination");
    }
    CUDA_CHECK(cudaMemcpyAsync(destination.data, source, destination.bytes(),
                               cudaMemcpyHostToDevice, stream));
}

// Materialize a contiguous [rows, columns] block of a fused projection output whose row blocks
// (q|k|gate|v for attention, q|k|v|z for GDN) are concatenated along dim 0. Dim 0 is the innermost
// (contiguous) axis of every activation tensor in this file, so a row sub-block of a wider parent is
// a strided window: at T == 1 the slice degenerates to a single column and stays contiguous, which
// is why the single-token decode path can alias it directly, but a batched prefill must materialize
// the block before handing it to ops that require contiguous operands (rmsnorm, RoPE, the paged
// attention, the split convolution, gated_delta_net).
void copy_row_block(const Tensor& parent, std::int32_t row_begin, std::int32_t rows, Tensor& dst,
                    cudaStream_t stream) {
    const std::int32_t parent_rows = parent.ne[0];
    const std::int32_t columns     = parent.ne[1];
    if (parent.ne[2] != 1 || parent.ne[3] != 1 || !parent.is_contiguous() || parent.data == nullptr) {
        throw std::invalid_argument("copy_row_block: parent must be a contiguous matrix");
    }
    if (rows <= 0 || columns <= 0 || row_begin < 0 || rows > parent_rows - row_begin) {
        throw std::invalid_argument("copy_row_block: invalid row range");
    }
    if (dst.dtype != parent.dtype || dst.ne[0] != rows || dst.ne[1] != columns || dst.ne[2] != 1 ||
        dst.ne[3] != 1 || !dst.is_contiguous() || dst.data == nullptr) {
        throw std::invalid_argument("copy_row_block: destination must be a contiguous [rows, cols]");
    }
    const std::size_t element = dtype_size(parent.dtype);
    CUDA_CHECK(cudaMemcpy2DAsync(dst.data, static_cast<std::size_t>(rows) * element,
                                 static_cast<const unsigned char*>(parent.data) +
                                     static_cast<std::size_t>(row_begin) * element,
                                 static_cast<std::size_t>(parent_rows) * element,
                                 static_cast<std::size_t>(rows) * element,
                                 static_cast<std::size_t>(columns), cudaMemcpyDeviceToDevice,
                                 stream));
}

// Inverse of copy_row_block: materialize a contiguous [rows, columns] source into the row block
// [row_begin, row_begin + rows) of a wider contiguous parent. The vocabulary-parallel output head
// needs it because dim 0 is the innermost axis: a shard projects its own vocabulary rows compactly
// and then stamps them into the full-logits buffer at its vocabulary offset.
void write_row_block(const Tensor& src, Tensor& parent, std::int32_t row_begin, cudaStream_t stream) {
    const std::int32_t parent_rows = parent.ne[0];
    const std::int32_t rows        = src.ne[0];
    const std::int32_t columns     = src.ne[1];
    if (parent.ne[2] != 1 || parent.ne[3] != 1 || !parent.is_contiguous() ||
        parent.data == nullptr) {
        throw std::invalid_argument("write_row_block: parent must be a contiguous matrix");
    }
    if (src.ne[2] != 1 || src.ne[3] != 1 || !src.is_contiguous() || src.data == nullptr ||
        src.dtype != parent.dtype) {
        throw std::invalid_argument("write_row_block: source must be a contiguous matrix");
    }
    if (rows <= 0 || columns <= 0 || row_begin < 0 || rows > parent_rows - row_begin) {
        throw std::invalid_argument("write_row_block: invalid row range");
    }
    const std::size_t element = dtype_size(parent.dtype);
    CUDA_CHECK(cudaMemcpy2DAsync(static_cast<unsigned char*>(parent.data) +
                                     static_cast<std::size_t>(row_begin) * element,
                                 static_cast<std::size_t>(parent_rows) * element, src.data,
                                 static_cast<std::size_t>(rows) * element,
                                 static_cast<std::size_t>(rows) * element,
                                 static_cast<std::size_t>(columns), cudaMemcpyDeviceToDevice,
                                 stream));
}

void require_tensor_shape(const Tensor& t, DType dtype, std::initializer_list<std::int32_t> shape,
                          const char* label) {
    if (t.dtype != dtype) { throw std::invalid_argument(std::string(label) + " dtype mismatch"); }
    int i = 0;
    for (const std::int32_t dim : shape) {
        if (t.ne[i] != dim) { throw std::invalid_argument(std::string(label) + " shape mismatch"); }
        ++i;
    }
    for (; i < 4; ++i) {
        if (t.ne[i] != 1) { throw std::invalid_argument(std::string(label) + " shape mismatch"); }
    }
    if (!t.is_contiguous()) {
        throw std::invalid_argument(std::string(label) + " must be contiguous");
    }
    if (t.data == nullptr) { throw std::invalid_argument(std::string(label) + " data is null"); }
}

void require_tensor_window(const Tensor& t, DType dtype, std::int32_t rows, std::int32_t cols,
                           const char* label) {
    if (cols <= 0) { throw std::invalid_argument(std::string(label) + " cols must be positive"); }
    if (t.dtype != dtype) { throw std::invalid_argument(std::string(label) + " dtype mismatch"); }
    if (t.ne[0] != rows || t.ne[1] < cols || t.ne[2] != 1 || t.ne[3] != 1) {
        throw std::invalid_argument(std::string(label) + " shape mismatch");
    }
    if (!t.is_contiguous()) {
        throw std::invalid_argument(std::string(label) + " must be contiguous");
    }
    if (t.data == nullptr) { throw std::invalid_argument(std::string(label) + " data is null"); }
}

Tensor matrix_window(Tensor& t, std::int32_t cols) {
    if (cols <= 0) { throw std::invalid_argument("matrix_window cols must be positive"); }
    if (t.ne[1] < cols || t.ne[2] != 1 || t.ne[3] != 1) {
        throw std::invalid_argument("matrix_window shape mismatch");
    }
    return t.slice(1, 0, cols);
}

class ScopedPositions {
public:
    ScopedPositions(const Tensor*& slot, const Tensor& positions) : slot_(slot) {
        slot_ = &positions;
    }

    ScopedPositions(const ScopedPositions&)            = delete;
    ScopedPositions& operator=(const ScopedPositions&) = delete;

    ~ScopedPositions() { slot_ = nullptr; }

private:
    const Tensor*& slot_;
};

class ScopedEnvelope {
public:
    ScopedEnvelope(const ops::CausalAttentionExecutionEnvelope*& slot,
                   const ops::CausalAttentionExecutionEnvelope& envelope)
        : slot_(slot) {
        slot_ = &envelope;
    }

    ScopedEnvelope(const ScopedEnvelope&)            = delete;
    ScopedEnvelope& operator=(const ScopedEnvelope&) = delete;

    ~ScopedEnvelope() { slot_ = nullptr; }

private:
    const ops::CausalAttentionExecutionEnvelope*& slot_;
};

template <class T>
class ScopedValue {
public:
    ScopedValue(T& slot, T value) : slot_(slot), previous_(slot) { slot_ = value; }

    ScopedValue(const ScopedValue&)            = delete;
    ScopedValue& operator=(const ScopedValue&) = delete;

    ~ScopedValue() { slot_ = previous_; }

private:
    T& slot_;
    T previous_;
};

} // namespace

void DFlashFeatureSink::begin(const Tensor& value) {
    const bool prefill = features != nullptr && positions != nullptr && batch_features == nullptr;
    const bool batch   = batch_features != nullptr && batch_lanes != nullptr &&
                       batch_valid_columns != nullptr && batch_width > 0 && batch_size > 0;
    if ((!prefill && !batch) || layers.empty()) {
        throw std::logic_error("DFlash feature sink is incomplete");
    }
    captured_mask = 0;
    active_tokens = batch ? batch_width * batch_size : value.ne[1];
    if (value.ne[1] != active_tokens) {
        throw std::logic_error("DFlash batch feature source has an invalid width");
    }
}

void DFlashFeatureSink::capture_layer(int layer, const Tensor& value, cudaStream_t stream) {
    const auto it = std::find(layers.begin(), layers.end(), layer);
    if (it == layers.end()) { return; }
    const std::size_t index = static_cast<std::size_t>(it - layers.begin());
    Tensor* destination     = batch_features != nullptr ? batch_features : features;
    if (layers.size() > 32 || active_tokens <= 0 || value.dtype != DType::BF16 ||
        destination == nullptr ||
        value.ne[0] * static_cast<std::int32_t>(layers.size()) != destination->ne[0] ||
        value.ne[1] != active_tokens) {
        throw std::logic_error("DFlash feature capture shape is invalid");
    }
    if (batch_features != nullptr) {
        Tensor source = value.view({value.ne[0], batch_width, batch_size});
        Tensor target =
            batch_features->slice(0, static_cast<std::int32_t>(index) * value.ne[0], value.ne[0]);
        ops::scatter_bf16_batch(source, *batch_lanes, *batch_valid_columns, target, stream);
        captured_mask |= 1U << index;
        return;
    }
    if (active_tokens > features->ne[1]) {
        throw std::logic_error("DFlash prefill feature capture exceeds its buffer");
    }
    const std::size_t element_bytes = dtype_size(DType::BF16);
    const std::size_t width_bytes   = static_cast<std::size_t>(value.ne[0]) * element_bytes;
    const std::size_t source_pitch  = static_cast<std::size_t>(value.nb[1]);
    const std::size_t target_pitch  = static_cast<std::size_t>(features->nb[1]);
    auto* target                    = static_cast<std::byte*>(features->data) + index * width_bytes;
    CUDA_CHECK(cudaMemcpy2DAsync(target, target_pitch, value.data, source_pitch, width_bytes,
                                 static_cast<std::size_t>(active_tokens), cudaMemcpyDeviceToDevice,
                                 stream));
    captured_mask |= 1U << index;
}

void DFlashFeatureSink::capture_positions(const Tensor& source, cudaStream_t stream) {
    const std::uint32_t complete_mask = layers.size() == 32 ? ~0U : ((1U << layers.size()) - 1U);
    if (captured_mask != complete_mask) {
        throw std::logic_error("DFlash target call did not publish every feature layer");
    }
    if (batch_features != nullptr) {
        if (source.dtype != DType::I32 || source.ne[0] != batch_width ||
            source.ne[1] != batch_size) {
            throw std::logic_error("DFlash batch feature positions are invalid");
        }
        return;
    }
    if (active_tokens <= 0 || source.dtype != DType::I32 || source.ne[0] != active_tokens ||
        positions == nullptr || active_tokens > positions->ne[0]) {
        throw std::logic_error("DFlash feature positions are invalid");
    }
    CUDA_CHECK(cudaMemcpyAsync(positions->data, source.data,
                               static_cast<std::size_t>(active_tokens) * sizeof(std::int32_t),
                               cudaMemcpyDeviceToDevice, stream));
}

void DFlashFeatureSink::consume_prefill_chunk(std::int32_t tokens, bool rewrite_checkpoint) {
    if (!consume_prefill || tokens != active_tokens) {
        throw std::logic_error("DFlash prefill feature consumer is unavailable");
    }
    Tensor feature_window  = features->slice(1, 0, tokens);
    Tensor position_window = positions->slice(0, 0, tokens);
    consume_prefill(feature_window, position_window, rewrite_checkpoint);
}

TextContext::TextContext(DeviceContext& ctx, const execution::Parameters& weights,
                         WorkspaceArena& work, qwen3_5::PagedKVCacheView kv,
                         LinearAttentionStatePool& state, qwen3_5::RoundState& io,
                         Tensor& prefill_hidden, std::uint32_t prefill_chunk,
                         std::uint32_t text_kv_base, qwen3_5::PagedKVCacheView mtp_kv,
                         const qwen3_5::PagedKVCache* batch_text_kv,
                         const qwen3_5::PagedKVCache* batch_mtp_kv)
    : ctx_(ctx), parameters_(weights), config_(weights.model.config().text), work_(work), kv_(kv),
      mtp_kv_(mtp_kv), state_(state), io_(io), prefill_hidden_(prefill_hidden),
      prefill_chunk_(prefill_chunk), text_kv_base_(text_kv_base), batch_text_kv_(batch_text_kv),
      batch_mtp_kv_(batch_mtp_kv) {
    if (prefill_chunk_ == 0 ||
        prefill_chunk_ > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::invalid_argument("TextContext effective prefill chunk must fit positive int32");
    }
    if (mtp_enabled() && !io_.mtp_decode && !io_.mtp) {
        throw std::invalid_argument("MTP TextContext requires MTP round state");
    }
    set_linear_state_slots(0, 0);
    embed_      = &parameters_.text.token_embedding;
    final_norm_ = &parameters_.text.final_norm;
    lm_head_    = &parameters_.text.output_head;
    mtp_        = parameters_.mtp ? &*parameters_.mtp : nullptr;
    if (mtp_enabled() && mtp_ == nullptr) {
        throw std::invalid_argument("MTP state requires selected MTP parameters");
    }
    if (parameters_.proposal) {
        const auto& p = *parameters_.proposal;
        set_proposal_head(
            &p.head, p.token_ids ? static_cast<const std::int32_t*>(p.token_ids->data) : nullptr,
            dimension(p.rows));
    }
}

TextContext::~TextContext() = default;

void TextContext::set_linear_state_slots(std::int32_t source_slot, std::int32_t destination_slot) {
    if (source_slot < 0 || source_slot >= state_.slot_count() || destination_slot < 0 ||
        destination_slot >= state_.slot_count()) {
        throw std::invalid_argument("TextContext Linear Attention slots are invalid");
    }
    linear_state_source_slot_      = source_slot;
    linear_state_destination_slot_ = destination_slot;
}

void TextContext::set_gdn_state_action(GdnStateAction action,
                                       const GdnReplayRecords* replay_records) {
    if ((action == GdnStateAction::RecordForReplay) != (replay_records != nullptr)) {
        throw std::invalid_argument("TextContext GDN state action has inconsistent records");
    }
    gdn_state_action_ = action;
    replay_records_   = replay_records;
}

void TextContext::set_tp_peer(TextContext* peer, tp::DevicePair* pair) {
    if (peer == nullptr || pair == nullptr) {
        throw std::invalid_argument("set_tp_peer: peer context and device pair must be set");
    }
    if (peer == this) { throw std::invalid_argument("set_tp_peer: a shard cannot be its own peer"); }
    peer_tp_ = peer;
    pair_tp_ = pair;
}

bool TextContext::embedding_is_split() const noexcept {
    return dimension(embed_->k) != dimension(config_.hidden_size);
}

void TextContext::embedding_full_width(const Tensor& ids, Tensor& out) {
    if (!embedding_is_split()) {
        ops::embedding(ids, *embed_, out, ctx_.stream);
        return;
    }
    if (peer_tp_ == nullptr || pair_tp_ == nullptr) {
        throw std::logic_error(
            "embedding_full_width: a column-split token embedding needs set_tp_peer");
    }
    // The peer ids are broadcast inside embedding_tp2 and a peer scratch destination is allocated
    // there, so only this shard's full-width [hidden, T] result is requested.
    embedding_tp2(*peer_tp_, *pair_tp_, ids, nullptr, out, nullptr);
}

void TextContext::merge_local_row_blocks(TextContext& peer, tp::DevicePair& pair,
                                         const Tensor& local, const Tensor& local_peer,
                                         Tensor& destination, Tensor& destination_peer,
                                         std::int32_t local_rows) {
    const std::int32_t full_rows = destination.ne[0];
    const std::int32_t columns   = destination.ne[1];
    if (local_rows <= 0 || local_rows * 2 != full_rows || columns <= 0 || shard_index_ < 0 ||
        peer.shard_index_ < 0 || peer.shard_index_ == shard_index_) {
        throw std::logic_error(
            "merge_local_row_blocks: a split weight needs exactly half the rows and two distinct "
            "shards");
    }
    const auto matrix = [columns](const Tensor& t, std::int32_t rows, const char* label) {
        if (t.dtype != DType::BF16 || t.ne[0] != rows || t.ne[1] != columns || t.ne[2] != 1 ||
            t.ne[3] != 1 || !t.is_contiguous() || t.data == nullptr) {
            throw std::invalid_argument(std::string("merge_local_row_blocks: ") + label +
                                        " must be a contiguous BF16 [rows, T] matrix");
        }
    };
    matrix(local, local_rows, "local block");
    matrix(local_peer, local_rows, "peer block");
    matrix(destination, full_rows, "destination");
    matrix(destination_peer, full_rows, "peer destination");

    ctx_.bind_to_current_thread();
    CUDA_CHECK(cudaMemsetAsync(destination.data, 0, destination.bytes(), ctx_.stream));
    write_row_block(local, destination, shard_index_ * local_rows, ctx_.stream);
    peer.ctx_.bind_to_current_thread();
    CUDA_CHECK(
        cudaMemsetAsync(destination_peer.data, 0, destination_peer.bytes(), peer.ctx_.stream));
    write_row_block(local_peer, destination_peer, peer.shard_index_ * local_rows, peer.ctx_.stream);
    ctx_.bind_to_current_thread();
    pair.allreduce(destination.data, destination_peer.data, destination.bytes(), ctx_.stream,
                   peer.ctx_.stream);
}

void TextContext::embedding_tp2(TextContext& peer, tp::DevicePair& pair, const Tensor& ids,
                                const Tensor* ids_peer, Tensor& x, Tensor* x_peer) {
    const std::int32_t hidden = dimension(config_.hidden_size);
    const std::int32_t local  = dimension(embed_->k);
    if (ids.dtype != DType::I32 || ids.ne[0] <= 0 || ids.ne[1] != 1 || ids.ne[2] != 1 ||
        ids.ne[3] != 1 || !ids.is_contiguous() || ids.data == nullptr) {
        throw std::invalid_argument("embedding_tp2: ids must be a contiguous I32 [T] tensor");
    }
    if (local == hidden) {
        // Replicated table: both shards hold every row, so each gathers its own copy.
        if (ids_peer == nullptr || x_peer == nullptr) {
            throw std::invalid_argument(
                "embedding_tp2: a replicated embedding needs both shards' ids and destinations");
        }
        ctx_.bind_to_current_thread();
        ops::embedding(ids, *embed_, x, ctx_.stream);
        peer.ctx_.bind_to_current_thread();
        ops::embedding(*ids_peer, *peer.embed_, *x_peer, peer.ctx_.stream);
        return;
    }
    if (local <= 0 || local * 2 != hidden || shard_index_ < 0 || peer.shard_index_ < 0 ||
        peer.shard_index_ == shard_index_ || dimension(peer.embed_->k) != local) {
        throw std::logic_error(
            "embedding_tp2: a column-split embedding needs half the hidden size per shard on two "
            "distinct shards");
    }
    const std::int32_t columns = x.ne[1];
    if (x.dtype != DType::BF16 || x.ne[0] != hidden || columns <= 0 || x.ne[2] != 1 || x.ne[3] != 1 ||
        !x.is_contiguous() || x.data == nullptr) {
        throw std::invalid_argument(
            "embedding_tp2: destination must be a contiguous BF16 [hidden, T] matrix");
    }
    ctx_.bind_to_current_thread();
    auto scope      = work_.scope();
    Tensor partial  = work_.alloc(DType::BF16, {local, columns});
    ops::embedding(ids, *embed_, partial, ctx_.stream);

    peer.ctx_.bind_to_current_thread();
    auto peer_scope          = peer.work_.scope();
    Tensor partial_peer      = peer.work_.alloc(DType::BF16, {local, columns});
    Tensor destination_peer  = x_peer != nullptr ? *x_peer
                                                 : peer.work_.alloc(DType::BF16, {hidden, columns});
    Tensor ids_peer_storage;
    Tensor ids_peer_view;
    if (ids_peer == nullptr) {
        // The peer needs the same token ids. The text path already binds them per shard; the MTP
        // stem only owns them here, so mirror a private copy into a zeroed peer buffer and sum the
        // pair - a broadcast that needs no host round trip.
        // The pair's all-reduce works in 16-byte units, so the broadcast buffer is padded up and
        // the tail zeroed on both sides; only the leading ids are read.
        const std::size_t id_bytes = static_cast<std::size_t>(ids.ne[0]) * sizeof(std::int32_t);
        const std::size_t broadcast_bytes = ((id_bytes + 15u) / 16u) * 16u;
        const std::int32_t words = static_cast<std::int32_t>(broadcast_bytes / sizeof(std::int32_t));
        ctx_.bind_to_current_thread();
        Tensor ids_staging = work_.alloc(DType::I32, {words});
        CUDA_CHECK(cudaMemsetAsync(ids_staging.data, 0, broadcast_bytes, ctx_.stream));
        CUDA_CHECK(cudaMemcpyAsync(ids_staging.data, ids.data, id_bytes,
                                   cudaMemcpyDeviceToDevice, ctx_.stream));
        peer.ctx_.bind_to_current_thread();
        ids_peer_storage = peer.work_.alloc(DType::I32, {words});
        CUDA_CHECK(cudaMemsetAsync(ids_peer_storage.data, 0, broadcast_bytes, peer.ctx_.stream));
        ctx_.bind_to_current_thread();
        pair.allreduce(ids_staging.data, ids_peer_storage.data, broadcast_bytes, ctx_.stream,
                       peer.ctx_.stream);
        // The gather validates ids against the output's token extent, so the peer reads only the
        // leading ids from the padded staging buffer.
        ids_peer_view = ids_peer_storage.slice(0, 0, ids.ne[0]);
        ids_peer      = &ids_peer_view;
    }
    peer.ctx_.bind_to_current_thread();
    ops::embedding(*ids_peer, *peer.embed_, partial_peer, peer.ctx_.stream);
    merge_local_row_blocks(peer, pair, partial, partial_peer, x, destination_peer, local);
}

void TextContext::mtp_forward_stem(const Tensor& ids, const Tensor& hidden,
                                   const Tensor* input_embeddings, Tensor& x, Tensor& ah) {
    cudaStream_t s     = ctx_.stream;
    const int T        = ids.ne[0] * ids.ne[1];
    Tensor flat_ids    = ids.view({T});
    Tensor flat_hidden = hidden.view({dimension(config_.hidden_size), T});

    auto roots = workspace::mtp_stem(work_, config_, T, input_embeddings == nullptr);
    Tensor emb;
    if (input_embeddings != nullptr) {
        if (input_embeddings->dtype != DType::BF16 ||
            input_embeddings->ne[0] != dimension(config_.hidden_size) ||
            input_embeddings->numel() !=
                static_cast<std::int64_t>(dimension(config_.hidden_size)) * T ||
            !input_embeddings->is_contiguous() || input_embeddings->data == nullptr) {
            throw std::invalid_argument("MTP input embeddings shape mismatch");
        }
        emb = input_embeddings->view({dimension(config_.hidden_size), T});
    } else {
        emb = roots.embedding;
        if (embedding_is_split()) {
            // The MTP layer runs on this shard alone, so the column split needs the peer's half:
            // broadcast the token ids and let it gather its columns, then merge.
            if (peer_tp_ == nullptr || pair_tp_ == nullptr) {
                throw std::logic_error(
                    "mtp_forward_stem: a column-split token embedding needs set_tp_peer");
            }
            embedding_tp2(*peer_tp_, *pair_tp_, flat_ids, nullptr, emb, nullptr);
        } else {
            ops::embedding(flat_ids, *embed_, emb, s);
        }
    }

    Tensor e = roots.normalized_embedding;
    Tensor h = roots.normalized_hidden;
    ops::rmsnorm(emb, mtp_->embedding_norm, config_.rms_norm_eps, true, e, s);
    ops::rmsnorm(flat_hidden, mtp_->hidden_norm, config_.rms_norm_eps, true, h, s);

    Tensor fc_in = roots.packed_input;
    ops::mtp_pack_fc_input(e, h, fc_in, s);

    x = roots.residual;
    project(fc_in, mtp_->input_projection, x, work_, s);

    ah = roots.attention_hidden;
    ops::rmsnorm(x, mtp_->input_norm, config_.rms_norm_eps, true, ah, s);
}

void TextContext::mtp_forward_tail(Tensor& x, const Tensor& ah, const Tensor& positions,
                                   const Tensor& rope_positions,
                                   ops::CausalAttentionExecutionEnvelope envelope,
                                   Tensor& mtp_hidden) {
    cudaStream_t s = ctx_.stream;
    const int T    = x.ne[1];

    const auto projection = workspace::mtp_attention_projection(work_, config_, T);
    Tensor q              = projection.query.view({dimension(config_.attention->head_dim),
                                                   dimension(config_.attention->num_attention_heads), T});
    Tensor k              = projection.key.view({dimension(config_.attention->head_dim),
                                                 dimension(config_.attention->num_key_value_heads), T});
    Tensor gate           = projection.gate.view({dimension(config_.attention->head_dim),
                                                  dimension(config_.attention->num_attention_heads), T});
    Tensor v              = projection.value.view({dimension(config_.attention->head_dim),
                                                   dimension(config_.attention->num_key_value_heads), T});
    Tensor q_flat         = q.view({dimension(config_.attention->query_width()), T});
    Tensor gate_flat      = gate.view({dimension(config_.attention->query_width()), T});
    Tensor k_flat         = k.view({dimension(config_.attention->key_width()), T});
    Tensor v_flat         = v.view({dimension(config_.attention->key_width()), T});
    mtp_projection(ah, mtp_->projection, *config_.attention, q_flat, gate_flat, k_flat, v_flat,
                   work_, s);

    const auto results = workspace::mtp_attention_results(work_, config_, T);
    Tensor qn =
        results.normalized_query.view({dimension(config_.attention->head_dim),
                                       dimension(config_.attention->num_attention_heads), T});
    Tensor kn = results.normalized_key.view({dimension(config_.attention->head_dim),
                                             dimension(config_.attention->num_key_value_heads), T});
    ops::rmsnorm(q, mtp_->query_norm, config_.rms_norm_eps, true, qn, s);
    ops::rmsnorm(k, mtp_->key_norm, config_.rms_norm_eps, true, kn, s);
    Tensor rope_for_op = active_sequence_batch_ != 0 ? rope_positions.view({T}) : rope_positions;
    text_rope(rope_for_op, *config_.rope_parameters, qn, kn, s);

    Tensor a = results.attention.view({dimension(config_.attention->head_dim),
                                       dimension(config_.attention->num_attention_heads), T});
    if (active_sequence_batch_ != 0) {
        const std::int32_t width = active_sequence_width_;
        if (width <= 0 || width * active_sequence_batch_ != T ||
            active_backend_kv_table_rows_ == nullptr || active_valid_columns_ == nullptr) {
            throw std::logic_error("MTP sequence batch binding is incomplete");
        }
        Tensor q_batch        = qn.view({dimension(config_.attention->head_dim),
                                         dimension(config_.attention->num_attention_heads), width,
                                         active_sequence_batch_});
        Tensor k_batch        = kn.view({dimension(config_.attention->head_dim),
                                         dimension(config_.attention->num_key_value_heads), width,
                                         active_sequence_batch_});
        Tensor v_batch        = v.view({dimension(config_.attention->head_dim),
                                        dimension(config_.attention->num_key_value_heads), width,
                                        active_sequence_batch_});
        Tensor a_batch        = a.view({dimension(config_.attention->head_dim),
                                        dimension(config_.attention->num_attention_heads), width,
                                        active_sequence_batch_});
        Tensor position_batch = positions.view({width, active_sequence_batch_});
        ops::causal_softmax_attention(
            q_batch, k_batch, v_batch, position_batch, *active_valid_columns_,
            *active_backend_kv_table_rows_,
            {dimension(config_.attention->head_dim),
             dimension(config_.attention->num_attention_heads),
             dimension(config_.attention->num_key_value_heads)},
            static_cast<float>(1.0 / std::sqrt(static_cast<double>(config_.attention->head_dim))),
            batch_mtp_kv_->batch_layer_view(0), envelope, work_, a_batch, s);
    } else {
        ops::causal_softmax_attention(
            qn, kn, v, positions, Tensor{}, io_.backend_kv_table_row,
            {dimension(config_.attention->head_dim),
             dimension(config_.attention->num_attention_heads),
             dimension(config_.attention->num_key_value_heads)},
            static_cast<float>(1.0 / std::sqrt(static_cast<double>(config_.attention->head_dim))),
            batch_mtp_kv_->batch_layer_view(0), envelope, work_, a, s);
    }
    ops::sigmoid_mul(gate, a, s);

    const auto post = workspace::mtp_post_attention(work_, config_, T);
    Tensor o        = post.output;
    project(a.view({dimension(config_.attention->query_width()), T}), mtp_->output, o, work_, s);
    ops::residual_add(o, x, s);

    Tensor mh = post.post_mixer_hidden;
    ops::rmsnorm(x, mtp_->post_attention_norm, config_.rms_norm_eps, true, mh, s);

    {
        auto post_mixer_scope = work_.scope();
        ffn(mh, mtp_->ffn, x, {}, work_, s, true);
    }

    Tensor flat_mtp_hidden = mtp_hidden.view({dimension(config_.hidden_size), T});
    ops::rmsnorm(x, mtp_->final_norm, config_.rms_norm_eps, true, flat_mtp_hidden, s);
}

void TextContext::mtp_forward_core(const Tensor& ids, const Tensor& hidden, const Tensor& positions,
                                   const Tensor& rope_positions,
                                   ops::CausalAttentionExecutionEnvelope envelope,
                                   Tensor& mtp_hidden, const Tensor* input_embeddings) {
    if (batch_mtp_kv_ == nullptr) { throw std::runtime_error("MTP forward is not enabled"); }
    nvtx::ScopedRange forward_range(nvtx::Name::MtpForward, nvtx::Category::Mtp,
                                    static_cast<std::uint64_t>(ids.numel()));
    auto scratch_scope = work_.scope();
    Tensor x;
    Tensor ah;
    mtp_forward_stem(ids, hidden, input_embeddings, x, ah);
    mtp_forward_tail(x, ah, positions, rope_positions, envelope, mtp_hidden);
}

void TextContext::mtp_prefill_chunk(const Tensor& ids, const Tensor& hidden,
                                    const Tensor* input_embeddings, const Tensor& positions,
                                    const Tensor& rope_positions,
                                    ops::CausalAttentionExecutionEnvelope envelope,
                                    bool final_chunk, Tensor* final_hidden, Tensor* logits,
                                    Tensor* draft_token) {
    if (!mtp_kv_.valid()) { throw std::runtime_error("MTP prefill is not enabled"); }
    const int T = ids.ne[0];
    if (T <= 0 || static_cast<std::uint32_t>(T) > prefill_chunk_) {
        throw std::invalid_argument("MTP prefill chunk T must be in [1,prefill_chunk]");
    }
    nvtx::ScopedRange mtp_prefill_range(nvtx::Name::PrefillMtpChunk, nvtx::Category::Mtp,
                                        static_cast<std::uint64_t>(T));
    require_tensor_shape(ids, DType::I32, {T}, "MTP prefill ids");
    require_tensor_shape(hidden, DType::BF16, {dimension(config_.hidden_size), T},
                         "MTP prefill hidden");
    require_tensor_shape(positions, DType::I32, {T}, "MTP prefill positions");
    if (rope_positions.dtype != DType::I32 || rope_positions.ne[0] != T ||
        (rope_positions.ne[1] != 1 && rope_positions.ne[1] != 3) || rope_positions.ne[2] != 1 ||
        rope_positions.ne[3] != 1 || !rope_positions.is_contiguous() ||
        rope_positions.data == nullptr) {
        throw std::invalid_argument("MTP prefill rope positions must be [T] or [T,3]");
    }
    if (final_chunk) {
        if (final_hidden == nullptr || logits == nullptr || draft_token == nullptr) {
            throw std::invalid_argument("MTP final prefill outputs are required");
        }
        require_tensor_shape(*final_hidden, DType::BF16, {dimension(config_.hidden_size), 1},
                             "MTP final prefill hidden");
        require_tensor_shape(*logits, DType::BF16, {dimension(config_.vocab_size), 1},
                             "MTP final prefill logits");
        require_tensor_shape(*draft_token, DType::I32, {1}, "MTP final prefill draft token");
    }

    cudaStream_t s     = ctx_.stream;
    auto scratch_scope = work_.scope();
    Tensor x_last;
    Tensor ah_last;
    if (final_chunk) {
        x_last  = work_.alloc(DType::BF16, {dimension(config_.hidden_size), 1});
        ah_last = work_.alloc(DType::BF16, {dimension(config_.hidden_size), 1});
    }

    {
        auto bulk_scope = work_.scope();
        Tensor x;
        Tensor ah;
        mtp_forward_stem(ids, hidden, input_embeddings, x, ah);

        Tensor k_flat = work_.alloc(DType::BF16, {dimension(config_.attention->key_width()), T});
        Tensor v_flat = work_.alloc(DType::BF16, {dimension(config_.attention->key_width()), T});
        mtp_kv_projection(ah, mtp_->projection, *config_.attention, k_flat, v_flat, work_, s);
        Tensor k = k_flat.view({dimension(config_.attention->head_dim),
                                dimension(config_.attention->num_key_value_heads), T});
        Tensor v = v_flat.view({dimension(config_.attention->head_dim),
                                dimension(config_.attention->num_key_value_heads), T});
        Tensor kn =
            work_.alloc(DType::BF16, {dimension(config_.attention->head_dim),
                                      dimension(config_.attention->num_key_value_heads), T});
        ops::rmsnorm(k, mtp_->key_norm, config_.rms_norm_eps, true, kn, s);
        text_rope(rope_positions, *config_.rope_parameters, kn, s);
        ops::kv_cache_append(kn, v, positions, mtp_kv_.layer_view(0), s);

        if (final_chunk) {
            const std::size_t column_bytes =
                static_cast<std::size_t>(dimension(config_.hidden_size)) * dtype_size(DType::BF16);
            const auto* x_src = static_cast<const unsigned char*>(x.data) +
                                static_cast<std::size_t>(T - 1) * column_bytes;
            const auto* ah_src = static_cast<const unsigned char*>(ah.data) +
                                 static_cast<std::size_t>(T - 1) * column_bytes;
            CUDA_CHECK(
                cudaMemcpyAsync(x_last.data, x_src, column_bytes, cudaMemcpyDeviceToDevice, s));
            CUDA_CHECK(
                cudaMemcpyAsync(ah_last.data, ah_src, column_bytes, cudaMemcpyDeviceToDevice, s));
        }
    }

    if (final_chunk) {
        Tensor q_flat = work_.alloc(DType::BF16, {dimension(config_.attention->query_width()), 1});
        Tensor gate_flat =
            work_.alloc(DType::BF16, {dimension(config_.attention->query_width()), 1});
        mtp_query_gate_projection(ah_last, mtp_->projection, *config_.attention, q_flat, gate_flat,
                                  work_, s);
        Tensor q    = q_flat.view({dimension(config_.attention->head_dim),
                                   dimension(config_.attention->num_attention_heads), 1});
        Tensor gate = gate_flat.view({dimension(config_.attention->head_dim),
                                      dimension(config_.attention->num_attention_heads), 1});
        Tensor qn =
            work_.alloc(DType::BF16, {dimension(config_.attention->head_dim),
                                      dimension(config_.attention->num_attention_heads), 1});
        ops::rmsnorm(q, mtp_->query_norm, config_.rms_norm_eps, true, qn, s);
        Tensor last_position = positions.slice(0, T - 1, 1);
        Tensor last_rope_position;
        if (rope_positions.ne[1] == 1) {
            last_rope_position = rope_positions.slice(0, T - 1, 1);
        } else {
            last_rope_position = work_.alloc(DType::I32, {1, 3});
            for (int axis = 0; axis < 3; ++axis) {
                const auto* src = static_cast<const std::int32_t*>(rope_positions.data) +
                                  static_cast<std::size_t>(axis) * T + (T - 1);
                auto* dst = static_cast<std::int32_t*>(last_rope_position.data) + axis;
                CUDA_CHECK(
                    cudaMemcpyAsync(dst, src, sizeof(std::int32_t), cudaMemcpyDeviceToDevice, s));
            }
        }
        text_rope(last_rope_position, *config_.rope_parameters, qn, s);

        Tensor a = work_.alloc(DType::BF16, {dimension(config_.attention->head_dim),
                                             dimension(config_.attention->num_attention_heads), 1});
        ops::causal_softmax_attention_cached(
            qn, last_position,
            {dimension(config_.attention->head_dim),
             dimension(config_.attention->num_attention_heads),
             dimension(config_.attention->num_key_value_heads)},
            static_cast<float>(1.0 / std::sqrt(static_cast<double>(config_.attention->head_dim))),
            mtp_kv_.layer_view(0), envelope, work_, a, s);
        ops::sigmoid_mul(gate, a, s);

        Tensor o = work_.alloc(DType::BF16, {dimension(config_.hidden_size), 1});
        project(a.view({dimension(config_.attention->query_width()), 1}), mtp_->output, o, work_,
                s);
        ops::residual_add(o, x_last, s);

        Tensor mh = work_.alloc(DType::BF16, {dimension(config_.hidden_size), 1});
        ops::rmsnorm(x_last, mtp_->post_attention_norm, config_.rms_norm_eps, true, mh, s);
        {
            auto post_mixer_scope = work_.scope();
            ffn(mh, mtp_->ffn, x_last, {}, work_, s, true);
        }
        ops::rmsnorm(x_last, mtp_->final_norm, config_.rms_norm_eps, true, *final_hidden, s);
        proposal_argmax(*final_hidden, *logits, *draft_token);
    }
}

void TextContext::proposal_argmax(const Tensor& hidden, Tensor& logits, Tensor& proposal_tokens) {
    auto proposal_scope = work_.scope();
    const int T         = hidden.ne[1];
    require_tensor_shape(hidden, DType::BF16, {dimension(config_.hidden_size), T},
                         "proposal hidden");
    require_tensor_shape(proposal_tokens, DType::I32, {T}, "proposal tokens");
    require_tensor_window(logits, DType::BF16, dimension(config_.vocab_size), T, "proposal logits");
    nvtx::ScopedRange proposal_range(nvtx::Name::MtpProposal, nvtx::Category::Mtp,
                                     static_cast<std::uint64_t>(T));
    if (proposal_head_ != nullptr) {
        const std::int32_t rows = dimension(proposal_head_->weight.n);
        if (rows == proposal_head_n_) {
            Tensor proposal_logits = work_.alloc(DType::BF16, {proposal_head_n_, T});
            project(hidden, *proposal_head_, proposal_logits, work_, ctx_.stream);
            ops::argmax(proposal_logits, proposal_tokens,
                        proposal_head_ids_
                            ? proposal_head_n_
                            : dimension(parameters_.model.resources().public_token_count),
                        ctx_.stream);
            if (proposal_head_ids_ != nullptr) {
                ops::proposal_remap_token_ids(proposal_tokens, proposal_head_ids_,
                                              proposal_head_n_, ctx_.stream);
            }
            return;
        }
        // Vocabulary-split draft head: each shard projects its own row block of the reduced
        // vocabulary into the draft logits, and the pair merges them back to [rows*2, T] on both
        // shards. Only this shard samples, and only this shard runs the MTP layer, so it hands the
        // peer the pre-head hidden state it needs.
        if (rows <= 0 || rows * 2 != proposal_head_n_ || peer_tp_ == nullptr || pair_tp_ == nullptr ||
            peer_tp_->proposal_head_ == nullptr ||
            dimension(peer_tp_->proposal_head_->weight.n) != rows) {
            throw std::logic_error(
                "proposal_argmax: a split proposal head needs half the rows on a peer that holds "
                "the matching half");
        }
        TextContext& peer                  = *peer_tp_;
        tp::DevicePair& pair               = *pair_tp_;
        const std::int32_t hidden_size = dimension(config_.hidden_size);
        ctx_.bind_to_current_thread();
        Tensor merged = work_.alloc(DType::BF16, {proposal_head_n_, T});
        Tensor hidden_local = work_.alloc(DType::BF16, {hidden_size, T});
        CUDA_CHECK(cudaMemcpyAsync(hidden_local.data, hidden.data, hidden_local.bytes(),
                                   cudaMemcpyDeviceToDevice, ctx_.stream));
        peer.ctx_.bind_to_current_thread();
        Tensor hidden_peer = peer.work_.alloc(DType::BF16, {hidden_size, T});
        CUDA_CHECK(cudaMemsetAsync(hidden_peer.data, 0, hidden_peer.bytes(), peer.ctx_.stream));
        ctx_.bind_to_current_thread();
        pair.allreduce(hidden_local.data, hidden_peer.data, hidden_local.bytes(), ctx_.stream,
                       peer.ctx_.stream);

        Tensor part = work_.alloc(DType::BF16, {rows, T});
        project(hidden_local, *proposal_head_, part, work_, ctx_.stream);
        peer.ctx_.bind_to_current_thread();
        Tensor part_peer = peer.work_.alloc(DType::BF16, {rows, T});
        project(hidden_peer, *peer.proposal_head_, part_peer, peer.work_, peer.ctx_.stream);
        Tensor merged_peer = peer.work_.alloc(DType::BF16, {proposal_head_n_, T});
        merge_local_row_blocks(peer, pair, part, part_peer, merged, merged_peer, rows);

        ctx_.bind_to_current_thread();
        ops::argmax(merged, proposal_tokens,
                    proposal_head_ids_
                        ? proposal_head_n_
                        : dimension(parameters_.model.resources().public_token_count),
                    ctx_.stream);
        if (proposal_head_ids_ != nullptr) {
            ops::proposal_remap_token_ids(proposal_tokens, proposal_head_ids_, proposal_head_n_,
                                          ctx_.stream);
        }
    } else {
        Tensor output_logits = matrix_window(logits, T);
        project(hidden, mtp_->output_head, output_logits, work_, ctx_.stream);
        ops::argmax(output_logits, proposal_tokens,
                    dimension(parameters_.model.resources().public_token_count), ctx_.stream);
    }
}

void TextContext::mtp_forward_batch(const Tensor& ids, const Tensor& hidden,
                                    const Tensor& positions,
                                    ops::CausalAttentionExecutionEnvelope envelope,
                                    Tensor& mtp_hidden, int logits_column, Tensor* logits,
                                    Tensor* draft_token, const Tensor* explicit_rope_positions,
                                    const Tensor* input_embeddings) {
    if (batch_mtp_kv_ == nullptr) { throw std::runtime_error("MTP forward is not enabled"); }
    const int T = ids.ne[0];
    if (T <= 0 || static_cast<std::uint32_t>(T) > prefill_chunk_) {
        throw std::invalid_argument("MTP batch T must be in [1,prefill_chunk]");
    }
    require_tensor_shape(ids, DType::I32, {T}, "MTP ids");
    require_tensor_shape(positions, DType::I32, {T}, "MTP positions");
    require_tensor_shape(hidden, DType::BF16, {dimension(config_.hidden_size), T}, "MTP hidden");
    require_tensor_shape(mtp_hidden, DType::BF16, {dimension(config_.hidden_size), T},
                         "MTP output hidden");
    if (logits_column >= T) { throw std::invalid_argument("MTP logits column out of range"); }
    if (logits_column >= 0) {
        if (logits == nullptr || draft_token == nullptr) {
            throw std::invalid_argument("MTP logits and draft_token outputs are required");
        }
        require_tensor_shape(*logits, DType::BF16, {dimension(config_.vocab_size), 1},
                             "MTP logits");
        require_tensor_shape(*draft_token, DType::I32, {1}, "MTP draft token");
    }

    auto position_scope = work_.scope();
    Tensor generated_rope_positions;
    const Tensor* rope_positions = explicit_rope_positions;
    if (rope_positions == nullptr) {
        generated_rope_positions = work_.alloc(DType::I32, {T});
        ops::offset_i32_positions(positions, io_.rope_delta, generated_rope_positions, ctx_.stream);
        rope_positions = &generated_rope_positions;
    } else if (rope_positions->dtype != DType::I32 || rope_positions->ne[0] != T ||
               (rope_positions->ne[1] != 1 && rope_positions->ne[1] != 3) ||
               rope_positions->ne[2] != 1 || rope_positions->ne[3] != 1 ||
               !rope_positions->is_contiguous() || rope_positions->data == nullptr) {
        throw std::invalid_argument("MTP explicit rope positions must be [T] or [T,3]");
    }
    mtp_forward_core(ids, hidden, positions, *rope_positions, envelope, mtp_hidden,
                     input_embeddings);

    if (logits_column >= 0) {
        auto logits_scope = work_.scope();
        Tensor col        = mtp_hidden.slice(1, logits_column, 1);
        proposal_argmax(col, *logits, *draft_token);
    }
}

void TextContext::mtp_forward_ar_step(const Tensor& token, const Tensor& previous_hidden,
                                      const Tensor& position,
                                      ops::CausalAttentionExecutionEnvelope envelope,
                                      Tensor& mtp_hidden, Tensor& logits, Tensor& draft_token) {
    if (batch_mtp_kv_ == nullptr) { throw std::runtime_error("MTP forward is not enabled"); }
    require_tensor_shape(token, DType::I32, {1}, "MTP AR token");
    require_tensor_shape(position, DType::I32, {1}, "MTP AR position");
    require_tensor_shape(previous_hidden, DType::BF16, {dimension(config_.hidden_size), 1},
                         "MTP AR previous hidden");
    require_tensor_shape(mtp_hidden, DType::BF16, {dimension(config_.hidden_size), 1},
                         "MTP AR output hidden");
    require_tensor_shape(logits, DType::BF16, {dimension(config_.vocab_size), 1}, "MTP AR logits");
    require_tensor_shape(draft_token, DType::I32, {1}, "MTP AR draft token");

    auto position_scope  = work_.scope();
    Tensor rope_position = work_.alloc(DType::I32, {1});
    ops::offset_i32_positions(position, io_.rope_delta, rope_position, ctx_.stream);
    mtp_forward_core(token, previous_hidden, position, rope_position, envelope, mtp_hidden,
                     nullptr);
    auto logits_scope = work_.scope();
    proposal_argmax(mtp_hidden, logits, draft_token);
}

void TextContext::ordinary_decode_batch(const Tensor& ids, const Tensor& cache_positions,
                                        const Tensor& rope_positions, const Tensor& kv_table_rows,
                                        const Tensor& linear_state_source_slots,
                                        const Tensor& linear_state_destination_slots,
                                        ops::CausalAttentionExecutionEnvelope envelope,
                                        Tensor& hidden, Tensor& logits) {
    const std::int32_t batch = ids.ne[0];
    if (batch <= 0 || batch > static_cast<std::int32_t>(kMaximumConcurrency)) {
        throw std::invalid_argument("ordinary decode batch size must be in [1,8]");
    }
    require_tensor_shape(ids, DType::I32, {batch}, "ordinary decode ids");
    require_tensor_shape(cache_positions, DType::I32, {batch}, "ordinary decode cache positions");
    require_tensor_shape(rope_positions, DType::I32, {batch}, "ordinary decode RoPE positions");
    require_tensor_shape(kv_table_rows, DType::I32, {batch}, "ordinary decode KV rows");
    require_tensor_shape(linear_state_source_slots, DType::I32, {batch},
                         "ordinary decode Linear Attention source slots");
    require_tensor_shape(linear_state_destination_slots, DType::I32, {batch},
                         "ordinary decode Linear Attention destination slots");
    require_tensor_shape(hidden, DType::BF16, {dimension(config_.hidden_size), batch},
                         "ordinary decode hidden");
    require_tensor_shape(logits, DType::BF16, {dimension(config_.vocab_size), batch},
                         "ordinary decode logits");

    cudaStream_t stream = ctx_.stream;
    work_.reset();
    {
        ScopedPositions cache_binding(active_cache_positions_, cache_positions);
        ScopedPositions rope_binding(active_rope_positions_, rope_positions);
        ScopedEnvelope envelope_binding(active_causal_attention_envelope_, envelope);
        ScopedValue<const Tensor*> kv_binding(active_kv_table_rows_, &kv_table_rows);
        ScopedValue<const Tensor*> source_binding(active_linear_state_source_slots_,
                                                  &linear_state_source_slots);
        ScopedValue<const Tensor*> destination_binding(active_linear_state_destination_slots_,
                                                       &linear_state_destination_slots);
        ScopedValue<std::int32_t> batch_binding(active_sequence_batch_, batch);
        ScopedValue<std::int32_t> width_binding(active_sequence_width_, 1);

        Tensor x = work_.alloc(DType::BF16, {dimension(config_.hidden_size), batch});
        ops::embedding(ids, *embed_, x, stream);
        NullTap tap;
        run_layers(x, Phase::Verify, tap);
        ops::rmsnorm(x, *final_norm_, config_.rms_norm_eps, true, hidden, stream);
        project(hidden, *lm_head_, logits, work_, stream);
    }
    work_.reset();
}

template <class Tap>
void TextContext::target_verify_batch_impl(const Tensor& ids, const Tensor& cache_positions,
                                           const Tensor& rope_positions,
                                           const Tensor& valid_columns, const Tensor& kv_table_rows,
                                           const Tensor& linear_state_source_slots,
                                           ops::CausalAttentionExecutionEnvelope envelope,
                                           Tensor& hidden, Tensor& logits, Tensor& target_tokens,
                                           Tap& tap) {
    const std::int32_t width = ids.ne[0];
    const std::int32_t batch = ids.ne[1];
    if (width <= 0 || width > static_cast<std::int32_t>(kDFlashDecodeMaximumWidth) || batch <= 0 ||
        batch > static_cast<std::int32_t>(kMaximumConcurrency)) {
        throw std::invalid_argument("target verify batch shape is outside the supported domain");
    }
    const std::int32_t columns = width * batch;
    require_tensor_shape(ids, DType::I32, {width, batch}, "target verify batch ids");
    require_tensor_shape(cache_positions, DType::I32, {width, batch},
                         "target verify batch cache positions");
    require_tensor_shape(rope_positions, DType::I32, {width, batch},
                         "target verify batch RoPE positions");
    require_tensor_shape(valid_columns, DType::I32, {batch}, "target verify batch valid columns");
    require_tensor_shape(kv_table_rows, DType::I32, {batch}, "target verify batch KV rows");
    require_tensor_shape(linear_state_source_slots, DType::I32, {batch},
                         "target verify batch Linear Attention slots");
    require_tensor_shape(hidden, DType::BF16, {dimension(config_.hidden_size), width, batch},
                         "target verify batch hidden");
    require_tensor_shape(logits, DType::BF16, {dimension(config_.vocab_size), width, batch},
                         "target verify batch logits");
    require_tensor_shape(target_tokens, DType::I32, {width, batch}, "target verify batch tokens");

    cudaStream_t stream = ctx_.stream;
    work_.reset();
    {
        ScopedPositions cache_binding(active_cache_positions_, cache_positions);
        ScopedPositions rope_binding(active_rope_positions_, rope_positions);
        ScopedEnvelope envelope_binding(active_causal_attention_envelope_, envelope);
        ScopedValue<const Tensor*> kv_binding(active_kv_table_rows_, &kv_table_rows);
        ScopedValue<const Tensor*> state_binding(active_linear_state_source_slots_,
                                                 &linear_state_source_slots);
        ScopedValue<const Tensor*> valid_binding(active_valid_columns_, &valid_columns);
        ScopedValue<std::int32_t> batch_binding(active_sequence_batch_, batch);
        ScopedValue<std::int32_t> width_binding(active_sequence_width_, width);

        Tensor x        = work_.alloc(DType::BF16, {dimension(config_.hidden_size), columns});
        Tensor flat_ids = ids.view({columns});
        ops::embedding(flat_ids, *embed_, x, stream);
        if constexpr (Tap::enabled) { tap.begin(x); }
        run_layers(x, Phase::Verify, tap);
        if constexpr (requires { tap.capture_positions(cache_positions, stream); }) {
            tap.capture_positions(cache_positions, stream);
        }
        Tensor flat_hidden = hidden.view({dimension(config_.hidden_size), columns});
        Tensor flat_logits = logits.view({dimension(config_.vocab_size), columns});
        Tensor flat_tokens = target_tokens.view({columns});
        ops::rmsnorm(x, *final_norm_, config_.rms_norm_eps, true, flat_hidden, stream);
        project(flat_hidden, *lm_head_, flat_logits, work_, stream);
        ops::argmax(flat_logits, flat_tokens,
                    dimension(parameters_.model.resources().public_token_count), stream);
    }
    work_.reset();
}

void TextContext::target_verify_batch(const Tensor& ids, const Tensor& cache_positions,
                                      const Tensor& rope_positions, const Tensor& valid_columns,
                                      const Tensor& kv_table_rows,
                                      const Tensor& linear_state_source_slots,
                                      ops::CausalAttentionExecutionEnvelope envelope,
                                      Tensor& hidden, Tensor& logits, Tensor& target_tokens) {
    NullTap tap;
    target_verify_batch_impl(ids, cache_positions, rope_positions, valid_columns, kv_table_rows,
                             linear_state_source_slots, envelope, hidden, logits, target_tokens,
                             tap);
}

void TextContext::target_verify_batch(const Tensor& ids, const Tensor& cache_positions,
                                      const Tensor& rope_positions, const Tensor& valid_columns,
                                      const Tensor& kv_table_rows,
                                      const Tensor& linear_state_source_slots,
                                      ops::CausalAttentionExecutionEnvelope envelope,
                                      Tensor& hidden, Tensor& logits, Tensor& target_tokens,
                                      DFlashFeatureSink& sink) {
    target_verify_batch_impl(ids, cache_positions, rope_positions, valid_columns, kv_table_rows,
                             linear_state_source_slots, envelope, hidden, logits, target_tokens,
                             sink);
}

void TextContext::mtp_forward_decode_batch(const Tensor& ids, const Tensor& hidden,
                                           const Tensor& cache_positions,
                                           const Tensor& rope_positions,
                                           const Tensor& valid_columns, const Tensor& kv_table_rows,
                                           ops::CausalAttentionExecutionEnvelope envelope,
                                           Tensor& mtp_hidden) {
    if (batch_mtp_kv_ == nullptr) { throw std::runtime_error("MTP forward is not enabled"); }
    const std::int32_t width = ids.ne[0];
    const std::int32_t batch = ids.ne[1];
    if (width <= 0 || width > static_cast<std::int32_t>(kMaximumMtpDraftTokens + 1) || batch <= 0 ||
        batch > static_cast<std::int32_t>(kMaximumConcurrency)) {
        throw std::invalid_argument("MTP decode batch shape is outside the supported domain");
    }
    require_tensor_shape(ids, DType::I32, {width, batch}, "MTP decode batch ids");
    require_tensor_shape(hidden, DType::BF16, {dimension(config_.hidden_size), width, batch},
                         "MTP decode batch target hidden");
    require_tensor_shape(cache_positions, DType::I32, {width, batch},
                         "MTP decode batch cache positions");
    require_tensor_shape(rope_positions, DType::I32, {width, batch},
                         "MTP decode batch RoPE positions");
    require_tensor_shape(valid_columns, DType::I32, {batch}, "MTP decode batch valid columns");
    require_tensor_shape(kv_table_rows, DType::I32, {batch}, "MTP decode batch KV rows");
    require_tensor_shape(mtp_hidden, DType::BF16, {dimension(config_.hidden_size), width, batch},
                         "MTP decode batch hidden");

    ScopedValue<const Tensor*> backend_binding(active_backend_kv_table_rows_, &kv_table_rows);
    ScopedValue<const Tensor*> valid_binding(active_valid_columns_, &valid_columns);
    ScopedValue<std::int32_t> batch_binding(active_sequence_batch_, batch);
    ScopedValue<std::int32_t> width_binding(active_sequence_width_, width);
    mtp_forward_core(ids, hidden, cache_positions, rope_positions, envelope, mtp_hidden, nullptr);
}

void TextContext::mtp_propose_batch(const Tensor& hidden, Tensor& logits, Tensor& draft_tokens) {
    const std::int32_t batch = hidden.ne[1];
    require_tensor_shape(hidden, DType::BF16, {dimension(config_.hidden_size), batch},
                         "MTP proposal batch hidden");
    require_tensor_shape(logits, DType::BF16, {dimension(config_.vocab_size), batch},
                         "MTP proposal batch logits");
    require_tensor_shape(draft_tokens, DType::I32, {batch}, "MTP proposal batch tokens");
    proposal_argmax(hidden, logits, draft_tokens);
}

void TextContext::attn_mix(const BlockParameters& w, Tensor& x, int fidx, Phase ph,
                           Tensor* delta) {
    const auto& p   = std::get<AttentionParameters>(w.mixer);
    const auto& cfg = shard_config();
    cudaStream_t s  = ctx_.stream;
    const int T    = x.ne[1];
    if (active_causal_attention_envelope_ == nullptr) {
        throw std::logic_error("Text GQA execution envelope is not set");
    }

    auto projection = workspace::text_attention_projection(work_, cfg, T);
    Tensor h        = projection.hidden;
    ops::rmsnorm(x, w.input_norm, cfg.rms_norm_eps, true, h, s);

    Tensor q         = projection.query.view({dimension(cfg.attention->head_dim),
                                              dimension(cfg.attention->num_attention_heads), T});
    Tensor gate      = projection.gate.view({dimension(cfg.attention->head_dim),
                                             dimension(cfg.attention->num_attention_heads), T});
    Tensor k         = projection.key.view({dimension(cfg.attention->head_dim),
                                            dimension(cfg.attention->num_key_value_heads), T});
    Tensor v         = projection.value.view({dimension(cfg.attention->head_dim),
                                              dimension(cfg.attention->num_key_value_heads), T});
    // The fused QKV+gate projection is a single GEMM whose output rows are laid out
    // [q | k | gate | v]. The per-shard fused weight (q/gate rows halved, k/v rows unchanged)
    // is run through the generic linear op, then the output is viewed into q/k/gate/v. This
    // mirrors the FFN-delta decomposition and avoids the shape-specific fused attn GEMM kernels
    // (which are registered for the full-model row counts only).
    const std::int32_t q_rows  = dimension(cfg.attention->query_width());
    const std::int32_t k_rows  = dimension(cfg.attention->key_width());
    const std::int32_t fused_n = 2 * q_rows + 2 * k_rows;
    Tensor fused = work_.alloc(DType::BF16, {fused_n, T});
    const auto& proj_w = std::get<LinearParameters>(p.projection);
    ops::linear(h, proj_w.weight, fused, proj_w.policy, work_, s);
    if (T > 1) {
        // Batched width: the four blocks are strided row windows of the fused output, so they are
        // materialized into the (already sized) workspace projection buffers.
        copy_row_block(fused, 0, q_rows, projection.query, s);
        copy_row_block(fused, q_rows, k_rows, projection.key, s);
        copy_row_block(fused, q_rows + k_rows, q_rows, projection.gate, s);
        copy_row_block(fused, 2 * q_rows + k_rows, k_rows, projection.value, s);
    } else {
        // Single column: the slices are contiguous, so q/k/gate/v alias the fused buffer directly.
        q    = fused.slice(0, 0, q_rows).view({dimension(cfg.attention->head_dim),
                                               dimension(cfg.attention->num_attention_heads), T});
        k    = fused.slice(0, q_rows, k_rows).view({dimension(cfg.attention->head_dim),
                                                    dimension(cfg.attention->num_key_value_heads), T});
        gate = fused.slice(0, q_rows + k_rows, q_rows)
                   .view({dimension(cfg.attention->head_dim),
                          dimension(cfg.attention->num_attention_heads), T});
        v    = fused.slice(0, 2 * q_rows + k_rows, k_rows)
                   .view({dimension(cfg.attention->head_dim),
                          dimension(cfg.attention->num_key_value_heads), T});
    }

    const auto results = workspace::text_attention_results(work_, cfg, T);
    Tensor qn =
        results.normalized_query.view({dimension(cfg.attention->head_dim),
                                       dimension(cfg.attention->num_attention_heads), T});
    Tensor kn = results.normalized_key.view({dimension(cfg.attention->head_dim),
                                             dimension(cfg.attention->num_key_value_heads), T});
    ops::rmsnorm(q, p.query_norm, cfg.rms_norm_eps, true, qn, s);
    ops::rmsnorm(k, p.key_norm, cfg.rms_norm_eps, true, kn, s);
    const Tensor& cache_positions =
        active_cache_positions_ != nullptr ? *active_cache_positions_ : io_.pos;
    const Tensor& rope_positions =
        active_rope_positions_ != nullptr ? *active_rope_positions_ : io_.rope_pos;
    Tensor rope_for_op = active_sequence_batch_ != 0 ? rope_positions.view({T}) : rope_positions;
    text_rope(rope_for_op, *cfg.rope_parameters, qn, kn, s);

    Tensor a = results.attention.view({dimension(cfg.attention->head_dim),
                                       dimension(cfg.attention->num_attention_heads), T});
    const Tensor& kv_table_rows =
        active_kv_table_rows_ != nullptr ? *active_kv_table_rows_ : io_.text_kv_table_row;
    if (active_sequence_batch_ != 0) {
        const std::int32_t width = active_sequence_width_;
        if (width <= 0 || width * active_sequence_batch_ != T) {
            throw std::logic_error("Text sequence batch binding does not match aggregate columns");
        }
        Tensor q_batch        = qn.view({dimension(cfg.attention->head_dim),
                                         dimension(cfg.attention->num_attention_heads), width,
                                         active_sequence_batch_});
        Tensor k_batch        = kn.view({dimension(cfg.attention->head_dim),
                                         dimension(cfg.attention->num_key_value_heads), width,
                                         active_sequence_batch_});
        Tensor v_batch        = v.view({dimension(cfg.attention->head_dim),
                                        dimension(cfg.attention->num_key_value_heads), width,
                                        active_sequence_batch_});
        Tensor a_batch        = a.view({dimension(cfg.attention->head_dim),
                                        dimension(cfg.attention->num_attention_heads), width,
                                        active_sequence_batch_});
        Tensor position_batch = cache_positions.view({width, active_sequence_batch_});
        const Tensor valid = active_valid_columns_ != nullptr ? *active_valid_columns_ : Tensor{};
        ops::causal_softmax_attention(
            q_batch, k_batch, v_batch, position_batch, valid, kv_table_rows,
            {dimension(cfg.attention->head_dim),
             dimension(cfg.attention->num_attention_heads),
             dimension(cfg.attention->num_key_value_heads)},
            static_cast<float>(1.0 / std::sqrt(static_cast<double>(cfg.attention->head_dim))),
            batch_text_kv_->batch_layer_view(fidx), *active_causal_attention_envelope_, work_,
            a_batch, s);
    } else {
        ops::causal_softmax_attention(
            qn, kn, v, cache_positions, Tensor{}, kv_table_rows,
            {dimension(cfg.attention->head_dim),
             dimension(cfg.attention->num_attention_heads),
             dimension(cfg.attention->num_key_value_heads)},
            static_cast<float>(1.0 / std::sqrt(static_cast<double>(cfg.attention->head_dim))),
            batch_text_kv_->batch_layer_view(fidx), *active_causal_attention_envelope_, work_, a,
            s);
    }
    ops::sigmoid_mul(gate, a, s);

    const Tensor a_flat = a.view({dimension(cfg.attention->query_width()), T});
    if (delta == nullptr) {
        ops::linear_add(a_flat, p.output.weight, x, p.output.policy, work_, s);
    } else {
        ops::linear(a_flat, p.output.weight, *delta, p.output.policy, work_, s);
    }
}

void TextContext::gdn_mix(const BlockParameters& w, Tensor& x, int gidx, Phase ph,
                          Tensor* delta) {
    const auto& p   = std::get<GdnParameters>(w.mixer);
    const auto& cfg = shard_config();
    cudaStream_t s  = ctx_.stream;
    const int T    = x.ne[1];

    const auto control = workspace::gdn_control(work_, cfg, T);
    Tensor h           = control.hidden;
    // The GDN gating projections (a/b) run in full on both shards: a_log/dt_bias are replicated
    // at the full 48 value heads, so g/beta must be allocated at the full head count (the
    // gating op hardcodes 48). The delta net below slices them to this shard's local heads.
    const std::int32_t full_value_heads = dimension(config_.gdn->linear_num_value_heads);
    Tensor g      = work_.alloc(DType::FP32, {full_value_heads, T});
    Tensor beta   = work_.alloc(DType::FP32, {full_value_heads, T});
    gdn_norm_control(x, w.input_norm, cfg.rms_norm_eps, p, h, g, beta, work_,
                     ctx_.execution_view());

    // The GDN projection is phase-dependent:
    // - Batched Verify (speculative decoding): fused shape-specific ops (gdn_projection_snapshot/
    //   record), registered for the full-model row counts. Uses workspace-allocated qc/kc/vc/z.
    // - Prefill (T>1, single-GPU): fused shape-specific op (gdn_projection), registered for the
    //   full-model row counts. Uses workspace-allocated qc/kc/vc/z + separate qkv buffer.
    // - Everything else, i.e. single-token decode (T=1) and every head-split shard width:
    //   decomposed GEMM (generic ops::linear on the shard's fused weight) + conv. The fused GEMM
    //   output [q|k|v|z] is sliced into the conv input (qkv) and the output gate (z), and qc/kc/vc
    //   are separate workspace buffers the conv writes, so the conv input never overlaps its
    //   outputs.
    Tensor qc, kc, vc, z;
    // A head-split shard has no fused record workspace for the width>1 op, so it takes the same
    // decomposed route as its prefill; the phase still selects the decode-equivalent kernels.
    const bool batched_verify =
        (ph == Phase::Verify) && (active_sequence_batch_ > 1 || active_sequence_width_ > 1) &&
        shard_config_ == nullptr;
    // Replay recording. The full model records through the fused width>1 ops above. A head-split
    // shard cannot: those ops are registered for the full 16384-row fused parent only, which is
    // why this route used to skip recording entirely. It records through the recurrent record op
    // instead, which is registered for the shard geometry (8 key heads, 24 value heads) and is the
    // geometry the fold plan is built for.
    const bool recording = gdn_state_action_ == GdnStateAction::RecordForReplay;
    if (recording && replay_records_ == nullptr) {
        throw std::logic_error("Replay-record GDN has no record storage");
    }
    if (recording && T < 2) {
        throw std::logic_error("Replay-record GDN requires at least two window columns");
    }
    const std::int32_t record_rows = active_sequence_batch_ > 0 ? active_sequence_batch_ : 1;
    GdnReplayRecordLayer records{};
    if (recording && shard_config_ != nullptr) {
        if (record_rows != 1) {
            throw std::logic_error("Replay-record GDN on a shard requires one record row");
        }
        records = replay_records_->layer(gidx, record_rows);
    }
    if (batched_verify) {
        if (active_sequence_batch_ == 0 || active_linear_state_source_slots_ == nullptr) {
            throw std::logic_error(
                "Verify GDN requires an explicit sequence batch and state slots");
        }
        const std::int32_t width = active_sequence_width_;
        if (width <= 0 || width * active_sequence_batch_ != T) {
            throw std::logic_error("GDN sequence batch binding does not match aggregate columns");
        }
        if (gdn_state_action_ == GdnStateAction::UpdateInPlace && width != 1) {
            throw std::logic_error("In-place batched GDN update requires width one");
        }
        const auto projection = workspace::gdn_projection(work_, cfg, T);
        z  = projection.output_gate.view({dimension(cfg.gdn->linear_value_head_dim),
                                          dimension(cfg.gdn->linear_num_value_heads), T});
        qc = projection.query;
        kc = projection.key;
        vc = projection.value;
        Tensor projection_input =
            h.view({dimension(cfg.hidden_size), width, active_sequence_batch_});
        Tensor query_output =
            qc.view({dimension(cfg.gdn->key_width()), width, active_sequence_batch_});
        Tensor key_output =
            kc.view({dimension(cfg.gdn->key_width()), width, active_sequence_batch_});
        Tensor value_output =
            vc.view({dimension(cfg.gdn->value_width()), width, active_sequence_batch_});
        Tensor gate_output =
            z.view({dimension(cfg.gdn->value_width()), width, active_sequence_batch_});
        Tensor conv_states = state_.layer_view(static_cast<std::uint32_t>(gidx)).conv;
        const Tensor valid = active_valid_columns_ != nullptr ? *active_valid_columns_ : Tensor{};
        if (gdn_state_action_ == GdnStateAction::RecordForReplay) {
            if (replay_records_ == nullptr) {
                throw std::logic_error("Replay-record GDN has no record storage");
            }
            GdnReplayRecordLayer records = replay_records_->layer(gidx, active_sequence_batch_);
            gdn_projection_record(projection_input, p, *cfg.gdn, conv_states, valid,
                                  *active_linear_state_source_slots_, records.conv, query_output,
                                  key_output, value_output, gate_output, work_, s);
        } else {
            gdn_projection_snapshot(projection_input, p, *cfg.gdn, conv_states, valid,
                                    *active_linear_state_source_slots_,
                                    *active_linear_state_destination_slots_, query_output,
                                    key_output, value_output, gate_output, work_, s);
        }
    } else if (T > 1 && shard_config_ == nullptr) {
        // Prefill (single-GPU, T>1): fused shape-specific GEMM + conv (original behavior). The
        // fused op is registered for the full-model fused row count only, so a head-split shard
        // (half the fused rows) takes the decomposed route below instead.
        const auto projection = workspace::gdn_projection(work_, cfg, T);
        z  = projection.output_gate.view({dimension(cfg.gdn->linear_value_head_dim),
                                          dimension(cfg.gdn->linear_num_value_heads), T});
        qc = projection.query;
        kc = projection.key;
        vc = projection.value;
        Tensor qkv    = workspace::gdn_prefill_conv(work_, cfg, T);
        Tensor z_flat = z.view({dimension(cfg.gdn->value_width()), T});
        gdn_projection(h, p, qkv, z_flat, work_, s);
        Tensor conv_state_in =
            state_.conv_slot(static_cast<std::uint32_t>(gidx), linear_state_source_slot_);
        Tensor conv_state_out =
            state_.conv_slot(static_cast<std::uint32_t>(gidx), linear_state_destination_slot_);
        ops::causal_conv1d_silu_split(qkv, p.convolution, conv_state_in, conv_state_out, qc, kc, vc,
                                      s);
    } else {
        // Single-token decode (T=1): decomposed GEMM (generic ops::linear on the per-shard
        // fused weight) + conv. The GEMM writes [q|k|v|z] into a single fused buffer; q/k/v are
        // sliced from it as the conv input, z is sliced as the output gate, and qc/kc/vc are
        // separate workspace buffers (conv outputs) so the conv input does not overlap them.
        const std::int32_t gdn_q_rows = dimension(cfg.gdn->key_width());
        const std::int32_t gdn_v_rows = dimension(cfg.gdn->value_width());
        const std::int32_t fused_n    = 2 * gdn_q_rows + 2 * gdn_v_rows;
        Tensor fused = work_.alloc(DType::BF16, {fused_n, T});
        const auto& proj_w = std::get<LinearParameters>(p.projection);
        ops::linear(h, proj_w.weight, fused, proj_w.policy, work_, s);
        auto projection = workspace::gdn_projection(work_, cfg, T);
        qc = projection.query;
        kc = projection.key;
        vc = projection.value;
        Tensor qkv;
        if (T > 1) {
            // Batched width: the [q|k|v] and [z] blocks are strided row windows of the fused
            // output, so they are materialized into contiguous buffers before the split
            // convolution (which requires a contiguous input) and the output gate.
            qkv = workspace::gdn_prefill_conv(work_, cfg, T);
            copy_row_block(fused, 0, 2 * gdn_q_rows + gdn_v_rows, qkv, s);
            copy_row_block(fused, 2 * gdn_q_rows + gdn_v_rows, gdn_v_rows, projection.output_gate,
                           s);
            z = projection.output_gate.view({dimension(cfg.gdn->linear_value_head_dim),
                                             dimension(cfg.gdn->linear_num_value_heads), T});
        } else {
            // Single column: the slices are contiguous and alias the fused buffer directly.
            qkv = fused.slice(0, 0, 2 * gdn_q_rows + gdn_v_rows);
            z = fused.slice(0, 2 * gdn_q_rows + gdn_v_rows, gdn_v_rows)
                    .view({dimension(cfg.gdn->linear_value_head_dim),
                           dimension(cfg.gdn->linear_num_value_heads), T});
        }
        Tensor conv_state_in =
            state_.conv_slot(static_cast<std::uint32_t>(gidx), linear_state_source_slot_);
        Tensor conv_state_out =
            state_.conv_slot(static_cast<std::uint32_t>(gidx), linear_state_destination_slot_);
        ops::causal_conv1d_silu_split(qkv, p.convolution, conv_state_in, conv_state_out, qc, kc, vc,
                                      s);
        if (recording && shard_config_ != nullptr) {
            // The fold rebuilds the convolution history as tail_3(old_history || record[0:committed]),
            // so the record is the raw pre-convolution window. This route already materializes that
            // buffer contiguously as [q|k|v], which is the record plane's channel order and layout.
            if (records.conv.data == nullptr || records.conv.bytes() != qkv.bytes()) {
                throw std::logic_error("GDN convolution record does not match the shard window");
            }
            CUDA_CHECK(cudaMemcpyAsync(records.conv.data, qkv.data, qkv.bytes(),
                                       cudaMemcpyDeviceToDevice, s));
        }
    }

    Tensor q_recurrent = qc.view({dimension(cfg.gdn->linear_key_head_dim),
                                  dimension(cfg.gdn->linear_num_key_heads), T});
    Tensor k_recurrent = kc.view({dimension(cfg.gdn->linear_key_head_dim),
                                  dimension(cfg.gdn->linear_num_key_heads), T});

    // The GDN gating projections run in full (48 heads, replicated a_log/dt_bias); slice
    // g/beta to this shard's local value heads (shard 0 = heads [0, H), shard 1 = [H, 2H)).
    const std::int32_t local_heads = dimension(cfg.gdn->linear_num_value_heads);
    const std::int32_t g_head0     = shard_index_ * local_heads;
    Tensor g_local    = g.slice(0, g_head0, local_heads);
    Tensor beta_local = beta.slice(0, g_head0, local_heads);
    if (T > 1) {
        // Batched width: the shard's heads are a strided row window of the full 48-head gating
        // output, and gated_delta_net requires contiguous g/beta.
        Tensor g_block    = work_.alloc(DType::FP32, {local_heads, T});
        Tensor beta_block = work_.alloc(DType::FP32, {local_heads, T});
        copy_row_block(g, g_head0, local_heads, g_block, s);
        copy_row_block(beta, g_head0, local_heads, beta_block, s);
        g_local    = g_block;
        beta_local = beta_block;
    }

    Tensor vv = vc.view({dimension(cfg.gdn->linear_value_head_dim),
                         dimension(cfg.gdn->linear_num_value_heads), T});
    Tensor o  = workspace::gdn_recurrent_output(work_, cfg, T)
                   .view({dimension(cfg.gdn->linear_value_head_dim),
                          dimension(cfg.gdn->linear_num_value_heads), T});
    // A head-split shard runs the decomposed recurrent path for the verify window too: the batched
    // verify views need the fused record workspace the shard does not have, and the recurrence is
    // sequential in both phases, so the shard's numerics do not depend on this branch.
    if (ph == Phase::Verify && shard_config_ == nullptr) {
        Tensor recurrent_states  = state_.layer_view(static_cast<std::uint32_t>(gidx)).recurrent;
        const std::int32_t width = active_sequence_width_;
        Tensor q_batch           = q_recurrent.view({dimension(cfg.gdn->linear_key_head_dim),
                                                     dimension(cfg.gdn->linear_num_key_heads), width,
                                                     active_sequence_batch_});
        Tensor k_batch           = k_recurrent.view({dimension(cfg.gdn->linear_key_head_dim),
                                                     dimension(cfg.gdn->linear_num_key_heads), width,
                                                     active_sequence_batch_});
        Tensor v_batch           = vv.view({dimension(cfg.gdn->linear_value_head_dim),
                                            dimension(cfg.gdn->linear_num_value_heads), width,
                                            active_sequence_batch_});
        Tensor g_batch =
            g_local.view({dimension(cfg.gdn->linear_num_value_heads), width, active_sequence_batch_});
        Tensor beta_batch = beta_local.view(
            {dimension(cfg.gdn->linear_num_value_heads), width, active_sequence_batch_});
        Tensor out_batch =
            o.view({dimension(cfg.gdn->linear_value_head_dim),
                    dimension(cfg.gdn->linear_num_value_heads), width, active_sequence_batch_});
        const Tensor valid = active_valid_columns_ != nullptr ? *active_valid_columns_ : Tensor{};
        if (gdn_state_action_ == GdnStateAction::RecordForReplay) {
            GdnReplayRecordLayer records = replay_records_->layer(gidx, active_sequence_batch_);
            ops::gated_delta_net_replay_record(
                q_batch, k_batch, v_batch, g_batch, beta_batch,
                static_cast<float>(
                    1.0 / std::sqrt(static_cast<double>(cfg.gdn->linear_key_head_dim))),
                recurrent_states, valid, *active_linear_state_source_slots_, records.key,
                records.value, records.gate, out_batch, s);
        } else {
            ops::gated_delta_net_batch_update(
                q_batch, k_batch, v_batch, g_batch, beta_batch,
                static_cast<float>(
                    1.0 / std::sqrt(static_cast<double>(cfg.gdn->linear_key_head_dim))),
                /*normalize_qk=*/true, recurrent_states, *active_linear_state_source_slots_,
                *active_linear_state_destination_slots_, out_batch, s);
        }
    } else if (recording && shard_config_ != nullptr) {
        // Head-split shard, replay recording. The record op reads the initial state from an absolute
        // slot and leaves every state slot untouched, so the window's advance is performed only by
        // the fold that replays the committed prefix. Its outputs are defined to be bit-identical to
        // the normalized gated_delta_net below, so the window's own columns keep the same logits.
        Tensor recurrent_states  = state_.layer_view(static_cast<std::uint32_t>(gidx)).recurrent;
        const std::int32_t width = T;
        Tensor q_batch           = q_recurrent.view({dimension(cfg.gdn->linear_key_head_dim),
                                                      dimension(cfg.gdn->linear_num_key_heads), width,
                                                      record_rows});
        Tensor k_batch           = k_recurrent.view({dimension(cfg.gdn->linear_key_head_dim),
                                                      dimension(cfg.gdn->linear_num_key_heads), width,
                                                      record_rows});
        Tensor v_batch           = vv.view({dimension(cfg.gdn->linear_value_head_dim),
                                            dimension(cfg.gdn->linear_num_value_heads), width,
                                            record_rows});
        Tensor g_batch =
            g_local.view({dimension(cfg.gdn->linear_num_value_heads), width, record_rows});
        Tensor beta_batch =
            beta_local.view({dimension(cfg.gdn->linear_num_value_heads), width, record_rows});
        Tensor out_batch = o.view({dimension(cfg.gdn->linear_value_head_dim),
                                   dimension(cfg.gdn->linear_num_value_heads), width, record_rows});
        Tensor initial_slots = work_.alloc(DType::I32, {record_rows});
        ops::set_i32_scalar(initial_slots, linear_state_source_slot_, s);
        ops::gated_delta_net_replay_record(
            q_batch, k_batch, v_batch, g_batch, beta_batch,
            static_cast<float>(1.0 / std::sqrt(static_cast<double>(cfg.gdn->linear_key_head_dim))),
            recurrent_states, Tensor{}, initial_slots, records.key, records.value, records.gate,
            out_batch, s);
    } else {
        Tensor recurrent_state_in =
            state_.recurrent_slot(static_cast<std::uint32_t>(gidx), linear_state_source_slot_);
        Tensor recurrent_state_out =
            state_.recurrent_slot(static_cast<std::uint32_t>(gidx), linear_state_destination_slot_);
        ops::gated_delta_net(
            q_recurrent, k_recurrent, vv, g_local, beta_local,
            static_cast<float>(1.0 /
                               std::sqrt(static_cast<double>(cfg.gdn->linear_key_head_dim))),
            /*normalize_qk=*/true, work_, recurrent_state_in, recurrent_state_out, o, s);
    }

    Tensor on = workspace::gdn_normalized_output(work_, cfg, T)
                    .view({dimension(cfg.gdn->linear_value_head_dim),
                           dimension(cfg.gdn->linear_num_value_heads), T});
    ops::gated_rmsnorm(o, p.norm, z, cfg.rms_norm_eps, on, s);

    const Tensor on_flat = on.view({dimension(cfg.gdn->value_width()), T});
    if (delta == nullptr) {
        ops::linear_add(on_flat, p.output.weight, x, p.output.policy, work_, s);
    } else {
        ops::linear(on_flat, p.output.weight, *delta, p.output.policy, work_, s);
    }
}

ops::SparseMoeHints TextContext::next_projection_hints(int layer) const {
    const auto next = static_cast<std::size_t>(layer) + 1;
    return next < parameters_.text.layers.size() ? parameters_.text.layers[next].projection_prefetch
                                                 : ops::SparseMoeHints{};
}

void TextContext::mlp_tail(const BlockParameters& weights, Tensor& x, Phase,
                           const ops::SparseMoeHints& hints) {
    Tensor h = workspace::post_mixer_hidden(work_, config_, x.ne[1]);
    ops::rmsnorm(x, weights.post_attention_norm, config_.rms_norm_eps, true, h, ctx_.stream);
    ffn(h, weights.ffn, x, hints, work_, ctx_.stream);
}

void TextContext::mixer_layer(const BlockParameters& block, Tensor& x, std::size_t layer,
                              Phase ph, Tensor* delta) {
    const bool prefill = ph == Phase::Prefill;
    const bool full    = config_.layer_types[layer] == MixerKind::FullAttention;
    const auto compact = dimension(config_.compact_layer_indices[layer]);
    nvtx::ScopedRange layer_range(
        full ? (prefill ? nvtx::Name::PrefillLayerFull : nvtx::Name::VerifyLayerFull)
             : (prefill ? nvtx::Name::PrefillLayerGdn : nvtx::Name::VerifyLayerGdn),
        full ? nvtx::Category::Attention : nvtx::Category::Gdn, layer);
    try {
        nvtx::ScopedRange mixer_range(
            full ? (prefill ? nvtx::Name::PrefillAttention : nvtx::Name::VerifyAttention)
                 : (prefill ? nvtx::Name::PrefillGdn : nvtx::Name::VerifyGdn),
            full ? nvtx::Category::Attention : nvtx::Category::Gdn, layer);
        auto scope = work_.scope();
        if (full) {
            attn_mix(block, x, compact, ph, delta);
        } else {
            gdn_mix(block, x, compact, ph, delta);
        }
    } catch (const std::exception& error) {
        throw std::runtime_error("text/layers/" + std::to_string(layer) +
                                 (prefill ? " prefill" : " verify") +
                                 " columns=" + std::to_string(x.ne[1]) + ": " + error.what());
    }
}

void TextContext::mlp_layer(const BlockParameters& block, Tensor& x, std::size_t layer,
                            Phase ph) {
    const bool prefill = ph == Phase::Prefill;
    try {
        nvtx::ScopedRange range(prefill ? nvtx::Name::PrefillPostMixer : nvtx::Name::VerifyPostMixer,
                                nvtx::Category::PostMixer, layer);
        auto scope = work_.scope();
        mlp_tail(block, x, ph, next_projection_hints(static_cast<int>(layer)));
    } catch (const std::exception& error) {
        throw std::runtime_error("text/layers/" + std::to_string(layer) +
                                 (prefill ? " prefill" : " verify") +
                                 " columns=" + std::to_string(x.ne[1]) + ": " + error.what());
    }
}

void TextContext::single_layer(const BlockParameters& block, Tensor& x, std::size_t layer,
                               Phase ph) {
    mixer_layer(block, x, layer, ph);
    mlp_layer(block, x, layer, ph);
}

void TextContext::tp_mixer_layer(const BlockParameters& block, Tensor& x, Tensor& delta,
                                 std::size_t layer, Phase ph) {
    mixer_layer(block, x, layer, ph, &delta);
}

void TextContext::tp_mlp_layer(const BlockParameters& block, Tensor& x, std::size_t layer,
                               Phase ph) {
    mlp_layer(block, x, layer, ph);
}

void TextContext::tp_mlp_delta(const BlockParameters& block, Tensor& x, Tensor& delta,
                               std::size_t layer, Phase ph) {
    const bool prefill = ph == Phase::Prefill;
    try {
        nvtx::ScopedRange range(prefill ? nvtx::Name::PrefillPostMixer : nvtx::Name::VerifyPostMixer,
                                nvtx::Category::PostMixer, layer);
        auto scope = work_.scope();
        Tensor h = workspace::post_mixer_hidden(work_, config_, x.ne[1]);
        ops::rmsnorm(x, block.post_attention_norm, config_.rms_norm_eps, true, h, ctx_.stream);
        ffn_delta(h, block.ffn, delta, next_projection_hints(static_cast<int>(layer)), work_,
                  ctx_.stream);
    } catch (const std::exception& error) {
        throw std::runtime_error("text/layers/" + std::to_string(layer) +
                                 (prefill ? " prefill" : " verify") +
                                 " columns=" + std::to_string(x.ne[1]) + ": " + error.what());
    }
}

template <class Tap>
void TextContext::run_layers(Tensor& x, Phase ph, Tap& tap) {
    for (std::size_t layer = 0; layer < parameters_.text.layers.size(); ++layer) {
        const auto& block = parameters_.text.layers[layer];
        single_layer(block, x, layer, ph);
        if constexpr (Tap::enabled) {
            tap.capture_layer(static_cast<int>(layer), x, ctx_.stream);
        }
    }
}

void TextContext::run_layers(Tensor& x, Phase ph) {
    NullTap tap;
    run_layers(x, ph, tap);
}

void TextContext::project_head_tp2(TextContext& peer, tp::DevicePair& pair, const Tensor& hidden,
                                   const Tensor& hidden_peer, Tensor& logits, Tensor& logits_peer) {
    const std::int32_t vocab = dimension(config_.vocab_size);
    const std::int32_t local = dimension(lm_head_->weight.n);
    if (local == vocab) {
        // Replicated head: both shards hold every row and their inputs agree exactly (replicated
        // mixers plus the all-reduced FFN delta), so each computes the full-vocabulary logits on
        // its own and no collective is needed.
        ctx_.bind_to_current_thread();
        project(hidden, *lm_head_, logits, work_, ctx_.stream);
        peer.ctx_.bind_to_current_thread();
        project(hidden_peer, *peer.lm_head_, logits_peer, peer.work_, peer.ctx_.stream);
        return;
    }
    if (local <= 0 || local * 2 != vocab || shard_index_ < 0 || peer.shard_index_ < 0 ||
        peer.shard_index_ == shard_index_) {
        throw std::logic_error(
            "project_head_tp2: a vocabulary-split output head needs exactly half the vocabulary and "
            "two distinct shards");
    }
    // Each shard projects only its own vocabulary rows into a compact block; the merge stamps them
    // into the full-logits buffers at their vocabulary offsets and sums the pair, which yields the
    // complete [V, T] logits on both shards.
    const std::int32_t columns = logits.ne[1];
    ctx_.bind_to_current_thread();
    auto scope      = work_.scope();
    Tensor partial  = work_.alloc(DType::BF16, {local, columns});
    project(hidden, *lm_head_, partial, work_, ctx_.stream);
    peer.ctx_.bind_to_current_thread();
    auto peer_scope     = peer.work_.scope();
    Tensor partial_peer = peer.work_.alloc(DType::BF16, {local, columns});
    project(hidden_peer, *peer.lm_head_, partial_peer, peer.work_, peer.ctx_.stream);
    merge_local_row_blocks(peer, pair, partial, partial_peer, logits, logits_peer, local);
}

void TextContext::forward_tp2(TextContext& peer, tp::DevicePair& pair, std::int32_t token,
                              std::int32_t position, Tensor& logits, Tensor& logits_peer,
                              Tensor* mtp_input_hidden) {
    const std::int32_t hidden = dimension(config_.hidden_size);
    const std::int32_t vocab  = dimension(config_.vocab_size);
    if (vocab % 2 != 0 || logits.ne[0] != vocab || logits.ne[1] != 1 ||
        logits_peer.ne[0] != vocab || logits_peer.ne[1] != 1) {
        throw std::invalid_argument("forward_tp2: logits buffers must be full-vocabulary [V,1]");
    }
    // No work_.reset() here: the caller owns the logits buffers in this arena, and resetting
    // would let the internal allocations overwrite them.
    // Bind the replicated-mixer state on both shards. A single token at position zero attends
    // only to itself, so the paged cache is empty and the GDN state slot is zeroed. The RAII
    // guards live until the end of the forward (a bind lambda would destroy them immediately),
    // and each shard's device is bound before any of its allocations or kernels.
    struct BindState {
        Tensor cache_positions;
        Tensor rope_positions;
        Tensor kv_table_rows;
        Tensor state_source;
        Tensor state_destination;
        ops::CausalAttentionExecutionEnvelope envelope{1, 1};
    };
    auto make_bind = [&](TextContext& c, WorkspaceArena& arena) {
        c.ctx_.bind_to_current_thread();
        BindState b;
        const auto envelope_tokens = static_cast<std::uint32_t>(position + 1);
        b.envelope                 = {envelope_tokens, envelope_tokens};
        b.cache_positions = arena.alloc(DType::I32, {1});
        ops::set_i32_scalar(b.cache_positions, position, c.ctx_.stream);
        b.rope_positions = arena.alloc(DType::I32, {1});
        ops::set_i32_scalar(b.rope_positions, position, c.ctx_.stream);
        b.kv_table_rows     = arena.alloc(DType::I32, {1});
        ops::set_i32_scalar(b.kv_table_rows, 0, c.ctx_.stream);
        b.state_source      = arena.alloc(DType::I32, {1});
        ops::set_i32_scalar(b.state_source, 0, c.ctx_.stream);
        b.state_destination = arena.alloc(DType::I32, {1});
        ops::set_i32_scalar(b.state_destination, 0, c.ctx_.stream);
        return b;
    };
    const BindState bind0 = make_bind(*this, work_);
    const BindState bind1 = make_bind(peer, peer.work_);
    ScopedPositions cache0(active_cache_positions_, bind0.cache_positions);
    ScopedPositions rope0(active_rope_positions_, bind0.rope_positions);
    ScopedEnvelope envelope0(active_causal_attention_envelope_, bind0.envelope);
    ScopedValue<const Tensor*> kv0(active_kv_table_rows_, &bind0.kv_table_rows);
    ScopedValue<const Tensor*> source0(active_linear_state_source_slots_, &bind0.state_source);
    ScopedValue<const Tensor*> destination0(active_linear_state_destination_slots_,
                                            &bind0.state_destination);
    ScopedValue<std::int32_t> batch0(active_sequence_batch_, 1);
    ScopedValue<std::int32_t> width0(active_sequence_width_, 1);
    ScopedPositions cache1(peer.active_cache_positions_, bind1.cache_positions);
    ScopedPositions rope1(peer.active_rope_positions_, bind1.rope_positions);
    ScopedEnvelope envelope1(peer.active_causal_attention_envelope_, bind1.envelope);
    ScopedValue<const Tensor*> kv1(peer.active_kv_table_rows_, &bind1.kv_table_rows);
    ScopedValue<const Tensor*> source1(peer.active_linear_state_source_slots_,
                                       &bind1.state_source);
    ScopedValue<const Tensor*> destination1(peer.active_linear_state_destination_slots_,
                                            &bind1.state_destination);
    ScopedValue<std::int32_t> batch1(peer.active_sequence_batch_, 1);
    ScopedValue<std::int32_t> width1(peer.active_sequence_width_, 1);

    ctx_.bind_to_current_thread();
    Tensor ids = work_.alloc(DType::I32, {1});
    ops::set_i32_scalar(ids, token, ctx_.stream);
    peer.ctx_.bind_to_current_thread();
    Tensor ids_peer = peer.work_.alloc(DType::I32, {1});
    ops::set_i32_scalar(ids_peer, token, peer.ctx_.stream);

    ctx_.bind_to_current_thread();
    Tensor x      = work_.alloc(DType::BF16, {hidden, 1});
    Tensor x_peer = peer.work_.alloc(DType::BF16, {hidden, 1});
    embedding_tp2(peer, pair, ids, &ids_peer, x, &x_peer);
    NullTap tap;
    run_layers_tp2(peer, pair, x, x_peer, Phase::Verify, tap);

    ctx_.bind_to_current_thread();
    Tensor hidden_out      = work_.alloc(DType::BF16, {hidden, 1});
    peer.ctx_.bind_to_current_thread();
    Tensor hidden_out_peer = peer.work_.alloc(DType::BF16, {hidden, 1});
    ctx_.bind_to_current_thread();
    ops::rmsnorm(x, *final_norm_, config_.rms_norm_eps, true, hidden_out, ctx_.stream);
    if (mtp_input_hidden != nullptr) {
        // The MTP layer consumes the final-norm hidden; hand the caller its own stable copy so this
        // context's workspace scope can end.
        if (mtp_input_hidden->dtype != DType::BF16 || mtp_input_hidden->ne[0] != hidden ||
            mtp_input_hidden->ne[1] != 1) {
            throw std::invalid_argument("forward_tp2: MTP hidden buffer must be [hidden,1] BF16");
        }
        CUDA_CHECK(cudaMemcpyAsync(mtp_input_hidden->data, hidden_out.data, hidden_out.bytes(),
                                   cudaMemcpyDeviceToDevice, ctx_.stream));
    }
    peer.ctx_.bind_to_current_thread();
    ops::rmsnorm(x_peer, *peer.final_norm_, config_.rms_norm_eps, true, hidden_out_peer,
                 peer.ctx_.stream);


    // The head is replicated or vocabulary-split; project_head_tp2 covers both, leaving the full
    // [V, 1] logits on either shard.
    project_head_tp2(peer, pair, hidden_out, hidden_out_peer, logits, logits_peer);
}

void TextContext::forward_tp2_decode_window(TextContext& peer, tp::DevicePair& pair,
                                            const std::int32_t* token, const std::int32_t* position,
                                            ops::CausalAttentionExecutionEnvelope envelope,
                                            Tensor& logits) {
    const std::int32_t hidden = dimension(config_.hidden_size);
    const std::int32_t vocab  = dimension(config_.vocab_size);
    if (token == nullptr || position == nullptr) {
        throw std::invalid_argument(
            "forward_tp2_decode_window requires pinned host token and position");
    }
    if (vocab % 2 != 0 || logits.ne[0] != vocab || logits.ne[1] != 1) {
        throw std::invalid_argument("forward_tp2_decode_window: logits must be [V,1]");
    }
    if (envelope.min_visible_keys == 0 || envelope.max_visible_keys < envelope.min_visible_keys) {
        throw std::invalid_argument("forward_tp2_decode_window: envelope is invalid");
    }
    // Same sequence as forward_tp2, with only its two per-round host values made capturable: the
    // decoded token and its absolute position now arrive through a memcpy node out of pinned host
    // memory, and the attention envelope is a capture parameter instead of the exact extent. The
    // envelope is a bound rather than a decision - the small-T route reads the real window from the
    // device-side positions and derives the active split count and the key partition from it, so a
    // bucket-wide envelope reduces to the same kernels and the same key ranges.
    struct BindState {
        Tensor ids;
        Tensor cache_positions;
        Tensor rope_positions;
        Tensor kv_table_rows;
        Tensor state_source;
        Tensor state_destination;
    };
    auto make_bind = [&](TextContext& card, WorkspaceArena& arena) {
        card.ctx_.bind_to_current_thread();
        BindState bind;
        bind.ids = arena.alloc(DType::I32, {1});
        copy_i32(token, bind.ids, card.ctx_.stream);
        bind.cache_positions = arena.alloc(DType::I32, {1});
        copy_i32(position, bind.cache_positions, card.ctx_.stream);
        bind.rope_positions = arena.alloc(DType::I32, {1});
        copy_i32(position, bind.rope_positions, card.ctx_.stream);
        bind.kv_table_rows = arena.alloc(DType::I32, {1});
        ops::set_i32_scalar(bind.kv_table_rows, 0, card.ctx_.stream);
        bind.state_source = arena.alloc(DType::I32, {1});
        ops::set_i32_scalar(bind.state_source, 0, card.ctx_.stream);
        bind.state_destination = arena.alloc(DType::I32, {1});
        ops::set_i32_scalar(bind.state_destination, 0, card.ctx_.stream);
        return bind;
    };
    const BindState bind0 = make_bind(*this, work_);
    const BindState bind1 = make_bind(peer, peer.work_);
    ScopedPositions cache0(active_cache_positions_, bind0.cache_positions);
    ScopedPositions rope0(active_rope_positions_, bind0.rope_positions);
    ScopedEnvelope envelope0(active_causal_attention_envelope_, envelope);
    ScopedValue<const Tensor*> kv0(active_kv_table_rows_, &bind0.kv_table_rows);
    ScopedValue<const Tensor*> source0(active_linear_state_source_slots_, &bind0.state_source);
    ScopedValue<const Tensor*> destination0(active_linear_state_destination_slots_,
                                            &bind0.state_destination);
    ScopedValue<std::int32_t> batch0(active_sequence_batch_, 1);
    ScopedValue<std::int32_t> width0(active_sequence_width_, 1);
    ScopedPositions cache1(peer.active_cache_positions_, bind1.cache_positions);
    ScopedPositions rope1(peer.active_rope_positions_, bind1.rope_positions);
    ScopedEnvelope envelope1(peer.active_causal_attention_envelope_, envelope);
    ScopedValue<const Tensor*> kv1(peer.active_kv_table_rows_, &bind1.kv_table_rows);
    ScopedValue<const Tensor*> source1(peer.active_linear_state_source_slots_, &bind1.state_source);
    ScopedValue<const Tensor*> destination1(peer.active_linear_state_destination_slots_,
                                            &bind1.state_destination);
    ScopedValue<std::int32_t> batch1(peer.active_sequence_batch_, 1);
    ScopedValue<std::int32_t> width1(peer.active_sequence_width_, 1);

    ctx_.bind_to_current_thread();
    Tensor x      = work_.alloc(DType::BF16, {hidden, 1});
    Tensor x_peer = peer.work_.alloc(DType::BF16, {hidden, 1});
    embedding_tp2(peer, pair, bind0.ids, &bind1.ids, x, &x_peer);
    NullTap tap;
    run_layers_tp2(peer, pair, x, x_peer, Phase::Verify, tap);

    ctx_.bind_to_current_thread();
    Tensor hidden_out      = work_.alloc(DType::BF16, {hidden, 1});
    peer.ctx_.bind_to_current_thread();
    Tensor hidden_out_peer = peer.work_.alloc(DType::BF16, {hidden, 1});
    ctx_.bind_to_current_thread();
    ops::rmsnorm(x, *final_norm_, config_.rms_norm_eps, true, hidden_out, ctx_.stream);
    peer.ctx_.bind_to_current_thread();
    ops::rmsnorm(x_peer, *peer.final_norm_, config_.rms_norm_eps, true, hidden_out_peer,
                 peer.ctx_.stream);
    Tensor logits_peer = peer.work_.alloc(DType::BF16, {vocab, 1});
    project_head_tp2(peer, pair, hidden_out, hidden_out_peer, logits, logits_peer);
}

std::int32_t TextContext::forward_tp2_token(TextContext& peer, tp::DevicePair& pair,
                                            std::int32_t token, std::int32_t position) {
    const std::int32_t vocab = dimension(config_.vocab_size);
    auto scope = work_.scope();
    Tensor logits      = work_.alloc(DType::BF16, {vocab, 1});
    Tensor logits_peer = peer.work_.alloc(DType::BF16, {vocab, 1});
    forward_tp2(peer, pair, token, position, logits, logits_peer);
    // forward_tp2 leaves the current device on the peer card; bind each shard before its
    // argmax so the device pointer resolves in the correct address space. The lm_head is
    // replicated, so the two shards' logits are bit-identical and their argmax must agree;
    // comparing both makes a shard divergence a hard failure at the sampling boundary.
    ctx_.bind_to_current_thread();
    Tensor sampled = work_.alloc(DType::I32, {1});
    ops::argmax(logits, sampled, vocab, ctx_.stream);
    peer.ctx_.bind_to_current_thread();
    Tensor sampled_peer = peer.work_.alloc(DType::I32, {1});
    ops::argmax(logits_peer, sampled_peer, vocab, peer.ctx_.stream);
    std::int32_t token_id      = 0;
    std::int32_t token_id_peer = 0;
    ctx_.bind_to_current_thread();
    CUDA_CHECK(cudaMemcpyAsync(&token_id, sampled.data, sizeof(std::int32_t),
                               cudaMemcpyDeviceToHost, ctx_.stream));
    CUDA_CHECK(cudaStreamSynchronize(ctx_.stream));
    peer.ctx_.bind_to_current_thread();
    CUDA_CHECK(cudaMemcpyAsync(&token_id_peer, sampled_peer.data, sizeof(std::int32_t),
                               cudaMemcpyDeviceToHost, peer.ctx_.stream));
    CUDA_CHECK(cudaStreamSynchronize(peer.ctx_.stream));
    if (token_id != token_id_peer) {
        throw std::logic_error("forward_tp2_token: shard argmax mismatch at position " +
                               std::to_string(position) + " (" + std::to_string(token_id) +
                               " vs " + std::to_string(token_id_peer) + ")");
    }
    return token_id;
}

void TextContext::forward_tp2_prefill(TextContext& peer, tp::DevicePair& pair,
                                      std::span<const int> ids, std::int32_t first_position,
                                      Tensor* logits, Tensor* logits_peer,
                                      Tensor* mtp_input_hidden, Tensor* logits_columns,
                                      Tensor* hidden_columns, Phase phase,
                                      const Tp2VisionChunk* vision, DFlashFeatureSink* sink) {
    const std::int32_t hidden = dimension(config_.hidden_size);
    const std::int32_t vocab  = dimension(config_.vocab_size);
    const std::int32_t tokens = static_cast<std::int32_t>(ids.size());
    if (tokens <= 0) { throw std::invalid_argument("forward_tp2_prefill requires tokens"); }
    if (first_position < 0) {
        throw std::invalid_argument("forward_tp2_prefill: position must be non-negative");
    }
    if (logits_columns == nullptr) {
        if (logits == nullptr || logits_peer == nullptr) {
            throw std::invalid_argument(
                "forward_tp2_prefill: a prefill chunk must request the last column logits");
        }
        if (logits->ne[0] != vocab || logits->ne[1] != 1 || logits_peer->ne[0] != vocab ||
            logits_peer->ne[1] != 1) {
            throw std::invalid_argument(
                "forward_tp2_prefill: logits buffers must be full-vocabulary [V,1]");
        }
    } else if (logits != nullptr || logits_peer != nullptr) {
        // The per-column buffer carries the last column as well, so a second lm_head read
        // would score exactly the same bytes again.
        throw std::invalid_argument(
            "forward_tp2_prefill: per-column logits replace the [V,1] last-column buffers");
    }
    // The execution envelope is the chunk's inclusive key extent: the chunk appends positions
    // [first_position, first_position + tokens) and its last query reads every one of them. This
    // is exactly the binding the validated single-device prefill chunk uses.
    const std::uint32_t visible_end =
        static_cast<std::uint32_t>(first_position) + static_cast<std::uint32_t>(tokens);
    if (vision != nullptr &&
        (vision->positions == nullptr ||
         vision->prompt_tokens < static_cast<std::size_t>(first_position) +
                                   static_cast<std::size_t>(tokens))) {
        throw std::invalid_argument(
            "forward_tp2_prefill: multimodal chunk has no prompt position table");
    }
    struct BindState {
        Tensor ids;
        Tensor positions;
        Tensor rope_positions;
        Tensor kv_table_rows;
        Tensor state_source;
        Tensor state_destination;
        ops::CausalAttentionExecutionEnvelope envelope{1, 1};
    };
    // Bind the replicated mixer state on both shards. The RAII guards live until the end of the
    // forward (a bind lambda would destroy them immediately), and each shard's device is bound
    // before any of its allocations or kernels. No work_.reset() here: the caller owns the logits
    // buffers in this arena, and resetting would let the internal allocations overwrite them.
    auto make_bind = [&](TextContext& c, WorkspaceArena& arena) {
        c.ctx_.bind_to_current_thread();
        BindState b;
        b.envelope    = {visible_end, visible_end};
        b.ids         = arena.alloc(DType::I32, {tokens});
        b.positions   = arena.alloc(DType::I32, {tokens});
        copy_i32(ids.data(), b.ids, c.ctx_.stream);
        ops::fill_i32_positions(b.positions, first_position, c.ctx_.stream);
        if (vision != nullptr) {
            // A multimodal prompt carries its own 3-axis (temporal, height, width) RoPE table for
            // every token; the cache position stays the plain absolute index, so the two bindings
            // stop coinciding for this request.
            // [tokens, 3]: the token index is contiguous, matching the single-device route's
            // [tokens, axes] binding (neo[1] == 3 is what makes the RoPE op read the MRoPE table).
            b.rope_positions = arena.alloc(DType::I32, {tokens, 3});
            std::vector<std::int32_t> host(static_cast<std::size_t>(3) * tokens);
            for (int axis = 0; axis < 3; ++axis) {
                std::copy_n(vision->positions +
                                static_cast<std::size_t>(axis) * vision->prompt_tokens +
                                static_cast<std::size_t>(first_position),
                            tokens, host.data() + static_cast<std::size_t>(axis) * tokens);
            }
            copy_i32(host.data(), b.rope_positions, c.ctx_.stream);
        } else {
            b.rope_positions = b.positions;
        }
        b.kv_table_rows = arena.alloc(DType::I32, {1});
        ops::set_i32_scalar(b.kv_table_rows, 0, c.ctx_.stream);
        b.state_source = arena.alloc(DType::I32, {1});
        ops::set_i32_scalar(b.state_source, 0, c.ctx_.stream);
        b.state_destination = arena.alloc(DType::I32, {1});
        ops::set_i32_scalar(b.state_destination, 0, c.ctx_.stream);
        return b;
    };
    const BindState bind0 = make_bind(*this, work_);
    const BindState bind1 = make_bind(peer, peer.work_);
    // Cache and RoPE positions coincide for a text prompt: the TP-2 path runs without a RoPE delta,
    // so the cache position of every chunk token is its own absolute position. A multimodal prompt
    // binds the prompt's 3-axis table instead (see make_bind).
    ScopedPositions cache0(active_cache_positions_, bind0.positions);
    ScopedPositions rope0(active_rope_positions_, bind0.rope_positions);
    ScopedEnvelope envelope0(active_causal_attention_envelope_, bind0.envelope);
    ScopedValue<const Tensor*> kv0(active_kv_table_rows_, &bind0.kv_table_rows);
    ScopedValue<const Tensor*> source0(active_linear_state_source_slots_, &bind0.state_source);
    ScopedValue<const Tensor*> destination0(active_linear_state_destination_slots_,
                                            &bind0.state_destination);
    ScopedPositions cache1(peer.active_cache_positions_, bind1.positions);
    ScopedPositions rope1(peer.active_rope_positions_, bind1.rope_positions);
    ScopedEnvelope envelope1(peer.active_causal_attention_envelope_, bind1.envelope);
    ScopedValue<const Tensor*> kv1(peer.active_kv_table_rows_, &bind1.kv_table_rows);
    ScopedValue<const Tensor*> source1(peer.active_linear_state_source_slots_,
                                      &bind1.state_source);
    ScopedValue<const Tensor*> destination1(peer.active_linear_state_destination_slots_,
                                            &bind1.state_destination);

    ctx_.bind_to_current_thread();
    Tensor x      = work_.alloc(DType::BF16, {hidden, tokens});
    Tensor x_peer = peer.work_.alloc(DType::BF16, {hidden, tokens});
    embedding_tp2(peer, pair, bind0.ids, &bind1.ids, x, &x_peer);
    if (vision != nullptr && vision->has_item()) {
        const auto& control = *vision->control;
        const auto scatter  = std::span<const std::int32_t>(control.scatter_indices);
        const auto begin    = std::lower_bound(scatter.begin(), scatter.end(), first_position);
        const auto stop     = std::lower_bound(begin, scatter.end(), first_position + tokens);
        const auto count    = static_cast<std::int32_t>(stop - begin);
        if (count != 0) {
            const std::int32_t merged = static_cast<std::int32_t>(control.merged_count);
            if (vision->embeddings == nullptr || vision->embeddings->dtype != DType::BF16 ||
                vision->embeddings->ne[0] != hidden || vision->embeddings->ne[1] != merged) {
                throw std::invalid_argument(
                    "forward_tp2_prefill: vision embeddings do not match the encoded item");
            }
            const std::int32_t visual_begin = static_cast<std::int32_t>(begin - scatter.begin());
            if (visual_begin + count > merged) {
                throw std::invalid_argument(
                    "forward_tp2_prefill: vision scatter runs past the encoded item");
            }
            std::vector<std::int32_t> local(static_cast<std::size_t>(count));
            for (std::int32_t i = 0; i < count; ++i) {
                local[static_cast<std::size_t>(i)] = begin[i] - first_position;
            }
            const std::size_t row_bytes = static_cast<std::size_t>(hidden) * sizeof(std::uint16_t);
            const std::size_t bytes     = row_bytes * static_cast<std::size_t>(count);
            // The Vision shard owns the encoded item, so only its side reads the handoff tensor: the
            // pair copies this chunk's contiguous column range into both arenas with one in-place
            // all-reduce over a zeroed text-shard buffer. That is exact (a sum with zero), keeps the
            // resident cost bounded by the chunk rather than by the item, and needs no second tower.
            ctx_.bind_to_current_thread();
            Tensor staging = work_.alloc(DType::BF16, {hidden, count});
            CUDA_CHECK(cudaMemsetAsync(staging.data, 0, bytes, ctx_.stream));
            peer.ctx_.bind_to_current_thread();
            Tensor source = peer.work_.alloc(DType::BF16, {hidden, count});
            // The tensor payload is untyped, so the byte offset needs an explicit byte pointer
            // (void arithmetic is a GCC extension).
            const auto* embeddings = static_cast<const std::byte*>(vision->embeddings->data);
            CUDA_CHECK(cudaMemcpyAsync(
                source.data, embeddings + static_cast<std::size_t>(visual_begin) * row_bytes, bytes,
                cudaMemcpyDeviceToDevice, peer.ctx_.stream));
            pair.allreduce(staging.data, source.data, bytes, ctx_.stream, peer.ctx_.stream);
            ctx_.bind_to_current_thread();
            Tensor indices = work_.alloc(DType::I32, {count});
            copy_i32(local.data(), indices, ctx_.stream);
            ops::scatter(staging, indices, x, ctx_.stream);
            peer.ctx_.bind_to_current_thread();
            Tensor indices_peer = peer.work_.alloc(DType::I32, {count});
            copy_i32(local.data(), indices_peer, peer.ctx_.stream);
            ops::scatter(source, indices_peer, x_peer, peer.ctx_.stream);
            ctx_.bind_to_current_thread();
        }
    }
    NullTap tap;
    auto run_layers = [&] {
        if (sink != nullptr) {
            run_layers_tp2(peer, pair, x, x_peer, phase, *sink);
        } else {
            run_layers_tp2(peer, pair, x, x_peer, phase, tap);
        }
    };
    if (sink != nullptr) { sink->begin(x); }
    if (phase == Phase::Verify) {
        // The verify window runs the phase the single-token decode path runs, so its per-column
        // logits agree with decode on near-ties. That phase needs the explicit sequence bindings the
        // decode path sets (one row, one column per window token, the live state as the source slot),
        // which prefill leaves unset because it is a single column.
        Tensor verify_source_slots = work_.alloc(DType::I32, {1});
        ops::set_i32_scalar(verify_source_slots, 0, ctx_.stream);
        // Batched MTP attention keys off active_sequence_batch_ != 0, so the window also needs the
        // per-row valid-column count and the backend KV row table that decode binds per row.
        Tensor verify_valid_columns = work_.alloc(DType::I32, {1});
        ops::set_i32_scalar(verify_valid_columns, tokens, ctx_.stream);
        Tensor verify_backend_rows = work_.alloc(DType::I32, {1});
        ops::set_i32_scalar(verify_backend_rows, 0, ctx_.stream);
        ScopedValue<std::int32_t> verify_batch(active_sequence_batch_, 1);
        ScopedValue<std::int32_t> verify_width(active_sequence_width_, tokens);
        ScopedValue<const Tensor*> verify_source(active_linear_state_source_slots_,
                                                 &verify_source_slots);
        ScopedValue<const Tensor*> verify_valid(active_valid_columns_, &verify_valid_columns);
        ScopedValue<const Tensor*> verify_backend(active_backend_kv_table_rows_,
                                                  &verify_backend_rows);
        // The verify path reads positions as [width, batch]; prefill binds them flat for its chunk.
        Tensor verify_positions = work_.alloc(DType::I32, {tokens, 1});
        ops::fill_i32_positions(verify_positions, first_position, ctx_.stream);
        ScopedPositions verify_cache(active_cache_positions_, verify_positions);
        ScopedPositions verify_rope(active_rope_positions_, verify_positions);
        ScopedValue<std::int32_t> verify_peer_batch(peer.active_sequence_batch_, 1);
        ScopedValue<std::int32_t> verify_peer_width(peer.active_sequence_width_, tokens);
        ScopedValue<const Tensor*> verify_peer_source(peer.active_linear_state_source_slots_,
                                                      &verify_source_slots);
        ScopedValue<const Tensor*> verify_peer_valid(peer.active_valid_columns_,
                                                     &verify_valid_columns);
        ScopedValue<const Tensor*> verify_peer_backend(peer.active_backend_kv_table_rows_,
                                                       &verify_backend_rows);
        ScopedPositions verify_peer_cache(peer.active_cache_positions_, verify_positions);
        ScopedPositions verify_peer_rope(peer.active_rope_positions_, verify_positions);
        run_layers();
    } else {
        run_layers();
    }
    if (sink != nullptr && phase != Phase::Verify) {
        // Positions are captured after the layers so the sink can check every feature layer published.
        sink->capture_positions(*active_cache_positions_, ctx_.stream);
        sink->consume_prefill_chunk(tokens, false);
    }

    ctx_.bind_to_current_thread();
    Tensor xf      = work_.alloc(DType::BF16, {hidden, tokens});
    peer.ctx_.bind_to_current_thread();
    Tensor xf_peer = peer.work_.alloc(DType::BF16, {hidden, tokens});
    ctx_.bind_to_current_thread();
    ops::rmsnorm(x, *final_norm_, config_.rms_norm_eps, true, xf, ctx_.stream);
    if (mtp_input_hidden != nullptr) {
        // The MTP layer's prefill chunk consumes the whole chunk's final-norm hidden.
        if (mtp_input_hidden->dtype != DType::BF16 || mtp_input_hidden->ne[0] != hidden ||
            mtp_input_hidden->ne[1] != tokens) {
            throw std::invalid_argument(
                "forward_tp2_prefill: MTP hidden buffer must be [hidden,T] BF16");
        }
        CUDA_CHECK(cudaMemcpyAsync(mtp_input_hidden->data, xf.data, xf.bytes(),
                                   cudaMemcpyDeviceToDevice, ctx_.stream));
    }
    peer.ctx_.bind_to_current_thread();
    ops::rmsnorm(x_peer, *peer.final_norm_, config_.rms_norm_eps, true, xf_peer, peer.ctx_.stream);

    // Only the chunk's last column feeds the first sample, exactly like the single-device prefill's
    // final chunk. project_head_tp2 covers both head layouts: replicated (two independent full
    // projections that agree exactly) and vocabulary-split (one half per shard plus an all-reduce).
    ctx_.bind_to_current_thread();
    if (logits_columns != nullptr) {
        // Speculative verification scores every column: column j predicts the token after the one
        // it reads, so it judges the draft placed at j+1. Its last column is exactly the logits
        // the [V,1] projection below would produce, so that read is not repeated. Only the driving
        // shard carries this buffer, so a split head needs a peer scratch window of the same size.
        if (logits_columns->dtype != DType::BF16 || logits_columns->ne[0] != vocab ||
            logits_columns->ne[1] != tokens) {
            throw std::invalid_argument(
                "forward_tp2_prefill: per-column logits buffer must be [V,T] BF16");
        }
        Tensor logits_columns_peer = peer.work_.alloc(DType::BF16, {vocab, tokens});
        project_head_tp2(peer, pair, xf, xf_peer, *logits_columns, logits_columns_peer);
    } else {
        project_head_tp2(peer, pair, xf.slice(1, tokens - 1, 1), xf_peer.slice(1, tokens - 1, 1),
                         *logits, *logits_peer);
    }
    if (hidden_columns != nullptr) {
        // The next round's MTP bridge consumes the final-norm hidden of the accepted column.
        if (hidden_columns->dtype != DType::BF16 || hidden_columns->ne[0] != hidden ||
            hidden_columns->ne[1] != tokens) {
            throw std::invalid_argument(
                "forward_tp2_prefill: per-column hidden buffer must be [hidden,T] BF16");
        }
        ctx_.bind_to_current_thread();
        CUDA_CHECK(cudaMemcpyAsync(hidden_columns->data, xf.data, xf.bytes(),
                                   cudaMemcpyDeviceToDevice, ctx_.stream));
    }
}

void TextContext::forward_tp2_window(TextContext& peer, tp::DevicePair& pair,
                                     const std::int32_t* ids, const std::int32_t* positions,
                                     ops::CausalAttentionExecutionEnvelope envelope,
                                     Tensor& logits_columns, Tensor* hidden_columns,
                                     DFlashFeatureSink* sink) {
    const std::int32_t hidden = dimension(config_.hidden_size);
    const std::int32_t vocab  = dimension(config_.vocab_size);
    if (ids == nullptr || positions == nullptr) {
        throw std::invalid_argument("forward_tp2_window requires pinned host ids and positions");
    }
    if (logits_columns.dtype != DType::BF16 || logits_columns.ne[0] != vocab) {
        throw std::invalid_argument("forward_tp2_window: logits columns must be [V,T] BF16");
    }
    const std::int32_t tokens = logits_columns.ne[1];
    if (tokens <= 0) { throw std::invalid_argument("forward_tp2_window requires tokens"); }
    if (envelope.min_visible_keys == 0 || envelope.max_visible_keys < envelope.min_visible_keys) {
        throw std::invalid_argument("forward_tp2_window: envelope does not cover the window");
    }
    if (hidden_columns != nullptr &&
        (hidden_columns->dtype != DType::BF16 || hidden_columns->ne[0] != hidden ||
         hidden_columns->ne[1] != tokens)) {
        throw std::invalid_argument("forward_tp2_window: hidden columns must be [hidden,T] BF16");
    }

    struct BindState {
        Tensor ids;
        Tensor positions;
        Tensor kv_table_rows;
        Tensor state_source;
        Tensor state_destination;
        ops::CausalAttentionExecutionEnvelope envelope{1, 1};
    };
    // Both shards bind their own copies of the window: the pair has no peer access, and a captured
    // sequence needs fixed addresses, so each shard's arena owns its [T] buffers and the only
    // per-round input is the pinned host source the memcpy node reads.
    auto make_bind = [&](TextContext& card, WorkspaceArena& arena) {
        card.ctx_.bind_to_current_thread();
        BindState bind;
        bind.envelope          = envelope;
        bind.ids               = arena.alloc(DType::I32, {tokens});
        bind.positions         = arena.alloc(DType::I32, {tokens});
        bind.kv_table_rows     = arena.alloc(DType::I32, {1});
        bind.state_source      = arena.alloc(DType::I32, {1});
        bind.state_destination = arena.alloc(DType::I32, {1});
        copy_i32(ids, bind.ids, card.ctx_.stream);
        copy_i32(positions, bind.positions, card.ctx_.stream);
        ops::set_i32_scalar(bind.kv_table_rows, 0, card.ctx_.stream);
        ops::set_i32_scalar(bind.state_source, 0, card.ctx_.stream);
        ops::set_i32_scalar(bind.state_destination, 0, card.ctx_.stream);
        return bind;
    };
    const BindState bind0 = make_bind(*this, work_);
    const BindState bind1 = make_bind(peer, peer.work_);
    // The window is a contiguous append at consecutive absolute positions, so its cache and RoPE
    // positions coincide: this path carries no RoPE delta and no multimodal position table.
    ScopedPositions cache0(active_cache_positions_, bind0.positions);
    ScopedPositions rope0(active_rope_positions_, bind0.positions);
    ScopedEnvelope envelope0(active_causal_attention_envelope_, bind0.envelope);
    ScopedValue<const Tensor*> kv0(active_kv_table_rows_, &bind0.kv_table_rows);
    ScopedValue<const Tensor*> source0(active_linear_state_source_slots_, &bind0.state_source);
    ScopedValue<const Tensor*> destination0(active_linear_state_destination_slots_,
                                            &bind0.state_destination);
    ScopedPositions cache1(peer.active_cache_positions_, bind1.positions);
    ScopedPositions rope1(peer.active_rope_positions_, bind1.positions);
    ScopedEnvelope envelope1(peer.active_causal_attention_envelope_, bind1.envelope);
    ScopedValue<const Tensor*> kv1(peer.active_kv_table_rows_, &bind1.kv_table_rows);
    ScopedValue<const Tensor*> source1(peer.active_linear_state_source_slots_,
                                       &bind1.state_source);
    ScopedValue<const Tensor*> destination1(peer.active_linear_state_destination_slots_,
                                            &bind1.state_destination);

    ctx_.bind_to_current_thread();
    Tensor x      = work_.alloc(DType::BF16, {hidden, tokens});
    peer.ctx_.bind_to_current_thread();
    Tensor x_peer = peer.work_.alloc(DType::BF16, {hidden, tokens});
    ctx_.bind_to_current_thread();
    embedding_tp2(peer, pair, bind0.ids, &bind1.ids, x, &x_peer);

    NullTap tap;
    if (sink != nullptr) {
        sink->begin(x);
        run_layers_tp2(peer, pair, x, x_peer, Phase::Prefill, *sink);
    } else {
        run_layers_tp2(peer, pair, x, x_peer, Phase::Prefill, tap);
    }

    ctx_.bind_to_current_thread();
    Tensor xf      = work_.alloc(DType::BF16, {hidden, tokens});
    peer.ctx_.bind_to_current_thread();
    Tensor xf_peer = peer.work_.alloc(DType::BF16, {hidden, tokens});
    ctx_.bind_to_current_thread();
    ops::rmsnorm(x, *final_norm_, config_.rms_norm_eps, true, xf, ctx_.stream);
    peer.ctx_.bind_to_current_thread();
    ops::rmsnorm(x_peer, *peer.final_norm_, config_.rms_norm_eps, true, xf_peer, peer.ctx_.stream);
    // Every column is scored, because column j judges the draft placed at j+1; the last column is
    // exactly the last column's logits, so no second lm_head read is needed.
    ctx_.bind_to_current_thread();
    Tensor logits_columns_peer = peer.work_.alloc(DType::BF16, {vocab, tokens});
    project_head_tp2(peer, pair, xf, xf_peer, logits_columns, logits_columns_peer);
    if (hidden_columns != nullptr) {
        // The next round's MTP bridge consumes the final-norm hidden of the accepted column.
        ctx_.bind_to_current_thread();
        CUDA_CHECK(cudaMemcpyAsync(hidden_columns->data, xf.data, xf.bytes(),
                                   cudaMemcpyDeviceToDevice, ctx_.stream));
    }
}

template <class Tap>
PrefillChunkResult
TextContext::prefill_impl(std::span<const int> ids, const TextPrefill* text_prefill,
                          const MultimodalPrefill* multimodal, Tap& tap, bool finalize_at_end) {
    runtime::ExecutionTimingRecorder timing;
    if (ids.empty()) { throw std::invalid_argument("TextContext::prefill requires tokens"); }
    if (ids.size() > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error("TextContext::prefill token count exceeds int32");
    }
    cudaStream_t s           = ctx_.stream;
    const int T              = static_cast<int>(ids.size());
    const int chunk          = static_cast<int>(prefill_chunk_);
    const std::uint32_t base = text_kv_base_;

    if (text_prefill != nullptr) {
        if (multimodal != nullptr || base != text_prefill->begin ||
            text_prefill->token_ids.size() < static_cast<std::size_t>(base) + ids.size()) {
            throw std::invalid_argument("text prefill chunk does not match its full prompt");
        }
    }
    if (multimodal != nullptr) {
        if (base != multimodal->begin ||
            multimodal->token_ids.size() < static_cast<std::size_t>(base) + ids.size()) {
            throw std::invalid_argument("multimodal prefill suffix does not match its cache base");
        }
        if (multimodal->positions.size() != 3 * multimodal->token_ids.size()) {
            throw std::invalid_argument("multimodal positions must have shape [3,T]");
        }
        if (multimodal->vision == nullptr) {
            throw std::invalid_argument("multimodal prefill requires a Vision session");
        }
        rope_delta_ = multimodal->rope_delta;
    } else if (text_kv_base_ == 0) {
        rope_delta_ = 0;
    }
    ops::set_i32_scalar(io_.rope_delta, rope_delta_, s);

    // Prefix-append prefill continues an existing cache: positions are absolute (start at the
    // resident length) and KV/GDN state is not reset. For a reset prefill base == 0.
    if (static_cast<std::uint64_t>(base) + static_cast<std::uint64_t>(T) >
        static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::overflow_error("TextContext::prefill absolute position exceeds int32");
    }
    const int base_i = static_cast<int>(base);

    const std::int64_t base64    = static_cast<std::int64_t>(base);
    const std::int64_t split_abs = prefill_split_frontier_;
    const bool has_split = split_abs > base64 && split_abs <= base64 + static_cast<std::int64_t>(T);
    const int split_rel  = has_split ? static_cast<int>(split_abs - base64) : -1;
    const bool prepare_mtp_prompt = mtp_enabled() && io_.mtp.has_value();
    if (prepare_mtp_prompt &&
        mtp_proposal_extent_ > static_cast<std::uint32_t>(io_.mtp->draft_tokens.ne[0])) {
        throw std::logic_error("MTP proposal extent exceeds the configured draft window");
    }
    int t0 = 0;
    for (; t0 < T;) {
        int len = std::min(chunk, T - t0);
        if (split_rel > 0 && t0 < split_rel && t0 + len > split_rel) { len = split_rel - t0; }
        work_.reset();

        VisionChunk vision_chunk;
        const std::uint32_t prompt_t0 = base + static_cast<std::uint32_t>(t0);
        if (multimodal != nullptr) {
            if (multimodal->vision == nullptr) {
                throw std::logic_error("multimodal prefill has no Vision session");
            }
            vision_chunk =
                multimodal->vision->prepare_chunk(prompt_t0, static_cast<std::uint32_t>(len));
            len = vision_chunk.length;
        }
        const bool is_last = finalize_at_end && (t0 + len == T);
        nvtx::ScopedRange chunk_range(nvtx::Name::PrefillChunk, nvtx::Category::Prefill,
                                      static_cast<std::uint64_t>(len));

        {
            std::vector<std::int32_t> local_scatter_indices;
            std::int32_t visual_begin = 0;
            if (vision_chunk.control != nullptr) {
                const auto scatter =
                    std::span<const std::int32_t>(vision_chunk.control->scatter_indices);
                const auto begin = std::lower_bound(scatter.begin(), scatter.end(), prompt_t0);
                const auto end   = std::lower_bound(begin, scatter.end(), prompt_t0 + len);
                const auto count = static_cast<std::int32_t>(end - begin);
                visual_begin     = static_cast<std::int32_t>(begin - scatter.begin());
                local_scatter_indices.resize(static_cast<std::size_t>(count));
                for (std::int32_t i = 0; i < count; ++i) {
                    local_scatter_indices[static_cast<std::size_t>(i)] =
                        begin[i] - static_cast<std::int32_t>(prompt_t0);
                }
            }

            const std::int32_t rope_axes = multimodal != nullptr ? 3 : (rope_delta_ != 0 ? 1 : 0);
            const auto roots             = workspace::text_prefill_roots(
                work_, config_, len, rope_axes,
                static_cast<std::int32_t>(local_scatter_indices.size()));
            Tensor ids_device = roots.ids;
            copy_i32(ids.data() + t0, ids_device, s);

            Tensor positions = roots.positions;
            ops::fill_i32_positions(positions, base_i + t0, s);

            Tensor rope_positions = positions;
            std::vector<std::int32_t> rope_positions_host;
            if (multimodal != nullptr) {
                rope_positions = roots.rope_positions;
                rope_positions_host.resize(static_cast<std::size_t>(3) * len);
                const std::size_t prompt_tokens = multimodal->token_ids.size();
                for (int axis = 0; axis < 3; ++axis) {
                    const auto* src = multimodal->positions.data() +
                                      static_cast<std::size_t>(axis) * prompt_tokens + prompt_t0;
                    std::copy_n(src, len,
                                rope_positions_host.data() + static_cast<std::size_t>(axis) * len);
                }
                copy_i32(rope_positions_host.data(), rope_positions, s);
            } else if (rope_delta_ != 0) {
                rope_positions = roots.rope_positions;
                ops::offset_i32_positions(positions, io_.rope_delta, rope_positions, s);
            }
            ScopedPositions scoped_cache(active_cache_positions_, positions);
            ScopedPositions scoped_rope(active_rope_positions_, rope_positions);
            const auto visible = static_cast<std::uint32_t>(base_i + t0 + len);
            const ops::CausalAttentionExecutionEnvelope chunk_envelope{visible, visible};
            ScopedEnvelope scoped_envelope(active_causal_attention_envelope_, chunk_envelope);

            Tensor x = roots.residual;
            ops::embedding(ids_device, *embed_, x, s);
            if (!local_scatter_indices.empty()) {
                Tensor indices_device = roots.scatter_indices;
                copy_i32(local_scatter_indices.data(), indices_device, s);
                Tensor embeddings = vision_chunk.embeddings.slice(
                    1, visual_begin, static_cast<std::int32_t>(local_scatter_indices.size()));
                ops::scatter(embeddings, indices_device, x, s);
            }
            if constexpr (Tap::enabled) { tap.begin(x); }
            run_layers(x, Phase::Prefill, tap);
            if constexpr (requires { tap.capture_positions(positions, s); }) {
                tap.capture_positions(positions, s);
            }

            Tensor xf = prefill_hidden_.data != nullptr
                            ? matrix_window(prefill_hidden_, len)
                            : work_.alloc(DType::BF16, {dimension(config_.hidden_size), len});
            ops::rmsnorm(x, *final_norm_, config_.rms_norm_eps, true, xf, s);

            if (is_last) {
                Tensor last_xf = xf.slice(1, len - 1, 1);
                Tensor logits  = matrix_window(io_.logits, 1);
                project(last_xf, *lm_head_, logits, work_, s);
                // Set io_.pos to the bonus token's absolute position (base + T) before picking so
                // the sampler RNG is keyed by it (prefill purpose keeps it distinct from the first
                // decode step, which reuses the same io_.pos).
                ops::set_i32_scalar(io_.pos, base_i + T, s);
                ops::set_i32_scalar(io_.rope_pos, base_i + T + rope_delta_, s);
                if (sampling_config_ != nullptr) {
                    ops::sample(logits, io_.token,
                                dimension(parameters_.model.resources().public_token_count),
                                sampling_config_, io_.pos, ops::kSamplePurposePrefill, work_, s);
                } else {
                    ops::argmax(logits, io_.token,
                                dimension(parameters_.model.resources().public_token_count), s);
                }
            }

            if (prepare_mtp_prompt) {
                const std::uint32_t alignment_tokens =
                    multimodal != nullptr ? static_cast<std::uint32_t>(multimodal->token_ids.size())
                    : text_prefill != nullptr
                        ? static_cast<std::uint32_t>(text_prefill->token_ids.size())
                        : static_cast<std::uint32_t>(T);
                const std::uint32_t alignment_begin =
                    multimodal != nullptr || text_prefill != nullptr
                        ? prompt_t0
                        : static_cast<std::uint32_t>(t0);
                const qwen3_5::MtpAlignmentWindow mtp_window = qwen3_5::plan_mtp_alignment_window(
                    alignment_tokens, alignment_begin, static_cast<std::uint32_t>(len));
                const std::span<const int> alignment_ids =
                    multimodal != nullptr     ? multimodal->token_ids
                    : text_prefill != nullptr ? text_prefill->token_ids
                                              : ids;
                const int prompt_columns =
                    len - static_cast<int>(mtp_window.final_column_uses_generated_token);
                Tensor mtp_ids = work_.alloc(DType::I32, {len});
                if (prompt_columns != 0) {
                    Tensor prompt_mtp_ids = mtp_ids.slice(0, 0, prompt_columns);
                    copy_i32(alignment_ids.data() + mtp_window.shifted_embedding_begin,
                             prompt_mtp_ids, s);
                }
                if (mtp_window.final_column_uses_generated_token) {
                    Tensor generated_mtp_id = mtp_ids.slice(0, len - 1, 1);
                    CUDA_CHECK(cudaMemcpyAsync(generated_mtp_id.data, io_.token.data,
                                               sizeof(std::int32_t), cudaMemcpyDeviceToDevice, s));
                }

                Tensor mtp_input_embeddings;
                const Tensor* mtp_input_embeddings_ptr = nullptr;
                if (multimodal != nullptr) {
                    mtp_input_embeddings =
                        work_.alloc(DType::BF16, {dimension(config_.hidden_size), len});
                    ops::embedding(mtp_ids, *embed_, mtp_input_embeddings, s);
                    if (vision_chunk.control != nullptr) {
                        const qwen3_5::MtpVisualOverlap overlap = qwen3_5::shifted_visual_overlap(
                            vision_chunk.control->scatter_indices, alignment_tokens, mtp_window);
                        if (!overlap.empty()) {
                            Tensor shifted_indices = workspace::visual_scatter_indices(
                                work_, static_cast<std::int32_t>(overlap.size()));
                            qwen3_5::detail::scatter_shifted_visual_embeddings(
                                mtp_input_embeddings, vision_chunk.embeddings, overlap,
                                shifted_indices, s);
                        }
                    }
                    mtp_input_embeddings_ptr = &mtp_input_embeddings;
                }
                if (is_last && mtp_proposal_extent_ != 0) {
                    Tensor logits = matrix_window(io_.logits, 1);
                    Tensor draft0 = io_.mtp->draft_tokens.slice(0, 0, 1);
                    mtp_prefill_chunk(mtp_ids, xf, mtp_input_embeddings_ptr, positions,
                                      rope_positions, chunk_envelope, true, &io_.mtp->ar_hidden,
                                      &logits, &draft0);

                    Tensor ar_position = io_.mtp->position.slice(0, 0, 1);
                    ops::set_i32_scalar(ar_position, base_i + T, s);
                    for (int i = 1; i < static_cast<int>(mtp_proposal_extent_); ++i) {
                        Tensor prev_token = io_.mtp->draft_tokens.slice(0, i - 1, 1);
                        Tensor next_token = io_.mtp->draft_tokens.slice(0, i, 1);
                        Tensor next_hidden =
                            work_.alloc(DType::BF16, {dimension(config_.hidden_size), 1});
                        const auto ar_visible = static_cast<std::uint32_t>(base_i + T + i);
                        const ops::CausalAttentionExecutionEnvelope ar_envelope{ar_visible,
                                                                                ar_visible};
                        mtp_forward_ar_step(prev_token, io_.mtp->ar_hidden, ar_position,
                                            ar_envelope, next_hidden, logits, next_token);
                        CUDA_CHECK(cudaMemcpyAsync(io_.mtp->ar_hidden.data, next_hidden.data,
                                                   io_.mtp->ar_hidden.bytes(),
                                                   cudaMemcpyDeviceToDevice, s));
                        ops::increment_i32_scalar(ar_position, s);
                    }
                } else {
                    mtp_prefill_chunk(mtp_ids, xf, mtp_input_embeddings_ptr, positions,
                                      rope_positions, chunk_envelope, false, nullptr, nullptr,
                                      nullptr);
                }
            }

            if (split_rel > 0 && t0 + len == split_rel &&
                rewrite_checkpoint_hidden_output_ != nullptr) {
                require_tensor_shape(*rewrite_checkpoint_hidden_output_, DType::BF16,
                                     {dimension(config_.hidden_size), 1},
                                     "rewrite checkpoint hidden output");
                const Tensor checkpoint_hidden = xf.slice(1, len - 1, 1);
                CUDA_CHECK(cudaMemcpyAsync(rewrite_checkpoint_hidden_output_->data,
                                           checkpoint_hidden.data, checkpoint_hidden.bytes(),
                                           cudaMemcpyDeviceToDevice, s));
            }
        }

        if constexpr (requires { tap.consume_prefill_chunk(len, false); }) {
            work_.reset();
            tap.consume_prefill_chunk(len, split_rel > 0 && t0 + len == split_rel);
        }

        t0 += len;
        break;
    }

    prefill_split_frontier_ = -1;

    timing.begin_wait();
    ctx_.synchronize();
    timing.end_wait();
    work_.reset();
    return PrefillChunkResult{.processed_tokens = static_cast<std::uint32_t>(t0),
                              .finalized        = finalize_at_end && t0 == T,
                              .timing           = timing.finish()};
}

PrefillChunkResult TextContext::prefill_chunk(std::span<const int> full_ids, std::uint32_t begin,
                                              std::uint32_t nominal_length, bool finalize_at_end) {
    if (begin >= full_ids.size() || nominal_length == 0 ||
        nominal_length > full_ids.size() - begin) {
        throw std::invalid_argument("text prefill chunk is outside the prompt");
    }
    const TextPrefill text_prefill{full_ids, begin};
    NullTap tap;
    return prefill_impl(full_ids.subspan(begin, nominal_length), &text_prefill, nullptr, tap,
                        finalize_at_end);
}

PrefillChunkResult TextContext::prefill_chunk(std::span<const int> full_ids, std::uint32_t begin,
                                              std::uint32_t nominal_length, bool finalize_at_end,
                                              DFlashFeatureSink& sink) {
    if (begin >= full_ids.size() || nominal_length == 0 ||
        nominal_length > full_ids.size() - begin) {
        throw std::invalid_argument("text prefill chunk is outside the prompt");
    }
    const TextPrefill text_prefill{full_ids, begin};
    return prefill_impl(full_ids.subspan(begin, nominal_length), &text_prefill, nullptr, sink,
                        finalize_at_end);
}

PrefillChunkResult TextContext::prefill_chunk(const qwen3_5::PreparedPromptData& input,
                                              std::uint32_t begin, std::uint32_t nominal_length,
                                              VisionPrefillSession& vision, bool finalize_at_end) {
    if (begin >= input.token_ids.size() || nominal_length == 0 ||
        nominal_length > input.token_ids.size() - begin) {
        throw std::invalid_argument("multimodal prefill chunk is outside the prompt");
    }
    const std::span<const int> tokens(input.token_ids);
    const MultimodalPrefill multimodal{tokens, input.positions, &vision, begin, input.rope_delta};
    NullTap tap;
    return prefill_impl(tokens.subspan(begin, nominal_length), nullptr, &multimodal, tap,
                        finalize_at_end);
}

PrefillChunkResult TextContext::prefill_chunk(const qwen3_5::PreparedPromptData& input,
                                              std::uint32_t begin, std::uint32_t nominal_length,
                                              VisionPrefillSession& vision, bool finalize_at_end,
                                              DFlashFeatureSink& sink) {
    if (begin >= input.token_ids.size() || nominal_length == 0 ||
        nominal_length > input.token_ids.size() - begin) {
        throw std::invalid_argument("multimodal prefill chunk is outside the prompt");
    }
    const std::span<const int> tokens(input.token_ids);
    const MultimodalPrefill multimodal{tokens, input.positions, &vision, begin, input.rope_delta};
    return prefill_impl(tokens.subspan(begin, nominal_length), nullptr, &multimodal, sink,
                        finalize_at_end);
}

} // namespace ninfer::models::qwen3_5::execution
