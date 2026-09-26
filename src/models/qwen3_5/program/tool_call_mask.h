#pragma once

#include "models/qwen3_5/frontend/output_session.h"
#include "models/qwen3_5/frontend/tool_call_constraint.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string_view>
#include <utility>
#include <vector>

namespace ninfer::models::qwen3_5::execution {

// One request's constrained-decoding binding. It couples the engine-owned OutputSession, which is
// the only place that exposes the committed Content byte stream and the reasoning/content split,
// with the immutable declared-name grammar. The Program holds only a raw pointer to it: the Engine
// keeps the object inside the request record for the whole request lifetime, so the binding stays
// valid exactly while its lane is active.
class ToolCallMask {
public:
    ToolCallMask(std::shared_ptr<frontend::ToolCallConstraint> constraint,
                 const OutputSession& output) noexcept
        : constraint_(std::move(constraint)), output_(&output) {}

    // The mask binds only while the committed output is content: the tool parser reads no reasoning
    // channel, so the grammar must not constrain the thinking phase either.
    [[nodiscard]] bool live() const noexcept {
        return constraint_ != nullptr && !output_->in_reasoning();
    }

    // Feeds the committed content bytes the grammar has not seen yet. The watermark makes this
    // idempotent across rounds, so a round that never samples still leaves the same position.
    void advance() {
        const std::string_view raw = output_->raw_content_text();
        if (raw.size() > fed_) {
            constraint_->feed(raw.substr(fed_));
            fed_ = raw.size();
        }
    }

    // The single-column mask for one sample position, built from the committed text alone.
    [[nodiscard]] bool build_single(std::size_t domain, std::vector<std::uint8_t>& mask) const {
        if (!representable(domain)) { return false; }
        return constraint_->build_mask(domain, mask);
    }

    // The mask for one speculative verify column, which sees the drafts of the earlier columns on
    // top of the committed text. A column that follows an accepted draft prefix therefore evaluates
    // the exact grammar position its candidate token is drawn from.
    [[nodiscard]] bool build_after(std::string_view prefix, std::size_t domain,
                                   std::vector<std::uint8_t>& mask) const {
        if (!representable(domain)) { return false; }
        return constraint_->build_mask_after(prefix, domain, mask);
    }

    // Decoded bytes of one vocabulary id, used to extend a speculative column from the drafts that
    // precede it.
    [[nodiscard]] std::string_view piece(std::size_t id) const noexcept {
        return id < constraint_->vocab_size() ? constraint_->piece(id) : std::string_view{};
    }

private:
    // A table wider than the packed logits domain means the artifact and tokenizer disagree. The
    // grammar is then not representable in this request's logits, so it degrades to unconstrained
    // instead of stranding the request the way the table's own domain check would.
    [[nodiscard]] bool representable(std::size_t domain) const noexcept {
        return constraint_->vocab_size() <= domain;
    }

    std::shared_ptr<frontend::ToolCallConstraint> constraint_;
    const OutputSession* output_ = nullptr;
    std::size_t fed_             = 0;
};

} // namespace ninfer::models::qwen3_5::execution
