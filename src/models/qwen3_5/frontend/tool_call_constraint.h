#pragma once

#include "models/qwen3_5/frontend/tokenizer.h"
#include "models/qwen3_5/frontend/tool_call_parser.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace ninfer::models::qwen3_5::frontend {

// One decoded byte piece per vocabulary id plus its special-token flag. The views point into the
// owning tokenizer, which must outlive the table. It is built once per Frontend and shared by every
// constrained request, so a constrained step pays no per-request vocabulary walk to build it.
struct ToolCallMaskTable {
    // Keeps the decoded bytes the piece views point into alive for the table's whole lifetime.
    std::shared_ptr<const Tokenizer> owner;
    std::vector<std::string_view> pieces;
    std::vector<std::uint8_t> special;

    [[nodiscard]] std::size_t size() const noexcept { return pieces.size(); }
};

// Builds the shared piece table, taking shared ownership of the tokenizer it reads.
[[nodiscard]] ToolCallMaskTable build_tool_call_mask_table(std::shared_ptr<const Tokenizer> tokenizer);

// Byte trie over a set of declared names. Node 0 is the root and kNoNode marks a missing edge.
class ToolCallNameTrie {
public:
    static constexpr std::uint32_t kNoNode = 0xffffffffu;

    void build(std::span<const std::string> names);

    [[nodiscard]] std::uint32_t root() const noexcept { return 0; }
    [[nodiscard]] std::uint32_t step(std::uint32_t node, unsigned char byte) const noexcept;
    [[nodiscard]] bool terminal(std::uint32_t node) const noexcept;
    // Index into the built name list of the name that ends here, or kNoNode.
    [[nodiscard]] std::uint32_t payload(std::uint32_t node) const noexcept;
    [[nodiscard]] bool empty() const noexcept { return nodes_.size() <= 1; }

private:
    struct Node {
        std::array<std::uint32_t, 256> next{};
        std::uint32_t payload = kNoNode;
        bool terminal         = false;
    };
    std::vector<Node> nodes_;
};

// Immutable declared-name grammar of one request's tool contract: the tool names and, per tool,
// the parameter names extracted from its JSON schema.
class ToolCallGrammar {
public:
    explicit ToolCallGrammar(const ToolCallOutputContract& contract);

    [[nodiscard]] const ToolCallNameTrie& tool_names() const noexcept { return tools_; }
    [[nodiscard]] const ToolCallNameTrie& parameter_names(std::uint32_t tool) const noexcept {
        return parameters_[tool];
    }
    [[nodiscard]] std::size_t tool_count() const noexcept { return parameters_.size(); }

private:
    ToolCallNameTrie tools_;
    std::vector<ToolCallNameTrie> parameters_;
};

// Byte-exact position inside the Qwen tool-call syntax. The grammar is fed the same content-channel
// byte stream the tool-call parser sees, so a name that spans several tokens, a partially decoded
// UTF-8 code point, or a token that straddles a structural literal cannot be misclassified.
class ToolCallGrammarState {
public:
    enum class Mode : std::uint8_t {
        Free,
        FunctionLiteral,
        FunctionName,
        FunctionClose,
        ParameterName,
        ParameterValue,
        ToolClose,
    };

    void feed(const ToolCallGrammar& grammar, std::string_view bytes);

    // True when the current position admits only declared names, with mask filled one byte per logits
    // row (1 = allowed). The logits domain is the packed embedding row count, which can be wider than
    // the tokenizer public domain; rows the tokenizer does not define are always excluded. A position
    // that can no longer advance is reported as unconstrained instead, because masking to it would
    // strand the request; the malformed text is then left to the parser.
    [[nodiscard]] bool build_mask(const ToolCallGrammar& grammar, const ToolCallMaskTable& table,
                                  std::size_t domain, std::vector<std::uint8_t>& mask) const;

    [[nodiscard]] Mode mode() const noexcept { return mode_; }
    [[nodiscard]] bool constrained() const noexcept;

private:
    void feed_byte(const ToolCallGrammar& grammar, unsigned char byte);

    Mode mode_             = Mode::Free;
    // Set when a required literal or name byte left the grammar. It is only read on the probe copy
    // build_mask feeds a candidate token into; the live state stops constraining once it is set.
    bool dead_             = false;
    std::uint8_t trigger_  = 0;
    std::uint8_t progress_ = 0;
    std::uint8_t alive_    = 0;
    std::uint8_t value_    = 0;
    std::uint32_t node_    = 0;
    std::uint32_t tool_    = 0;
};

// One request's constrained-decoding state: the shared vocabulary table, the immutable grammar,
// and the mutable position the decode loop advances with every committed content byte.
class ToolCallConstraint {
public:
    ToolCallConstraint(std::shared_ptr<const ToolCallMaskTable> table,
                       const ToolCallOutputContract& contract);

    void feed(std::string_view bytes) { state_.feed(grammar_, bytes); }

    [[nodiscard]] bool build_mask(std::size_t domain, std::vector<std::uint8_t>& mask) const {
        return state_.build_mask(grammar_, *table_, domain, mask);
    }

    // The mask for one speculative verify column, which sees the drafts of the columns before it on
    // top of the committed text. Columns that follow an accepted draft prefix therefore evaluate the
    // exact grammar position their candidate token is drawn from.
    [[nodiscard]] bool build_mask_after(std::string_view prefix, std::size_t domain,
                                        std::vector<std::uint8_t>& mask) const;

    [[nodiscard]] const ToolCallGrammar& grammar() const noexcept { return grammar_; }
    [[nodiscard]] std::size_t vocab_size() const noexcept { return table_->size(); }
    // Decoded bytes of one vocabulary id, used to extend a speculative column from the drafts that
    // precede it.
    [[nodiscard]] std::string_view piece(std::size_t id) const noexcept {
        return table_->pieces[id];
    }

private:
    std::shared_ptr<const ToolCallMaskTable> table_;
    ToolCallGrammar grammar_;
    ToolCallGrammarState state_;
};

} // namespace ninfer::models::qwen3_5::frontend
