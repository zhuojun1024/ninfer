#include "models/qwen3_5/frontend/tool_call_constraint.h"

#include <stdexcept>
#include <utility>

namespace ninfer::models::qwen3_5::frontend {
namespace {

constexpr bool is_format_whitespace(unsigned char byte) {
    return byte == ' ' || byte == '\t' || byte == '\r' || byte == '\n';
}

} // namespace

ToolCallMaskTable build_tool_call_mask_table(std::shared_ptr<const Tokenizer> tokenizer) {
    if (tokenizer == nullptr) {
        throw std::invalid_argument("tool-call mask table requires a tokenizer");
    }
    ToolCallMaskTable table;
    table.owner             = std::move(tokenizer);
    const std::size_t vocab = table.owner->vocab_size();
    table.pieces.reserve(vocab);
    table.special.reserve(vocab);
    for (std::size_t index = 0; index < vocab; ++index) {
        const int id = static_cast<int>(index);
        if (table.owner->is_valid_token(id)) {
            const DecodedTokenView view = table.owner->decoded_token(id);
            table.pieces.push_back(view.bytes);
            table.special.push_back(view.special ? std::uint8_t{1} : std::uint8_t{0});
        } else {
            // A packed logits domain can expose rows the checkpoint does not define; no
            // constrained position may ever select one of them.
            table.pieces.push_back(std::string_view{});
            table.special.push_back(1);
        }
    }
    return table;
}

void ToolCallNameTrie::build(std::span<const std::string> names) {
    nodes_.clear();
    nodes_.emplace_back();
    for (auto& edge : nodes_[0].next) { edge = kNoNode; }
    for (std::size_t index = 0; index < names.size(); ++index) {
        std::uint32_t node = 0;
        for (const char raw : names[index]) {
            const auto byte      = static_cast<unsigned char>(raw);
            std::uint32_t child  = nodes_[node].next[byte];
            if (child == kNoNode) {
                child = static_cast<std::uint32_t>(nodes_.size());
                nodes_.emplace_back();
                for (auto& edge : nodes_.back().next) { edge = kNoNode; }
                nodes_[node].next[byte] = child;
            }
            node = child;
        }
        nodes_[node].terminal = true;
        nodes_[node].payload  = static_cast<std::uint32_t>(index);
    }
}

std::uint32_t ToolCallNameTrie::step(std::uint32_t node, unsigned char byte) const noexcept {
    if (node >= nodes_.size()) { return kNoNode; }
    return nodes_[node].next[byte];
}

bool ToolCallNameTrie::terminal(std::uint32_t node) const noexcept {
    return node < nodes_.size() && nodes_[node].terminal;
}

std::uint32_t ToolCallNameTrie::payload(std::uint32_t node) const noexcept {
    return node < nodes_.size() ? nodes_[node].payload : kNoNode;
}

ToolCallGrammar::ToolCallGrammar(const ToolCallOutputContract& contract) {
    std::vector<std::string> names;
    names.reserve(contract.tools.size());
    for (const ToolCallOutputContract::Tool& tool : contract.tools) { names.push_back(tool.name); }
    tools_.build(names);

    parameters_.resize(contract.tools.size());
    for (std::size_t index = 0; index < contract.tools.size(); ++index) {
        std::vector<std::string> parameter_names;
        parameter_names.reserve(contract.tools[index].parameters.size());
        for (const ToolCallOutputContract::Parameter& parameter :
             contract.tools[index].parameters) {
            parameter_names.push_back(parameter.name);
        }
        parameters_[index].build(parameter_names);
    }
}

void ToolCallGrammarState::feed(const ToolCallGrammar& grammar, std::string_view bytes) {
    for (const char raw : bytes) { feed_byte(grammar, static_cast<unsigned char>(raw)); }
}

bool ToolCallGrammarState::constrained() const noexcept {
    return mode_ == Mode::FunctionName || mode_ == Mode::ParameterName;
}

bool ToolCallGrammarState::build_mask(const ToolCallGrammar& grammar, const ToolCallMaskTable& table,
                                      std::size_t domain, std::vector<std::uint8_t>& mask) const {
    if (domain < table.size()) {
        throw std::invalid_argument("tool-call mask domain is narrower than the mask table");
    }
    switch (mode_) {
    case Mode::FunctionLiteral:
    case Mode::FunctionName:
    case Mode::FunctionClose:
    case Mode::ParameterName:
    case Mode::ToolClose:
        break;
    // Free text and an open parameter value admit every token, so they carry no mask. A value is
    // left to the parser in this phase, and Free is where a tool call ends and prose resumes.
    default:
        return false;
    }

    // A candidate token is legal exactly when feeding its decoded bytes leaves the grammar on a
    // literal or name path. One state copy per candidate is what makes a name that spans several
    // tokens, a half-decoded code point, or a token that straddles a structural literal all fall
    // out of the same check instead of needing separate cases.
    mask.assign(domain, 0);
    bool advances = false;
    for (std::size_t id = 0; id < table.size(); ++id) {
        if (table.special[id] != 0) { continue; }
        const std::string_view piece = table.pieces[id];
        if (piece.empty()) { continue; }
        ToolCallGrammarState probe = *this;
        probe.dead_                = false;
        probe.feed(grammar, piece);
        if (probe.dead_) { continue; }
        mask[id] = 1;
        for (const char byte : piece) {
            if (!is_format_whitespace(static_cast<unsigned char>(byte))) {
                advances = true;
                break;
            }
        }
    }
    // Whitespace is skipped wherever it is legal, so a position that admits nothing but whitespace
    // can never reach a literal or a declared name; masking to it would let the model emit spaces
    // forever. Every other rejected position would strand the request the same way. Reporting both
    // as unconstrained keeps generation alive and leaves the malformed text to the parser.
    return advances;
}

void ToolCallGrammarState::feed_byte(const ToolCallGrammar& grammar, unsigned char byte) {
    switch (mode_) {
    case Mode::Free: {
        // Rolling match of the lazy trigger. The trigger has no self-overlapping prefix, so a
        // mismatch restarts at the current byte.
        if (trigger_ != 0) {
            if (byte == static_cast<unsigned char>(kToolOpen[trigger_])) {
                if (++trigger_ == kToolOpen.size()) {
                    trigger_  = 0;
                    progress_ = 0;
                    dead_     = false;
                    mode_     = Mode::FunctionLiteral;
                }
                return;
            }
            trigger_ = 0;
        }
        if (byte == static_cast<unsigned char>(kToolOpen[0])) { trigger_ = 1; }
        return;
    }
    case Mode::FunctionLiteral: {
        if (progress_ == 0 && is_format_whitespace(byte)) { return; }
        if (byte == static_cast<unsigned char>(kFunctionOpen[progress_])) {
            if (++progress_ == kFunctionOpen.size()) {
                progress_ = 0;
                node_     = grammar.tool_names().root();
                mode_     = Mode::FunctionName;
            }
            return;
        }
        trigger_ = 0;
        dead_    = true;
        mode_    = Mode::Free;
        return;
    }
    case Mode::FunctionName: {
        const ToolCallNameTrie& trie = grammar.tool_names();
        if (trie.terminal(node_) && byte == static_cast<unsigned char>('>')) {
            tool_     = trie.payload(node_);
            progress_ = 0;
            alive_    = 0x03u;
            mode_     = Mode::FunctionClose;
            return;
        }
        const std::uint32_t next = trie.step(node_, byte);
        if (next == ToolCallNameTrie::kNoNode) {
            // The name left the declared set. Nothing is left to constrain for this call; later
            // bytes may still open a fresh one.
            trigger_ = 0;
            dead_    = true;
            mode_    = Mode::Free;
            return;
        }
        node_ = next;
        return;
    }
    case Mode::FunctionClose: {
        if (progress_ == 0 && is_format_whitespace(byte)) { return; }
        std::uint8_t alive = 0;
        if ((alive_ & 0x01u) != 0 && progress_ < kFunctionClose.size() &&
            byte == static_cast<unsigned char>(kFunctionClose[progress_])) {
            alive |= 0x01u;
        }
        if ((alive_ & 0x02u) != 0 && progress_ < kParamOpen.size() &&
            byte == static_cast<unsigned char>(kParamOpen[progress_])) {
            alive |= 0x02u;
        }
        alive_ = alive;
        if (alive == 0) {
            trigger_ = 0;
            dead_    = true;
            mode_    = Mode::Free;
            return;
        }
        ++progress_;
        if (progress_ == kFunctionClose.size() && (alive_ & 0x01u) != 0) {
            alive_    = 0;
            progress_ = 0;
            mode_     = Mode::ToolClose;
            return;
        }
        if (progress_ == kParamOpen.size() && (alive_ & 0x02u) != 0) {
            alive_    = 0;
            progress_ = 0;
            value_    = 0;
            if (grammar.parameter_names(tool_).empty()) {
                // The selected tool declares no parameters, so this marker can never complete.
                trigger_ = 0;
                dead_    = true;
                mode_    = Mode::Free;
                return;
            }
            node_ = grammar.parameter_names(tool_).root();
            mode_ = Mode::ParameterName;
            return;
        }
        return;
    }
    case Mode::ParameterName: {
        const ToolCallNameTrie& trie = grammar.parameter_names(tool_);
        if (trie.terminal(node_) && byte == static_cast<unsigned char>('>')) {
            value_ = 0;
            mode_  = Mode::ParameterValue;
            return;
        }
        const std::uint32_t next = trie.step(node_, byte);
        if (next == ToolCallNameTrie::kNoNode) {
            trigger_ = 0;
            dead_    = true;
            mode_    = Mode::Free;
            return;
        }
        node_ = next;
        return;
    }
    case Mode::ParameterValue: {
        // Free text while the value is open; only the closing marker is tracked. A nested
        // parameter marker inside a value is not modelled, because the value itself is left to
        // the parser in this phase.
        if (byte == static_cast<unsigned char>(kParamClose[value_])) {
            if (++value_ == kParamClose.size()) {
                value_    = 0;
                progress_ = 0;
                alive_    = 0x03u;
                mode_     = Mode::FunctionClose;
            }
            return;
        }
        value_ = 0;
        if (byte == static_cast<unsigned char>(kParamClose[0])) { value_ = 1; }
        return;
    }
    case Mode::ToolClose: {
        if (progress_ == 0 && is_format_whitespace(byte)) { return; }
        if (byte == static_cast<unsigned char>(kToolClose[progress_])) {
            if (++progress_ == kToolClose.size()) {
                progress_ = 0;
                trigger_  = 0;
                mode_     = Mode::Free;
            }
            return;
        }
        trigger_ = 0;
        dead_    = true;
        mode_    = Mode::Free;
        return;
    }
    }
}

ToolCallConstraint::ToolCallConstraint(std::shared_ptr<const ToolCallMaskTable> table,
                                       const ToolCallOutputContract& contract)
    : table_(std::move(table)), grammar_(contract) {
    if (table_ == nullptr) {
        throw std::invalid_argument("tool-call constraint requires a vocabulary mask table");
    }
}

bool ToolCallConstraint::build_mask_after(std::string_view prefix, std::size_t domain,
                                          std::vector<std::uint8_t>& mask) const {
    ToolCallGrammarState state = state_;
    state.feed(grammar_, prefix);
    return state.build_mask(grammar_, *table_, domain, mask);
}

} // namespace ninfer::models::qwen3_5::frontend
