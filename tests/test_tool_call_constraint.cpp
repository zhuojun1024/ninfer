// Declared-name constrained decoding: the byte automaton must stay on the declared tool and
// parameter names, and must leave free text and open parameter values alone.
#include "models/qwen3_5/frontend/tool_call_constraint.h"

#include <cstddef>
#include <cstdint>
#include <deque>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace fi = ninfer::models::qwen3_5::frontend;

namespace {

int failures = 0;

void check(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

// Owns the piece strings a synthetic vocabulary table points into.
struct FakeVocabulary {
    std::deque<std::string> owned;
    fi::ToolCallMaskTable table;

    std::size_t add(std::string text, bool special = false) {
        owned.push_back(std::move(text));
        table.pieces.push_back(owned.back());
        table.special.push_back(special ? std::uint8_t{1} : std::uint8_t{0});
        return table.pieces.size() - 1;
    }
};

fi::ToolCallOutputContract two_tool_contract() {
    fi::ToolCallOutputContract contract;
    contract.enforce_declared_names = true;
    const auto add_tool = [&contract](std::string name, std::vector<std::string> parameters) {
        fi::ToolCallOutputContract::Tool tool;
        tool.name = std::move(name);
        for (std::string& parameter : parameters) {
            tool.parameters.push_back(
                fi::ToolCallOutputContract::Parameter{.name = std::move(parameter)});
        }
        contract.tools.push_back(std::move(tool));
    };
    add_tool("run_code", {"code", "timeout"});
    add_tool("todo_write", {"todos"});
    return contract;
}

std::vector<std::size_t> allowed_ids(const std::vector<std::uint8_t>& mask) {
    std::vector<std::size_t> ids;
    for (std::size_t index = 0; index < mask.size(); ++index) {
        if (mask[index] != 0) { ids.push_back(index); }
    }
    return ids;
}

bool allows(const std::vector<std::uint8_t>& mask, std::size_t id) {
    return id < mask.size() && mask[id] != 0;
}

} // namespace

int main() {
    FakeVocabulary vocabulary;
    const std::size_t run              = vocabulary.add("run");
    const std::size_t underscore_code  = vocabulary.add("_code");
    const std::size_t run_code         = vocabulary.add("run_code");
    const std::size_t todo             = vocabulary.add("todo");
    const std::size_t underscore_write = vocabulary.add("_write");
    const std::size_t todo_write       = vocabulary.add("todo_write");
    const std::size_t tool             = vocabulary.add("tool");
    const std::size_t greater          = vocabulary.add(">");
    const std::size_t eog              = vocabulary.add("<|im_end|>", true);
    const std::size_t code             = vocabulary.add("code");
    const std::size_t timeout          = vocabulary.add("timeout");
    const std::size_t todos            = vocabulary.add("todos");
    const std::size_t function_open    = vocabulary.add("<function=");
    const std::size_t function_close   = vocabulary.add("</function>");
    const std::size_t parameter_open   = vocabulary.add("<parameter=");
    const std::size_t parameter_close  = vocabulary.add("</parameter>");
    const std::size_t tool_open        = vocabulary.add("<tool_call>");
    const std::size_t tool_close       = vocabulary.add("</tool_call>");
    const std::size_t newline          = vocabulary.add("\n");
    const std::size_t equals_to        = vocabulary.add("=to");
    const std::size_t equals_tool      = vocabulary.add("=tool");

    const auto table = std::make_shared<const fi::ToolCallMaskTable>(vocabulary.table);
    const fi::ToolCallOutputContract contract = two_tool_contract();
    fi::ToolCallConstraint constraint(table, contract);
    std::vector<std::uint8_t> mask;
    // The logits domain here is the vocabulary itself; the packed wider domain has its own case below.
    const std::size_t domain = vocabulary.table.size();

    // Lazy trigger: nothing is constrained until <tool_call> completes, then the closing literal
    // must follow.
    constraint.feed("<tool");
    check(!constraint.build_mask(domain, mask), "pre-trigger text must stay unconstrained");
    constraint.feed("_call>");
    check(constraint.build_mask(domain, mask), "the function literal must be constrained");
    check(allows(mask, function_open) && allows(mask, newline),
          "only the <function= literal or whitespace may follow <tool_call>");
    check(!allows(mask, tool_open) && !allows(mask, run) && !allows(mask, tool),
          "nothing but the function marker may follow the trigger");

    constraint.feed("\n<function=");
    check(constraint.build_mask(domain, mask), "the function name must be constrained");
    check(allowed_ids(mask) == std::vector<std::size_t>({run, run_code, todo, todo_write}),
          "only declared tool names may start the function name");
    check(!allows(mask, eog), "the end-of-generation token must never be allowed inside a call");

    // A name that spans several tokens stays constrained token by token.
    constraint.feed("run");
    check(constraint.build_mask(domain, mask), "a partial name must stay constrained");
    check(allows(mask, underscore_code) && !allows(mask, greater) &&
              !allows(mask, underscore_write),
          "only continuations of run_code may extend the partial name");

    // Once the name is complete, the closing marker is the only legal continuation.
    constraint.feed("_code");
    check(constraint.build_mask(domain, mask), "a complete name must stay constrained");
    check(allowed_ids(mask) == std::vector<std::size_t>({greater}),
          "a complete name admits only the closing marker");

    constraint.feed(">\n<parameter=");
    check(constraint.build_mask(domain, mask), "the parameter name must be constrained");
    check(allowed_ids(mask) == std::vector<std::size_t>({code, timeout}),
          "only parameters declared by the selected tool may start the parameter name");
    check(!allows(mask, todos), "another tool parameter must not leak into this tool");

    // The value is free, and only the closing marker is tracked.
    constraint.feed("code>\nprint(1)\n");
    check(!constraint.build_mask(domain, mask), "an open parameter value must stay unconstrained");

    // The framing literals are constrained too, so a call always closes the way the parser reads it.
    constraint.feed("</parameter>");
    check(constraint.build_mask(domain, mask), "the parameter close must be constrained");
    check(allows(mask, parameter_open) && allows(mask, function_close),
          "only another parameter or the function close may follow a value");
    check(!allows(mask, parameter_close), "a stray parameter close must be rejected");

    constraint.feed("\n</function>");
    check(constraint.build_mask(domain, mask), "the tool close must be constrained");
    check(allows(mask, tool_close) && !allows(mask, function_close),
          "only the tool close may follow the function close");

    constraint.feed("\n</tool_call>");
    check(!constraint.build_mask(domain, mask), "a finished call must return to free text");

    // A token that straddles the structural literal into an undeclared name is rejected while the
    // literal is still being matched, which is the case a token-level prefix tree cannot see.
    fi::ToolCallConstraint straddle(table, contract);
    straddle.feed("<tool_call>\n<function");
    check(straddle.build_mask(domain, mask), "a partial function literal must be constrained");
    check(allows(mask, equals_to) && !allows(mask, equals_tool),
          "a literal-straddling token may only enter a declared name");

    // An off-track name leaves the position unconstrained instead of stranding the request.
    fi::ToolCallConstraint off_track(table, contract);
    off_track.feed("<tool_call>\n<function=xx");
    check(!off_track.build_mask(domain, mask), "an undeclared name prefix must stay unconstrained");

    // A grammar whose name has no available piece reports unconstrained rather than an empty mask.
    fi::ToolCallOutputContract missing = two_tool_contract();
    missing.tools.front().name         = "zzz_absent";
    missing.tools.resize(1);
    fi::ToolCallConstraint absent(table, missing);
    absent.feed("<tool_call>\n<function=");
    check(!absent.build_mask(domain, mask),
          "a name the vocabulary cannot spell must not produce an empty mask");

    // Whitespace never advances the grammar, so a position that admits nothing else is a dead end
    // too: masking to it would let the model emit spaces until the context runs out.
    FakeVocabulary whitespace_only;
    whitespace_only.add("\n");
    whitespace_only.add(" ");
    whitespace_only.add("run");
    const auto narrow_table = std::make_shared<const fi::ToolCallMaskTable>(whitespace_only.table);
    fi::ToolCallConstraint narrow(narrow_table, contract);
    narrow.feed("<tool_call>");
    check(!narrow.build_mask(narrow_table->size(), mask),
          "a position that admits only whitespace must stay unconstrained");

    // The logits domain is the packed embedding row count, so it can be wider than the tokenizer
    // public domain the table covers. Rows in between carry no vocabulary piece and must never win.
    const std::size_t packed_domain = domain + 8;
    std::vector<std::uint8_t> wide(packed_domain, 1);
    fi::ToolCallConstraint packed(table, contract);
    packed.feed("<tool_call>\n<function=");
    check(packed.build_mask(packed_domain, wide), "a packed logits domain must stay constrained");
    check(allows(wide, run) && allows(wide, todo), "declared names stay reachable in the wider domain");
    bool padding_excluded = true;
    for (std::size_t id = domain; id < wide.size(); ++id) { padding_excluded = padding_excluded && wide[id] == 0; }
    check(padding_excluded, "rows the tokenizer does not define must be excluded");

    // A second call after the first one re-enters the constrained region.
    fi::ToolCallConstraint repeated(table, contract);
    repeated.feed("<tool_call>\n<function=todo_write>\n</function>\n</tool_call>\n");
    repeated.feed("<tool_call>\n<function=");
    check(repeated.build_mask(domain, mask), "a second call must be constrained again");
    check(allows(mask, todo), "the second call still selects from the declared names");

    check(table->size() == vocabulary.table.size(), "the shared table must be observable");

    std::cout << (failures == 0 ? "OK" : "FAIL") << " tool-call constraint\n";
    return failures == 0 ? 0 : 1;
}
