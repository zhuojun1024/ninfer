// The guard that lets a re-rendered assistant turn be served from the tokens this lineage generated.
//
// It is the only thing standing between a client's replay and the KV that belongs to the answer
// NInfer wrote, so what it must reject is tested as carefully as what it must accept: the template
// trims the reasoning and the content and writes its own separators, which a client echoing the
// answer cannot reproduce byte for byte, but every other difference is a different turn.

#include "runtime/engine/turn_replay.h"

#include <iostream>
#include <string>
#include <string_view>
#include <vector>

namespace {

using ninfer::runtime::same_rendered_turn;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

// One turn as the template renders it: the reasoning, the canonical close, the content, one call.
std::string turn(std::string_view reasoning, std::string_view content, std::string_view call) {
    std::string out(reasoning);
    out += "\n</think>\n\n";
    out += content;
    if (!call.empty()) {
        out += "\n\n<tool_call>\n<function=write>\n<parameter=path>\n";
        out += call;
        out += "\n</parameter>\n<parameter=content>\nconst value = 1;\n</parameter>\n"
               "</function>\n</tool_call>";
    }
    out += "<|im_end|>\n";
    return out;
}

// Which histories a replayed answer may be matched against.
//
// The batched route retires a lane's own lineage with its pages (release_lane_kv), so the lane that
// serves the next turn of a conversation is routinely not the lane that served the last one. The
// catalog entry the recall is about to restore is then the only history left, and a replayed answer
// that is not matched against it is prefilled again from the last prompt end - the regression this
// selection exists to prevent.
int check_adoption_candidates() {
    using ninfer::runtime::AdoptionCandidate;
    using ninfer::runtime::AdoptionSource;
    using ninfer::runtime::adoption_candidates;
    int failures = 0;

    const std::vector<ninfer::TokenId> incoming = {1, 2, 3, 4, 5, 6};

    // A retired lane, with the conversation sitting in the catalog.
    {
        const std::vector<ninfer::TokenId> history   = {1, 2, 3, 4, 5, 6, 7, 8};
        const AdoptionSource sources[]               = {{history, 4, false, true, 7}};
        std::size_t divergence                       = 0;
        const std::vector<AdoptionCandidate> found   = adoption_candidates(incoming, sources, divergence);
        failures += check(found.size() == 1 && found[0].source == 0 && found[0].shared == 6,
                          "a stored catalog entry was not offered to a retired lane");
        failures += check(divergence == 6, "the agreement of a considered source was not recorded");
    }

    // A lane that still holds its own lineage offers that one and nothing else, even when a catalog
    // entry agrees further: the lineage is the state this lane's KV really carries.
    {
        const std::vector<ninfer::TokenId> lineage = {1, 2, 3, 4, 5};
        const std::vector<ninfer::TokenId> entry   = {1, 2, 3, 4, 5, 6, 7};
        const AdoptionSource sources[]             = {{lineage, 2, true, false, -1},
                                                      {entry, 2, false, true, 3}};
        std::size_t divergence                     = 0;
        const std::vector<AdoptionCandidate> found = adoption_candidates(incoming, sources, divergence);
        failures += check(found.size() == 1 && found[0].source == 0 && found[0].shared == 5,
                          "a live lineage did not hide the catalog");
        // Only the offered source is measured: the trace reports the decision the scan can act on.
        failures += check(divergence == 5, "a hidden source was measured instead of the live lineage");
    }

    // Neither resident nor stored: nothing to restore, so nothing to adopt.
    {
        const std::vector<ninfer::TokenId> history = {1, 2, 3, 4, 5, 6, 7, 8};
        const AdoptionSource sources[]             = {{history, 4, false, false, 0}};
        std::size_t divergence                     = 0;
        failures += check(adoption_candidates(incoming, sources, divergence).empty(),
                          "a source with no KV to restore was offered");
        failures += check(divergence == 0, "a source with no KV to restore was measured");
    }

    // Deepest agreement first, and a tie goes to the longer history.
    {
        const std::vector<ninfer::TokenId> deep   = {1, 2, 3, 4, 5, 6, 7};
        const std::vector<ninfer::TokenId> shallow = {1, 2, 3, 9};
        const std::vector<ninfer::TokenId> longer  = {1, 2, 3, 4, 5, 6, 7, 8};
        const std::vector<ninfer::TokenId> tied    = {1, 2, 3, 4, 5, 6};
        const AdoptionSource order[]               = {{deep, 2, false, true, 0},
                                                      {shallow, 2, false, true, 1}};
        const AdoptionSource ties[]                = {{tied, 2, false, true, 0},
                                                      {longer, 2, false, true, 1}};
        std::size_t divergence                     = 0;
        const std::vector<AdoptionCandidate> found = adoption_candidates(incoming, order, divergence);
        failures += check(found.size() == 2 && found[0].source == 0 && found[1].source == 1,
                          "the candidates were not ordered by agreement");
        divergence                                 = 0;
        const std::vector<AdoptionCandidate> same  = adoption_candidates(incoming, ties, divergence);
        failures += check(same.size() == 2 && same[0].source == 1 && same[1].source == 0,
                          "a tie was not broken by the longer history");
    }

    // The boundaries a history has to carry to be worth anything: a turn that starts at zero, one
    // that starts past its own end, one the prompt never reaches, and one with no tokens at all.
    {
        const std::vector<ninfer::TokenId> history = {1, 2, 3, 4, 5, 6, 7, 8};
        const std::vector<ninfer::TokenId> empty;
        const std::vector<ninfer::TokenId> short_prompt = {1, 2, 3};
        const AdoptionSource no_begin[]     = {{history, 0, false, true, 0}};
        const AdoptionSource past_end[]     = {{history, 9, false, true, 0}};
        const AdoptionSource beyond_prompt[] = {{history, 5, false, true, 0}};
        const AdoptionSource nothing[]      = {{empty, 2, false, true, 0}};
        std::size_t divergence              = 0;
        failures += check(adoption_candidates(incoming, no_begin, divergence).empty(),
                          "a history with no turn boundary was offered");
        failures += check(adoption_candidates(incoming, past_end, divergence).empty(),
                          "a turn starting past the end of its own history was offered");
        failures += check(adoption_candidates(short_prompt, beyond_prompt, divergence).empty(),
                          "a turn the incoming prompt never reaches was offered");
        failures += check(adoption_candidates(incoming, nothing, divergence).empty(),
                          "an empty history was offered");
    }

    // The prompt parts from the history before the turn even starts: a different conversation, which
    // must not be adopted however far the turn itself might line up.
    {
        const std::vector<ninfer::TokenId> history = {9, 9, 9, 9, 9, 9};
        const AdoptionSource sources[]             = {{history, 3, false, true, 0}};
        std::size_t divergence                     = 0;
        failures += check(adoption_candidates(incoming, sources, divergence).empty(),
                          "a history the prompt parts from before the turn was offered");
        failures += check(divergence == 0, "the agreement with a rejected history was not recorded");
    }

    return failures;
}

} // namespace

int main() {
    int failures = 0;
    const std::string canonical = turn("I will write the file.", "Here it is:", "app.js");

    // A replay the renderer reproduces exactly.
    failures += check(same_rendered_turn(canonical, canonical),
                      "an identical replay was rejected");

    // The generated bytes carry the model's own framing: an extra newline before the close and a
    // single one after it, where the template writes two. Same turn, so the tokens this lineage
    // generated may be reused.
    std::string generated = turn("I will write the file.", "Here it is:", "app.js");
    const std::size_t close_at = generated.find("\n</think>");
    generated.replace(close_at, std::string_view("\n</think>").size(), "\n\n</think>");
    const std::size_t after_close = generated.find("</think>") + std::string_view("</think>").size();
    generated.erase(after_close, generated.find_first_not_of('\n', after_close) - after_close);
    generated.insert(after_close, "\n");
    failures += check(same_rendered_turn(generated, canonical),
                      "framing whitespace around the close was treated as a different turn");
    failures += check(same_rendered_turn(canonical, generated),
                      "the comparison is not symmetric in the framing runs");

    // The content differs by one character: a different turn.
    const std::string other_content = turn("I will write the file.", "Here it is!", "app.js");
    failures += check(!same_rendered_turn(generated, other_content),
                      "an edited content byte was accepted as the same turn");

    // Whitespace inside the content is content, not framing.
    const std::string inner_space = turn("I will write the file.", "Here  it is:", "app.js");
    failures += check(!same_rendered_turn(generated, inner_space),
                      "interior content whitespace was treated as framing");

    // A reordered argument object, an edited value, or a different path is a different turn: the
    // tool-call region is compared byte for byte.
    const std::string other_call = turn("I will write the file.", "Here it is:", "app.ts");
    failures += check(!same_rendered_turn(generated, other_call),
                      "an edited tool argument was accepted as the same turn");

    // A turn with no call where the generated one had one.
    const std::string no_call = turn("I will write the file.", "Here it is:", "");
    failures += check(!same_rendered_turn(generated, no_call),
                      "a missing tool call was accepted as the same turn");
    failures += check(!same_rendered_turn(no_call, generated),
                      "an extra tool call was accepted as the same turn");

    // Two calls: the second one has to be there as well, and it sits before the turn's end where
    // the template puts it.
    std::string two_calls = turn("I will write the file.", "Here it is:", "app.js");
    two_calls.insert(two_calls.find("<|im_end|>"),
                     "\n\n<tool_call>\n<function=read>\n<parameter=path>\napp.js\n"
                     "</parameter>\n</function>\n</tool_call>");
    failures += check(!same_rendered_turn(generated, two_calls),
                      "a second tool call was ignored by the comparison");

    // A turn the budget stopped inside its reasoning: it never wrote the close marker, and the
    // template writes one for the reasoning it renders. That marker is the template's, so the turn is
    // still the same one - a turn carrying an answer of its own is not.
    const std::string cut_off = "I will write the file.\n";
    failures += check(same_rendered_turn(cut_off, "I will write the file.\n</think>\n\n"),
                      "a cut-off turn with only the template's close was treated as a different turn");
    failures += check(!same_rendered_turn(cut_off, "I will write the file.\n</think>\n\nHere it is."),
                      "a cut-off turn carrying an answer the model never wrote was accepted");
    failures += check(!same_rendered_turn(cut_off, "I will write another file.\n</think>\n\n"),
                      "a cut-off turn with different reasoning was accepted");
    failures += check(!same_rendered_turn(cut_off, canonical),
                      "a cut-off turn whose rendering carries a tool call was accepted");

    // Reasoning differences are differences, framing aside.
    const std::string other_reasoning = turn("I will write another file.", "Here it is:", "app.js");
    failures += check(!same_rendered_turn(generated, other_reasoning),
                      "an edited reasoning byte was accepted as the same turn");

    failures += check_adoption_candidates();

    if (failures == 0) { std::cout << "turn replay guard ok\n"; }
    return failures == 0 ? 0 : 1;
}
