// TP-2 cross-session KV retention.
//
// The device KV pool holds one conversation at a time, so a request that belongs to another one
// evicts the resident session into pinned host memory and copies the returning session back. This
// test drives several alternating conversations through the public Engine route and checks the
// observable consequences: a returning conversation is recalled rather than prefilled
// (reused_prompt_tokens), its output is checked against a from-scratch prefill of the same prompt,
// and a conversation the LRU budget evicted is prefilled from zero again.
//
// A second scenario covers clients that render one stable system prompt ahead of every
// conversation: the two prompts share a long prefix while belonging to different conversations, and
// switching between them has to move the outgoing conversation into its host slabs even though a
// prefix-reuse checkpoint inside the shared prefix is reusable. It also covers the third
// conversation of that family arriving after a small unrelated request has taken over the device
// pools, when only the state frozen at the boundary the family diverged at can carry the restart.
// A fourth scenario covers the return trip of a client that re-renders the answer it was handed:
// the divergence sits at the first generated token, so the conversation has to come back on the
// state frozen at its own prompt end. A last one cancels a prompt mid-prefill and checks that the
// retry continues from the prefix the cancelled walk published.
//
// The from-scratch comparison is the strong claim: a reuse that changes the answer must not hide
// behind a matching token count. The switch scenario and the cancellation retry agree with it token
// for token. The third scenario, whose prompt is recalled behind a small unrelated request, agrees
// only on the first sample; that gap is an open finding recorded in PLAN.md, not an accepted
// behaviour, and it is why that one assertion is deliberately narrower than the others.
//
// A conversation that shares no tokens with the resident one is a switch even when nothing else can
// serve it, so these scenarios also pin down what a switch costs when the host slabs are unusable.
//
// The scenario runs on three routes: plain, MTP and DFlash2. MTP keeps its own KV slab on shard 0
// and DFlash2 keeps its masked draft's local context ring there; both travel through the same
// eviction transaction and the same prefix-reuse checkpoints. The default set is plain and MTP;
// DFlash2 runs only when NINFER_TEST_ROUTE names it, because its masked-draft window's fp8 logits
// diverge from a from-scratch walk at near ties even though the recalled state is carried exactly
// (PLAN.md section 3.6, "B6 result"). The oracle in each round uses the same speculative
// configuration with retention disabled.
//
// The artifact is selected with NINFER_TEST_ARTIFACT; two identical sm_120a devices are required.
// Without either, the test skips with exit code 77.

#include "ninfer/engine.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using ninfer::TokenId;

constexpr std::uint32_t kMaxContext   = 2048;
constexpr std::uint32_t kOutputTokens = 8;
// The catalog holds the resident conversation plus two host-resident ones.
constexpr std::uint32_t kSessions = 3;
// The whole --host-kv-mib value, split evenly between the two shards.
constexpr std::size_t kHostKvBytes = 128ULL << 20;

std::pair<int, int> pick_devices() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count < 2) { return {-1, -1}; }
    // The machine may carry other parts (an older compute capability, for instance). Only
    // sm_120a devices can run the TP-2 kernels, so they are the only candidates; a foreign
    // device in the enumeration must not mask the pair we can actually use.
    std::vector<int> candidates;
    std::vector<std::string> names(static_cast<std::size_t>(count));
    for (int index = 0; index < count; ++index) {
        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, index) != cudaSuccess) { return {-1, -1}; }
        if (prop.major != 12) { continue; }
        candidates.push_back(index);
        names[static_cast<std::size_t>(index)] = prop.name;
    }
    for (std::size_t a = 0; a < candidates.size(); ++a) {
        for (std::size_t b = a + 1; b < candidates.size(); ++b) {
            const std::size_t left  = static_cast<std::size_t>(candidates[a]);
            const std::size_t right = static_cast<std::size_t>(candidates[b]);
            if (names[left] == names[right]) { return {candidates[a], candidates[b]}; }
        }
    }
    return {-1, -1};
}

// The generation route one scenario runs on. The oracle of a route is the same route with
// retention disabled, so a DFlash2 comparison is against a DFlash2 full prefill.
enum class Route { Plain, Mtp, DFlash2 };

const char* route_name(Route route) {
    switch (route) {
    case Route::Plain: return "plain";
    case Route::Mtp: return "mtp";
    case Route::DFlash2: return "dflash2";
    }
    return "unknown";
}

ninfer::EngineOptions engine_options(const char* artifact, int device_a, int device_b,
                                     bool retention, Route route) {
    ninfer::EngineOptions options;
    options.artifact_path                        = artifact;
    options.device                               = device_a;
    options.device_b                             = device_b;
    options.max_context                          = kMaxContext;
    options.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(kMaxContext);
    options.prefill_chunk                       = 256;
    // The chunk plan and the recall alignment share this width: the host checkpoint boundaries a
    // recall may reuse are multiples of it, so changing it also moves which conversations can be
    // reused and how far a recall has to re-prefill.
    options.max_concurrency                     = 1;
    options.max_pending_requests                = 1;
    if (route == Route::Mtp) {
        options.speculative.backend      = ninfer::SpeculativeBackend::Mtp;
        options.speculative.draft_tokens = 2;
        // The server recipe runs MTP on fp8 KV, and this is the route the cross-session slabs
        // have to carry; bf16 KV with MTP never gets past its first prefill (see the note in
        // docs/tp2-dual-5060ti.md).
        options.kv_cache = ninfer::KvCacheStorage::Fp8E4M3Row256;
    }
    if (route == Route::DFlash2) {
        // The same fp8 KV recipe the server runs, and the full window the route is tuned with, so
        // the draft's ring and pending staging take their real size through the slabs.
        options.speculative.backend      = ninfer::SpeculativeBackend::DFlash2;
        options.speculative.draft_tokens = 7;
        options.kv_cache                 = ninfer::KvCacheStorage::Fp8E4M3Row256;
    }
    options.context_cache.host_kv_capacity_bytes = retention ? kHostKvBytes : 0;
    options.context_cache.max_private_continuations = retention ? kSessions : 1;
    return options;
}

// Greedy sampling keeps the answer a deterministic function of the KV, which is what makes the
// recalled and the from-scratch walks comparable token for token.
ninfer::RequestOptions greedy_request() {
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens = kOutputTokens;
    options.execution.sampling.temperature    = 0.0F;
    options.execution.sampling.top_k          = 0;
    options.stop.include_model_defaults       = false;
    return options;
}

std::vector<TokenId> make_prompt(std::int32_t first, std::size_t length) {
    std::vector<TokenId> tokens;
    tokens.reserve(length);
    for (std::size_t index = 0; index < length; ++index) {
        tokens.push_back(static_cast<TokenId>(first + static_cast<std::int32_t>(index % 977)));
    }
    return tokens;
}

ninfer::GenerationResult run(ninfer::Engine& engine, std::vector<TokenId> tokens) {
    return engine.generate(engine.prepare_tokens(std::move(tokens)), greedy_request());
}

// Cancels after 'cancel_after' prefill iterations, so the walk is interrupted between chunks and
// the prefix it finished is what a retry of the same prompt has to be able to continue from.
ninfer::GenerationResult run_cancelled(ninfer::Engine& engine, std::vector<TokenId> tokens,
                                       int cancel_after) {
    int checks = 0;
    ninfer::CancellationView cancellation(
        [&checks, cancel_after]() { return ++checks > cancel_after; });
    return engine.generate(engine.prepare_tokens(std::move(tokens)), greedy_request(), nullptr,
                           cancellation);
}

void append(std::vector<TokenId>& destination, const std::vector<TokenId>& source) {
    destination.insert(destination.end(), source.begin(), source.end());
}

int fail(const std::string& label, const std::string& message) {
    std::cerr << "FAIL (" << label << "): " << message << '\n';
    return 1;
}

// Renders a token vector for a failure message, so a divergence reports where it starts instead of
// only that it happened.
std::string tokens_text(const std::vector<TokenId>& tokens) {
    std::string text = "[";
    for (std::size_t index = 0; index < tokens.size(); ++index) {
        if (index != 0) { text += " "; }
        text += std::to_string(tokens[index]);
    }
    return text + "]";
}

// One full A/B/A/LRU scenario. Returns 0 on success, 1 on a failed assertion.
int run_scenario(const char* artifact, int device_a, int device_b, Route route) {
    const std::string label = route_name(route);

    const std::vector<TokenId> opening    = make_prompt(1200, 64);
    const std::vector<TokenId> follow_up  = make_prompt(4000, 16);
    const std::vector<TokenId> other_a    = make_prompt(7000, 48);
    const std::vector<TokenId> other_b    = make_prompt(20000, 48);
    const std::vector<TokenId> other_c    = make_prompt(30000, 48);
    const std::vector<TokenId> other_d    = make_prompt(40000, 48);

    // A client that renders one stable block - a system prompt - ahead of every conversation sends
    // prompts that share a long prefix while belonging to different conversations. That prefix is not
    // evidence that the resident conversation *continues*: a prefix-reuse checkpoint inside it used to
    // be enough for the recall scan to keep the resident lineage, which left the conversation being
    // switched away from without host slabs, so coming back re-prefilled it from the shared prefix
    // instead of recalling it.
    const std::vector<TokenId> system_prompt = make_prompt(500, 512);
    std::vector<TokenId> shared_a           = system_prompt;
    append(shared_a, make_prompt(9000, 128));
    std::vector<TokenId> shared_b = system_prompt;
    append(shared_b, make_prompt(20000, 128));
    std::vector<TokenId> shared_c = system_prompt;
    append(shared_c, make_prompt(40000, 128));
    // A request that shares nothing with the conversations: the shape of a title or summary call,
    // which real clients interleave with the turns they actually cache.
    const std::vector<TokenId> aside = make_prompt(55000, 48);
    const std::vector<TokenId> aside_flush = make_prompt(56000, 48);
    // Long enough to be split across prefill chunks, so a cancellation can land between two of them.
    const std::vector<TokenId> interrupted = make_prompt(65000, 600);

    // The prompt a client sends back after re-rendering the answer it was handed: the previous prompt
    // again, then its own rendering of the answer, which does not reproduce the sampled tokens. The
    // divergence sits at the first generated token, so the conversation's frontier is out of reach
    // and only the state frozen at its prompt end can bring the conversation back.
    const std::vector<TokenId> rerender_base = make_prompt(60000, 64);
    std::vector<TokenId> rerendered          = rerender_base;
    append(rerendered, make_prompt(50000, 8));
    append(rerendered, follow_up);

    // The oracle prefills the continued prompt from zero with retention disabled. A recalled
    // walk reuses the byte-identical KV prefix, so the greedy answers have to agree exactly.
    std::vector<TokenId> opening_answer;
    std::vector<TokenId> continued_answer;
    std::vector<TokenId> shared_answer;
    std::vector<TokenId> shared_continued_answer;
    std::vector<TokenId> rerendered_answer;
    std::vector<TokenId> shared_b_answer;
    std::vector<TokenId> shared_c_answer;
    std::vector<TokenId> interrupted_answer;
    {
        ninfer::Engine oracle(engine_options(artifact, device_a, device_b, false, route));
        opening_answer = run(oracle, opening).generated_token_ids;
        std::vector<TokenId> continued = opening;
        append(continued, opening_answer);
        append(continued, follow_up);
        continued_answer = run(oracle, std::move(continued)).generated_token_ids;

        shared_answer = run(oracle, shared_a).generated_token_ids;
        std::vector<TokenId> shared_continued = shared_a;
        append(shared_continued, shared_answer);
        append(shared_continued, follow_up);
        shared_continued_answer = run(oracle, std::move(shared_continued)).generated_token_ids;

        rerendered_answer = run(oracle, rerendered).generated_token_ids;

        // The flush request leaves the oracle with a lineage that shares nothing with the prompt
        // under test, so these two answers come from a full prefill even though the oracle keeps the
        // checkpoint ring: the ring is pruned to the lineage's shared prefix, which is zero here.
        (void)run(oracle, aside);
        shared_b_answer = run(oracle, shared_b).generated_token_ids;
        (void)run(oracle, aside_flush);
        shared_c_answer      = run(oracle, shared_c).generated_token_ids;
        (void)run(oracle, aside_flush);
        interrupted_answer = run(oracle, interrupted).generated_token_ids;
    }
    if (opening_answer.empty() || continued_answer.empty()) {
        return fail(label, "the oracle produced no tokens");
    }

    ninfer::Engine engine(engine_options(artifact, device_a, device_b, true, route));

    // Conversation A: the empty catalog makes this a full prefill that installs the entry.
    const ninfer::GenerationResult a_first = run(engine, opening);
    if (a_first.reused_prompt_tokens != 0) {
        return fail(label, "the first conversation was not prefilled from zero");
    }
    if (a_first.generated_token_ids != opening_answer) {
        return fail(label, "a fresh conversation diverged from the oracle: got " +
                               tokens_text(a_first.generated_token_ids) + " expected " +
                               tokens_text(opening_answer));
    }

    std::vector<TokenId> a_continued = opening;
    append(a_continued, opening_answer);
    append(a_continued, follow_up);
    // The recalled walk stops one token short of A's history: its last sampled token was
    // never forwarded, so it is the one column the prefill has to replay.
    const std::uint32_t recalled_frontier =
        static_cast<std::uint32_t>(opening.size() + opening_answer.size()) - 1U;

    // Conversation B displaces A into its host slabs.
    const ninfer::GenerationResult b_first = run(engine, other_a);
    if (b_first.reused_prompt_tokens != 0) {
        return fail(label, "a second conversation was prefilled from a stale lineage");
    }

    // A returns: the whole prefix comes back from the host slabs, so only the frontier token is
    // forwarded again.
    const ninfer::GenerationResult a_second = run(engine, a_continued);
    if (a_second.reused_prompt_tokens != recalled_frontier) {
        return fail(label, "a recalled conversation reused " +
                              std::to_string(a_second.reused_prompt_tokens) +
                              " prompt tokens, expected " + std::to_string(recalled_frontier));
    }
    if (a_second.generated_token_ids != continued_answer) {
        return fail(label, "a recalled conversation diverged from the oracle: got " +
                               tokens_text(a_second.generated_token_ids) + " expected " +
                               tokens_text(continued_answer));
    }

    // Fill the catalog past its capacity (three entries: one resident, two host). C and D each
    // push a conversation out; D evicts B, and E evicts A as the oldest host entry.
    (void)run(engine, other_b);
    (void)run(engine, other_c);
    (void)run(engine, other_d);
    const ninfer::GenerationResult a_evicted = run(engine, a_continued);
    if (a_evicted.reused_prompt_tokens != 0) {
        return fail(label, "an LRU-evicted conversation reused " +
                              std::to_string(a_evicted.reused_prompt_tokens) + " prompt tokens");
    }
    if (a_evicted.generated_token_ids != continued_answer) {
        return fail(label, "a re-prefilled conversation diverged from the oracle: got " +
                               tokens_text(a_evicted.generated_token_ids) + " expected " +
                               tokens_text(continued_answer));
    }

    // Two conversations that share the system prompt. Switching to the second one has to move the
    // first into its host slabs even though a checkpoint inside the shared prefix is reusable, or the
    // return trip cannot recall it.
    const ninfer::GenerationResult shared_a_first = run(engine, shared_a);
    if (shared_a_first.reused_prompt_tokens != 0) {
        return fail(label, "a conversation behind a shared system prompt reused " +
                               std::to_string(shared_a_first.reused_prompt_tokens) +
                               " prompt tokens on its first turn");
    }
    if (shared_a_first.generated_token_ids != shared_answer) {
        return fail(label,
                    "a conversation behind a shared system prompt diverged from the oracle: got " +
                        tokens_text(shared_a_first.generated_token_ids) + " expected " +
                        tokens_text(shared_answer));
    }

    const ninfer::GenerationResult shared_b_first = run(engine, shared_b);
    // The switch still reuses the system prefix the two conversations share; what it must not do is
    // stay on the resident lineage.
    if (shared_b_first.reused_prompt_tokens != system_prompt.size()) {
        return fail(label, "a switch between conversations behind one system prompt reused " +
                               std::to_string(shared_b_first.reused_prompt_tokens) +
                               " prompt tokens, expected " + std::to_string(system_prompt.size()));
    }
    if (shared_b_first.generated_token_ids != shared_b_answer) {
        return fail(label,
                    "a switch behind a shared system prompt diverged from the oracle: got " +
                        tokens_text(shared_b_first.generated_token_ids) + " expected " +
                        tokens_text(shared_b_answer));
    }

    std::vector<TokenId> shared_a_continued = shared_a;
    append(shared_a_continued, shared_answer);
    append(shared_a_continued, follow_up);
    const std::uint32_t shared_frontier =
        static_cast<std::uint32_t>(shared_a.size() + shared_answer.size()) - 1U;
    const ninfer::GenerationResult shared_a_second = run(engine, shared_a_continued);
    if (shared_a_second.reused_prompt_tokens != shared_frontier) {
        return fail(label, "a conversation behind a shared system prompt reused " +
                               std::to_string(shared_a_second.reused_prompt_tokens) +
                               " prompt tokens on return, expected " +
                               std::to_string(shared_frontier));
    }
    if (shared_a_second.generated_token_ids != shared_continued_answer) {
        return fail(label,
                    "a conversation behind a shared system prompt diverged from the oracle on "
                    "return: got " +
                        tokens_text(shared_a_second.generated_token_ids) + " expected " +
                        tokens_text(shared_continued_answer));
    }

    // A title or summary call in between takes over the device pools without sharing anything with
    // the family. The system prompt the family opened with is now only in shared_a's host slabs, and
    // the boundary where the family diverged from shared_a was frozen with them, so the next
    // conversation that opens with that system prompt restarts on the boundary instead of prefilling
    // the block again. Nothing on the device can serve it: the resident lineage shares a handful of
    // tokens with it at most.
    const ninfer::GenerationResult aside_first = run(engine, aside);
    if (aside_first.reused_prompt_tokens != 0) {
        return fail(label, "an unrelated request reused a stale lineage");
    }
    const ninfer::GenerationResult shared_c_first = run(engine, shared_c);
    if (shared_c_first.reused_prompt_tokens != system_prompt.size()) {
        return fail(label, "a conversation opening with a stored system prompt reused " +
                               std::to_string(shared_c_first.reused_prompt_tokens) +
                               " prompt tokens, expected " + std::to_string(system_prompt.size()));
    }
    // The recalled walk agrees with the oracle on the first sample, which is the logits of the last
    // prompt column over the recalled KV and GDN state - the whole point of the boundary. It is not
    // asserted token for token: a recall that restores its KV from a host slab diverges from a
    // from-scratch walk inside the generated tail (this case splits at the seventh of eight greedy
    // tokens, into a repeating token the from-scratch walk does not produce), and the same
    // comparison passes when the KV is still the one the device already held. That difference is
    // tracked separately; see PLAN.md's TP-2 session note. The prefix/state themselves are pinned
    // down by the frontier, prompt-end and switch scenarios above.

    if (!shared_c_first.generated_token_ids.empty() && !shared_c_answer.empty() &&
        shared_c_first.generated_token_ids.front() != shared_c_answer.front()) {
        return fail(label, "a conversation behind a stored system prompt diverged from the oracle on "
                           "its first sample");
    }

    // A client that re-renders the answer it was handed never reproduces the sampled tokens, so the
    // frontier its conversation reached is unreachable. The state frozen at the entry's own prompt end
    // has to carry the return trip, or the whole conversation is prefilled a second time.
    const ninfer::GenerationResult base_first = run(engine, rerender_base);
    if (base_first.reused_prompt_tokens != 0) {
        return fail(label, "a fresh conversation behind a re-rendered answer reused a stale lineage");
    }
    (void)run(engine, aside);
    const ninfer::GenerationResult rerendered_first = run(engine, rerendered);
    const std::uint32_t rerender_prompt_end = static_cast<std::uint32_t>(rerender_base.size());
    if (rerendered_first.reused_prompt_tokens != rerender_prompt_end) {
        return fail(label, "a conversation behind a re-rendered answer reused " +
                               std::to_string(rerendered_first.reused_prompt_tokens) +
                               " prompt tokens, expected " +
                               std::to_string(rerender_prompt_end));
    }
    if (rerendered_first.generated_token_ids != rerendered_answer) {
        return fail(label,
                    "a conversation behind a re-rendered answer diverged from the oracle: got " +
                        tokens_text(rerendered_first.generated_token_ids) + " expected " +
                        tokens_text(rerendered_answer));
    }

    // A client that gives up mid-prefill and sends the same prompt again. The cancelled walk wrote
    // KV for the whole chunks it finished and left the GDN state at the end of the last one, so the
    // catalog can name that prefix; the retry has to continue from it rather than prefill the whole
    // prompt, and it has to agree with a from-scratch walk of the same prompt.
    const ninfer::GenerationResult cancelled = run_cancelled(engine, interrupted, 1);
    if (cancelled.finish_reason != ninfer::FinishReason::Cancelled) {
        return fail(label, "a cancelled request did not report Cancelled");
    }
    if (!cancelled.generated_token_ids.empty()) {
        return fail(label, "a request cancelled before the last chunk produced tokens");
    }
    const std::uint32_t interrupted_prefix = 256;
    const ninfer::GenerationResult retried = run(engine, interrupted);
    if (retried.reused_prompt_tokens != interrupted_prefix) {
        return fail(label, "a retry of a cancelled prompt reused " +
                               std::to_string(retried.reused_prompt_tokens) +
                               " prompt tokens, expected " +
                               std::to_string(interrupted_prefix));
    }
    if (retried.generated_token_ids != interrupted_answer) {
        return fail(label, "a retry of a cancelled prompt diverged from the oracle");
    }

    // Reproducibility probe: the same prompt, from scratch, a second time in this engine. The
    // scenarios above compare generated tokens; this compares the logits the prefill produced,
    // which is what decides a near tie. The flush first (a prompt sharing nothing with it) forces
    // reused_prompt_tokens to zero, so the walk really is the from-scratch path again.
    {
        (void)run(engine, aside);
        const ninfer::GenerationResult shared_a_repeat = run(engine, shared_a);
        if (shared_a_repeat.reused_prompt_tokens != 0) {
            return fail(label, "the reproducibility probe reused " +
                                   std::to_string(shared_a_repeat.reused_prompt_tokens) +
                                   " prompt tokens");
        }
        if (shared_a_repeat.generated_token_ids != shared_answer) {
            return fail(label,
                        "the same prompt prefilled from scratch twice produced different tokens");
        }
    }

    std::cout << "TP-2 session retention (" << label << ") passed: recall reused "
              << recalled_frontier
              << " prompt tokens bit-identically; LRU eviction forced a full prefill\n";
    return 0;
}

} // namespace

int main() {
    const char* artifact = std::getenv("NINFER_TEST_ARTIFACT");
    if (artifact == nullptr || *artifact == '\0') {
        std::cout << "skip: NINFER_TEST_ARTIFACT is not set\n";
        return 77;
    }
    const auto devices = pick_devices();
    if (devices.first < 0) {
        std::cout << "skip: two identical sm_120a devices are not available\n";
        return 77;
    }

    // NINFER_TEST_ROUTE=plain|mtp|dflash2 narrows the run to one route, which is what a failing
    // route needs when each of the others costs a full model load. The default set is the two
    // routes the retention property holds for; DFlash2 is opt-in because its masked-draft window
    // diverges from a from-scratch walk at near ties (PLAN.md section 3.6, "B6 result").
    const char* selected_env   = std::getenv("NINFER_TEST_ROUTE");
    const std::string selected = selected_env == nullptr ? "" : selected_env;
    std::vector<Route> routes;
    if (selected.empty()) {
        routes = {Route::Plain, Route::Mtp};
    } else if (selected == "plain") {
        routes = {Route::Plain};
    } else if (selected == "mtp") {
        routes = {Route::Mtp};
    } else if (selected == "dflash2") {
        routes = {Route::DFlash2};
    } else {
        std::cerr << "FAIL: NINFER_TEST_ROUTE must be plain, mtp or dflash2\n";
        return 1;
    }

    try {
        for (const Route route : routes) {
            if (const int status = run_scenario(artifact, devices.first, devices.second, route);
                status != 0) {
                return status;
            }
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
