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
// behind a matching token count. Every recall that keeps the oracle's draft pattern - the switch
// scenario, the cancellation retry, DFlash2 on the prefill grid, and a shallow mid-chunk recall the
// masked draft rounds back down to that grid - agrees token for token. A recall whose masked draft is
// declined runs target-only rounds instead: a different draft pattern,
// so its near ties resolve its own way and only its boundary crossing is pinned (compare_recall).
//
// A conversation that shares no tokens with the resident one is a switch even when nothing else can
// serve it, so these scenarios also pin down what a switch costs when the host slabs are unusable.
//
// The scenario runs on three routes: plain, MTP and DFlash2. MTP keeps its own KV slab on shard 0
// and DFlash2 keeps its masked draft's local context ring there; all three travel through the same
// eviction transaction and the same prefix-reuse checkpoints, and all three run by default.
// NINFER_TEST_ROUTE narrows the run to one route when one needs a focused pass. The oracle in each
// round uses the same speculative configuration with retention disabled.
//
// The artifact is selected with NINFER_TEST_ARTIFACT; two identical sm_120a devices are required.
// Without either, the test skips with exit code 77.

#include "ninfer/engine.h"

#include <cuda_runtime.h>

#include <algorithm>

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

// The scenario that pins what the reuse gate charges for an unaligned boundary: one long turn that
// is deliberately longer than the conversation's average turn, continued with a budget far larger
// than anything the conversation has generated. The gate has to price the conversation's own turns,
// or the budget its client asked for puts the boundary out of reach for good.
constexpr std::uint32_t kBurstBudget    = 60;
constexpr std::uint32_t kContinueBudget = 1024;

// The prefill chunk the scenario pins; `normalize_engine_options` keeps it and the core's recall
// grid is the same width, so a boundary that is not a multiple of it sits inside a chunk.
constexpr std::uint32_t kPrefillChunk = 256;

// A DFlash2 recall whose boundary is not on the prefill grid cannot license the masked draft
// directly: the ring beside the restored state belongs to a differently chunked walk, so its
// proposals would not be licensed against a from-scratch walk's windows. The masked draft rounds
// such a boundary down to the aligned scan's result instead whenever the clip stays within its
// budget - one chunk, or sixteen tokens per token the request may still generate
// (tp2_generation_core.cpp, the reuse fallback's gate). The recall then re-prefills that clip through
// the from-scratch chunk plan, which keeps the ring canonical and the draft licensed, so the answer
// has to reproduce the oracle's in full (see compare_recall). Only a deeper mid-chunk lineage keeps
// the mid-chunk restore and declines the draft: the request runs target-only rounds, deterministic in
// themselves, and their boundary crossing matches the oracle, but the tail is a different draft
// pattern's walk.
constexpr std::uint32_t kDraftDeclineClip =
    kPrefillChunk > 16U * kOutputTokens ? kPrefillChunk : 16U * kOutputTokens;
bool draft_rounds_down(Route route, std::uint32_t boundary) {
    return route == Route::DFlash2 && boundary != 0 && boundary % kPrefillChunk != 0 &&
           boundary <= kDraftDeclineClip;
}
bool draft_declined(Route route, std::uint32_t boundary) {
    return route == Route::DFlash2 && boundary != 0 && boundary % kPrefillChunk != 0 &&
           !draft_rounds_down(route, boundary);
}
// The exit-path scan prices an unaligned boundary against the masked draft: the restored ring is the
// one a differently chunked walk froze, so taking the boundary runs the request target-only for as
// long as it generates. A boundary only wins when the prefill it skips beats sixteen tokens per token
// the request will still generate, and what it will generate is the conversation's own average turn
// rather than the budget its client asked for (tp2_generation_core.cpp). A resident continuation's
// frontier is unaligned and its clip is the distance back to the previous prompt's own chunk start,
// so the scenarios have to pin which of the two boundaries each route lands on.
bool resident_tail_rounds_down(Route route, std::uint32_t frontier, std::uint32_t aligned,
                               std::uint32_t expected_turn) {
    return route == Route::DFlash2 && frontier != aligned &&
           frontier - aligned <= std::max(kPrefillChunk, 16U * expected_turn);
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

// The same walk with a chosen output budget: the gate reads the request's budget, so a scenario that
// pins how the gate prices a boundary has to control it.
ninfer::GenerationResult run_budget(ninfer::Engine& engine, std::vector<TokenId> tokens,
                                    std::uint32_t budget) {
    ninfer::RequestOptions options            = greedy_request();
    options.execution.requested_output_tokens = budget;
    return engine.generate(engine.prepare_tokens(std::move(tokens)), options);
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

// Pins a recall's answer against the from-scratch oracle. Every recall the masked draft keeps -
// every route, and DFlash2 whenever it stays on the prefill grid or rounds a shallow clip back down
// to it - has to match token for token: the walk keeps the oracle's draft pattern, so its verify
// windows are the same execution shape and the KV rows they write are the same bytes. A declined
// DFlash2 recall (see draft_declined) runs target-only rounds instead - a
// different draft pattern and therefore a different execution shape, the same property that makes
// MTP windows and plain decodes differ (docs/tp2-dual-5060ti.md) - and a changed draft pattern
// shifts which way near ties resolve. The re-rendered-answer scenario demonstrates it
// deterministically: its third token forks, identically in every run. A rounded-down recall (see
// draft_rounds_down) is licensed but re-prefills from zero in a warm engine, which is the ulp-level
// shape difference the golden probe warns about; it pins its boundary crossing the same way. A recall
// in either class still owes the oracle the boundary crossing - its first sample is the logits of the
// prefix it starts from - so that is what is pinned for it.
int compare_recall(const std::string& label, const char* what, Route route, std::uint32_t boundary,
                   const ninfer::GenerationResult& got, const std::vector<TokenId>& expected) {
    const bool declined = draft_declined(route, boundary);
    const bool replayed = draft_rounds_down(route, boundary);
    if (got.draft_context_declined != declined) {
        return fail(label, std::string(what) +
                               (declined ? " did not decline its draft"
                                         : " declined its draft on an aligned boundary"));
    }
    // A rounded-down recall re-prefills its clip from zero in a warm engine. That is a licensed walk -
    // the draft stays live, which the flag above pins - but its KV rows carry this prefill's reduction
    // shape, and the project only guarantees a trajectory reproduces across engine warmth on a server
    // that has served the same shapes (tools/win_port/r52_ab.ps1 records the property). The boundary
    // crossing is what it still owes the oracle.
    if (declined || replayed) {
        if (got.generated_token_ids.empty() || expected.empty() ||
            got.generated_token_ids.front() != expected.front()) {
            return fail(label, std::string(what) + " diverged from the oracle on its first sample: got " +
                                   tokens_text(got.generated_token_ids) + " expected " +
                                   tokens_text(expected));
        }
        return 0;
    }
    if (got.generated_token_ids != expected) {
        return fail(label, std::string(what) + " diverged from the oracle: got " +
                               tokens_text(got.generated_token_ids) + " expected " +
                               tokens_text(expected));
    }
    return 0;
}

// One full A/B/A/LRU scenario. Returns 0 on success, 1 on a failed assertion.
// The first token the prefill samples is a generated token like every later one, so it has to reach
// the output policy. The policy owns the published text, the reasoning/content split and the
// stop-token decision, and a token that bypasses it is never published at all. The pools, the
// session catalog and the published answer then describe histories one token apart, so a client that
// replays an answer never reaches the tail of it - no cache setting can repair that, which is why
// the usage counter is the observable to pin: with the thinking phase open, a request whose whole
// budget is one token has to report that one token as reasoning.
int check_first_token_published(const char* artifact, int device_a, int device_b, Route route) {
    const std::string label = std::string("first token (") + route_name(route) + ")";
    ninfer::Engine engine(engine_options(artifact, device_a, device_b, false, route));
    ninfer::PromptInput input;
    input.options.enable_thinking = true;
    ninfer::ChatMessage user;
    user.role = ninfer::ChatRole::User;
    ninfer::MessagePart question;
    question.kind = ninfer::MessagePartKind::Text;
    question.text = "How many trailing zeros does 100! have?";
    user.parts.push_back(std::move(question));
    input.messages.push_back(std::move(user));

    ninfer::RequestOptions request            = greedy_request();
    request.execution.requested_output_tokens = 1;
    const ninfer::GenerationResult result =
        engine.generate(engine.prepare(std::move(input)), request);
    if (result.generated_token_ids.size() != 1) {
        return fail(label, "a one-token budget produced " +
                               std::to_string(result.generated_token_ids.size()) + " tokens");
    }
    if (result.reasoning_tokens != 1) {
        return fail(label, "the first generated token was not published: " +
                               std::to_string(result.reasoning_tokens) +
                               " reasoning tokens for 1 generated token");
    }
    return 0;
}

int run_scenario(const char* artifact, int device_a, int device_b, Route route) {
    const std::string label = route_name(route);
    // Every route keeps cross-session retention here; the oracle beside it runs retention-off.

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

    // A conversation that continues in place: the client sends the previous prompt, the sampled
    // tokens it was handed, and its next turn, with no other conversation in between, so the
    // resident lineage - not a host slab - has to serve it. Its previous prompt ends inside a
    // prefill chunk, so the committed frontier sits off the reuse grid and the aligned scan can only
    // offer the rewind snapshot at that prompt's own chunk start, with the whole generated answer
    // behind it. The tail is reachable only through the exit-path scan, which is what this pins.
    const std::vector<TokenId> resident_base = make_prompt(61000, 1000);

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
    std::vector<TokenId> resident_base_answer;
    std::vector<TokenId> resident_answer;
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

        resident_base_answer = run(oracle, resident_base).generated_token_ids;
        std::vector<TokenId> resident_continued = resident_base;
        append(resident_continued, resident_base_answer);
        append(resident_continued, follow_up);
        resident_answer = run(oracle, std::move(resident_continued)).generated_token_ids;
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
    const std::uint32_t a_second_reuse = recalled_frontier;
    if (draft_rounds_down(route, recalled_frontier)) {
        // The masked draft trades this shallow mid-chunk reuse for its ring, so the recall re-prefills
        // from the aligned scan's result (nothing else is cached here, so that is zero) and has to
        // reproduce the oracle's answer in full below.
        if (a_second.reused_prompt_tokens != 0) {
            return fail(label, "a rounded-down recall reused " +
                                   std::to_string(a_second.reused_prompt_tokens) +
                                   " prompt tokens, expected 0");
        }
    } else if (a_second.reused_prompt_tokens != a_second_reuse) {
        return fail(label, "a recalled conversation reused " +
                               std::to_string(a_second.reused_prompt_tokens) +
                               " prompt tokens, expected " + std::to_string(a_second_reuse));
    }
    if (const int status = compare_recall(label, "a recalled conversation", route,
                                          recalled_frontier, a_second, continued_answer);
        status != 0) {
        return status;
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
    const std::uint32_t shared_a_second_reuse = shared_frontier;
    if (shared_a_second.reused_prompt_tokens != shared_a_second_reuse) {
        return fail(label, "a conversation behind a shared system prompt reused " +
                               std::to_string(shared_a_second.reused_prompt_tokens) +
                               " prompt tokens on return, expected " +
                               std::to_string(shared_a_second_reuse));
    }
    if (const int status = compare_recall(label, "a conversation behind a shared system prompt",
                                          route, shared_frontier, shared_a_second,
                                          shared_continued_answer);
        status != 0) {
        return status;
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
    const std::uint32_t shared_c_first_reuse =
        static_cast<std::uint32_t>(system_prompt.size());
    if (shared_c_first.reused_prompt_tokens != shared_c_first_reuse) {
        return fail(label, "a conversation opening with a stored system prompt reused " +
                               std::to_string(shared_c_first.reused_prompt_tokens) +
                               " prompt tokens, expected " + std::to_string(shared_c_first_reuse));
    }
    // The recalled walk has to agree with the oracle token for token. This is the case that used to
    // split inside its generated tail (PLAN.md 3.6, "B6 result"): the host-slab recall lands on the
    // prefill grid, so its masked draft stays licensed and its verify windows have to reproduce a
    // from-scratch walk exactly.
    if (const int status = compare_recall(
            label, "a conversation opening with a stored system prompt", route, shared_c_first_reuse,
            shared_c_first, shared_c_answer);
        status != 0) {
        return status;
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
    const std::uint32_t rerendered_first_reuse = rerender_prompt_end;
    if (draft_rounds_down(route, rerender_prompt_end)) {
        // Same trade as the shallow recall above, and the oracle walks the same from-scratch plan.
        if (rerendered_first.reused_prompt_tokens != 0) {
            return fail(label, "a rounded-down return reused " +
                                   std::to_string(rerendered_first.reused_prompt_tokens) +
                                   " prompt tokens, expected 0");
        }
    } else if (rerendered_first.reused_prompt_tokens != rerendered_first_reuse) {
        return fail(label, "a conversation behind a re-rendered answer reused " +
                               std::to_string(rerendered_first.reused_prompt_tokens) +
                               " prompt tokens, expected " +
                               std::to_string(rerendered_first_reuse));
    }
    if (const int status = compare_recall(label, "a conversation behind a re-rendered answer",
                                          route, rerender_prompt_end, rerendered_first,
                                          rerendered_answer);
        status != 0) {
        return status;
    }

    // The resident conversation: no other conversation ran in between, so the device pools still hold
    // exactly this lineage and the continuation has to start at the committed frontier rather than at
    // the chunk start behind the previous prompt. The masked draft rounds the frontier down to that
    // chunk start while the clip stays inside its budget, so the two routes pin two boundaries.
    const ninfer::GenerationResult resident_first = run(engine, resident_base);
    if (resident_first.reused_prompt_tokens != 0) {
        return fail(label, "a resident conversation started from a stale lineage");
    }
    if (resident_first.generated_token_ids != resident_base_answer) {
        return fail(label, "a fresh resident conversation diverged from the oracle: got " +
                               tokens_text(resident_first.generated_token_ids) + " expected " +
                               tokens_text(resident_base_answer));
    }
    std::vector<TokenId> resident_continued = resident_base;
    append(resident_continued, resident_first.generated_token_ids);
    append(resident_continued, follow_up);
    const std::uint32_t resident_frontier = static_cast<std::uint32_t>(
                                                resident_base.size() +
                                                resident_first.generated_token_ids.size()) -
                                            1U;
    const std::uint32_t resident_aligned =
        static_cast<std::uint32_t>(resident_base.size()) / kPrefillChunk * kPrefillChunk;
    // The conversation has completed exactly one turn, of kOutputTokens tokens, so its average turn
    // is the same number the request's budget is.
    const std::uint32_t resident_reuse =
        resident_tail_rounds_down(route, resident_frontier, resident_aligned, kOutputTokens)
            ? resident_aligned
            : resident_frontier;
    const ninfer::GenerationResult resident_second = run(engine, resident_continued);
    if (resident_second.reused_prompt_tokens != resident_reuse) {
        return fail(label, "a resident conversation's next turn reused " +
                               std::to_string(resident_second.reused_prompt_tokens) +
                               " prompt tokens, expected " + std::to_string(resident_reuse));
    }
    if (const int status = compare_recall(label, "a resident conversation's next turn", route,
                                          resident_reuse, resident_second, resident_answer);
        status != 0) {
        return status;
    }
    // A conversation whose client asks for far more than it spends. The gate used to charge the
    // request's whole budget for every token it might still generate, so a client asking for its
    // model's maximum put an unaligned boundary out of reach for good. Five short turns and one long
    // one give the conversation an average to price against, and the long turn's own saving is what
    // the continuation is offered. Only the conversation's average keeps that boundary reachable:
    // priced against the budget the continuation asks for, it is not.
    std::vector<TokenId> burst_history = make_prompt(63000, 386);
    std::uint64_t burst_generated      = 0;
    std::uint32_t burst_turns          = 0;
    const auto charge                  = [&](const ninfer::GenerationResult& shaped) {
        if (!shaped.generated_token_ids.empty()) {
            burst_generated += shaped.generated_token_ids.size();
            ++burst_turns;
        }
    };
    for (int turn = 0; turn < 5; ++turn) {
        const ninfer::GenerationResult short_turn = run(engine, burst_history);
        charge(short_turn);
        append(burst_history, short_turn.generated_token_ids);
        append(burst_history, follow_up);
    }
    const std::uint32_t burst_prompt_end = static_cast<std::uint32_t>(burst_history.size());
    const ninfer::GenerationResult burst = run_budget(engine, burst_history, kBurstBudget);
    charge(burst);
    append(burst_history, burst.generated_token_ids);
    append(burst_history, follow_up);
    if (burst_turns == 0) { return fail(label, "the scenario's conversation generated nothing"); }
    const std::uint32_t burst_frontier =
        burst_prompt_end + static_cast<std::uint32_t>(burst.generated_token_ids.size()) - 1U;
    const std::uint32_t burst_aligned  = burst_prompt_end / kPrefillChunk * kPrefillChunk;
    const std::uint32_t burst_saving   = burst_frontier - burst_aligned;
    const std::uint32_t burst_expected = static_cast<std::uint32_t>(burst_generated / burst_turns);
    // The budget-priced rule this scenario is here to distinguish would have taken the boundary
    // already if the saving were this deep, so the scenario only means something while it is not.
    if (burst_saving > std::max(kPrefillChunk, 16U * kContinueBudget)) {
        return fail(label, "the scenario no longer tells the budget from the average: saving " +
                               std::to_string(burst_saving) + " already beats budget " +
                               std::to_string(kContinueBudget));
    }
    if (burst_saving <= std::max(kPrefillChunk, 16U * burst_expected)) {
        return fail(label, "the scenario's long turn no longer pays for its own boundary: saving " +
                               std::to_string(burst_saving) + " against average " +
                               std::to_string(burst_expected));
    }
    const ninfer::GenerationResult burst_continued =
        run_budget(engine, burst_history, kContinueBudget);
    if (burst_continued.reused_prompt_tokens != burst_frontier) {
        return fail(label, "a conversation continued past its long turn reused " +
                               std::to_string(burst_continued.reused_prompt_tokens) +
                               " prompt tokens, expected its frontier " +
                               std::to_string(burst_frontier));
    }
    if (burst_continued.draft_context_declined != (route == Route::DFlash2)) {
        return fail(label, "an off-grid continuation did not price its own draft: declined=" +
                               std::to_string(burst_continued.draft_context_declined));
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
    const std::uint32_t retried_reuse = interrupted_prefix;
    if (retried.reused_prompt_tokens != retried_reuse) {
        return fail(label, "a retry of a cancelled prompt reused " +
                               std::to_string(retried.reused_prompt_tokens) +
                               " prompt tokens, expected " +
                               std::to_string(retried_reuse));
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
    // route needs when each of the others costs a full model load. The default set is all three.
    const char* selected_env   = std::getenv("NINFER_TEST_ROUTE");
    const std::string selected = selected_env == nullptr ? "" : selected_env;
    std::vector<Route> routes;
    if (selected.empty()) {
        routes = {Route::Plain, Route::Mtp, Route::DFlash2};
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
            if (const int status = check_first_token_published(artifact, devices.first,
                                                               devices.second, route);
                status != 0) {
                return status;
            }
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
