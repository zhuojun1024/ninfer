// TP-2 cross-session KV retention.
//
// The device KV pool holds one conversation at a time, so a request that belongs to another one
// evicts the resident session into pinned host memory and copies the returning session back. This
// test drives several alternating conversations through the public Engine route and checks the
// observable consequences: a returning conversation is recalled rather than prefilled
// (reused_prompt_tokens), its output matches a from-scratch prefill token for token, and a
// conversation the LRU budget evicted is prefilled from zero again.
//
// The scenario runs twice: on the plain route and with MTP enabled, because the MTP layer keeps
// its own KV slab on shard 0 and that slab travels through the same eviction transaction. The
// oracle in each round uses the same speculative configuration with retention disabled.
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

ninfer::EngineOptions engine_options(const char* artifact, int device_a, int device_b,
                                     bool retention, bool mtp) {
    ninfer::EngineOptions options;
    options.artifact_path                        = artifact;
    options.device                               = device_a;
    options.device_b                             = device_b;
    options.max_context                          = kMaxContext;
    options.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(kMaxContext);
    options.prefill_chunk                       = 256;
    options.max_concurrency                     = 1;
    options.max_pending_requests                = 1;
    if (mtp) {
        options.speculative.backend      = ninfer::SpeculativeBackend::Mtp;
        options.speculative.draft_tokens = 2;
        // The server recipe runs MTP on fp8 KV, and this is the route the cross-session slabs
        // have to carry; bf16 KV with MTP never gets past its first prefill (see the note in
        // docs/tp2-dual-5060ti.md).
        options.kv_cache = ninfer::KvCacheStorage::Fp8E4M3Row256;
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

void append(std::vector<TokenId>& destination, const std::vector<TokenId>& source) {
    destination.insert(destination.end(), source.begin(), source.end());
}

int fail(const std::string& label, const std::string& message) {
    std::cerr << "FAIL (" << label << "): " << message << '\n';
    return 1;
}

// One full A/B/A/LRU scenario. Returns 0 on success, 1 on a failed assertion.
int run_scenario(const char* artifact, int device_a, int device_b, bool mtp) {
    const std::string label = mtp ? "mtp" : "plain";

    const std::vector<TokenId> opening    = make_prompt(1200, 64);
    const std::vector<TokenId> follow_up  = make_prompt(4000, 16);
    const std::vector<TokenId> other_a    = make_prompt(7000, 48);
    const std::vector<TokenId> other_b    = make_prompt(20000, 48);
    const std::vector<TokenId> other_c    = make_prompt(30000, 48);
    const std::vector<TokenId> other_d    = make_prompt(40000, 48);

    // The oracle prefills the continued prompt from zero with retention disabled. A recalled
    // walk reuses the byte-identical KV prefix, so the greedy answers have to agree exactly.
    std::vector<TokenId> opening_answer;
    std::vector<TokenId> continued_answer;
    {
        ninfer::Engine oracle(engine_options(artifact, device_a, device_b, false, mtp));
        opening_answer = run(oracle, opening).generated_token_ids;
        std::vector<TokenId> continued = opening;
        append(continued, opening_answer);
        append(continued, follow_up);
        continued_answer = run(oracle, std::move(continued)).generated_token_ids;
    }
    if (opening_answer.empty() || continued_answer.empty()) {
        return fail(label, "the oracle produced no tokens");
    }

    ninfer::Engine engine(engine_options(artifact, device_a, device_b, true, mtp));

    // Conversation A: the empty catalog makes this a full prefill that installs the entry.
    const ninfer::GenerationResult a_first = run(engine, opening);
    if (a_first.reused_prompt_tokens != 0) {
        return fail(label, "the first conversation was not prefilled from zero");
    }
    if (a_first.generated_token_ids != opening_answer) {
        return fail(label, "a fresh conversation diverged from the oracle");
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
        return fail(label, "a recalled conversation diverged from the oracle");
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
        return fail(label, "a re-prefilled conversation diverged from the oracle");
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

    // NINFER_TEST_ROUTE=plain|mtp narrows the run to one route, which is what a failing route
    // needs when the other one costs a full model load.
    const char* route    = std::getenv("NINFER_TEST_ROUTE");
    const std::string selected = route == nullptr ? "" : route;
    const bool only_mtp   = selected == "mtp";
    const bool only_plain = selected == "plain";

    try {
        if (!only_mtp) {
            if (const int status = run_scenario(artifact, devices.first, devices.second, false);
                status != 0) {
                return status;
            }
        }
        if (!only_plain) {
            if (const int status = run_scenario(artifact, devices.first, devices.second, true);
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
