// Single-instance determinism probe for the TP-2 DFlash2 route (PLAN.md section 3.6, "B6 result").
//
// The session scenario compares a retention-enabled Engine against a from-scratch oracle Engine in
// one process, and its shared-system-prompt switch flips about two runs out of five: the Engine built
// second in the process disagrees with the one built first, while an identically configured walk in
// the compiler's own order stays put. This probe builds ONE Engine, replays the same request sequence
// on it (each continuation rendered from this engine's own answer, which is what a client does) and
// prints every walk's generated tokens plus a digest of them all. Running it in independent processes
// answers whether the product shape - one Engine per process - is deterministic at all, which is the
// question that decides whether the route can be released.

#include "ninfer/engine.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

using TokenId = std::int32_t;

constexpr std::uint32_t kMaxContext   = 2048;
constexpr std::uint32_t kOutputTokens = 8;
constexpr std::size_t kHostKvBytes    = 128ULL << 20;
constexpr std::uint32_t kSessions     = 3;

std::pair<int, int> pick_devices() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count < 2) { return {-1, -1}; }
    std::vector<std::string> names(static_cast<std::size_t>(count));
    for (int index = 0; index < count; ++index) {
        cudaDeviceProp prop{};
        cudaGetDeviceProperties(&prop, index);
        names[static_cast<std::size_t>(index)] = prop.name;
    }
    for (int a = 0; a < count; ++a) {
        for (int b = a + 1; b < count; ++b) {
            if (names[static_cast<std::size_t>(a)] == names[static_cast<std::size_t>(b)]) {
                return {a, b};
            }
        }
    }
    return {-1, -1};
}

ninfer::EngineOptions engine_options(const char* artifact, int device_a, int device_b) {
    ninfer::EngineOptions options;
    options.artifact_path                          = artifact;
    options.device                                 = device_a;
    options.device_b                               = device_b;
    options.max_context                            = kMaxContext;
    options.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(kMaxContext);
    options.prefill_chunk                          = 256;
    options.max_concurrency                        = 1;
    options.max_pending_requests                   = 1;
    options.speculative.backend                    = ninfer::SpeculativeBackend::DFlash2;
    options.speculative.draft_tokens               = 7;
    options.kv_cache                               = ninfer::KvCacheStorage::Fp8E4M3Row256;
    options.context_cache.host_kv_capacity_bytes   = kHostKvBytes;
    options.context_cache.max_private_continuations = kSessions;
    return options;
}

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

void append(std::vector<TokenId>& destination, const std::vector<TokenId>& source) {
    destination.insert(destination.end(), source.begin(), source.end());
}

std::string tokens_text(const std::vector<TokenId>& tokens) {
    std::string text = "[";
    for (std::size_t index = 0; index < tokens.size(); ++index) {
        if (index != 0) { text += " "; }
        text += std::to_string(tokens[index]);
    }
    return text + "]";
}

} // namespace

int main(int argc, char** argv) {
    const char* artifact = argc > 1 ? argv[1] : std::getenv("NINFER_TEST_ARTIFACT");
    if (artifact == nullptr) {
        std::cout << "skip: set NINFER_TEST_ARTIFACT or pass the artifact path\n";
        return 77;
    }
    const auto [device_a, device_b] = pick_devices();
    if (device_a < 0) {
        std::cout << "skip: two identical sm_120a devices are not available\n";
        return 77;
    }

    try {
        ninfer::Engine engine(engine_options(artifact, device_a, device_b));

        std::uint64_t digest = 1469598103934665603ULL;
        const auto digest_walk = [&digest](const ninfer::GenerationResult& result) {
            for (const TokenId token : result.generated_token_ids) {
                digest = (digest ^ static_cast<std::uint32_t>(token)) * 1099511628211ULL;
            }
            return digest;
        };
        const auto walk = [&engine, &digest_walk](const char* name,
                                                  std::vector<TokenId> prompt) {
            const ninfer::GenerationResult result =
                engine.generate(engine.prepare_tokens(std::move(prompt)), greedy_request());
            std::cout << "[solo] " << name << " reused=" << result.reused_prompt_tokens
                      << " declined=" << (result.draft_context_declined ? 1 : 0)
                      << " tokens=" << tokens_text(result.generated_token_ids) << '\n';
            digest_walk(result);
            return result;
        };

        const std::vector<TokenId> opening   = make_prompt(1200, 64);
        const std::vector<TokenId> follow_up = make_prompt(4000, 16);
        const std::vector<TokenId> other_a   = make_prompt(7000, 48);
        const std::vector<TokenId> other_b   = make_prompt(20000, 48);
        const std::vector<TokenId> other_c   = make_prompt(30000, 48);
        const std::vector<TokenId> other_d   = make_prompt(40000, 48);
        const std::vector<TokenId> system_prompt = make_prompt(500, 512);
        std::vector<TokenId> shared_a        = system_prompt;
        append(shared_a, make_prompt(9000, 128));
        std::vector<TokenId> shared_b        = system_prompt;
        append(shared_b, make_prompt(20000, 128));

        std::vector<TokenId> a_continued = opening;
        append(a_continued, walk("opening", opening).generated_token_ids);
        append(a_continued, follow_up);
        walk("other_a", other_a);
        walk("a_continued", a_continued);
        walk("other_b", other_b);
        walk("other_c", other_c);
        walk("other_d", other_d);
        walk("a_continued_evicted", a_continued);
        walk("shared_a", shared_a);
        walk("shared_b", shared_b);

        std::cout << "[solo] digest=0x" << std::hex << digest << std::dec << '\n';
        std::cout << "[solo] PASS\n";
        return 0;
    } catch (const std::exception& error) {
        // The route is construction-refused while its walk is not deterministic (PLAN.md 3.6, "B6
        // result"). This probe is the harness that measures it the moment that gate opens, so a
        // refused route is a skip rather than a failure.
        if (std::string(error.what()).find("dflash2 is withheld") != std::string::npos) {
            std::cout << "skip: the DFlash2 route is construction-refused; this probe measures it "
                         "once the option gate opens (PLAN.md 3.6)\n";
            return 77;
        }
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
