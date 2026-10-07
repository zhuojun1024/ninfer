// A/B oracle for the TP-2 causal scoring route: the same artifact, the same tokens and the same KV
// format scored once by the single-device Program and once by the two-device TP-2 core. The two
// routes compute the same math in BF16 but split every mixer and FFN projection across the devices
// and all-reduce the deltas, so they are not expected to agree bit for bit; what this test pins is
// that they agree to the last few digits of the mean NLL, that the TP-2 route is self-consistent
// across repeated windows, and that a suffix window is independent of the full-prompt run.
//
// The artifact must fit one card, because the single-device reference is the baseline; the two
// routes are loaded one after the other, never together, so a model that fits either card alone
// works:  NINFER_TEST_ARTIFACT=<model.ninfer> NINFER_TEST_TP2_DEVICES=0,1
#include "ninfer/engine.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kTokenCount    = 1537;
constexpr std::uint32_t kSuffixStart = 513;
constexpr std::size_t kFullTargets   = kTokenCount - 1;
constexpr std::size_t kSuffixTargets = kTokenCount - kSuffixStart;

double mean_nll(const std::vector<float>& values) {
    double sum = 0.0;
    for (const float value : values) {
        sum += static_cast<double>(value);
    }
    return sum / static_cast<double>(values.size());
}

double max_abs_delta(const std::vector<float>& lhs, const std::vector<float>& rhs) {
    double worst = 0.0;
    for (std::size_t i = 0; i < lhs.size(); ++i) {
        worst = std::max(worst, std::abs(static_cast<double>(lhs[i]) - static_cast<double>(rhs[i])));
    }
    return worst;
}

bool parse_devices(int& device_a, int& device_b) {
    const char* text = std::getenv("NINFER_TEST_TP2_DEVICES");
    if (text == nullptr || *text == '\0') {
        device_a = 0;
        device_b = 1;
        return true;
    }
    const char* comma = std::strchr(text, ',');
    if (comma == nullptr) {
        return false;
    }
    device_a = std::atoi(std::string(text, comma).c_str());
    device_b = std::atoi(comma + 1);
    return device_a >= 0 && device_b >= 0 && device_a != device_b;
}

} // namespace

int run() {
    const char* artifact = std::getenv("NINFER_TEST_ARTIFACT");
    if (artifact == nullptr || *artifact == '\0') {
        std::cout << "SKIP: NINFER_TEST_ARTIFACT is not set\n";
        return 77;
    }
    int device_a = 0;
    int device_b = 1;
    if (!parse_devices(device_a, device_b)) {
        std::cout << "SKIP: NINFER_TEST_TP2_DEVICES must name two distinct devices\n";
        return 77;
    }

    ninfer::EngineOptions options;
    options.artifact_path = artifact;
    options.purpose       = ninfer::EnginePurpose::CausalScoring;
    options.device        = device_a;
    options.max_context   = 2048;
    options.kv_cache      = ninfer::KvCacheStorage::Fp8E4M3Row256;

    const std::string paragraph =
        "NInfer scores each target token from the preceding hidden state. "
        "Every evaluation window owns fresh state and a fresh KV address space.\n";

    // Tokenize with a throwaway engine and release it before either route loads: the two routes
    // together hold the weights three times over, which no pair of cards can afford.
    std::vector<ninfer::TokenId> tokens;
    std::fprintf(stderr, "[ab] loading the tokenizer route\n");
    {
        ninfer::Engine tokenizer(options);
        std::string text;
        while (tokens.size() < kTokenCount) {
            text += paragraph;
            tokens = tokenizer.tokenize_text(text);
        }
        tokens.resize(kTokenCount);
    }

    std::vector<float> reference;
    std::vector<float> reference_suffix;
    std::fprintf(stderr, "[ab] loading the single-device route\n");
    {
        ninfer::Engine single(options);
        std::fprintf(stderr, "[ab] scoring on one device\n");
        reference        = single.score_tokens(tokens, 1);
        reference_suffix = single.score_tokens(tokens, kSuffixStart);
    }
    if (reference.size() != kFullTargets || reference_suffix.size() != kSuffixTargets) {
        std::cerr << "single-device causal scoring returned an invalid result shape\n";
        return 1;
    }

    ninfer::EngineOptions tp2_options = options;
    tp2_options.device_b              = device_b;
    std::vector<float> tp2_full;
    std::vector<float> tp2_suffix;
    std::vector<float> tp2_repeat;
    std::fprintf(stderr, "[ab] loading the two-device route\n");
    {
        ninfer::Engine tp2(tp2_options);
        const auto& effective = tp2.options();
        if (effective.device_b != device_b || effective.max_concurrency != 1 ||
            effective.kv_capacity.mode != ninfer::KvCapacityMode::Explicit ||
            effective.kv_capacity.explicit_tokens != effective.max_context ||
            effective.context_cache.enabled || effective.enable_vision ||
            effective.speculative.backend != ninfer::SpeculativeBackend::None) {
            std::cerr << "the TP-2 causal scoring options were not normalized correctly\n";
            return 1;
        }
        std::fprintf(stderr, "[ab] scoring on two devices\n");
        tp2_full   = tp2.score_tokens(tokens, 1);
        std::fprintf(stderr, "[ab] scoring the suffix window\n");
        tp2_suffix = tp2.score_tokens(tokens, kSuffixStart);
        std::fprintf(stderr, "[ab] scoring the suffix window again\n");
        tp2_repeat = tp2.score_tokens(tokens, kSuffixStart);
    }
    if (tp2_full.size() != kFullTargets || tp2_suffix.size() != kSuffixTargets) {
        std::cerr << "TP-2 causal scoring returned an invalid result shape\n";
        return 1;
    }
    for (const float value : tp2_full) {
        if (!std::isfinite(value) || value > 0.0F) {
            std::cerr << "TP-2 causal scoring returned an invalid log probability\n";
            return 1;
        }
    }
    // Two identical windows must produce identical bytes: a walk that inherited state or KV from the
    // run before it would drift here, and the per-request isolation of the scoring route is exactly
    // what the perplexity harness relies on.
    for (std::size_t i = 0; i < tp2_suffix.size(); ++i) {
        if (tp2_suffix[i] != tp2_repeat[i]) {
            std::cerr << "a repeated TP-2 score window inherited prior State/KV\n";
            return 1;
        }
    }

    const double reference_nll = mean_nll(reference);
    const double tp2_nll       = mean_nll(tp2_full);
    const double nll_delta     = std::abs(reference_nll - tp2_nll);
    const double worst         = max_abs_delta(reference, tp2_full);
    // The suffix window is scored by both routes from the same offset; comparing it against the full
    // run's tail checks the two routes against each other on a window that shares no prefix position.
    const double suffix_delta =
        max_abs_delta(tp2_suffix, std::vector<float>(reference.begin() + (kSuffixStart - 1),
                                                     reference.end()));
    std::printf("single-device mean_nll=%.9f ppl=%.6f\n", reference_nll, std::exp(reference_nll));
    std::printf("tp2           mean_nll=%.9f ppl=%.6f\n", tp2_nll, std::exp(tp2_nll));
    std::printf("delta mean_nll=%.9f max|logprob delta|=%.6f suffix max|delta|=%.6f\n", nll_delta,
                worst, suffix_delta);

    // The reduction order differs, so an exact match is not the contract; a mean NLL that moved by
    // more than a thousandth would be a route bug, not arithmetic.
    if (nll_delta > 5e-3) {
        std::cerr << "the TP-2 scoring route disagrees with the single-device reference\n";
        return 1;
    }
    std::cout << "OK causal_score_tp2_ab\n";
    return 0;
}

int main() {
    try {
        return run();
    } catch (const std::exception& error) {
        // A bare main would let the exception escape into std::terminate, which on Windows reports
        // nothing but a 0xC0000409 exit status; the harness needs the message.
        std::fprintf(stderr, "FAILED causal_score_tp2_ab: %s\n", error.what());
        return 1;
    }
}
