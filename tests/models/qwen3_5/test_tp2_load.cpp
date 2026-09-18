#include "artifact/reader.h"
#include "core/device.h"
#include "models/qwen3_5/execution/parameters.h"
#include "models/qwen3_5/load.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

namespace {

using namespace ninfer;
using namespace ninfer::models;
namespace qwen = ninfer::models::qwen3_5;

std::pair<int, int> pick_devices() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count < 2) { return {-1, -1}; }
    std::vector<std::string> names(count);
    for (int i = 0; i < count; ++i) {
        cudaDeviceProp prop{};
        cudaGetDeviceProperties(&prop, i);
        names[i] = prop.name;
    }
    for (int a = 0; a < count; ++a) {
        for (int b = a + 1; b < count; ++b) {
            if (names[a] == names[b]) { return {a, b}; }
        }
    }
    return {0, 1};
}

void check_shard_shapes(const qwen::execution::Parameters& parameters, const qwen::Model& model,
                        int shard, bool expect_mtp) {
    const auto& weights = model.weights();
    if (expect_mtp) {
        // MTP weights are replicated whole on both shards: the proposal runs on shard 0 alone (its
        // input is bit-identical on both shards), so no part of the MTP layer may be halved.
        if (!weights.mtp.has_value()) {
            throw std::runtime_error("shard " + std::to_string(shard) +
                                     " has no MTP weights under --spec mtp");
        }
        const auto& mtp  = *weights.mtp;
        const auto& proj = model.weight(mtp.input_projection).view;
        if (proj.shape[0] != 5120 || proj.shape[1] != 10240) {
            throw std::runtime_error("shard " + std::to_string(shard) +
                                     " MTP input_projection is not replicated [5120,10240]");
        }
        const auto* mixer = std::get_if<qwen::AttentionWeights>(&mtp.layer.mixer);
        if (mixer == nullptr) { throw std::runtime_error("MTP layer mixer is not attention"); }
        const auto& q = model.weight(mixer->query).view;
        if (q.shape[0] != 6144 || q.shape[1] != 5120) {
            throw std::runtime_error("shard " + std::to_string(shard) +
                                     " MTP attention query is not replicated [6144,5120]");
        }
    }
    const auto layer0   = weights.text.layers[0];
    const auto* dense   = std::get_if<qwen::DenseWeights>(&layer0.ffn);
    if (!dense) { throw std::runtime_error("layer 0 FFN is not dense"); }
    // gate/up: full [17408,5120] -> shard [8704,5120].
    const auto& gate = model.weight(dense->gate).view;
    if (gate.shape[0] != 8704 || gate.shape[1] != 5120) {
        throw std::runtime_error("shard " + std::to_string(shard) +
                                 " gate shape is not [8704,5120]");
    }
    // down: full [5120,17408] -> shard [5120,8704].
    const auto& down = model.weight(dense->down).view;
    if (down.shape[0] != 5120 || down.shape[1] != 8704) {
        throw std::runtime_error("shard " + std::to_string(shard) +
                                 " down shape is not [5120,8704]");
    }
    // output_head: replicated in full. The head is weight-tied to the token embedding, and the
    // TP-2 forward computes the full-vocabulary logits independently on each shard (the final
    // hidden state is identical on both shards, so no head all-reduce is needed).
    const auto& head = model.weight(weights.text.output_head).view;
    if (head.shape[0] != 248320 || head.shape[1] != 5120) {
        throw std::runtime_error("shard " + std::to_string(shard) +
                                 " output_head shape is not [248320,5120]");
    }
    // The head-split mixer weights are per-shard halves (attention q [3072,5120] for a
    // full-attn layer: 12 of 24 q heads; GDN q [1024,5120] for a GDN layer: 8 of 16 k heads).
    const auto& mixer = layer0.mixer;
    if (const auto* attention = std::get_if<qwen::AttentionWeights>(&mixer)) {
        const auto& q = model.weight(attention->query).view;
        if (q.shape[0] != 3072 || q.shape[1] != 5120) {
            throw std::runtime_error("shard " + std::to_string(shard) +
                                     " attention query is not per-shard [3072,5120]");
        }
    } else {
        const auto& gdn = std::get<qwen::GdnWeights>(mixer);
        const auto& q   = model.weight(gdn.query).view;
        if (q.shape[0] != 1024 || q.shape[1] != 5120) {
            throw std::runtime_error("shard " + std::to_string(shard) +
                                     " GDN query is not per-shard [1024,5120]");
        }
    }
    (void)parameters;
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::filesystem::path path;
        LoadOptions options;
        options.vision = false;
        options.speculative = SpeculativeBackend::None;
        for (int i = 1; i < argc; ++i) {
            const std::string arg(argv[i]);
            const auto value = [&]() -> std::string {
                if (++i >= argc) { throw std::invalid_argument("missing value for " + arg); }
                return argv[i];
            };
            if (arg == "--artifact") {
                path = value();
            } else if (arg == "--spec") {
                const std::string backend = value();
                if (backend == "mtp") {
                    options.speculative = SpeculativeBackend::Mtp;
                } else {
                    throw std::invalid_argument("--spec supports mtp only");
                }
            } else if (arg == "--help") {
                std::cout << "--artifact PATH [--spec mtp]\n";
                return 0;
            } else {
                throw std::invalid_argument("unknown argument " + arg);
            }
        }
        if (path.empty()) {
            std::cout << "SKIP: supply an explicit --artifact\n";
            return 77;
        }
        const auto [dev0, dev1] = pick_devices();
        if (dev0 < 0) {
            std::cout << "SKIP: need two GPUs\n";
            return 77;
        }
        std::unique_ptr<DeviceContext> device0(new DeviceContext(dev0));
        std::unique_ptr<DeviceContext> device1(new DeviceContext(dev1));
        artifact::Reader reader(path);
        auto plan = qwen::plan_load(reader, options);
        auto [model0, model1] =
            qwen::materialize_model_tp2(std::move(plan), *device0, *device1, nullptr);
        // Construct both shard Parameters: this is the blocker validated this round.
        qwen::execution::Parameters parameters0(*model0);
        qwen::execution::Parameters parameters1(*model1);
        const bool expect_mtp = options.speculative == SpeculativeBackend::Mtp;
        check_shard_shapes(parameters0, *model0, 0, expect_mtp);
        check_shard_shapes(parameters1, *model1, 1, expect_mtp);
        std::cout << path.filename().string() << ": TP-2 dual-shard load passed "
                  << "devices=" << dev0 << "," << dev1 << " layers="
                  << model0->weights().text.layers.size() << " shard_gate=[8704,5120] "
                  << "shard_down=[5120,8704] head=[248320,5120] replicated\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
