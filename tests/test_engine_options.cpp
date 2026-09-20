#include "runtime/engine/model_instance.h"

#include "ninfer/types.h"

#include <iostream>

namespace {

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

ninfer::EngineOptions generation_options() {
    ninfer::EngineOptions options;
    options.purpose = ninfer::EnginePurpose::Generation;
    return options;
}

} // namespace

// The TP-2 generation route runs without a context cache, but its generation core stores
// prefix-reuse checkpoints in pinned host memory (one host StateImage per checkpoint), so that
// route keeps the host state-image budget. Any other disabled-cache route has no user for that
// memory and drops it.
int main() {
    int failures = 0;

    {
        ninfer::EngineOptions options          = generation_options();
        options.device_b                       = 1;
        options.context_cache.host_state_slots = 16;
        const ninfer::EngineOptions resolved = ninfer::runtime::normalize_engine_options(options);
        failures += check(resolved.device_b == 1, "TP-2 route lost its second device");
        failures += check(resolved.max_concurrency == 1, "TP-2 route did not normalize concurrency");
        failures += check(!resolved.context_cache.enabled, "TP-2 route kept the context cache");
        failures += check(resolved.context_cache.host_state_slots == 16,
                          "TP-2 route dropped the host state-image budget");
        failures += check(resolved.context_cache.device_state_slots.value_or(1) == 0,
                          "TP-2 route kept device state-image slots");
        failures += check(resolved.context_cache.host_kv_capacity_bytes == 0,
                          "TP-2 route kept a host KV arena");
    }
    {
        ninfer::EngineOptions options = generation_options();
        options.device_b              = 1;
        const ninfer::EngineOptions resolved = ninfer::runtime::normalize_engine_options(options);
        failures += check(resolved.context_cache.host_state_slots == ninfer::kDefaultHostStateSlots,
                          "TP-2 route did not keep the default host state-image budget");
    }
    {
        // CausalScoring is not the TP-2 generation route: no core there keeps host state images.
        ninfer::EngineOptions options          = generation_options();
        options.purpose                        = ninfer::EnginePurpose::CausalScoring;
        options.device_b                       = 1;
        options.context_cache.host_state_slots = 16;
        const ninfer::EngineOptions resolved = ninfer::runtime::normalize_engine_options(options);
        failures += check(resolved.context_cache.host_state_slots == 0,
                          "a non-TP-2 disabled-cache route kept the host state-image budget");
    }

    if (failures != 0) {
        std::cerr << failures << " engine option checks failed\n";
        return 1;
    }
    std::cout << "engine options: ok\n";
    return 0;
}
