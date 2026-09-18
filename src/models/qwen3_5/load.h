#pragma once

#include "core/tp/tp_materialize.h"
#include "models/qwen3_5/model.h"

#include <filesystem>
#include <memory>
#include <span>
#include <utility>
#include <vector>

namespace ninfer::artifact {
class Reader;
struct ParameterReference;
} // namespace ninfer::artifact

namespace ninfer::models::qwen3_5 {

// Cold load plan borrows its Reader until materialization. Its selected Host bytes and parsed
// resources already have owners; no Program or device allocation is needed for plan_load.
class LoadPlan {
public:
    ~LoadPlan();
    LoadPlan(LoadPlan&&) noexcept;
    LoadPlan& operator=(LoadPlan&&) noexcept;
    LoadPlan(const LoadPlan&)            = delete;
    LoadPlan& operator=(const LoadPlan&) = delete;

    [[nodiscard]] const Config& config() const;
    [[nodiscard]] const ModelWeights& weights() const;
    [[nodiscard]] const FrontendResources& resources() const;
    [[nodiscard]] const artifact::MaterializationPlan& materialization() const;
    [[nodiscard]] const artifact::ParameterReference& parameter(WeightId id) const;
    [[nodiscard]] std::span<const WeightUse> uses(WeightId id) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    explicit LoadPlan(std::unique_ptr<Impl> impl);
    friend LoadPlan plan_load(const artifact::Reader&, LoadOptions);
    friend std::unique_ptr<Model> materialize_model(LoadPlan&&, DeviceContext&,
                                                    const StartupObserver*);
    friend std::pair<std::unique_ptr<Model>, std::unique_ptr<Model>>
    materialize_model_tp2(LoadPlan&&, DeviceContext&, DeviceContext&, const StartupObserver*);
};

[[nodiscard]] LoadPlan plan_load(const artifact::Reader& reader, LoadOptions options = {});
[[nodiscard]] std::unique_ptr<Model> materialize_model(LoadPlan&& plan, DeviceContext& device,
                                                       const StartupObserver* observer = nullptr);
[[nodiscard]] std::unique_ptr<Model> load_model(const std::filesystem::path& path,
                                                LoadOptions options, DeviceContext& device,
                                                const StartupObserver* observer = nullptr);

// TP-2 load: materialize the plan onto two GPUs (split per spec) and build one Model per shard.
// device0 receives shard 0, device1 receives shard 1. The split spec is derived from the
// artifact directory and the parsed text config.
[[nodiscard]] std::pair<std::unique_ptr<Model>, std::unique_ptr<Model>>
materialize_model_tp2(LoadPlan&& plan, DeviceContext& device0, DeviceContext& device1,
                      const StartupObserver* observer = nullptr);

} // namespace ninfer::models::qwen3_5
