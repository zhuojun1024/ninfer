#include "models/qwen3_5/model.h"

#include <stdexcept>
#include <utility>

namespace ninfer::models::qwen3_5 {

Model::Model(Config config, LoadOptions options, ModelWeights weights,
             std::vector<BoundWeight> bound, FrontendResources resources, InstanceInfo info,
             artifact::MaterializedArtifact backing)
    : backing_(std::move(backing)), config_(std::move(config)), options_(options),
      weights_(std::move(weights)), bound_(std::move(bound)), resources_(std::move(resources)),
      info_(std::move(info)) {}

Model::~Model() = default;

ops::WeightInput Model::input(WeightUseId id) const {
    const auto& parameter = weight(id.parameter);
    const auto& use       = parameter.uses.at(id.use_index);
    ops::WeightInput result{parameter.view, use.policy, use.activation_input_divisor};
    // A GGUF matrix whose stored columns permute its input's carries the gather as its own device
    // weight; the consuming Op reads it through Weight::input_columns.
    if (use.input_columns) {
        const auto& columns = weight(*use.input_columns).view;
        result.input_columns =
            weight_tensor(columns, {static_cast<std::int32_t>(columns.shape[0])});
    }
    return result;
}

ops::WeightInput Model::input(WeightId id) const {
    if (weight(id).uses.size() != 1) {
        throw std::invalid_argument("weight input requires an explicit mathematical use");
    }
    return input(WeightUseId{id, 0});
}

std::unique_ptr<Model> make_shard_model(Config config, LoadOptions options, ModelWeights weights,
                                        std::vector<BoundWeight> bound, FrontendResources resources,
                                        InstanceInfo info, artifact::MaterializedArtifact backing) {
    return std::unique_ptr<Model>(new Model(std::move(config), options, std::move(weights),
                                            std::move(bound), std::move(resources),
                                            std::move(info), std::move(backing)));
}

} // namespace ninfer::models::qwen3_5
