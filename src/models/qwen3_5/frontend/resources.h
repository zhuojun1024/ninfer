#pragma once

#include "models/qwen3_5/config.h"
#include "models/qwen3_5/frontend/tokenizer.h"

#include <memory>
#include <string>

namespace ninfer::models::qwen3_5 {

// Owned byte payloads for the frontend resources. The tokenizer and chat template are parsed
// from these at load time. Owning the bytes (rather than borrowing a Host backing) keeps each
// Model self-contained, which the TP-2 shard path requires: the shard backings hold only device
// objects, so a borrowed view would dangle once the load plan is destroyed.
struct FrontendResources {
    std::string tokenizer_json;
    std::string tokenizer_config_json;
    std::string chat_template_jinja;
    std::string generation_config_json;
    std::string preprocessor_config_json;
    std::string video_preprocessor_config_json;
    std::shared_ptr<const frontend::Tokenizer> tokenizer;
    std::uint32_t public_token_count = 0;
};

void parse_resources(FrontendResources& resources, const Config& config);

} // namespace ninfer::models::qwen3_5
