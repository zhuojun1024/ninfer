#include "artifact/reader.h"
#include "artifact/schema.h"
#include <iostream>
int main(int argc, char** argv) {
    if (argc < 2) { return 2; }
    ninfer::artifact::Reader reader(argv[1]);
    const auto& dir = reader.directory();
    for (const auto& name : {"text", "mtp", "dflash", "dflash2"}) {
        try {
            const auto& c = dir.component(name);
            const auto& cfg = c.config;
            auto it = cfg.find("num_hidden_layers");
            long long nhl = (it != cfg.end() && it->is_number()) ? it->get<long long>() : -1;
            auto mt = cfg.find("model_type");
            std::string mty = (mt != cfg.end() && mt->is_string()) ? mt->get<std::string>() : "?";
            std::cout << name << " num_hidden_layers=" << nhl << " model_type=" << mty << "\n";
        } catch (const std::exception& e) {
            std::cout << name << " (none)\n";
        }
    }
    return 0;
}
