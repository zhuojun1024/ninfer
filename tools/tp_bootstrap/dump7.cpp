#include "artifact/reader.h"
#include "artifact/schema.h"
#include <iostream>
#include <string>
int main(int argc, char** argv) {
    if (argc < 2) { return 2; }
    ninfer::artifact::Reader reader(argv[1]);
    const auto& dir = reader.directory();
    auto info = [&](std::size_t idx) -> std::string {
        const auto& o = dir.object({idx});
        if (auto* t = std::get_if<ninfer::artifact::TensorObject>(&o)) {
            std::string s = t->format;
            for (auto d : t->shape) s += "[" + std::to_string(d) + "]";
            return s;
        }
        return std::string("RES");
    };
    // whole-object text params: real object index via parts[0]
    const char* names[] = {
        "text/layers/0/mlp/down", "text/layers/0/gdn/output",
        "text/layers/11/attention/output", "text/token_embedding",
        "text/output_head", "text/final_norm", "text/layers/0/input_norm",
        "text/layers/0/gdn/convolution", "text/layers/0/gdn/norm",
        "text/layers/0/gdn/a_log", "text/layers/0/gdn/dt_bias",
        "text/layers/11/attention/query_norm",
    };
    for (auto name : names) {
        auto it = dir.bindings.find(name);
        if (it == dir.bindings.end()) { std::cout << name << " MISSING\n"; continue; }
        const auto& b = it->second;
        std::size_t idx = b.parts.empty() ? 0 : b.parts[0].object.index;
        std::cout << name << " whole=" << b.whole_object << " nparts=" << b.parts.size()
                  << " obj" << idx << "(" << info(idx) << ") elems=" << b.elements << "\n";
    }
    return 0;
}
