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
        if (auto* t = std::get_if<ninfer::artifact::TensorObject>(&o))
            return t->format + " " + std::to_string(t->shape[0]) + "x" + std::to_string(t->shape.size()>1?t->shape[1]:0);
        return std::string("res");
    };
    const char* names[] = {
        "text/layers/0/mlp/down", "text/layers/0/gdn/output",
        "text/layers/11/attention/output", "text/token_embedding",
        "text/output_head", "text/final_norm", "text/layers/0/input_norm",
        "text/layers/0/gdn/convolution", "text/layers/0/gdn/norm",
        "text/layers/11/attention/query_norm", "text/layers/0/gdn/a_log",
    };
    for (auto name : names) {
        auto it = dir.bindings.find(name);
        if (it == dir.bindings.end()) { std::cout << name << " MISSING\n"; continue; }
        const auto& b = it->second;
        std::cout << name << " whole=" << b.whole_object;
        if (!b.whole_object) {
            for (auto& p : b.parts) std::cout << " obj" << p.object.index << "(" << info(p.object.index) << ")";
        }
        std::cout << " elements=" << b.elements << "\n";
    }
    return 0;
}
