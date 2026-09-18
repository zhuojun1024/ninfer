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
        return std::string("res");
    };
    for (auto& [name, b] : dir.bindings) {
        if (name.rfind("text/", 0) != 0) continue;
        bool top = name.find("layers/") == std::string::npos;
        bool l0  = name.find("layers/0/") != std::string::npos;
        bool l11 = name.find("layers/11/") != std::string::npos;
        if (!top && !l0 && !l11) continue;
        std::cout << name << " whole=" << b.whole_object;
        if (!b.whole_object) {
            for (auto& p : b.parts) std::cout << " obj" << p.object.index << "(" << info(p.object.index) << ")";
        } else {
            std::cout << " obj" << b.whole_object << "(" << info(b.whole_object) << ")";
        }
        std::cout << "\n";
    }
    return 0;
}
