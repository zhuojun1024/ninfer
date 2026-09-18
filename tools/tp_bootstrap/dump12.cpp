#include "artifact/reader.h"
#include "artifact/schema.h"
#include <iostream>
#include <string>
int main(int argc, char** argv) {
    if (argc < 2) { return 2; }
    ninfer::artifact::Reader reader(argv[1]);
    const auto& dir = reader.directory();
    auto fmt_of = [&](std::size_t idx) -> std::string {
        const auto& o = dir.object({idx});
        if (auto* t = std::get_if<ninfer::artifact::TensorObject>(&o))
            return t->format + " " + std::to_string(t->shape.size()>=2?t->shape[0]:0) + "x" + std::to_string(t->shape.size()>=2?t->shape[1]:0);
        return "res";
    };
    // Print top-level text bindings (embedding, output_head, norms) with object indices.
    for (const auto& [name, b] : dir.bindings) {
        if (name.rfind("text/", 0) != 0) continue;
        if (name.rfind("text/layers/", 0) == 0) continue;
        std::cout << name;
        if (b.whole_object) { std::cout << " whole obj" << b.whole_object << " (" << fmt_of(b.whole_object) << ") elements=" << b.elements << "\n"; continue; }
        std::cout << " parts=[";
        for (auto& p : b.parts) std::cout << "obj" << p.object.index << "(" << fmt_of(p.object.index) << "):" << p.begin << "-" << p.end << " ";
        std::cout << "] elements=" << b.elements << "\n";
    }
    return 0;
}
