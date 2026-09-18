#include "artifact/reader.h"
#include "artifact/schema.h"

#include <iostream>
#include <string>

int main(int argc, char** argv) {
    if (argc < 2) { std::cerr << "usage: dump_artifact <path.ninfer> [layer_prefix]\n"; return 2; }
    ninfer::artifact::Reader reader(argv[1]);
    const auto& dir = reader.directory();
    std::cout << "objects: " << dir.objects.size() << "\n";
    for (std::size_t i = 0; i < dir.objects.size(); ++i) {
        const auto& o = dir.object({i});
        if (auto* t = std::get_if<ninfer::artifact::TensorObject>(&o)) {
            std::cout << "[" << i << "] " << t->id << " shape=";
            for (auto d : t->shape) std::cout << d << ",";
            std::cout << " fmt=" << t->format << " layout=" << t->layout << " bytes=" << t->bytes << "\n";
        }
    }
    std::cout << "\nbindings (layer 0 + embedding + head):\n";
    for (const auto& [name, b] : dir.bindings) {
        const bool interesting =
            name.find("layers/0/") != std::string::npos ||
            name.find("token_embedding") != std::string::npos ||
            name.find("output_head") != std::string::npos ||
            name.find("final_norm") != std::string::npos;
        if (!interesting) { continue; }
        std::cout << name << " whole=" << b.whole_object;
        if (!b.whole_object) {
            std::cout << " parts=[";
            for (auto& p : b.parts) std::cout << p.object.index << ":" << p.begin << "-" << p.end << " ";
            std::cout << "]";
        }
        std::cout << " elements=" << b.elements << "\n";
    }
    return 0;
}
