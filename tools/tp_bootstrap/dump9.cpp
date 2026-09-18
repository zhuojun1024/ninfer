#include "artifact/reader.h"
#include "artifact/schema.h"
#include <iostream>
#include <string>
#include <map>
int main(int argc, char** argv) {
    if (argc < 2) { return 2; }
    ninfer::artifact::Reader reader(argv[1]);
    const auto& dir = reader.directory();
    // For FFN-shaped objects, print the binding names that reference them.
    for (std::size_t idx = 0; idx < dir.objects.size(); ++idx) {
        const auto* t = std::get_if<ninfer::artifact::TensorObject>(&dir.objects[idx]);
        if (!t || t->shape.size() < 2) { continue; }
        const auto n = t->shape[0], k = t->shape[1];
        const bool ffn = (n == 5120 && k == 17408) || (n == 34816 && k == 5120);
        if (!ffn) { continue; }
        std::string names;
        for (const auto& [name, binding] : dir.bindings) {
            if (name.rfind("text/", 0) != 0) { continue; }
            bool ref = binding.whole_object && binding.whole_object == idx;
            if (!ref) { for (const auto& p : binding.parts) { if (p.object.index == idx) { ref = true; break; } } }
            if (ref) { names += name + " "; }
        }
        std::cout << "obj" << idx << " " << t->format << " " << names << "\n";
    }
    return 0;
}
