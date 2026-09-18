#include "artifact/reader.h"
#include "artifact/schema.h"
#include <iostream>
#include <string>
int main(int argc, char** argv) {
    if (argc < 2) { return 2; }
    ninfer::artifact::Reader reader(argv[1]);
    const auto& dir = reader.directory();
    // List every tensor object whose shape is an FFN shape ([5120,17408] down or [34816,5120] gate_up),
    // with its format, and whether it is referenced by a text/ binding.
    auto is_text = [&](std::size_t idx) {
        for (const auto& [name, binding] : dir.bindings) {
            if (name.rfind("text/", 0) != 0) { continue; }
            if (binding.whole_object && binding.whole_object == idx) { return true; }
            for (const auto& p : binding.parts) { if (p.object.index == idx) { return true; } }
        }
        return false;
    };
    for (std::size_t idx = 0; idx < dir.objects.size(); ++idx) {
        const auto* t = std::get_if<ninfer::artifact::TensorObject>(&dir.objects[idx]);
        if (!t || t->shape.size() < 2) { continue; }
        const auto n = t->shape[0], k = t->shape[1];
        const bool down = (n == 5120 && k == 17408);
        const bool gateup = (n == 34816 && k == 5120);
        if (!down && !gateup) { continue; }
        std::cout << "obj" << idx << " " << (down ? "down[5120,17408]" : "gate_up[34816,5120]")
                  << " fmt=" << t->format << " text_ref=" << (is_text(idx) ? 1 : 0) << "\n";
    }
    return 0;
}
