#include "artifact/reader.h"
#include "artifact/schema.h"
#include <iostream>
#include <string>
int main(int argc, char** argv) {
    if (argc < 2) { return 2; }
    ninfer::artifact::Reader reader(argv[1]);
    const auto& dir = reader.directory();
    std::cout << "object_count=" << dir.objects.size() << "\n";
    for (std::size_t i = 0; i < dir.objects.size(); ++i) {
        const auto& o = dir.object({i});
        if (auto* t = std::get_if<ninfer::artifact::TensorObject>(&o)) {
            std::string s = "T " + t->format;
            for (auto d : t->shape) s += "[" + std::to_string(d) + "]";
            std::cout << i << " " << s << "\n";
        } else {
            std::cout << i << " R (resource)\n";
        }
    }
    return 0;
}
