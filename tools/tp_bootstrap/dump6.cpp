#include "artifact/reader.h"
#include "artifact/schema.h"
#include <iostream>
#include <string>
int main(int argc, char** argv) {
    if (argc < 2) { return 2; }
    ninfer::artifact::Reader reader(argv[1]);
    const auto& dir = reader.directory();
    int shown = 0;
    for (std::size_t i = 0; i < dir.objects.size() && shown < 12; ++i) {
        const auto& o = dir.object({i});
        if (auto* r = std::get_if<ninfer::artifact::ResourceObject>(&o)) {
            auto geo = reader.geometry({i});
            std::string s = "R enc=" + r->encoding + " bytes=" + std::to_string(r->bytes);
            s += " geo_fmt=" + std::to_string((int)geo.format);
            for (auto d : geo.shape) s += "[" + std::to_string(d) + "]";
            s += " layout=" + std::to_string((int)geo.layout);
            std::cout << i << " " << s << "\n";
            ++shown;
        }
    }
    return 0;
}
