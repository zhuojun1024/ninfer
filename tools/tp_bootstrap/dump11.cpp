#include "artifact/reader.h"
#include "artifact/materializer.h"
#include "core/tp/tp_materialize.h"
#include "core/tp/weight_splitter.h"
#include "core/weight.h"
#include "models/qwen3_5/load.h"
#include "models/qwen3_5/load/tp_split_spec.h"

#include <cstdint>
#include <iostream>

using namespace ninfer;
using namespace ninfer::artifact;
using namespace ninfer::tp;
using namespace ninfer::models::qwen3_5;

namespace {

WeightGeometry shard_geometry(const WeightGeometry& full, std::int32_t n, std::int32_t k) {
    WeightGeometry g = full;
    g.shape    = {static_cast<std::uint64_t>(n), static_cast<std::uint64_t>(k)};
    g.elements = static_cast<std::uint64_t>(n) * k;
    const auto align256 = [](std::uint64_t v) { return (v + 255) / 256 * 256; };
    if (full.layout == QuantLayout::BlockScaleK16M128x4) {
        g.code_bytes_per_row = static_cast<std::uint64_t>(k) / 2;
        g.code_bytes         = g.elements / 2;
        g.scale_offset       = align256(g.code_bytes);
        g.scale_bytes        = g.elements / 16;
        g.divisor_offset     = g.scale_offset + g.scale_bytes;
        g.bytes              = g.divisor_offset + 4;
    } else if (full.layout == QuantLayout::RowScale) {
        g.code_bytes_per_row = static_cast<std::uint64_t>(k);
        g.code_bytes         = g.elements;
        g.scale_offset       = align256(g.code_bytes);
        g.scale_bytes        = static_cast<std::uint64_t>(n) * 2;
        g.divisor_offset     = 0;
        g.bytes              = g.scale_offset + g.scale_bytes;
    } else {
        const std::uint64_t wb = full.format == QType::BF16 ? 2 : 4;
        g.code_bytes = g.bytes = g.elements * wb;
        g.divisor_offset       = 0;
    }
    return g;
}

std::uint64_t object_shard_bytes(const Reader& reader, const DevicePlacement& placement,
                                 const TPSplitSpec& spec) {
    const auto& geometry = reader.geometry(placement.object);
    const std::uint64_t n = geometry.shape.size() >= 1 ? geometry.shape[0] : 1;
    const std::uint64_t k = geometry.shape.size() >= 2 ? geometry.shape[1] : 1;
    const auto kind = spec.kind(placement.object);
    if (kind == WeightSplitKind::Replicated) { return geometry.bytes; }
    std::int32_t shard_n = static_cast<std::int32_t>(n);
    std::int32_t shard_k = static_cast<std::int32_t>(k);
    if (kind == WeightSplitKind::GatherRows || kind == WeightSplitKind::GatherCols) {
        const auto* split = spec.find(placement.object);
        if (split && !split->parts.empty()) {
            std::int32_t half = 0;
            for (const auto& part : split->parts) { half += part.row_count / 2; }
            if (kind == WeightSplitKind::GatherRows) { shard_n = half; }
            else { shard_k = half; }
        }
    } else if (kind == WeightSplitKind::ColumnParallel) {
        shard_n = static_cast<std::int32_t>(n / 2);
    } else if (kind == WeightSplitKind::RowParallel) {
        shard_k = static_cast<std::int32_t>(k / 2);
    }
    return shard_geometry(geometry, shard_n, shard_k).bytes;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) { return 2; }
    Reader reader(argv[1]);
    ninfer::models::LoadOptions options;
    options.vision      = false;
    options.speculative = SpeculativeBackend::None;
    auto plan = plan_load(reader, options);
    const auto spec = loading::build_tp_split_spec(reader.directory(), plan.config().text);
    const auto& plan_m = plan.materialization();

    std::uint64_t total = 0;
    std::uint64_t by_kind[5] = {0, 0, 0, 0, 0};
    std::size_t count_by_kind[5] = {0, 0, 0, 0, 0};
    for (const auto& placement : plan_m.device_objects) {
        const auto kind = spec.kind(placement.object);
        const std::uint64_t b = object_shard_bytes(reader, placement, spec);
        total += b;
        total += 256;
        const int ki = static_cast<int>(kind);
        if (ki >= 0 && ki < 5) { by_kind[ki] += b; ++count_by_kind[ki]; }
    }
    const char* kind_names[5] = {"ColumnParallel", "RowParallel", "Replicated", "GatherRows",
                                 "GatherCols"};
    std::cout << "device_objects = " << plan_m.device_objects.size() << "\n";
    for (int i = 0; i < 5; ++i) {
        if (count_by_kind[i]) {
            std::cout << "  " << kind_names[i] << " count=" << count_by_kind[i]
                      << " bytes=" << by_kind[i] << " (" << (by_kind[i] / 1024 / 1024) << " MiB)\n";
        }
    }
    std::cout << "shard_capacity (weights) = " << total << " bytes = "
              << (total / 1024 / 1024) << " MiB = " << (total / 1048576.0) / 1024.0 << " GiB\n";
    return 0;
}
