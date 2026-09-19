#include "core/tp/weight_splitter.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace ninfer::tp {
namespace {

std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

// NVFP4 payload geometry derived from (n, k): code = n*k/2, scale plane at
// align_up(code, 256), scale = n*k/16, divisor (4 bytes) at the end.
struct Nvfp4Geometry {
    std::uint64_t code_bytes;
    std::uint64_t scale_offset;
    std::uint64_t scale_bytes;
    std::uint64_t divisor_offset;
    std::uint64_t payload_bytes;
    static Nvfp4Geometry of(std::int32_t n, std::int32_t k) {
        Nvfp4Geometry g;
        g.code_bytes     = static_cast<std::uint64_t>(n) * k / 2;
        g.scale_offset   = align_up(g.code_bytes, 256);
        g.scale_bytes    = static_cast<std::uint64_t>(n) * k / 16;
        g.divisor_offset = g.scale_offset + g.scale_bytes;
        g.payload_bytes  = g.divisor_offset + 4;
        return g;
    }
};

// Grouped RowSplit payload geometry derived from (n, k): per-row code words, an optional per-row
// high-bit plane, and one FP16 scale word per 64- (or 32-) value group, each plane 256-byte
// aligned. Mirrors weight_geometry's RowSplit branch.
struct GroupedGeometry {
    std::uint64_t code_bytes_per_row  = 0;
    std::uint64_t high_bytes_per_row  = 0;
    std::uint64_t scale_bytes_per_row = 0;
    std::uint64_t code_bytes          = 0;
    std::uint64_t high_offset         = 0;
    std::uint64_t high_bytes          = 0;
    std::uint64_t scale_offset        = 0;
    std::uint64_t scale_bytes         = 0;
    std::uint64_t payload_bytes       = 0;

    static GroupedGeometry of(QType format, std::int32_t n, std::int32_t k) {
        std::uint64_t group = 0, high_per_group = 0;
        switch (format) {
        case QType::Q4_G64_FP16: group = 64; break;
        case QType::Q5_G64_FP16: group = 64; high_per_group = 8; break;
        case QType::Q6_G64_FP16: group = 64; high_per_group = 16; break;
        case QType::Q8_G32_FP16: group = 32; break;
        default: throw std::invalid_argument("weight splitter: not a grouped RowSplit format");
        }
        const std::uint64_t groups = align_up(k, 128) / group;
        GroupedGeometry g;
        g.code_bytes_per_row  = groups * 32;
        g.high_bytes_per_row  = groups * high_per_group;
        g.scale_bytes_per_row = groups * 2;
        g.code_bytes          = static_cast<std::uint64_t>(n) * g.code_bytes_per_row;
        g.high_offset         = align_up(g.code_bytes, 256);
        g.high_bytes          = static_cast<std::uint64_t>(n) * g.high_bytes_per_row;
        g.scale_offset        = align_up(g.high_offset + g.high_bytes, 256);
        g.scale_bytes         = static_cast<std::uint64_t>(n) * g.scale_bytes_per_row;
        g.payload_bytes       = g.scale_offset + g.scale_bytes;
        return g;
    }
};

// Grouped RowSplit row slice: rows [row_begin, row_begin + row_count). Every plane is contiguous
// per row block, so the slice is one span copy per plane.
WeightShard slice_grouped_rows(std::span<const std::uint8_t> full, const Weight& full_weight,
                              std::int32_t row_begin, std::int32_t row_count) {
    const std::int32_t k = full_weight.k;
    if (row_count <= 0 || row_begin < 0 || row_begin + row_count > full_weight.n) {
        throw std::invalid_argument("weight splitter: invalid grouped row slice");
    }
    const GroupedGeometry full_geo  = GroupedGeometry::of(full_weight.qtype, full_weight.n, k);
    const GroupedGeometry shard_geo = GroupedGeometry::of(full_weight.qtype, row_count, k);

    WeightShard shard;
    shard.payload.assign(shard_geo.payload_bytes, 0);
    std::memcpy(shard.payload.data(),
                full.data() + static_cast<std::size_t>(row_begin) * full_geo.code_bytes_per_row,
                shard_geo.code_bytes);
    if (shard_geo.high_bytes_per_row != 0) {
        std::memcpy(shard.payload.data() + shard_geo.high_offset,
                    full.data() + full_geo.high_offset +
                        static_cast<std::size_t>(row_begin) * full_geo.high_bytes_per_row,
                    shard_geo.high_bytes);
    }
    std::memcpy(shard.payload.data() + shard_geo.scale_offset,
                full.data() + full_geo.scale_offset +
                    static_cast<std::size_t>(row_begin) * full_geo.scale_bytes_per_row,
                shard_geo.scale_bytes);

    shard.weight = full_weight;
    shard.weight.n               = row_count;
    shard.weight.shape[0]        = row_count;
    shard.weight.padded_shape[0] = row_count;
    shard.weight.payload         = shard.payload.data();
    shard.weight.payload_bytes   = shard.payload.size();
    shard.weight.qdata           = shard.payload.data();
    shard.weight.qhigh =
        shard_geo.high_bytes_per_row != 0 ? shard.payload.data() + shard_geo.high_offset : nullptr;
    shard.weight.scales          = shard.payload.data() + shard_geo.scale_offset;
    shard.weight.high_plane_bytes = shard_geo.high_bytes;
    return shard;
}

// NVFP4 row slice: rows [row_begin, row_begin + row_count). row_count must be a multiple of
// 128. The code plane and the scale plane are each contiguous for a row block (the scale
// tile index is row_tile * k_tiles + scale_tile, so a row block owns a contiguous scale span).
WeightShard slice_nvfp4_rows(std::span<const std::uint8_t> full, const Weight& full_weight,
                             std::int32_t row_begin, std::int32_t row_count) {
    const std::int32_t k = full_weight.k;
    if ((row_count % 128) != 0 || row_begin < 0 || row_begin + row_count > full_weight.n) {
        throw std::invalid_argument("weight splitter: invalid NVFP4 row slice");
    }
    const Nvfp4Geometry full_geo = Nvfp4Geometry::of(full_weight.n, k);
    const Nvfp4Geometry shard_geo = Nvfp4Geometry::of(row_count, k);

    WeightShard shard;
    shard.payload.assign(shard_geo.payload_bytes, 0);
    std::memcpy(shard.payload.data(), full.data() + static_cast<std::size_t>(row_begin) * k / 2,
                shard_geo.code_bytes);
    std::memcpy(shard.payload.data() + shard_geo.scale_offset,
                full.data() + full_geo.scale_offset + static_cast<std::size_t>(row_begin) * k / 16,
                shard_geo.scale_bytes);
    std::memcpy(shard.payload.data() + shard_geo.divisor_offset,
                full.data() + full_geo.divisor_offset, 4);

    shard.weight = full_weight;
    shard.weight.n               = row_count;
    shard.weight.shape[0]        = row_count;
    shard.weight.padded_shape[0] = row_count;
    shard.weight.payload         = shard.payload.data();
    shard.weight.payload_bytes   = shard.payload.size();
    shard.weight.qdata           = shard.payload.data();
    shard.weight.scales          = shard.payload.data() + shard_geo.scale_offset;
    return shard;
}

// NVFP4 column slice: columns [col_begin, col_begin + col_count). col_begin and col_count must
// be multiples of 64. The code plane is a per-row strided copy; the scale plane is a per-tile
// copy where shard scale-tile st maps to full scale-tile (col_begin / 64 + st).
WeightShard slice_nvfp4_cols(std::span<const std::uint8_t> full, const Weight& full_weight,
                             std::int32_t col_begin, std::int32_t col_count) {
    const std::int32_t n = full_weight.n, k = full_weight.k;
    if ((col_begin % 64) != 0 || (col_count % 64) != 0 || col_begin < 0 ||
        col_begin + col_count > k) {
        throw std::invalid_argument("weight splitter: invalid NVFP4 column slice");
    }
    const Nvfp4Geometry full_geo  = Nvfp4Geometry::of(n, k);
    const Nvfp4Geometry shard_geo = Nvfp4Geometry::of(n, col_count);

    WeightShard shard;
    shard.payload.assign(shard_geo.payload_bytes, 0);
    for (std::int32_t r = 0; r < n; ++r) {
        std::memcpy(shard.payload.data() + static_cast<std::size_t>(r) * col_count / 2,
                    full.data() + static_cast<std::size_t>(r) * k / 2 + col_begin / 2,
                    col_count / 2);
    }
    const std::int32_t k_tiles_full  = k / 64;
    const std::int32_t k_tiles_shard = col_count / 64;
    const std::int32_t row_tiles     = n / 128;
    for (std::int32_t rt = 0; rt < row_tiles; ++rt) {
        for (std::int32_t st = 0; st < k_tiles_shard; ++st) {
            const std::size_t shard_off =
                shard_geo.scale_offset + static_cast<std::size_t>(rt * k_tiles_shard + st) * 512;
            const std::size_t full_off =
                full_geo.scale_offset +
                static_cast<std::size_t>(rt * k_tiles_full + col_begin / 64 + st) * 512;
            std::memcpy(shard.payload.data() + shard_off, full.data() + full_off, 512);
        }
    }
    std::memcpy(shard.payload.data() + shard_geo.divisor_offset,
                full.data() + full_geo.divisor_offset, 4);

    shard.weight = full_weight;
    shard.weight.k               = col_count;
    shard.weight.shape[1]        = col_count;
    shard.weight.padded_shape[1] = col_count;
    shard.weight.payload         = shard.payload.data();
    shard.weight.payload_bytes   = shard.payload.size();
    shard.weight.qdata           = shard.payload.data();
    shard.weight.scales          = shard.payload.data() + shard_geo.scale_offset;
    return shard;
}

// BF16 contiguous payload: row-major n*k*2 bytes. Column split (row slice) is a contiguous
// first/second half; row split (column slice) is a per-row strided copy.
// FP8 row-scale row slice: the code plane is n*k bytes (one byte per element) and the scale
// plane is n*2 bytes (one BF16 per row), so a row block is two contiguous copies.
WeightShard slice_fp8_rows(std::span<const std::uint8_t> full, const Weight& full_weight,
                           std::int32_t row_begin, std::int32_t row_count) {
    const std::int32_t k = full_weight.k;
    if (row_begin < 0 || row_begin + row_count > full_weight.n) {
        throw std::invalid_argument("weight splitter: invalid FP8 row slice");
    }
    const std::uint64_t full_code_bytes = static_cast<std::uint64_t>(full_weight.n) * k;
    const std::uint64_t full_scale_offset = (full_code_bytes + 255) / 256 * 256;
    const std::uint64_t shard_code_bytes  = static_cast<std::uint64_t>(row_count) * k;
    const std::uint64_t shard_scale_offset = (shard_code_bytes + 255) / 256 * 256;
    // Match shard_geometry's RowScale layout: code plane, 256-byte-aligned scale plane.
    WeightShard shard;
    shard.payload.assign(static_cast<std::size_t>(shard_scale_offset + static_cast<std::uint64_t>(row_count) * 2), 0);
    std::memcpy(shard.payload.data(),
                full.data() + static_cast<std::size_t>(row_begin) * k,
                static_cast<std::size_t>(shard_code_bytes));
    std::memcpy(shard.payload.data() + shard_scale_offset,
                full.data() + full_scale_offset + static_cast<std::size_t>(row_begin) * 2,
                static_cast<std::size_t>(row_count) * 2);
    shard.weight = full_weight;
    shard.weight.n               = row_count;
    shard.weight.shape[0]        = row_count;
    shard.weight.padded_shape[0] = row_count;
    // One BF16 multiplier per row, so the row-scale plane and its row stride follow the new row
    // count (the plane validator checks both against weight.n).
    shard.weight.scale_ne[0]     = row_count;
    const std::int64_t shard_scale_stride = static_cast<std::int64_t>(row_count) * 2;
    shard.weight.scale_nb[1]              = shard_scale_stride;
    shard.weight.scale_nb[2]              = shard_scale_stride;
    shard.weight.scale_nb[3]              = shard_scale_stride;
    shard.weight.payload         = shard.payload.data();
    shard.weight.payload_bytes   = shard.payload.size();
    shard.weight.qdata           = shard.payload.data();
    shard.weight.scales          = shard.payload.data() + shard_scale_offset;
    return shard;
}

// FP8 row-scale column slice: the code plane is row-major (k bytes per row), so a column block
// is a per-row strided copy; the scale plane is one BF16 per row and is copied whole.
WeightShard slice_fp8_cols(std::span<const std::uint8_t> full, const Weight& full_weight,
                           std::int32_t col_begin, std::int32_t col_count) {
    const std::int32_t n = full_weight.n, k = full_weight.k;
    if (col_begin < 0 || col_begin + col_count > k) {
        throw std::invalid_argument("weight splitter: invalid FP8 column slice");
    }
    const std::uint64_t full_code_bytes = static_cast<std::uint64_t>(n) * k;
    const std::uint64_t full_scale_offset = (full_code_bytes + 255) / 256 * 256;
    const std::uint64_t shard_code_bytes  = static_cast<std::uint64_t>(n) * col_count;
    const std::uint64_t shard_scale_offset = (shard_code_bytes + 255) / 256 * 256;
    // Match shard_geometry's RowScale layout: code plane, 256-byte-aligned scale plane.
    WeightShard shard;
    shard.payload.assign(static_cast<std::size_t>(shard_scale_offset + static_cast<std::uint64_t>(n) * 2), 0);
    for (std::int32_t r = 0; r < n; ++r) {
        std::memcpy(shard.payload.data() + static_cast<std::size_t>(r) * col_count,
                    full.data() + static_cast<std::size_t>(r) * k + col_begin,
                    static_cast<std::size_t>(col_count));
    }
    std::memcpy(shard.payload.data() + shard_scale_offset,
                full.data() + full_scale_offset,
                static_cast<std::size_t>(n) * 2);
    shard.weight = full_weight;
    shard.weight.k               = col_count;
    shard.weight.shape[1]        = col_count;
    shard.weight.padded_shape[1] = col_count;
    // The row scale spans the whole (now halved) row, so its group follows the new column count.
    shard.weight.group_size      = static_cast<std::uint32_t>(col_count);
    shard.weight.group           = col_count;
    shard.weight.payload         = shard.payload.data();
    shard.weight.payload_bytes   = shard.payload.size();
    shard.weight.qdata           = shard.payload.data();
    shard.weight.scales          = shard.payload.data() + shard_scale_offset;
    return shard;
}

WeightShard slice_bf16_rows(std::span<const std::uint8_t> full, const Weight& full_weight,
                            std::int32_t row_begin, std::int32_t row_count) {
    const std::int32_t k = full_weight.k;
    if (row_begin < 0 || row_begin + row_count > full_weight.n) {
        throw std::invalid_argument("weight splitter: invalid BF16 row slice");
    }
    const std::uint64_t row_bytes = static_cast<std::uint64_t>(k) * 2;
    WeightShard shard;
    shard.payload.assign(static_cast<std::size_t>(row_count) * row_bytes, 0);
    std::memcpy(shard.payload.data(),
                full.data() + static_cast<std::size_t>(row_begin) * row_bytes,
                static_cast<std::size_t>(row_count) * row_bytes);
    shard.weight = full_weight;
    shard.weight.n               = row_count;
    shard.weight.shape[0]        = row_count;
    shard.weight.padded_shape[0] = row_count;
    shard.weight.payload         = shard.payload.data();
    shard.weight.payload_bytes   = shard.payload.size();
    shard.weight.qdata           = shard.payload.data();
    shard.weight.scales          = nullptr;
    return shard;
}

WeightShard slice_bf16_cols(std::span<const std::uint8_t> full, const Weight& full_weight,
                            std::int32_t col_begin, std::int32_t col_count) {
    const std::int32_t n = full_weight.n, k = full_weight.k;
    if (col_begin < 0 || col_begin + col_count > k) {
        throw std::invalid_argument("weight splitter: invalid BF16 column slice");
    }
    const std::uint64_t shard_row_bytes = static_cast<std::uint64_t>(col_count) * 2;
    const std::uint64_t full_row_bytes  = static_cast<std::uint64_t>(k) * 2;
    WeightShard shard;
    shard.payload.assign(static_cast<std::size_t>(n) * shard_row_bytes, 0);
    for (std::int32_t r = 0; r < n; ++r) {
        std::memcpy(shard.payload.data() + static_cast<std::size_t>(r) * shard_row_bytes,
                    full.data() + static_cast<std::size_t>(r) * full_row_bytes + col_begin * 2,
                    shard_row_bytes);
    }
    shard.weight = full_weight;
    shard.weight.k               = col_count;
    shard.weight.shape[1]        = col_count;
    shard.weight.padded_shape[1] = col_count;
    shard.weight.payload         = shard.payload.data();
    shard.weight.payload_bytes   = shard.payload.size();
    shard.weight.qdata           = shard.payload.data();
    shard.weight.scales          = nullptr;
    return shard;
}


// FP8 row-scale payload geometry: code = n*k bytes (k per row), scale plane at
// align_up(code, 256), scale = n*2 bytes (2 per row). No divisor.
struct Fp8Geometry {
    std::uint64_t code_bytes;
    std::uint64_t scale_offset;
    std::uint64_t scale_bytes;
    std::uint64_t payload_bytes;
    static Fp8Geometry of(std::int32_t n, std::int32_t k) {
        Fp8Geometry g;
        g.code_bytes     = static_cast<std::uint64_t>(n) * k;
        g.scale_offset   = align_up(g.code_bytes, 256);
        g.scale_bytes    = static_cast<std::uint64_t>(n) * 2;
        g.payload_bytes  = g.scale_offset + g.scale_bytes;
        return g;
    }
};

// BF16 contiguous payload geometry: n*k*2 bytes, no scale plane.
struct Bf16Geometry {
    std::uint64_t payload_bytes;
    static Bf16Geometry of(std::int32_t n, std::int32_t k) {
        Bf16Geometry g;
        g.payload_bytes = static_cast<std::uint64_t>(n) * k * 2;
        return g;
    }
};

// Per-part row gather: for each part, copy the `shard`-th half of its rows (code + scale)
// planes) into the shard payload, concatenating in part order. The shard keeps the parent's
// plane structure with n = sum(row_count / 2).
WeightShard gather_rows_impl(std::span<const std::uint8_t> full_payload,
                               const Weight& full_weight, std::span<const RowPart> parts,
                               int shard) {
    if (shard != 0 && shard != 1) {
        throw std::invalid_argument("weight splitter: gather shard must be 0 or 1");
    }
    const std::int32_t k = full_weight.k;
    std::int32_t total_n = 0;
    for (const auto& p : parts) {
        if (p.row_count % 2 != 0 || p.row_begin < 0 || p.row_begin + p.row_count > full_weight.n) {
            throw std::invalid_argument("weight splitter: invalid gather part");
        }
        total_n += p.row_count / 2;
    }

    WeightShard shard_out;
    shard_out.weight = full_weight;
    shard_out.weight.n               = total_n;
    shard_out.weight.shape[0]        = total_n;
    shard_out.weight.padded_shape[0] = total_n;

    if (full_weight.qtype == QType::NVFP4) {
        if (full_weight.layout != QuantLayout::BlockScaleK16M128x4) {
            throw std::invalid_argument("weight splitter: NVFP4 gather requires BlockScale");
        }
        for (const auto& p : parts) {
            if ((p.row_count / 2) % 128 != 0) {
                throw std::invalid_argument("weight splitter: NVFP4 gather part half must be 128-multiple");
            }
        }
        const Nvfp4Geometry full_geo  = Nvfp4Geometry::of(full_weight.n, k);
        const Nvfp4Geometry shard_geo = Nvfp4Geometry::of(total_n, k);
        shard_out.payload.assign(shard_geo.payload_bytes, 0);
        std::size_t code_off = 0, scale_off = shard_geo.scale_offset;
        for (const auto& p : parts) {
            const std::int32_t rb = p.row_begin + shard * (p.row_count / 2);
            const std::int32_t rc = p.row_count / 2;
            std::memcpy(shard_out.payload.data() + code_off,
                        full_payload.data() + static_cast<std::size_t>(rb) * k / 2,
                        static_cast<std::size_t>(rc) * k / 2);
            std::memcpy(shard_out.payload.data() + scale_off,
                        full_payload.data() + full_geo.scale_offset + static_cast<std::size_t>(rb) * k / 16,
                        static_cast<std::size_t>(rc) * k / 16);
            code_off  += static_cast<std::size_t>(rc) * k / 2;
            scale_off += static_cast<std::size_t>(rc) * k / 16;
        }
        std::memcpy(shard_out.payload.data() + shard_geo.divisor_offset,
                    full_payload.data() + full_geo.divisor_offset, 4);
        shard_out.weight.payload         = shard_out.payload.data();
        shard_out.weight.payload_bytes   = shard_out.payload.size();
        shard_out.weight.qdata           = shard_out.payload.data();
        shard_out.weight.scales          = shard_out.payload.data() + shard_geo.scale_offset;
        return shard_out;
    }

    if (full_weight.qtype == QType::FP8_E4M3FN_ROW_BF16) {
        if (full_weight.layout != QuantLayout::RowScale) {
            throw std::invalid_argument("weight splitter: FP8 gather requires RowScale");
        }
        const Fp8Geometry full_geo  = Fp8Geometry::of(full_weight.n, k);
        const Fp8Geometry shard_geo = Fp8Geometry::of(total_n, k);
        shard_out.payload.assign(shard_geo.payload_bytes, 0);
        std::size_t code_off = 0, scale_off = shard_geo.scale_offset;
        for (const auto& p : parts) {
            const std::int32_t rb = p.row_begin + shard * (p.row_count / 2);
            const std::int32_t rc = p.row_count / 2;
            std::memcpy(shard_out.payload.data() + code_off,
                        full_payload.data() + static_cast<std::size_t>(rb) * k,
                        static_cast<std::size_t>(rc) * k);
            std::memcpy(shard_out.payload.data() + scale_off,
                        full_payload.data() + full_geo.scale_offset + static_cast<std::size_t>(rb) * 2,
                        static_cast<std::size_t>(rc) * 2);
            code_off  += static_cast<std::size_t>(rc) * k;
            scale_off += static_cast<std::size_t>(rc) * 2;
        }
        shard_out.weight.payload         = shard_out.payload.data();
        shard_out.weight.payload_bytes   = shard_out.payload.size();
        shard_out.weight.qdata           = shard_out.payload.data();
        shard_out.weight.scales          = shard_out.payload.data() + shard_geo.scale_offset;
        return shard_out;
    }

    if (full_weight.qtype == QType::BF16) {
        if (full_weight.layout != QuantLayout::Contiguous) {
            throw std::invalid_argument("weight splitter: BF16 gather requires Contiguous");
        }
        const Bf16Geometry shard_geo = Bf16Geometry::of(total_n, k);
        shard_out.payload.assign(shard_geo.payload_bytes, 0);
        std::size_t code_off = 0;
        for (const auto& p : parts) {
            const std::int32_t rb = p.row_begin + shard * (p.row_count / 2);
            const std::int32_t rc = p.row_count / 2;
            std::memcpy(shard_out.payload.data() + code_off,
                        full_payload.data() + static_cast<std::size_t>(rb) * k * 2,
                        static_cast<std::size_t>(rc) * k * 2);
            code_off += static_cast<std::size_t>(rc) * k * 2;
        }
        shard_out.weight.payload         = shard_out.payload.data();
        shard_out.weight.payload_bytes   = shard_out.payload.size();
        shard_out.weight.qdata           = shard_out.payload.data();
        shard_out.weight.scales          = nullptr;
        return shard_out;
    }

    throw std::invalid_argument("weight splitter: unsupported qtype for gather");
}

// Per-part column gather (K dimension): for each part, copy the shard-th half of its columns
// into the shard payload, concatenating in part order. The shard keeps the parent's row count
// with k = sum(row_count / 2). Supported for BF16 Contiguous (the GDN causal conv, whose
// physical object is [taps, channels]); other formats throw.
WeightShard gather_cols_impl(std::span<const std::uint8_t> full_payload,
                               const Weight& full_weight, std::span<const RowPart> parts,
                               int shard) {
    if (shard != 0 && shard != 1) {
        throw std::invalid_argument("weight splitter: gather shard must be 0 or 1");
    }
    if (full_weight.qtype != QType::BF16 || full_weight.layout != QuantLayout::Contiguous) {
        throw std::invalid_argument("weight splitter: column gather requires BF16 Contiguous");
    }
    const std::int32_t n = full_weight.n, k = full_weight.k;
    std::int32_t total_k = 0;
    for (const auto& p : parts) {
        if (p.row_count % 2 != 0 || p.row_begin < 0 || p.row_begin + p.row_count > k) {
            throw std::invalid_argument("weight splitter: invalid column gather part");
        }
        total_k += p.row_count / 2;
    }

    WeightShard shard_out;
    shard_out.weight = full_weight;
    shard_out.weight.k               = total_k;
    shard_out.weight.shape[1]        = total_k;
    shard_out.weight.padded_shape[1] = total_k;

    const std::uint64_t full_row_bytes  = static_cast<std::uint64_t>(k) * 2;
    const std::uint64_t shard_row_bytes = static_cast<std::uint64_t>(total_k) * 2;
    shard_out.payload.assign(static_cast<std::size_t>(n) * shard_row_bytes, 0);
    std::size_t col_off = 0;
    for (const auto& p : parts) {
        const std::int32_t cb = p.row_begin + shard * (p.row_count / 2);
        const std::int32_t cc = p.row_count / 2;
        for (std::int32_t r = 0; r < n; ++r) {
            std::memcpy(shard_out.payload.data() +
                            static_cast<std::size_t>(r) * shard_row_bytes + col_off * 2,
                        full_payload.data() +
                            static_cast<std::size_t>(r) * full_row_bytes + cb * 2,
                        static_cast<std::size_t>(cc) * 2);
        }
        col_off += cc;
    }
    shard_out.weight.payload         = shard_out.payload.data();
    shard_out.weight.payload_bytes   = shard_out.payload.size();
    shard_out.weight.qdata           = shard_out.payload.data();
    shard_out.weight.scales          = nullptr;
    return shard_out;
}

} // namespace

WeightShard gather_weight_rows(std::span<const std::uint8_t> full_payload,
                               const Weight& full_weight, std::span<const RowPart> parts,
                               int shard) {
    return gather_rows_impl(full_payload, full_weight, parts, shard);
}

WeightShard gather_weight_cols(std::span<const std::uint8_t> full_payload,
                               const Weight& full_weight, std::span<const RowPart> parts,
                               int shard) {
    return gather_cols_impl(full_payload, full_weight, parts, shard);
}

std::vector<WeightShard> split_weight(std::span<const std::uint8_t> full_payload,
                                      const Weight& full_weight, WeightSplitKind kind) {
    if (full_weight.qtype == QType::NVFP4) {
        if (full_weight.layout != QuantLayout::BlockScaleK16M128x4) {
            throw std::invalid_argument("weight splitter: NVFP4 parent must be BlockScale");
        }
        if (kind == WeightSplitKind::ColumnParallel) {
            if ((full_weight.n % 256) != 0) {
                throw std::invalid_argument("weight splitter: column split requires n % 256 == 0");
            }
            return {slice_nvfp4_rows(full_payload, full_weight, 0, full_weight.n / 2),
                    slice_nvfp4_rows(full_payload, full_weight, full_weight.n / 2,
                                     full_weight.n / 2)};
        }
        if ((full_weight.k % 128) != 0) {
            throw std::invalid_argument("weight splitter: row split requires k % 128 == 0");
        }
        return {slice_nvfp4_cols(full_payload, full_weight, 0, full_weight.k / 2),
                slice_nvfp4_cols(full_payload, full_weight, full_weight.k / 2, full_weight.k / 2)};
    }
    if (full_weight.layout == QuantLayout::RowSplit) {
        // Grouped integer formats only need the row split: they carry no column-blocked scale
        // plane, so a column slice would repack code words across group boundaries.
        if (kind != WeightSplitKind::ColumnParallel) {
            throw std::invalid_argument("weight splitter: grouped formats support only a row split");
        }
        if ((full_weight.n % 2) != 0) {
            throw std::invalid_argument("weight splitter: column split requires even n");
        }
        return {slice_grouped_rows(full_payload, full_weight, 0, full_weight.n / 2),
                slice_grouped_rows(full_payload, full_weight, full_weight.n / 2,
                                   full_weight.n / 2)};
    }
    if (full_weight.qtype == QType::BF16) {
        if (kind == WeightSplitKind::ColumnParallel) {
            if ((full_weight.n % 2) != 0) {
                throw std::invalid_argument("weight splitter: column split requires even n");
            }
            return {slice_bf16_rows(full_payload, full_weight, 0, full_weight.n / 2),
                    slice_bf16_rows(full_payload, full_weight, full_weight.n / 2,
                                    full_weight.n / 2)};
        }
        if ((full_weight.k % 2) != 0) {
            throw std::invalid_argument("weight splitter: row split requires even k");
        }
        return {slice_bf16_cols(full_payload, full_weight, 0, full_weight.k / 2),
                slice_bf16_cols(full_payload, full_weight, full_weight.k / 2, full_weight.k / 2)};
    }
    if (full_weight.qtype == QType::FP8_E4M3FN_ROW_BF16) {
        // The output head is FP8 row-scale (vocabulary row split); the mixed-precision FFN down
        // projections are also FP8 row-scale and need the intermediate column split.
        if (kind == WeightSplitKind::ColumnParallel) {
            if ((full_weight.n % 2) != 0) {
                throw std::invalid_argument("weight splitter: column split requires even n");
            }
            return {slice_fp8_rows(full_payload, full_weight, 0, full_weight.n / 2),
                    slice_fp8_rows(full_payload, full_weight, full_weight.n / 2, full_weight.n / 2)};
        }
        if ((full_weight.k % 2) != 0) {
            throw std::invalid_argument("weight splitter: row split requires even k");
        }
        return {slice_fp8_cols(full_payload, full_weight, 0, full_weight.k / 2),
                slice_fp8_cols(full_payload, full_weight, full_weight.k / 2, full_weight.k / 2)};
    }
    throw std::invalid_argument("weight splitter: unsupported qtype");
}

} // namespace ninfer::tp
