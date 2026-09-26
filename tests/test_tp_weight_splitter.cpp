// TP-2 GGUF block splitter. A GGUF block weight may only be cut on block boundaries: a block holds
// its codes and its scale together, so a piece that begins or ends inside one has no representation
// without requantising. Every expected payload below is recomputed from gguf_block_shape() itself, so
// splitter arithmetic that disagreed with the block geometry could not match it.

#include "core/tp/weight_splitter.h"
#include "core/weight.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace ninfer;

int failures = 0;

void check(bool condition, const std::string& what) {
    if (!condition) {
        std::cerr << "FAIL: " << what << '\n';
        ++failures;
    }
}

template <typename Fn>
void expect_throws(Fn&& fn, const std::string& what) {
    try {
        fn();
    } catch (const std::invalid_argument& error) {
        const std::string message = error.what();
        if (message.find("GGUF") == std::string::npos) {
            std::cerr << "FAIL: " << what << ": the message does not name GGUF: " << message << '\n';
            ++failures;
        }
        return;
    }
    std::cerr << "FAIL: " << what << ": did not throw\n";
    ++failures;
}

std::vector<std::uint8_t> random_payload(std::size_t size, std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::vector<std::uint8_t> bytes(size);
    for (auto& byte : bytes) { byte = static_cast<std::uint8_t>(rng() & 0xFFU); }
    return bytes;
}

std::uint64_t rows_bytes(QType format, std::int32_t k) {
    const GgufBlockShape block = gguf_block_shape(format);
    return static_cast<std::uint64_t>(k / block.elements) * block.bytes;
}

Weight block_weight(QType format, std::int32_t n, std::int32_t k,
                    std::span<const std::uint8_t> payload) {
    const GgufBlockShape block = gguf_block_shape(format);
    Weight weight;
    weight.qtype           = format;
    weight.layout          = QuantLayout::GgufBlocks;
    weight.ndim            = 2;
    weight.n               = n;
    weight.k               = k;
    weight.shape[0]        = n;
    weight.shape[1]        = k;
    weight.padded_shape[0] = n;
    weight.padded_shape[1] = k;
    weight.group_size      = static_cast<std::uint32_t>(block.elements);
    weight.group           = block.elements;
    weight.payload         = payload.data();
    weight.payload_bytes   = payload.size();
    weight.qdata           = payload.data();
    return weight;
}

std::vector<std::uint8_t> expect_rows(const std::vector<std::uint8_t>& full, QType format,
                                      std::int32_t k, std::int32_t row_begin,
                                      std::int32_t row_count) {
    const std::uint64_t row_bytes = rows_bytes(format, k);
    const auto first = full.begin() + static_cast<std::ptrdiff_t>(row_begin * row_bytes);
    return {first, first + static_cast<std::ptrdiff_t>(row_count * row_bytes)};
}

std::vector<std::uint8_t> expect_cols(const std::vector<std::uint8_t>& full, QType format,
                                      std::int32_t n, std::int32_t k, std::int32_t col_begin,
                                      std::int32_t col_count) {
    const GgufBlockShape block   = gguf_block_shape(format);
    const std::uint64_t full_rb  = rows_bytes(format, k);
    const std::uint64_t part_rb  = static_cast<std::uint64_t>(col_count / block.elements) * block.bytes;
    const std::uint64_t byte_off = static_cast<std::uint64_t>(col_begin / block.elements) * block.bytes;
    std::vector<std::uint8_t> out(static_cast<std::size_t>(n) * part_rb, 0);
    for (std::int32_t r = 0; r < n; ++r) {
        std::memcpy(out.data() + static_cast<std::size_t>(r) * part_rb,
                    full.data() + static_cast<std::size_t>(r) * full_rb + byte_off, part_rb);
    }
    return out;
}

void split_case(QType format, std::int32_t n, std::int32_t k, std::uint32_t seed) {
    const std::string label = "qtype " + std::to_string(static_cast<int>(format)) +
                              " n=" + std::to_string(n) + " k=" + std::to_string(k);
    const std::uint64_t row_bytes = rows_bytes(format, k);
    const std::vector<std::uint8_t> full =
        random_payload(static_cast<std::size_t>(row_bytes) * static_cast<std::size_t>(n), seed);
    const Weight weight = block_weight(format, n, k, full);

    // N split: whole rows move, so each half is one contiguous run of the code plane.
    const std::vector<tp::WeightShard> column =
        tp::split_weight(full, weight, tp::WeightSplitKind::ColumnParallel);
    check(column.size() == 2, label + ": N split yields two shards");
    check(column[0].payload == expect_rows(full, format, k, 0, n / 2), label + ": first N half");
    check(column[1].payload == expect_rows(full, format, k, n / 2, n / 2), label + ": second N half");
    check(column[0].weight.n == n / 2 && column[0].weight.k == k, label + ": first N half shape");
    check(column[1].weight.n == n / 2 && column[1].weight.k == k, label + ": second N half shape");

    // K split: whole blocks of every row.
    const std::vector<tp::WeightShard> row =
        tp::split_weight(full, weight, tp::WeightSplitKind::RowParallel);
    check(row.size() == 2, label + ": K split yields two shards");
    check(row[0].payload == expect_cols(full, format, n, k, 0, k / 2), label + ": first K half");
    check(row[1].payload == expect_cols(full, format, n, k, k / 2, k / 2), label + ": second K half");
    check(row[0].weight.k == k / 2 && row[1].weight.k == k / 2, label + ": K half shape");
    check(row[0].weight.n == n && row[1].weight.n == n, label + ": K half keeps every row");

    // The K halves must rejoin to the parent's code plane byte for byte.
    const std::size_t half_bytes = static_cast<std::size_t>(rows_bytes(format, k / 2));
    std::vector<std::uint8_t> rejoined(full.size(), 0);
    for (std::int32_t r = 0; r < n; ++r) {
        const std::size_t base = static_cast<std::size_t>(r) * row_bytes;
        const std::size_t half = static_cast<std::size_t>(r) * half_bytes;
        std::memcpy(rejoined.data() + base, row[0].payload.data() + half, half_bytes);
        std::memcpy(rejoined.data() + base + half_bytes, row[1].payload.data() + half, half_bytes);
    }
    check(rejoined == full, label + ": the K halves rejoin to the parent");
}

void gather_rows_case(QType format) {
    const std::int32_t n = 128, k = 256;
    const std::string label = "row gather qtype " + std::to_string(static_cast<int>(format));
    const std::uint64_t row_bytes = rows_bytes(format, k);
    const std::vector<std::uint8_t> full =
        random_payload(static_cast<std::size_t>(row_bytes) * static_cast<std::size_t>(n), 0xA7U);
    const Weight weight = block_weight(format, n, k, full);
    // Two parts tiling the parent's rows, exactly as the fused q/k/v/z parents do.
    const std::vector<tp::RowPart> parts{{0, 64}, {64, 64}};
    const tp::WeightShard shard0 = tp::gather_weight_rows(full, weight, parts, 0);
    const tp::WeightShard shard1 = tp::gather_weight_rows(full, weight, parts, 1);

    auto expect_gather = [&](std::int32_t half) {
        std::vector<std::uint8_t> out;
        for (const auto& part : parts) {
            const auto run = expect_rows(full, format, k, part.row_begin + half * (part.row_count / 2),
                                         part.row_count / 2);
            out.insert(out.end(), run.begin(), run.end());
        }
        return out;
    };
    check(shard0.payload == expect_gather(0), label + ": first half of every part");
    check(shard1.payload == expect_gather(1), label + ": second half of every part");
    check(shard0.weight.n == 64 && shard1.weight.n == 64, label + ": half the gathered rows");
    check(shard0.weight.k == k && shard1.weight.k == k, label + ": the gathered rows keep K");
}

void gather_cols_case(QType format) {
    const std::int32_t n = 4, k = 1024;
    const std::string label = "column gather qtype " + std::to_string(static_cast<int>(format));
    const std::uint64_t row_bytes = rows_bytes(format, k);
    const std::vector<std::uint8_t> full =
        random_payload(static_cast<std::size_t>(row_bytes) * static_cast<std::size_t>(n), 0x3DU);
    const Weight weight = block_weight(format, n, k, full);
    // The GDN causal conv: [taps, q | k | v] channel blocks, each halved.
    const std::vector<tp::RowPart> parts{{0, 512}, {512, 512}};
    const tp::WeightShard shard0 = tp::gather_weight_cols(full, weight, parts, 0);
    const tp::WeightShard shard1 = tp::gather_weight_cols(full, weight, parts, 1);

    // The shard is row-major: each row holds, in part order, this shard's half of that part's
    // columns, so the expectation interleaves per row rather than concatenating whole parts.
    const GgufBlockShape block = gguf_block_shape(format);
    const std::uint64_t full_rb = rows_bytes(format, k);
    auto expect_gather = [&](std::int32_t half) {
        std::vector<std::uint8_t> out;
        for (std::int32_t r = 0; r < n; ++r) {
            for (const auto& part : parts) {
                const std::int32_t cb = part.row_begin + half * (part.row_count / 2);
                const std::int32_t cc = part.row_count / 2;
                const std::uint64_t byte_off =
                    static_cast<std::uint64_t>(cb / block.elements) * block.bytes;
                const std::uint64_t part_bytes =
                    static_cast<std::uint64_t>(cc / block.elements) * block.bytes;
                const auto* src = full.data() + static_cast<std::size_t>(r) * full_rb + byte_off;
                out.insert(out.end(), src, src + part_bytes);
            }
        }
        return out;
    };
    check(shard0.payload == expect_gather(0), label + ": first half of every channel block");
    check(shard1.payload == expect_gather(1), label + ": second half of every channel block");
    check(shard0.weight.k == 512 && shard1.weight.k == 512, label + ": half the gathered columns");
    check(shard0.weight.n == n && shard1.weight.n == n, label + ": the conv keeps its taps");
}

void diagnostic_cases() {
    // 768 values is a whole number of 256-value blocks, but its halves are not: a head-parallel K
    // split of a GGUF block weight has no representation and must say so.
    const std::int32_t n = 32, k = 768;
    const std::vector<std::uint8_t> full =
        random_payload(static_cast<std::size_t>(rows_bytes(QType::GGUF_IQ4_XS, k)) *
                           static_cast<std::size_t>(n),
                       0x11U);
    const Weight weight = block_weight(QType::GGUF_IQ4_XS, n, k, full);
    expect_throws([&] { (void)tp::split_weight(full, weight, tp::WeightSplitKind::RowParallel); },
                  "a K split at a non-block boundary");
    expect_throws([&] { (void)tp::require_gguf_block_columns(QType::GGUF_IQ4_XS, 128, "test"); },
                  "a partial block column count");
    expect_throws([&] { (void)tp::require_gguf_block_columns(QType::BF16, 256, "test"); },
                  "a non-GGUF format");
}

} // namespace

int main() {
    // Q8_0 carries 32-value blocks; IQ4_XS and Q4_K carry 256-value blocks.
    split_case(QType::GGUF_Q8_0, 64, 512, 0x51U);
    split_case(QType::GGUF_IQ4_XS, 64, 1024, 0x52U);
    split_case(QType::GGUF_Q4_K, 32, 1024, 0x53U);
    gather_rows_case(QType::GGUF_IQ4_XS);
    gather_cols_case(QType::GGUF_IQ4_XS);
    diagnostic_cases();
    if (failures != 0) {
        std::cerr << failures << " checks failed\n";
        return 1;
    }
    std::cout << "TP-2 GGUF block splitter: block-exact row/column splits and diagnostics passed\n";
    return 0;
}
