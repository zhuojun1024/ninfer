#include "core/device.h"
#include "ninfer/ops/linear_topk.h"

#include "ops/op_tester.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <utility>
#include <vector>

namespace {

using namespace ninfer;
using namespace ninfer::test;

constexpr int kTopK       = 16;
constexpr int kCandidates = 2 * kTopK;

// The Op's documented total order: descending score, exact ties by the lower global token id.
std::vector<std::pair<float, std::int32_t>>
oracle_merge(const std::vector<std::pair<float, std::int32_t>>& first,
             const std::vector<std::pair<float, std::int32_t>>& second) {
    std::vector<std::pair<float, std::int32_t>> all = first;
    all.insert(all.end(), second.begin(), second.end());
    std::sort(all.begin(), all.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.first > rhs.first || (lhs.first == rhs.first && lhs.second < rhs.second);
    });
    all.resize(kTopK);
    return all;
}

// Twelve candidates sit strictly above 4.0 across the two lists, and twenty tie at exactly 4.0, so
// the last four slots are a boundary tie that only the ids can break. Each list also carries its own
// four-way run of equal scores below the cut.
void make_lists(int column, std::vector<std::pair<float, std::int32_t>>& first,
                std::vector<std::pair<float, std::int32_t>>& second) {
    first.clear();
    second.clear();
    for (int rank = 0; rank < kTopK; ++rank) {
        const float high = 16.0F - static_cast<float>(rank);
        const float low  = 10.5F - static_cast<float>(rank);
        first.emplace_back(rank < 6 ? high : 4.0F, 200 + rank * 3 + column);
        second.emplace_back(rank < 6 ? low : 4.0F, 100 + rank * 3 + column);
    }
}

int run_case(int columns) {
    const std::size_t elements = static_cast<std::size_t>(kTopK) * columns;
    std::vector<std::int32_t> ids_a(elements), ids_b(elements);
    std::vector<float> scores_a(elements), scores_b(elements);
    std::vector<std::pair<float, std::int32_t>> expected(elements);
    int failures = 0;
    for (int column = 0; column < columns; ++column) {
        std::vector<std::pair<float, std::int32_t>> first, second;
        make_lists(column, first, second);
        for (int rank = 0; rank < kTopK; ++rank) {
            // linear_topk's payload layout: a column's sixteen candidates are contiguous.
            const std::size_t at = static_cast<std::size_t>(column) * kTopK + rank;
            scores_a[at] = first[rank].first;
            ids_a[at]    = first[rank].second;
            scores_b[at] = second[rank].first;
            ids_b[at]    = second[rank].second;
        }
        const auto merged = oracle_merge(first, second);
        for (int rank = 0; rank < kTopK; ++rank) {
            expected[static_cast<std::size_t>(column) * kTopK + rank] = merged[rank];
        }
    }

    DeviceBuffer a_ids(elements * sizeof(std::int32_t));
    DeviceBuffer a_scores(elements * sizeof(float));
    DeviceBuffer b_ids(elements * sizeof(std::int32_t));
    DeviceBuffer b_scores(elements * sizeof(float));
    a_ids.copy_from_host(ids_a.data(), a_ids.bytes);
    a_scores.copy_from_host(scores_a.data(), a_scores.bytes);
    b_ids.copy_from_host(ids_b.data(), b_ids.bytes);
    b_scores.copy_from_host(scores_b.data(), b_scores.bytes);
    GuardedDeviceBuffer out_ids(elements * sizeof(std::int32_t));
    GuardedDeviceBuffer out_scores(elements * sizeof(float));
    out_ids.fill(0xcd);
    out_scores.fill(0xcd);

    Tensor t_a_ids(a_ids.p, DType::I32, {kTopK, columns});
    Tensor t_a_scores(a_scores.p, DType::FP32, {kTopK, columns});
    Tensor t_b_ids(b_ids.p, DType::I32, {kTopK, columns});
    Tensor t_b_scores(b_scores.p, DType::FP32, {kTopK, columns});
    Tensor t_out_ids(out_ids.data(), DType::I32, {kTopK, columns});
    Tensor t_out_scores(out_scores.data(), DType::FP32, {kTopK, columns});
    ops::merge_topk_candidates(t_a_ids, t_a_scores, t_b_ids, t_b_scores, t_out_ids, t_out_scores,
                               nullptr);
    cuda_synchronize();

    const auto got_ids    = from_device<std::int32_t>(out_ids.data(), elements);
    const auto got_scores = from_device<std::uint32_t>(out_scores.data(), elements);
    for (std::size_t index = 0; index < elements; ++index) {
        std::uint32_t want_bits = 0;
        std::memcpy(&want_bits, &expected[index].first, sizeof(want_bits));
        if (got_ids[index] != expected[index].second || got_scores[index] != want_bits) {
            std::cerr << "merge_topk_candidates mismatch at column " << (index / kTopK)
                      << " rank " << (index % kTopK) << ": got " << got_ids[index] << '/'
                      << got_scores[index] << " want " << expected[index].second << '/'
                      << want_bits << '\n';
            ++failures;
            break;
        }
    }
    failures += out_ids.verify_guards("merged ids tail");
    failures += out_scores.verify_guards("merged scores tail");

    // The Op refuses overlapping outputs instead of aliasing its inputs.
    bool rejected = false;
    try {
        ops::merge_topk_candidates(t_a_ids, t_a_scores, t_b_ids, t_b_scores, t_a_ids, t_out_scores,
                                   nullptr);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    if (!rejected) {
        std::cerr << "merge_topk_candidates accepted an output aliasing its input\n";
        ++failures;
    }
    cuda_synchronize();
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        int failures = 0;
        failures += run_case(1);
        failures += run_case(7);
        failures += run_case(2 * 7);
        std::cout << (failures == 0 ? "OK" : "FAIL") << " merge_topk_candidates\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "merge_topk_candidates: " << error.what() << '\n';
        return 1;
    }
}
