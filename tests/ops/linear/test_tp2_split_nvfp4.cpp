#include "ops/linear/linear_test_common.h"

#include "core/arena.h"
#include "core/tensor.h"
#include "core/tp/weight_splitter.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

// TP-2 split-Op oracle. Splits a full NVFP4 linear weight with the production
// ninfer::tp::split_weight, runs the registered half-size Linear on each shard, combines
// (concat for column-parallel, sum for row-parallel) and compares against an independent
// FP64 reference of the full Op. This proves the production splitter + half-size shapes
// reproduce the full Op exactly.
//
// Memory layout convention (matching cpu_linear_gemm_fp64 and the kernel epilogue):
//   weight [n, k]   row-major:  weight[row * k + col]
//   activation [t, k] token-major: activation[token * k + col]
//   output [t, n]   token-major:  output[token * n + row]

namespace {

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::test::linear;
using namespace ninfer::test::quantized_weight;

constexpr double kBf16UnitRoundoff = 1.0 / 256.0;

std::vector<std::uint16_t> make_activation(std::int32_t k, std::int32_t t, std::uint32_t seed) {
    std::vector<std::uint16_t> bits(static_cast<std::size_t>(k) * t);
    std::vector<float> values(bits.size());
    fill_uniform(values, seed, -4.0F, 4.0F);
    round_to_bf16(values);
    for (std::size_t i = 0; i < bits.size(); ++i) bits[i] = f32_to_bf16(values[i]);
    return bits;
}

std::vector<float> activation_as_float(const std::vector<std::uint16_t>& bits) {
    std::vector<float> out(bits.size());
    for (std::size_t i = 0; i < bits.size(); ++i) out[i] = bf16_to_f32(bits[i]);
    return out;
}

// Runs the A16 convenience Linear on a shard. Output is token-major [t, n].
std::vector<double> run_shard(const std::vector<std::uint16_t>& activation_bits,
                              const tp::WeightShard& shard, std::int32_t k, std::int32_t t) {
    DeviceBuffer dx(activation_bits.size() * sizeof(std::uint16_t));
    dx.copy_from_host(activation_bits.data(), dx.bytes);
    DeviceBuffer dw(shard.payload.size());
    dw.copy_from_host(shard.payload.data(), dw.bytes);
    Weight weight = shard.weight;
    weight.payload = dw.p;
    weight.qdata   = dw.p;
    // Scale offset is deterministic from (n, k): align_up(n*k/2, 256).
    const std::uint64_t scale_off = (static_cast<std::uint64_t>(weight.n) * weight.k / 2 + 255) / 256 * 256;
    weight.scales = static_cast<const std::uint8_t*>(dw.p) + scale_off;
    const std::size_t out_words = static_cast<std::size_t>(weight.n) * t;
    DeviceBuffer dout(out_words * sizeof(std::uint16_t));
    Tensor input(dx.p, DType::BF16, {k, t});
    Tensor output(dout.p, DType::BF16, {weight.n, t});
    ops::linear(input, weight, output, nullptr);
    cuda_check(cudaDeviceSynchronize(), "tp oracle: synchronize shard linear");
    return from_device_bf16(dout.p, out_words);
}

int check_column_parallel(std::int32_t n, std::int32_t k, std::uint32_t seed, std::int32_t t) {
    const std::string label = "TP column [" + std::to_string(2 * n) + "," + std::to_string(k) +
                              "] T=" + std::to_string(t);
    PackedWeight full = make_nvfp4_weight(2 * n, k, seed);
    std::vector<std::int32_t> all_rows(2 * n);
    std::iota(all_rows.begin(), all_rows.end(), 0);
    const std::vector<float> ref_weight = materialize_rows_fp32(full, all_rows);
    const std::vector<std::uint16_t> bits = make_activation(k, t, seed + 1U);
    const std::vector<float> activation   = activation_as_float(bits);
    // Full reference: [t, 2n] token-major.
    std::vector<double> reference(static_cast<std::size_t>(2 * n) * t);
    cpu_linear_gemm_fp64(ref_weight.data(), activation.data(), reference.data(), 2 * n, k, t);

    const std::vector<tp::WeightShard> shards =
        tp::split_weight(full.payload, full.weight, tp::WeightSplitKind::ColumnParallel);
    const std::vector<double> out0 = run_shard(bits, shards[0], k, t);
    const std::vector<double> out1 = run_shard(bits, shards[1], k, t);

    // Interleave-extract references from [t, 2n] full reference.
    std::vector<double> ref0(static_cast<std::size_t>(n) * t);
    std::vector<double> ref1(static_cast<std::size_t>(n) * t);
    for (std::int32_t token = 0; token < t; ++token) {
        for (std::int32_t row = 0; row < n; ++row) {
            ref0[static_cast<std::size_t>(token) * n + row] =
                reference[static_cast<std::size_t>(token) * 2 * n + row];
            ref1[static_cast<std::size_t>(token) * n + row] =
                reference[static_cast<std::size_t>(token) * 2 * n + n + row];
        }
    }

    const ReductionCriterion criterion{kBf16UnitRoundoff, kBf16UnitRoundoff,
                                       2.0 * kBf16UnitRoundoff};
    int failures = 0;
    failures += verify_reduction(label + " shard0", out0, ref0, criterion);
    failures += verify_reduction(label + " shard1", out1, ref1, criterion);
    return failures;
}

int check_row_parallel(std::int32_t n, std::int32_t k, std::uint32_t seed, std::int32_t t) {
    const std::string label = "TP row [" + std::to_string(n) + "," + std::to_string(2 * k) +
                              "] T=" + std::to_string(t);
    PackedWeight full = make_nvfp4_weight(n, 2 * k, seed);
    std::vector<std::int32_t> all_rows(n);
    std::iota(all_rows.begin(), all_rows.end(), 0);
    const std::vector<float> ref_weight = materialize_rows_fp32(full, all_rows);
    const std::vector<std::uint16_t> bits = make_activation(2 * k, t, seed + 1U);
    const std::vector<float> activation   = activation_as_float(bits);
    std::vector<double> reference(static_cast<std::size_t>(n) * t);
    cpu_linear_gemm_fp64(ref_weight.data(), activation.data(), reference.data(), n, 2 * k, t);

    const std::vector<tp::WeightShard> shards =
        tp::split_weight(full.payload, full.weight, tp::WeightSplitKind::RowParallel);
    // Interleave-split activation [t, 2k] into two [t, k] halves.
    std::vector<std::uint16_t> bits0(static_cast<std::size_t>(k) * t);
    std::vector<std::uint16_t> bits1(static_cast<std::size_t>(k) * t);
    for (std::int32_t token = 0; token < t; ++token) {
        for (std::int32_t col = 0; col < k; ++col) {
            bits0[static_cast<std::size_t>(token) * k + col] =
                bits[static_cast<std::size_t>(token) * 2 * k + col];
            bits1[static_cast<std::size_t>(token) * k + col] =
                bits[static_cast<std::size_t>(token) * 2 * k + k + col];
        }
    }
    const std::vector<double> out0 = run_shard(bits0, shards[0], k, t);
    const std::vector<double> out1 = run_shard(bits1, shards[1], k, t);

    std::vector<double> combined(out0.size());
    for (std::size_t i = 0; i < combined.size(); ++i) combined[i] = out0[i] + out1[i];

    const ReductionCriterion criterion{2.0 * kBf16UnitRoundoff, 2.0 * kBf16UnitRoundoff,
                                       4.0 * kBf16UnitRoundoff};
    return verify_reduction(label + " all-reduce", combined, reference, criterion);
}

} // namespace

int main() {
    if (!ninfer::test::linear::cuda_available()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        int failures = 0;
        for (int t : {1, 8, 32}) {
            failures += check_column_parallel(7168, 5120, 901U, t);
            failures += check_column_parallel(8192, 5120, 903U, t);
            failures += check_column_parallel(17408, 5120, 904U, t);
        }
        for (int t : {1, 8, 32}) {
            failures += check_row_parallel(5120, 3072, 905U, t);
            failures += check_row_parallel(5120, 8704, 907U, t);
        }
        std::cout << (failures == 0 ? "OK" : "FAIL") << " TP-2 split Linear oracle\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "TP-2 split Linear oracle: " << error.what() << '\n';
        return 1;
    }
}
