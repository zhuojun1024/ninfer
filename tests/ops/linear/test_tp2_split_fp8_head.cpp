#include "ops/linear/linear_test_common.h"

#include "core/arena.h"
#include "core/tensor.h"
#include "core/tp/weight_splitter.h"
#include "ops/op_tester.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

// TP-2 vocabulary-parallel output-head oracle. The engine splits the FP8 row-scale output head with
// the production ninfer::tp::split_weight (ColumnParallel), runs the registered half-size Linear on
// each shard and assembles the full-vocabulary logits by stamping shard 0's rows at [0, n) and
// shard 1's at [n, 2n) into zeroed [2n, t] buffers and summing the pair. Sample selection and the
// speculative accept/reject read those assembled logits, so this test qualifies the route directly
// against the same Linear run with the full weight on the same device: the merge must reproduce it
// exactly, not merely closely.
//
// Memory layout convention (matching cpu_linear_gemm_fp64 and the kernel epilogue):
//   weight [n, k]   row-major:    weight[row * k + col]
//   activation [t, k] token-major: activation[token * k + col]
//   output [t, n]   token-major:  output[token * n + row]

namespace {

using namespace ninfer;
using namespace ninfer::test;
using namespace ninfer::test::linear;
using namespace ninfer::test::quantized_weight;

std::vector<std::uint16_t> make_activation(std::int32_t k, std::int32_t t, std::uint32_t seed) {
    std::vector<std::uint16_t> bits(static_cast<std::size_t>(k) * t);
    std::vector<float> values(bits.size());
    fill_uniform(values, seed, -4.0F, 4.0F);
    round_to_bf16(values);
    for (std::size_t i = 0; i < bits.size(); ++i) bits[i] = f32_to_bf16(values[i]);
    return bits;
}

// Runs the A16 convenience Linear for one weight/payload pair. The output is token-major [t, n].
// The plane payload is the host payload the weight's plane pointers refer to (the full artifact
// payload, or the splitter's shard payload); the device copy keeps the same plane offsets.
std::vector<double> run_linear(const std::vector<std::uint16_t>& activation_bits,
                               const Weight& weight_in, std::span<const std::uint8_t> plane_payload,
                               std::int32_t k, std::int32_t t) {
    DeviceBuffer dx(activation_bits.size() * sizeof(std::uint16_t));
    dx.copy_from_host(activation_bits.data(), dx.bytes);
    DeviceBuffer dw(plane_payload.size());
    dw.copy_from_host(plane_payload.data(), dw.bytes);
    const auto* base = reinterpret_cast<const std::uint8_t*>(weight_in.payload);
    const auto offset_of = [base](const void* plane) {
        return static_cast<std::ptrdiff_t>(static_cast<const std::uint8_t*>(plane) - base);
    };
    Weight weight = weight_in;
    weight.payload = dw.p;
    weight.qdata   = static_cast<std::uint8_t*>(dw.p) + offset_of(weight_in.qdata);
    weight.scales  = static_cast<std::uint8_t*>(dw.p) + offset_of(weight_in.scales);
    const std::size_t out_words = static_cast<std::size_t>(weight.n) * t;
    DeviceBuffer dout(out_words * sizeof(std::uint16_t));
    Tensor input(dx.p, DType::BF16, {k, t});
    Tensor output(dout.p, DType::BF16, {weight.n, t});
    ops::linear(input, weight, output, nullptr);
    cuda_check(cudaDeviceSynchronize(), "tp fp8 head oracle: synchronize shard linear");
    return from_device_bf16(dout.p, out_words);
}

int check_vocabulary_parallel(std::int32_t shard_rows, std::int32_t k, std::uint32_t seed,
                              std::int32_t t) {
    const std::string label = "TP vocab head [" + std::to_string(2 * shard_rows) + "," +
                              std::to_string(k) + "] T=" + std::to_string(t);
    PackedWeight full = make_fp8_weight(2 * shard_rows, k, seed);
    const std::vector<std::uint16_t> bits = make_activation(k, t, seed + 1U);

    // Oracle: the same Linear over the whole vocabulary on the same device.
    std::vector<double> reference;
    try {
        reference = run_linear(bits, full.weight, full.payload, k, t);
    } catch (const std::exception& error) {
        throw std::runtime_error(label + ": full-weight oracle rejected the weight: " + error.what());
    }

    const std::vector<tp::WeightShard> shards =
        tp::split_weight(full.payload, full.weight, tp::WeightSplitKind::ColumnParallel);
    if (shards.size() != 2 || shards[0].weight.n != shard_rows ||
        shards[1].weight.n != shard_rows || shards[0].weight.k != k || shards[1].weight.k != k) {
        throw std::runtime_error(label + ": split geometry is not two equal row blocks");
    }
    std::vector<double> out0;
    std::vector<double> out1;
    try {
        out0 = run_linear(bits, shards[0].weight, shards[0].payload, k, t);
        out1 = run_linear(bits, shards[1].weight, shards[1].payload, k, t);
    } catch (const std::exception& error) {
        throw std::runtime_error(label + ": split shard rejected the weight: " + error.what());
    }

    int failures = 0;
    for (std::int32_t token = 0; token < t && failures == 0; ++token) {
        for (std::int32_t row = 0; row < 2 * shard_rows; ++row) {
            const std::size_t local  = static_cast<std::size_t>(row % shard_rows);
            const auto* shard        = row < shard_rows ? &out0 : &out1;
            const double got         = (*shard)[static_cast<std::size_t>(token) * shard_rows + local];
            const double want =
                reference[static_cast<std::size_t>(token) * 2 * shard_rows + row];
            if (got != want) {
                std::cerr << label << ": merged logits differ at token " << token << " row " << row
                          << " (" << got << " vs " << want << ")\n";
                ++failures;
                break;
            }
        }
    }
    if (failures == 0) { std::cout << "  " << label << ": exact\n"; }
    return failures;
}

} // namespace

int main() {
    if (!ninfer::test::linear::cuda_available()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }
    try {
        int failures = 0;
        // The served artifact's head: 248320 vocabulary rows, hidden 5120. T covers the decode
        // column and the speculative verify window width.
        for (int t : {1, 3, 5}) {
            failures += check_vocabulary_parallel(124160, 5120, 951U, t);
        }
        for (int t : {1, 4}) {
            failures += check_vocabulary_parallel(8192, 5120, 953U, t);
        }
        std::cout << (failures == 0 ? "OK" : "FAIL") << " TP-2 vocabulary-parallel head oracle\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "TP-2 vocabulary-parallel head oracle: " << error.what() << '\n';
        return 1;
    }
}
