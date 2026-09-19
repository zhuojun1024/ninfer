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

// TP-2 grouped-format split oracle. The reduced proposal head is a Q4_G64_FP16 table whose rows are
// vocabulary-parallel across the pair. NInfer splits it with the production
// ninfer::tp::split_weight (ColumnParallel), runs the registered half-size Linear on each shard and
// samples a draft from the rows stamped back at their global positions. This test qualifies that
// route against the same Linear run with the full weight on the same device: the split must
// reproduce it exactly, not merely closely.
//
// Grouped codecs pack per-row code words, optional high bits and one FP16 scale per group into
// separate planes, so the split is a pure plane-span copy. The oracle therefore also pins the
// payload geometry each shard is expected to have.
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
    if (weight_in.qhigh != nullptr) {
        weight.qhigh = static_cast<std::uint8_t*>(dw.p) + offset_of(weight_in.qhigh);
    }
    const std::size_t out_words = static_cast<std::size_t>(weight.n) * t;
    DeviceBuffer dout(out_words * sizeof(std::uint16_t));
    Tensor input(dx.p, DType::BF16, {k, t});
    Tensor output(dout.p, DType::BF16, {weight.n, t});
    ops::linear(input, weight, output, nullptr);
    cuda_check(cudaDeviceSynchronize(), "tp grouped head oracle: synchronize shard linear");
    return from_device_bf16(dout.p, out_words);
}

int check_vocabulary_parallel(std::int32_t shard_rows, std::int32_t k, std::uint32_t seed,
                              std::int32_t t) {
    const std::string label = "TP grouped head [" + std::to_string(2 * shard_rows) + "," +
                              std::to_string(k) + "] T=" + std::to_string(t);
    PackedWeight full = make_q4_g64_fp16_weight(2 * shard_rows, k, seed);
    const std::vector<std::uint16_t> bits = make_activation(k, t, seed + 1U);

    // Oracle: the same Linear over the whole reduced vocabulary on the same device.
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
    // Every shard carries exactly the planes its own geometry asks for.
    for (int shard = 0; shard < 2; ++shard) {
        const std::uint64_t shape[2] = {static_cast<std::uint64_t>(shard_rows),
                                        static_cast<std::uint64_t>(k)};
        const WeightGeometry geo =
            weight_geometry(full.weight.qtype, full.weight.layout, shape);
        if (shards[shard].payload.size() != geo.bytes ||
            shards[shard].weight.payload_bytes != geo.bytes) {
            throw std::runtime_error(label + ": shard " + std::to_string(shard) +
                                     " payload is not the shard geometry size");
        }
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
            const double want        = reference[static_cast<std::size_t>(token) * 2 * shard_rows + row];
            if (got != want) {
                std::cerr << label << ": draft logits differ at token " << token << " row " << row
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
        // The served proposal head exactly: [131072, 5120] reduced-vocabulary rows, 65536 per
        // shard. T covers the single-token draft step and the two-token draft window.
        for (int t : {1, 2}) {
            failures += check_vocabulary_parallel(65536, 5120, 971U, t);
        }
        std::cout << (failures == 0 ? "OK" : "FAIL") << " TP-2 grouped-format head oracle\n";
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "TP-2 grouped-format head oracle: " << error.what() << '\n';
        return 1;
    }
}
