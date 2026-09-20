// Bit-exact public-contract qualification for apply_token_mask. Expected values are derived from
// the logical input state; no launcher or kernel implementation is used as an oracle.
#include "ninfer/ops/token_mask.h"
#include "ops/op_tester.h"

#include <cstdint>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr std::int32_t kRows = 5;
constexpr std::int32_t kCols = 3;

int mask_contract(cudaStream_t stream) {
    const std::vector<float> values = {1.0F,   -2.5F, 0.0F,  3.25F, 100.0F,
                                       -0.5F,  7.0F,  8.0F,  9.0F,  10.0F,
                                       -11.0F, 12.0F, 13.0F, 14.0F, 15.0F};
    std::vector<std::uint16_t> input(values.size());
    for (std::size_t index = 0; index < values.size(); ++index) {
        input[index] = f32_to_bf16(values[index]);
    }
    const std::vector<std::uint8_t> host_mask = {1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1};

    std::vector<std::uint16_t> expected = input;
    const std::uint16_t negative_infinity =
        f32_to_bf16(-std::numeric_limits<float>::infinity());
    for (std::size_t index = 0; index < expected.size(); ++index) {
        if (host_mask[index] == 0) { expected[index] = negative_infinity; }
    }

    GuardedDeviceBuffer logits_buffer(input.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer mask_buffer(host_mask.size());
    logits_buffer.copy_from_host(input.data(), input.size() * sizeof(std::uint16_t));
    mask_buffer.copy_from_host(host_mask.data(), host_mask.size());

    Tensor logits(logits_buffer.data(), DType::BF16, {kRows, kCols});
    Tensor mask(mask_buffer.data(), DType::U8, {kRows, kCols});
    ops::apply_token_mask(logits, mask, stream);
    cuda_synchronize(stream);

    int failures = 0;
    failures += verify_exact("apply_token_mask logits",
                             from_device<std::uint16_t>(logits_buffer.data(), expected.size()),
                             expected);
    failures += verify_exact("apply_token_mask mask stays read-only",
                             from_device<std::uint8_t>(mask_buffer.data(), host_mask.size()),
                             host_mask);
    failures += logits_buffer.verify_guards("apply_token_mask logits");
    failures += mask_buffer.verify_guards("apply_token_mask mask");
    return failures;
}

int all_masked_contract(cudaStream_t stream) {
    const std::vector<float> values = {4.0F, -4.0F, 0.5F, 2.0F, -8.0F, 6.0F};
    std::vector<std::uint16_t> input(values.size());
    for (std::size_t index = 0; index < values.size(); ++index) {
        input[index] = f32_to_bf16(values[index]);
    }
    const std::vector<std::uint8_t> host_mask(values.size(), 0);
    const std::uint16_t negative_infinity =
        f32_to_bf16(-std::numeric_limits<float>::infinity());
    const std::vector<std::uint16_t> expected(values.size(), negative_infinity);

    GuardedDeviceBuffer logits_buffer(input.size() * sizeof(std::uint16_t));
    GuardedDeviceBuffer mask_buffer(host_mask.size());
    logits_buffer.copy_from_host(input.data(), input.size() * sizeof(std::uint16_t));
    mask_buffer.copy_from_host(host_mask.data(), host_mask.size());

    Tensor logits(logits_buffer.data(), DType::BF16, {2, 3});
    Tensor mask(mask_buffer.data(), DType::U8, {2, 3});
    ops::apply_token_mask(logits, mask, stream);
    cuda_synchronize(stream);

    int failures = 0;
    failures += verify_exact("apply_token_mask all-rejected column",
                             from_device<std::uint16_t>(logits_buffer.data(), expected.size()),
                             expected);
    return failures;
}

int validation_contract(cudaStream_t stream) {
    GuardedDeviceBuffer logits_buffer(4 * sizeof(std::uint16_t));
    GuardedDeviceBuffer mask_buffer(4);
    logits_buffer.fill(0);
    mask_buffer.fill(1);

    Tensor logits(logits_buffer.data(), DType::BF16, {4, 1});
    Tensor mask(mask_buffer.data(), DType::U8, {4, 1});
    Tensor reshaped(mask_buffer.data(), DType::U8, {2, 2});
    Tensor wrong_dtype(logits_buffer.data(), DType::FP32, {4, 1});
    Tensor aliasing(logits_buffer.data(), DType::U8, {4, 1});
    Tensor rank_three(mask_buffer.data(), DType::U8, {2, 1, 2});

    int failures = 0;
    const auto rejects = [&failures, stream, &logits, &mask](const char* label,
                                                              const Tensor& candidate) {
        try {
            ops::apply_token_mask(logits, candidate, stream);
            std::cerr << "FAIL: apply_token_mask accepted " << label << '\n';
            ++failures;
        } catch (const std::invalid_argument&) {
        }
    };
    rejects("a mask with a mismatched shape", reshaped);
    rejects("a rank-3 mask", rank_three);
    try {
        ops::apply_token_mask(wrong_dtype, mask, stream);
        std::cerr << "FAIL: apply_token_mask accepted a non-BF16 logits tensor\n";
        ++failures;
    } catch (const std::invalid_argument&) {
    }
    rejects("an aliasing mask", aliasing);

    // A well-formed call still succeeds after the rejected ones.
    ops::apply_token_mask(logits, mask, stream);
    cuda_synchronize(stream);
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    cudaStream_t stream = nullptr;
    cuda_check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate");

    int failures = 0;
    failures += mask_contract(stream);
    failures += all_masked_contract(stream);
    failures += validation_contract(stream);

    cuda_check(cudaStreamDestroy(stream), "cudaStreamDestroy");
    std::cout << (failures == 0 ? "OK" : "FAIL") << " token mask public contract\n";
    return failures == 0 ? 0 : 1;
}
