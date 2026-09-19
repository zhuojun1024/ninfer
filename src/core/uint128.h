#pragma once

#include <cstdint>

#if defined(_MSC_VER) && !defined(__clang__)
#    include <intrin.h>
#endif

namespace ninfer {

// Runtime accounting needs 128-bit intermediates (prefill attention pairs, context transfer traffic)
// because the exact product of two 64-bit counts can exceed 64 bits while the reported value is a
// saturated 64-bit count. MSVC has no __int128, so the two multiplications the callers need are
// spelled with _umul128 there and with the native extension elsewhere.
struct Uint128 {
    std::uint64_t low  = 0;
    std::uint64_t high = 0;

    [[nodiscard]] static constexpr Uint128 from(std::uint64_t value) noexcept {
        return Uint128{value, 0};
    }

    [[nodiscard]] static Uint128 multiply(std::uint64_t left, std::uint64_t right) noexcept {
#if defined(_MSC_VER) && !defined(__clang__)
        Uint128 result;
        result.low = ::_umul128(left, right, &result.high);
        return result;
#else
        const unsigned __int128 product = static_cast<unsigned __int128>(left) * right;
        return Uint128{static_cast<std::uint64_t>(product),
                       static_cast<std::uint64_t>(product >> 64)};
#endif
    }

    // Saturating addition: an overflow of the 128-bit sum reports the maximum instead of wrapping.
    [[nodiscard]] static Uint128 saturating_add(Uint128 left, Uint128 right) noexcept {
        Uint128 sum{left.low + right.low, left.high + right.high};
        if (sum.low < left.low) { ++sum.high; }
        if (sum.high < left.high) { return kMaximum; }
        return sum;
    }

    [[nodiscard]] constexpr Uint128 halved() const noexcept {
        return Uint128{(low >> 1) | (high << 63), high >> 1};
    }

    // bits is in 0..63, which is all the callers need.
    [[nodiscard]] constexpr Uint128 shifted_right(unsigned bits) const noexcept {
        if (bits == 0) { return *this; }
        return Uint128{(low >> bits) | (high << (64U - bits)), high >> bits};
    }

    [[nodiscard]] constexpr bool fits_u64() const noexcept { return high == 0; }

    [[nodiscard]] friend constexpr bool operator==(Uint128 left, Uint128 right) noexcept = default;

    [[nodiscard]] friend constexpr bool operator<(Uint128 left, Uint128 right) noexcept {
        return left.high != right.high ? left.high < right.high : left.low < right.low;
    }

    static const Uint128 kMaximum;
};

inline constexpr Uint128 Uint128::kMaximum{~std::uint64_t{0}, ~std::uint64_t{0}};

} // namespace ninfer
