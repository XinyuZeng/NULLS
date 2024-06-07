#pragma once
#include <xsimd/xsimd.hpp>

namespace bits {
template <typename T, typename U>
constexpr inline T roundUp(T value, U factor) {
    return (value + (factor - 1)) / factor * factor;
}

constexpr inline uint64_t lowMask(int32_t bits) { return (1UL << bits) - 1; }

constexpr inline uint64_t highMask(int32_t bits) {
    return lowMask(bits) << (64 - bits);
}
} // namespace bits

namespace null_revisit::detail {
extern int32_t byteSetBits[256][8];
} // namespace null_revisit::detail

inline const int32_t *byteSetBits(uint8_t byte) {
    return null_revisit::detail::byteSetBits[byte];
}

enum class SpacedExpandSIMDMode { ALWAYS_SCALAR, ALWAYS_SIMD, ADAPTIVE };

template <typename A, SpacedExpandSIMDMode simd_mode>
extern int32_t spacedExpandSSE(const uint32_t *__restrict dense_in,
                               const uint64_t *bits, int32_t bits_length,
                               uint32_t *__restrict spaced_out, const A &);

template <typename A, bool is_simd>
extern int32_t bmToSV(const uint64_t *bits, int32_t bits_length,
                      uint32_t *__restrict index_out, const A &);