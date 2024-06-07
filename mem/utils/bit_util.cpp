#include "bit_util.h"

#include "xsimd/types/xsimd_avx2_register.hpp"

/// Code vendored from Velox, Apache License
namespace null_revisit::detail {
template <typename A = xsimd::default_arch>
constexpr int32_t batchByteSize(const A & = {}) {
    return sizeof(xsimd::types::simd_register<int8_t, A>);
}
constexpr int32_t kPadding = std::max(32, batchByteSize<xsimd::avx2>());
alignas(kPadding) int32_t byteSetBits[256][8];
} // namespace null_revisit::detail

bool initByteSetBits() {
    for (int32_t i = 0; i < 256; ++i) {
        int32_t *entry = null_revisit::detail::byteSetBits[i];
        int32_t fill = 0;
        for (auto b = 0; b < 8; ++b) {
            if (i & (1 << b)) {
                entry[fill++] = b;
            }
        }
        for (; fill < 8; ++fill) {
            entry[fill] = fill;
        }
    }
    return true;
}

[[maybe_unused]] static bool initByteSetBitsResult = initByteSetBits();