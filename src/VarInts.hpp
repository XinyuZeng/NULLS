#pragma once
#include <cstddef>
#include <cstdint>

namespace null_revisit {
inline uint32_t encodeZigZagValInt32(int32_t n) { return (n << 1) ^ (n >> 31); }
inline int32_t decodeZigZagValInt32(uint32_t n) {
    return (n >> 1) ^ (-(n & 1));
}
} // namespace null_revisit
