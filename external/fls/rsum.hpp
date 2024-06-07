#ifndef RSUM_HPP
#define RSUM_HPP

#include <cstdint>

namespace generated {
namespace rsum {
namespace fallback {
namespace scalar {
void rsum(const uint8_t *__restrict in, uint8_t *__restrict out,
          const uint8_t *__restrict base);
void rsum(const uint16_t *__restrict in, uint16_t *__restrict out,
          const uint16_t *__restrict base);
void rsum(const uint32_t *__restrict in, uint32_t *__restrict out,
          const uint32_t *__restrict base);
void rsum(const uint64_t *__restrict in, uint64_t *__restrict out,
          const uint64_t *__restrict base);
} // namespace scalar
namespace unit64 {
void rsum(const uint8_t *__restrict in, uint8_t *__restrict out,
          const uint8_t *__restrict base);
void rsum(const uint16_t *__restrict in, uint16_t *__restrict out,
          const uint16_t *__restrict base);
void rsum(const uint32_t *__restrict in, uint32_t *__restrict out,
          const uint32_t *__restrict base);
void rsum(const uint64_t *__restrict in, uint64_t *__restrict out,
          const uint64_t *__restrict base);
} // namespace unit64
} // namespace fallback

namespace helper {
namespace scalar {
void rsum(const uint8_t *__restrict in, uint8_t *__restrict out,
          const uint8_t *__restrict base);
void rsum(const uint16_t *__restrict in, uint16_t *__restrict out,
          const uint16_t *__restrict base);
void rsum(const uint32_t *__restrict in, uint32_t *__restrict out,
          const uint32_t *__restrict base);
void rsum(const uint64_t *__restrict in, uint64_t *__restrict out,
          const uint64_t *__restrict base);
} // namespace scalar
} // namespace helper

namespace arm64v8 {
namespace neon {
void rsum(const uint8_t *__restrict in, uint8_t *__restrict out,
          const uint8_t *__restrict base);
void rsum(const uint16_t *__restrict in, uint16_t *__restrict out,
          const uint16_t *__restrict base);
void rsum(const uint32_t *__restrict in, uint32_t *__restrict out,
          const uint32_t *__restrict base);
void rsum(const uint64_t *__restrict in, uint64_t *__restrict out,
          const uint64_t *__restrict base);
} // namespace neon
} // namespace arm64v8

namespace x86_64 {
namespace avx2 {
void rsum(const uint8_t *__restrict in, uint8_t *__restrict out,
          const uint8_t *__restrict base);
void rsum(const uint16_t *__restrict in, uint16_t *__restrict out,
          const uint16_t *__restrict base);
void rsum(const uint32_t *__restrict in, uint32_t *__restrict out,
          const uint32_t *__restrict base);
void rsum(const uint64_t *__restrict in, uint64_t *__restrict out,
          const uint64_t *__restrict base);
} // namespace avx2

namespace sse {
void rsum(const uint8_t *__restrict in, uint8_t *__restrict out,
          const uint8_t *__restrict base);
void rsum(const uint16_t *__restrict in, uint16_t *__restrict out,
          const uint16_t *__restrict base);
void rsum(const uint32_t *__restrict in, uint32_t *__restrict out,
          const uint32_t *__restrict base);
void rsum(const uint64_t *__restrict in, uint64_t *__restrict out,
          const uint64_t *__restrict base);
} // namespace sse

namespace avx512bw {
void rsum(const uint8_t *__restrict in, uint8_t *__restrict out,
          const uint8_t *__restrict base);
void rsum(const uint16_t *__restrict in, uint16_t *__restrict out,
          const uint16_t *__restrict base);
void rsum(const uint32_t *__restrict in, uint32_t *__restrict out,
          const uint32_t *__restrict base);
void rsum(const uint64_t *__restrict in, uint64_t *__restrict out,
          const uint64_t *__restrict base);
} // namespace avx512bw
namespace avx512bw_dm {
void rsum(const uint32_t *__restrict a_in_p, uint32_t *__restrict a_out_p,
          const uint32_t *__restrict a_base_p);
}
namespace avx512bw_d4 {
void rsum(const uint32_t *__restrict a_in_p, uint32_t *__restrict a_out_p,
          const uint32_t *__restrict a_base_p);
}
} // namespace x86_64
} // namespace rsum
} // namespace generated

#endif
