#include "macros.hpp"
#include "rsum.hpp"
#include <immintrin.h>
#define _mm128 _mm
#define _mm_set1_epi64 _mm_set1_epi64x
#define _mm256_set1_epi64 _mm256_set1_epi64x
namespace generated {
namespace rsum::x86_64 {
namespace avx512bw {
void rsum(const uint8_t *__restrict a_in_p, uint8_t *__restrict a_out_p,
          const uint8_t *__restrict a_base_p) {
  [[maybe_unused]] auto out = reinterpret_cast<__m512i *>(a_out_p);
  [[maybe_unused]] const auto in = reinterpret_cast<const __m512i *>(a_in_p);
  [[maybe_unused]] const auto base =
      reinterpret_cast<const __m512i *>(a_base_p);
  [[maybe_unused]] __m512i register_0;
  [[maybe_unused]] __m512i tmp_0;
  [[maybe_unused]] __m512i base_0;
  for (int i = 0; i < 2; ++i) {
    register_0 = _mm512_loadu_si512(in + (0 * 2) + (i * 1) + 0);
    tmp_0 = _mm512_loadu_si512(base + (0 * 2) + (i * 1));
    tmp_0 = tmp_0 + register_0;
    _mm512_storeu_si512(out + (i * 1) + (0 * 2) + 0, tmp_0);
    register_0 = _mm512_loadu_si512(in + (0 * 2) + (i * 1) + 2);
    tmp_0 = tmp_0 + register_0;
    _mm512_storeu_si512(out + (i * 1) + (0 * 2) + 2, tmp_0);
    register_0 = _mm512_loadu_si512(in + (0 * 2) + (i * 1) + 4);
    tmp_0 = tmp_0 + register_0;
    _mm512_storeu_si512(out + (i * 1) + (0 * 2) + 4, tmp_0);
    register_0 = _mm512_loadu_si512(in + (0 * 2) + (i * 1) + 6);
    tmp_0 = tmp_0 + register_0;
    _mm512_storeu_si512(out + (i * 1) + (0 * 2) + 6, tmp_0);
    register_0 = _mm512_loadu_si512(in + (0 * 2) + (i * 1) + 8);
    tmp_0 = tmp_0 + register_0;
    _mm512_storeu_si512(out + (i * 1) + (0 * 2) + 8, tmp_0);
    register_0 = _mm512_loadu_si512(in + (0 * 2) + (i * 1) + 10);
    tmp_0 = tmp_0 + register_0;
    _mm512_storeu_si512(out + (i * 1) + (0 * 2) + 10, tmp_0);
    register_0 = _mm512_loadu_si512(in + (0 * 2) + (i * 1) + 12);
    tmp_0 = tmp_0 + register_0;
    _mm512_storeu_si512(out + (i * 1) + (0 * 2) + 12, tmp_0);
    register_0 = _mm512_loadu_si512(in + (0 * 2) + (i * 1) + 14);
    tmp_0 = tmp_0 + register_0;
    _mm512_storeu_si512(out + (i * 1) + (0 * 2) + 14, tmp_0);
  }
}
void rsum(const uint16_t *__restrict a_in_p, uint16_t *__restrict a_out_p,
          const uint16_t *__restrict a_base_p) {
  [[maybe_unused]] auto out = reinterpret_cast<__m512i *>(a_out_p);
  [[maybe_unused]] const auto in = reinterpret_cast<const __m512i *>(a_in_p);
  [[maybe_unused]] const auto base =
      reinterpret_cast<const __m512i *>(a_base_p);
  [[maybe_unused]] __m512i register_0;
  [[maybe_unused]] __m512i tmp_0;
  [[maybe_unused]] __m512i base_0;
  for (int i = 0; i < 2; ++i) {
    register_0 = _mm512_loadu_si512(in + (0 * 2) + (i * 1) + 0);
    tmp_0 = _mm512_loadu_si512(base + (0 * 2) + (i * 1));
    tmp_0 = tmp_0 + register_0;
    _mm512_storeu_si512(out + (i * 1) + (0 * 2) + 0, tmp_0);
    register_0 = _mm512_loadu_si512(in + (0 * 2) + (i * 1) + 4);
    tmp_0 = tmp_0 + register_0;
    _mm512_storeu_si512(out + (i * 1) + (0 * 2) + 4, tmp_0);
    register_0 = _mm512_loadu_si512(in + (0 * 2) + (i * 1) + 8);
    tmp_0 = tmp_0 + register_0;
    _mm512_storeu_si512(out + (i * 1) + (0 * 2) + 8, tmp_0);
    register_0 = _mm512_loadu_si512(in + (0 * 2) + (i * 1) + 12);
    tmp_0 = tmp_0 + register_0;
    _mm512_storeu_si512(out + (i * 1) + (0 * 2) + 12, tmp_0);
    register_0 = _mm512_loadu_si512(in + (0 * 2) + (i * 1) + 16);
    tmp_0 = tmp_0 + register_0;
    _mm512_storeu_si512(out + (i * 1) + (0 * 2) + 16, tmp_0);
    register_0 = _mm512_loadu_si512(in + (0 * 2) + (i * 1) + 20);
    tmp_0 = tmp_0 + register_0;
    _mm512_storeu_si512(out + (i * 1) + (0 * 2) + 20, tmp_0);
    register_0 = _mm512_loadu_si512(in + (0 * 2) + (i * 1) + 24);
    tmp_0 = tmp_0 + register_0;
    _mm512_storeu_si512(out + (i * 1) + (0 * 2) + 24, tmp_0);
    register_0 = _mm512_loadu_si512(in + (0 * 2) + (i * 1) + 28);
    tmp_0 = tmp_0 + register_0;
    _mm512_storeu_si512(out + (i * 1) + (0 * 2) + 28, tmp_0);
    register_0 = _mm512_loadu_si512(in + (0 * 2) + (i * 1) + 2);
    tmp_0 = tmp_0 + register_0;
    _mm512_storeu_si512(out + (i * 1) + (0 * 2) + 2, tmp_0);
    register_0 = _mm512_loadu_si512(in + (0 * 2) + (i * 1) + 6);
    tmp_0 = tmp_0 + register_0;
    _mm512_storeu_si512(out + (i * 1) + (0 * 2) + 6, tmp_0);
    register_0 = _mm512_loadu_si512(in + (0 * 2) + (i * 1) + 10);
    tmp_0 = tmp_0 + register_0;
    _mm512_storeu_si512(out + (i * 1) + (0 * 2) + 10, tmp_0);
    register_0 = _mm512_loadu_si512(in + (0 * 2) + (i * 1) + 14);
    tmp_0 = tmp_0 + register_0;
    _mm512_storeu_si512(out + (i * 1) + (0 * 2) + 14, tmp_0);
    register_0 = _mm512_loadu_si512(in + (0 * 2) + (i * 1) + 18);
    tmp_0 = tmp_0 + register_0;
    _mm512_storeu_si512(out + (i * 1) + (0 * 2) + 18, tmp_0);
    register_0 = _mm512_loadu_si512(in + (0 * 2) + (i * 1) + 22);
    tmp_0 = tmp_0 + register_0;
    _mm512_storeu_si512(out + (i * 1) + (0 * 2) + 22, tmp_0);
    register_0 = _mm512_loadu_si512(in + (0 * 2) + (i * 1) + 26);
    tmp_0 = tmp_0 + register_0;
    _mm512_storeu_si512(out + (i * 1) + (0 * 2) + 26, tmp_0);
    register_0 = _mm512_loadu_si512(in + (0 * 2) + (i * 1) + 30);
    tmp_0 = tmp_0 + register_0;
    _mm512_storeu_si512(out + (i * 1) + (0 * 2) + 30, tmp_0);
  }
}
void rsum(const uint32_t *__restrict a_in_p, uint32_t *__restrict a_out_p,
          const uint32_t *__restrict a_base_p) {
  [[maybe_unused]] auto out = reinterpret_cast<__m512i *>(a_out_p);
  [[maybe_unused]] const auto in = reinterpret_cast<const __m512i *>(a_in_p);
  [[maybe_unused]] const auto base =
      reinterpret_cast<const __m512i *>(a_base_p);
  [[maybe_unused]] __m512i register_0;
  [[maybe_unused]] __m512i tmp_0;
  [[maybe_unused]] __m512i base_0;
  for (int i = 0; i < 2; ++i) {
    register_0 = _mm512_loadu_si512(in + (0 * 2) + (i * 1) + 0);
    tmp_0 = _mm512_loadu_si512(base + (0 * 2) + (i * 1));
    tmp_0 = tmp_0 + register_0;
    _mm512_storeu_si512(out + (i * 1) + (0 * 2) + 0, tmp_0);
    register_0 = _mm512_loadu_si512(in + (0 * 2) + (i * 1) + 8);
    tmp_0 = tmp_0 + register_0;
    _mm512_storeu_si512(out + (i * 1) + (0 * 2) + 8, tmp_0);
    register_0 = _mm512_loadu_si512(in + (0 * 2) + (i * 1) + 16);
    tmp_0 = tmp_0 + register_0;
    _mm512_storeu_si512(out + (i * 1) + (0 * 2) + 16, tmp_0);
    register_0 = _mm512_loadu_si512(in + (0 * 2) + (i * 1) + 24);
    tmp_0 = tmp_0 + register_0;
    _mm512_storeu_si512(out + (i * 1) + (0 * 2) + 24, tmp_0);
    register_0 = _mm512_loadu_si512(in + (0 * 2) + (i * 1) + 32);
    tmp_0 = tmp_0 + register_0;
    _mm512_storeu_si512(out + (i * 1) + (0 * 2) + 32, tmp_0);
    register_0 = _mm512_loadu_si512(in + (0 * 2) + (i * 1) + 40);
    tmp_0 = tmp_0 + register_0;
    _mm512_storeu_si512(out + (i * 1) + (0 * 2) + 40, tmp_0);
    register_0 = _mm512_loadu_si512(in + (0 * 2) + (i * 1) + 48);
    tmp_0 = tmp_0 + register_0;
    _mm512_storeu_si512(out + (i * 1) + (0 * 2) + 48, tmp_0);
    register_0 = _mm512_loadu_si512(in + (0 * 2) + (i * 1) + 56);
    tmp_0 = tmp_0 + register_0;
    _mm512_storeu_si512(out + (i * 1) + (0 * 2) + 56, tmp_0);
    register_0 = _mm512_loadu_si512(in + (0 * 2) + (i * 1) + 4);
    tmp_0 = tmp_0 + register_0;
    _mm512_storeu_si512(out + (i * 1) + (0 * 2) + 4, tmp_0);
    register_0 = _mm512_loadu_si512(in + (0 * 2) + (i * 1) + 12);
    tmp_0 = tmp_0 + register_0;
    _mm512_storeu_si512(out + (i * 1) + (0 * 2) + 12, tmp_0);
    register_0 = _mm512_loadu_si512(in + (0 * 2) + (i * 1) + 20);
    tmp_0 = tmp_0 + register_0;
    _mm512_storeu_si512(out + (i * 1) + (0 * 2) + 20, tmp_0);
    register_0 = _mm512_loadu_si512(in + (0 * 2) + (i * 1) + 28);
    tmp_0 = tmp_0 + register_0;
    _mm512_storeu_si512(out + (i * 1) + (0 * 2) + 28, tmp_0);
    register_0 = _mm512_loadu_si512(in + (0 * 2) + (i * 1) + 36);
    tmp_0 = tmp_0 + register_0;
    _mm512_storeu_si512(out + (i * 1) + (0 * 2) + 36, tmp_0);
    register_0 = _mm512_loadu_si512(in + (0 * 2) + (i * 1) + 44);
    tmp_0 = tmp_0 + register_0;
    _mm512_storeu_si512(out + (i * 1) + (0 * 2) + 44, tmp_0);
    register_0 = _mm512_loadu_si512(in + (0 * 2) + (i * 1) + 52);
    tmp_0 = tmp_0 + register_0;
    _mm512_storeu_si512(out + (i * 1) + (0 * 2) + 52, tmp_0);
    register_0 = _mm512_loadu_si512(in + (0 * 2) + (i * 1) + 60);
    tmp_0 = tmp_0 + register_0;
    _mm512_storeu_si512(out + (i * 1) + (0 * 2) + 60, tmp_0);
    register_0 = _mm512_loadu_si512(in + (0 * 2) + (i * 1) + 2);
    tmp_0 = tmp_0 + register_0;
    _mm512_storeu_si512(out + (i * 1) + (0 * 2) + 2, tmp_0);
    register_0 = _mm512_loadu_si512(in + (0 * 2) + (i * 1) + 10);
    tmp_0 = tmp_0 + register_0;
    _mm512_storeu_si512(out + (i * 1) + (0 * 2) + 10, tmp_0);
    register_0 = _mm512_loadu_si512(in + (0 * 2) + (i * 1) + 18);
    tmp_0 = tmp_0 + register_0;
    _mm512_storeu_si512(out + (i * 1) + (0 * 2) + 18, tmp_0);
    register_0 = _mm512_loadu_si512(in + (0 * 2) + (i * 1) + 26);
    tmp_0 = tmp_0 + register_0;
    _mm512_storeu_si512(out + (i * 1) + (0 * 2) + 26, tmp_0);
    register_0 = _mm512_loadu_si512(in + (0 * 2) + (i * 1) + 34);
    tmp_0 = tmp_0 + register_0;
    _mm512_storeu_si512(out + (i * 1) + (0 * 2) + 34, tmp_0);
    register_0 = _mm512_loadu_si512(in + (0 * 2) + (i * 1) + 42);
    tmp_0 = tmp_0 + register_0;
    _mm512_storeu_si512(out + (i * 1) + (0 * 2) + 42, tmp_0);
    register_0 = _mm512_loadu_si512(in + (0 * 2) + (i * 1) + 50);
    tmp_0 = tmp_0 + register_0;
    _mm512_storeu_si512(out + (i * 1) + (0 * 2) + 50, tmp_0);
    register_0 = _mm512_loadu_si512(in + (0 * 2) + (i * 1) + 58);
    tmp_0 = tmp_0 + register_0;
    _mm512_storeu_si512(out + (i * 1) + (0 * 2) + 58, tmp_0);
    register_0 = _mm512_loadu_si512(in + (0 * 2) + (i * 1) + 6);
    tmp_0 = tmp_0 + register_0;
    _mm512_storeu_si512(out + (i * 1) + (0 * 2) + 6, tmp_0);
    register_0 = _mm512_loadu_si512(in + (0 * 2) + (i * 1) + 14);
    tmp_0 = tmp_0 + register_0;
    _mm512_storeu_si512(out + (i * 1) + (0 * 2) + 14, tmp_0);
    register_0 = _mm512_loadu_si512(in + (0 * 2) + (i * 1) + 22);
    tmp_0 = tmp_0 + register_0;
    _mm512_storeu_si512(out + (i * 1) + (0 * 2) + 22, tmp_0);
    register_0 = _mm512_loadu_si512(in + (0 * 2) + (i * 1) + 30);
    tmp_0 = tmp_0 + register_0;
    _mm512_storeu_si512(out + (i * 1) + (0 * 2) + 30, tmp_0);
    register_0 = _mm512_loadu_si512(in + (0 * 2) + (i * 1) + 38);
    tmp_0 = tmp_0 + register_0;
    _mm512_storeu_si512(out + (i * 1) + (0 * 2) + 38, tmp_0);
    register_0 = _mm512_loadu_si512(in + (0 * 2) + (i * 1) + 46);
    tmp_0 = tmp_0 + register_0;
    _mm512_storeu_si512(out + (i * 1) + (0 * 2) + 46, tmp_0);
    register_0 = _mm512_loadu_si512(in + (0 * 2) + (i * 1) + 54);
    tmp_0 = tmp_0 + register_0;
    _mm512_storeu_si512(out + (i * 1) + (0 * 2) + 54, tmp_0);
    register_0 = _mm512_loadu_si512(in + (0 * 2) + (i * 1) + 62);
    tmp_0 = tmp_0 + register_0;
    _mm512_storeu_si512(out + (i * 1) + (0 * 2) + 62, tmp_0);
  }
}
void rsum(const uint64_t *__restrict a_in_p, uint64_t *__restrict a_out_p,
          const uint64_t *__restrict a_base_p) {
  [[maybe_unused]] auto out = reinterpret_cast<__m512i *>(a_out_p);
  [[maybe_unused]] const auto in = reinterpret_cast<const __m512i *>(a_in_p);
  [[maybe_unused]] const auto base =
      reinterpret_cast<const __m512i *>(a_base_p);
  [[maybe_unused]] __m512i register_0;
  [[maybe_unused]] __m512i tmp_0;
  [[maybe_unused]] __m512i base_0;
  for (int i = 0; i < 2; ++i) {
    register_0 = _mm512_loadu_si512(in + (0 * 2) + (i * 1) + 0);
    tmp_0 = _mm512_loadu_si512(base + (0 * 2) + (i * 1));
    tmp_0 = tmp_0 + register_0;
    _mm512_storeu_si512(out + (i * 1) + (0 * 2) + 0, tmp_0);
    register_0 = _mm512_loadu_si512(in + (0 * 2) + (i * 1) + 16);
    tmp_0 = tmp_0 + register_0;
    _mm512_storeu_si512(out + (i * 1) + (0 * 2) + 16, tmp_0);
    register_0 = _mm512_loadu_si512(in + (0 * 2) + (i * 1) + 32);
    tmp_0 = tmp_0 + register_0;
    _mm512_storeu_si512(out + (i * 1) + (0 * 2) + 32, tmp_0);
    register_0 = _mm512_loadu_si512(in + (0 * 2) + (i * 1) + 48);
    tmp_0 = tmp_0 + register_0;
    _mm512_storeu_si512(out + (i * 1) + (0 * 2) + 48, tmp_0);
    register_0 = _mm512_loadu_si512(in + (0 * 2) + (i * 1) + 64);
    tmp_0 = tmp_0 + register_0;
    _mm512_storeu_si512(out + (i * 1) + (0 * 2) + 64, tmp_0);
    register_0 = _mm512_loadu_si512(in + (0 * 2) + (i * 1) + 80);
    tmp_0 = tmp_0 + register_0;
    _mm512_storeu_si512(out + (i * 1) + (0 * 2) + 80, tmp_0);
    register_0 = _mm512_loadu_si512(in + (0 * 2) + (i * 1) + 96);
    tmp_0 = tmp_0 + register_0;
    _mm512_storeu_si512(out + (i * 1) + (0 * 2) + 96, tmp_0);
    register_0 = _mm512_loadu_si512(in + (0 * 2) + (i * 1) + 112);
    tmp_0 = tmp_0 + register_0;
    _mm512_storeu_si512(out + (i * 1) + (0 * 2) + 112, tmp_0);
    register_0 = _mm512_loadu_si512(in + (0 * 2) + (i * 1) + 8);
    tmp_0 = tmp_0 + register_0;
    _mm512_storeu_si512(out + (i * 1) + (0 * 2) + 8, tmp_0);
    register_0 = _mm512_loadu_si512(in + (0 * 2) + (i * 1) + 24);
    tmp_0 = tmp_0 + register_0;
    _mm512_storeu_si512(out + (i * 1) + (0 * 2) + 24, tmp_0);
    register_0 = _mm512_loadu_si512(in + (0 * 2) + (i * 1) + 40);
    tmp_0 = tmp_0 + register_0;
    _mm512_storeu_si512(out + (i * 1) + (0 * 2) + 40, tmp_0);
    register_0 = _mm512_loadu_si512(in + (0 * 2) + (i * 1) + 56);
    tmp_0 = tmp_0 + register_0;
    _mm512_storeu_si512(out + (i * 1) + (0 * 2) + 56, tmp_0);
    register_0 = _mm512_loadu_si512(in + (0 * 2) + (i * 1) + 72);
    tmp_0 = tmp_0 + register_0;
    _mm512_storeu_si512(out + (i * 1) + (0 * 2) + 72, tmp_0);
    register_0 = _mm512_loadu_si512(in + (0 * 2) + (i * 1) + 88);
    tmp_0 = tmp_0 + register_0;
    _mm512_storeu_si512(out + (i * 1) + (0 * 2) + 88, tmp_0);
    register_0 = _mm512_loadu_si512(in + (0 * 2) + (i * 1) + 104);
    tmp_0 = tmp_0 + register_0;
    _mm512_storeu_si512(out + (i * 1) + (0 * 2) + 104, tmp_0);
    register_0 = _mm512_loadu_si512(in + (0 * 2) + (i * 1) + 120);
    tmp_0 = tmp_0 + register_0;
    _mm512_storeu_si512(out + (i * 1) + (0 * 2) + 120, tmp_0);
    register_0 = _mm512_loadu_si512(in + (0 * 2) + (i * 1) + 4);
    tmp_0 = tmp_0 + register_0;
    _mm512_storeu_si512(out + (i * 1) + (0 * 2) + 4, tmp_0);
    register_0 = _mm512_loadu_si512(in + (0 * 2) + (i * 1) + 20);
    tmp_0 = tmp_0 + register_0;
    _mm512_storeu_si512(out + (i * 1) + (0 * 2) + 20, tmp_0);
    register_0 = _mm512_loadu_si512(in + (0 * 2) + (i * 1) + 36);
    tmp_0 = tmp_0 + register_0;
    _mm512_storeu_si512(out + (i * 1) + (0 * 2) + 36, tmp_0);
    register_0 = _mm512_loadu_si512(in + (0 * 2) + (i * 1) + 52);
    tmp_0 = tmp_0 + register_0;
    _mm512_storeu_si512(out + (i * 1) + (0 * 2) + 52, tmp_0);
    register_0 = _mm512_loadu_si512(in + (0 * 2) + (i * 1) + 68);
    tmp_0 = tmp_0 + register_0;
    _mm512_storeu_si512(out + (i * 1) + (0 * 2) + 68, tmp_0);
    register_0 = _mm512_loadu_si512(in + (0 * 2) + (i * 1) + 84);
    tmp_0 = tmp_0 + register_0;
    _mm512_storeu_si512(out + (i * 1) + (0 * 2) + 84, tmp_0);
    register_0 = _mm512_loadu_si512(in + (0 * 2) + (i * 1) + 100);
    tmp_0 = tmp_0 + register_0;
    _mm512_storeu_si512(out + (i * 1) + (0 * 2) + 100, tmp_0);
    register_0 = _mm512_loadu_si512(in + (0 * 2) + (i * 1) + 116);
    tmp_0 = tmp_0 + register_0;
    _mm512_storeu_si512(out + (i * 1) + (0 * 2) + 116, tmp_0);
    register_0 = _mm512_loadu_si512(in + (0 * 2) + (i * 1) + 12);
    tmp_0 = tmp_0 + register_0;
    _mm512_storeu_si512(out + (i * 1) + (0 * 2) + 12, tmp_0);
    register_0 = _mm512_loadu_si512(in + (0 * 2) + (i * 1) + 28);
    tmp_0 = tmp_0 + register_0;
    _mm512_storeu_si512(out + (i * 1) + (0 * 2) + 28, tmp_0);
    register_0 = _mm512_loadu_si512(in + (0 * 2) + (i * 1) + 44);
    tmp_0 = tmp_0 + register_0;
    _mm512_storeu_si512(out + (i * 1) + (0 * 2) + 44, tmp_0);
    register_0 = _mm512_loadu_si512(in + (0 * 2) + (i * 1) + 60);
    tmp_0 = tmp_0 + register_0;
    _mm512_storeu_si512(out + (i * 1) + (0 * 2) + 60, tmp_0);
    register_0 = _mm512_loadu_si512(in + (0 * 2) + (i * 1) + 76);
    tmp_0 = tmp_0 + register_0;
    _mm512_storeu_si512(out + (i * 1) + (0 * 2) + 76, tmp_0);
    register_0 = _mm512_loadu_si512(in + (0 * 2) + (i * 1) + 92);
    tmp_0 = tmp_0 + register_0;
    _mm512_storeu_si512(out + (i * 1) + (0 * 2) + 92, tmp_0);
    register_0 = _mm512_loadu_si512(in + (0 * 2) + (i * 1) + 108);
    tmp_0 = tmp_0 + register_0;
    _mm512_storeu_si512(out + (i * 1) + (0 * 2) + 108, tmp_0);
    register_0 = _mm512_loadu_si512(in + (0 * 2) + (i * 1) + 124);
    tmp_0 = tmp_0 + register_0;
    _mm512_storeu_si512(out + (i * 1) + (0 * 2) + 124, tmp_0);
    register_0 = _mm512_loadu_si512(in + (0 * 2) + (i * 1) + 2);
    tmp_0 = tmp_0 + register_0;
    _mm512_storeu_si512(out + (i * 1) + (0 * 2) + 2, tmp_0);
    register_0 = _mm512_loadu_si512(in + (0 * 2) + (i * 1) + 18);
    tmp_0 = tmp_0 + register_0;
    _mm512_storeu_si512(out + (i * 1) + (0 * 2) + 18, tmp_0);
    register_0 = _mm512_loadu_si512(in + (0 * 2) + (i * 1) + 34);
    tmp_0 = tmp_0 + register_0;
    _mm512_storeu_si512(out + (i * 1) + (0 * 2) + 34, tmp_0);
    register_0 = _mm512_loadu_si512(in + (0 * 2) + (i * 1) + 50);
    tmp_0 = tmp_0 + register_0;
    _mm512_storeu_si512(out + (i * 1) + (0 * 2) + 50, tmp_0);
    register_0 = _mm512_loadu_si512(in + (0 * 2) + (i * 1) + 66);
    tmp_0 = tmp_0 + register_0;
    _mm512_storeu_si512(out + (i * 1) + (0 * 2) + 66, tmp_0);
    register_0 = _mm512_loadu_si512(in + (0 * 2) + (i * 1) + 82);
    tmp_0 = tmp_0 + register_0;
    _mm512_storeu_si512(out + (i * 1) + (0 * 2) + 82, tmp_0);
    register_0 = _mm512_loadu_si512(in + (0 * 2) + (i * 1) + 98);
    tmp_0 = tmp_0 + register_0;
    _mm512_storeu_si512(out + (i * 1) + (0 * 2) + 98, tmp_0);
    register_0 = _mm512_loadu_si512(in + (0 * 2) + (i * 1) + 114);
    tmp_0 = tmp_0 + register_0;
    _mm512_storeu_si512(out + (i * 1) + (0 * 2) + 114, tmp_0);
    register_0 = _mm512_loadu_si512(in + (0 * 2) + (i * 1) + 10);
    tmp_0 = tmp_0 + register_0;
    _mm512_storeu_si512(out + (i * 1) + (0 * 2) + 10, tmp_0);
    register_0 = _mm512_loadu_si512(in + (0 * 2) + (i * 1) + 26);
    tmp_0 = tmp_0 + register_0;
    _mm512_storeu_si512(out + (i * 1) + (0 * 2) + 26, tmp_0);
    register_0 = _mm512_loadu_si512(in + (0 * 2) + (i * 1) + 42);
    tmp_0 = tmp_0 + register_0;
    _mm512_storeu_si512(out + (i * 1) + (0 * 2) + 42, tmp_0);
    register_0 = _mm512_loadu_si512(in + (0 * 2) + (i * 1) + 58);
    tmp_0 = tmp_0 + register_0;
    _mm512_storeu_si512(out + (i * 1) + (0 * 2) + 58, tmp_0);
    register_0 = _mm512_loadu_si512(in + (0 * 2) + (i * 1) + 74);
    tmp_0 = tmp_0 + register_0;
    _mm512_storeu_si512(out + (i * 1) + (0 * 2) + 74, tmp_0);
    register_0 = _mm512_loadu_si512(in + (0 * 2) + (i * 1) + 90);
    tmp_0 = tmp_0 + register_0;
    _mm512_storeu_si512(out + (i * 1) + (0 * 2) + 90, tmp_0);
    register_0 = _mm512_loadu_si512(in + (0 * 2) + (i * 1) + 106);
    tmp_0 = tmp_0 + register_0;
    _mm512_storeu_si512(out + (i * 1) + (0 * 2) + 106, tmp_0);
    register_0 = _mm512_loadu_si512(in + (0 * 2) + (i * 1) + 122);
    tmp_0 = tmp_0 + register_0;
    _mm512_storeu_si512(out + (i * 1) + (0 * 2) + 122, tmp_0);
    register_0 = _mm512_loadu_si512(in + (0 * 2) + (i * 1) + 6);
    tmp_0 = tmp_0 + register_0;
    _mm512_storeu_si512(out + (i * 1) + (0 * 2) + 6, tmp_0);
    register_0 = _mm512_loadu_si512(in + (0 * 2) + (i * 1) + 22);
    tmp_0 = tmp_0 + register_0;
    _mm512_storeu_si512(out + (i * 1) + (0 * 2) + 22, tmp_0);
    register_0 = _mm512_loadu_si512(in + (0 * 2) + (i * 1) + 38);
    tmp_0 = tmp_0 + register_0;
    _mm512_storeu_si512(out + (i * 1) + (0 * 2) + 38, tmp_0);
    register_0 = _mm512_loadu_si512(in + (0 * 2) + (i * 1) + 54);
    tmp_0 = tmp_0 + register_0;
    _mm512_storeu_si512(out + (i * 1) + (0 * 2) + 54, tmp_0);
    register_0 = _mm512_loadu_si512(in + (0 * 2) + (i * 1) + 70);
    tmp_0 = tmp_0 + register_0;
    _mm512_storeu_si512(out + (i * 1) + (0 * 2) + 70, tmp_0);
    register_0 = _mm512_loadu_si512(in + (0 * 2) + (i * 1) + 86);
    tmp_0 = tmp_0 + register_0;
    _mm512_storeu_si512(out + (i * 1) + (0 * 2) + 86, tmp_0);
    register_0 = _mm512_loadu_si512(in + (0 * 2) + (i * 1) + 102);
    tmp_0 = tmp_0 + register_0;
    _mm512_storeu_si512(out + (i * 1) + (0 * 2) + 102, tmp_0);
    register_0 = _mm512_loadu_si512(in + (0 * 2) + (i * 1) + 118);
    tmp_0 = tmp_0 + register_0;
    _mm512_storeu_si512(out + (i * 1) + (0 * 2) + 118, tmp_0);
    register_0 = _mm512_loadu_si512(in + (0 * 2) + (i * 1) + 14);
    tmp_0 = tmp_0 + register_0;
    _mm512_storeu_si512(out + (i * 1) + (0 * 2) + 14, tmp_0);
    register_0 = _mm512_loadu_si512(in + (0 * 2) + (i * 1) + 30);
    tmp_0 = tmp_0 + register_0;
    _mm512_storeu_si512(out + (i * 1) + (0 * 2) + 30, tmp_0);
    register_0 = _mm512_loadu_si512(in + (0 * 2) + (i * 1) + 46);
    tmp_0 = tmp_0 + register_0;
    _mm512_storeu_si512(out + (i * 1) + (0 * 2) + 46, tmp_0);
    register_0 = _mm512_loadu_si512(in + (0 * 2) + (i * 1) + 62);
    tmp_0 = tmp_0 + register_0;
    _mm512_storeu_si512(out + (i * 1) + (0 * 2) + 62, tmp_0);
    register_0 = _mm512_loadu_si512(in + (0 * 2) + (i * 1) + 78);
    tmp_0 = tmp_0 + register_0;
    _mm512_storeu_si512(out + (i * 1) + (0 * 2) + 78, tmp_0);
    register_0 = _mm512_loadu_si512(in + (0 * 2) + (i * 1) + 94);
    tmp_0 = tmp_0 + register_0;
    _mm512_storeu_si512(out + (i * 1) + (0 * 2) + 94, tmp_0);
    register_0 = _mm512_loadu_si512(in + (0 * 2) + (i * 1) + 110);
    tmp_0 = tmp_0 + register_0;
    _mm512_storeu_si512(out + (i * 1) + (0 * 2) + 110, tmp_0);
    register_0 = _mm512_loadu_si512(in + (0 * 2) + (i * 1) + 126);
    tmp_0 = tmp_0 + register_0;
    _mm512_storeu_si512(out + (i * 1) + (0 * 2) + 126, tmp_0);
  }
}
} // namespace avx512bw
namespace avx512bw_dm {
void rsum(const uint32_t *__restrict a_in_p, uint32_t *__restrict a_out_p,
          const uint32_t *__restrict a_base_p) {
  [[maybe_unused]] auto out = reinterpret_cast<__m512i *>(a_out_p);
  [[maybe_unused]] const auto in = reinterpret_cast<const __m512i *>(a_in_p);
  [[maybe_unused]] const auto base =
      reinterpret_cast<const __m512i *>(a_base_p);
  [[maybe_unused]] __m512i register_0;
  [[maybe_unused]] __m512i tmp_0;
  [[maybe_unused]] __m512i base_0;

  register_0 = _mm512_loadu_si512(in);
  tmp_0 = _mm512_set1_epi32(*a_base_p);
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out, tmp_0);
  register_0 = _mm512_loadu_si512(in + 1);

  // tried _mm512_set1_epi32(*(reinterpret_cast<uint32_t *>(&tmp_0) + 15));
  // slow.
  tmp_0 = _mm512_set1_epi32(_mm_cvtsi128_si32(_mm512_castsi512_si128(tmp_0)));
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 1, tmp_0);

  register_0 = _mm512_loadu_si512(in + 2);
  tmp_0 = _mm512_set1_epi32(_mm_cvtsi128_si32(_mm512_castsi512_si128(tmp_0)));
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 2, tmp_0);

  register_0 = _mm512_loadu_si512(in + 3);
  tmp_0 = _mm512_set1_epi32(_mm_cvtsi128_si32(_mm512_castsi512_si128(tmp_0)));
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 3, tmp_0);

  register_0 = _mm512_loadu_si512(in + 4);
  tmp_0 = _mm512_set1_epi32(_mm_cvtsi128_si32(_mm512_castsi512_si128(tmp_0)));
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 4, tmp_0);

  register_0 = _mm512_loadu_si512(in + 5);
  tmp_0 = _mm512_set1_epi32(_mm_cvtsi128_si32(_mm512_castsi512_si128(tmp_0)));
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 5, tmp_0);

  register_0 = _mm512_loadu_si512(in + 6);
  tmp_0 = _mm512_set1_epi32(_mm_cvtsi128_si32(_mm512_castsi512_si128(tmp_0)));
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 6, tmp_0);

  register_0 = _mm512_loadu_si512(in + 7);
  tmp_0 = _mm512_set1_epi32(_mm_cvtsi128_si32(_mm512_castsi512_si128(tmp_0)));
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 7, tmp_0);

  register_0 = _mm512_loadu_si512(in + 8);
  tmp_0 = _mm512_set1_epi32(_mm_cvtsi128_si32(_mm512_castsi512_si128(tmp_0)));
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 8, tmp_0);

  register_0 = _mm512_loadu_si512(in + 9);
  tmp_0 = _mm512_set1_epi32(_mm_cvtsi128_si32(_mm512_castsi512_si128(tmp_0)));
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 9, tmp_0);

  register_0 = _mm512_loadu_si512(in + 10);
  tmp_0 = _mm512_set1_epi32(_mm_cvtsi128_si32(_mm512_castsi512_si128(tmp_0)));
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 10, tmp_0);

  register_0 = _mm512_loadu_si512(in + 11);
  tmp_0 = _mm512_set1_epi32(_mm_cvtsi128_si32(_mm512_castsi512_si128(tmp_0)));
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 11, tmp_0);

  register_0 = _mm512_loadu_si512(in + 12);
  tmp_0 = _mm512_set1_epi32(_mm_cvtsi128_si32(_mm512_castsi512_si128(tmp_0)));
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 12, tmp_0);

  register_0 = _mm512_loadu_si512(in + 13);
  tmp_0 = _mm512_set1_epi32(_mm_cvtsi128_si32(_mm512_castsi512_si128(tmp_0)));
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 13, tmp_0);

  register_0 = _mm512_loadu_si512(in + 14);
  tmp_0 = _mm512_set1_epi32(_mm_cvtsi128_si32(_mm512_castsi512_si128(tmp_0)));
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 14, tmp_0);

  register_0 = _mm512_loadu_si512(in + 15);
  tmp_0 = _mm512_set1_epi32(_mm_cvtsi128_si32(_mm512_castsi512_si128(tmp_0)));
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 15, tmp_0);

  register_0 = _mm512_loadu_si512(in + 16);
  tmp_0 = _mm512_set1_epi32(_mm_cvtsi128_si32(_mm512_castsi512_si128(tmp_0)));
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 16, tmp_0);

  register_0 = _mm512_loadu_si512(in + 17);
  tmp_0 = _mm512_set1_epi32(_mm_cvtsi128_si32(_mm512_castsi512_si128(tmp_0)));
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 17, tmp_0);

  register_0 = _mm512_loadu_si512(in + 18);
  tmp_0 = _mm512_set1_epi32(_mm_cvtsi128_si32(_mm512_castsi512_si128(tmp_0)));
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 18, tmp_0);

  register_0 = _mm512_loadu_si512(in + 19);
  tmp_0 = _mm512_set1_epi32(_mm_cvtsi128_si32(_mm512_castsi512_si128(tmp_0)));
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 19, tmp_0);

  register_0 = _mm512_loadu_si512(in + 20);
  tmp_0 = _mm512_set1_epi32(_mm_cvtsi128_si32(_mm512_castsi512_si128(tmp_0)));
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 20, tmp_0);

  register_0 = _mm512_loadu_si512(in + 21);
  tmp_0 = _mm512_set1_epi32(_mm_cvtsi128_si32(_mm512_castsi512_si128(tmp_0)));
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 21, tmp_0);

  register_0 = _mm512_loadu_si512(in + 22);
  tmp_0 = _mm512_set1_epi32(_mm_cvtsi128_si32(_mm512_castsi512_si128(tmp_0)));
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 22, tmp_0);

  register_0 = _mm512_loadu_si512(in + 23);
  tmp_0 = _mm512_set1_epi32(_mm_cvtsi128_si32(_mm512_castsi512_si128(tmp_0)));
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 23, tmp_0);

  register_0 = _mm512_loadu_si512(in + 24);
  tmp_0 = _mm512_set1_epi32(_mm_cvtsi128_si32(_mm512_castsi512_si128(tmp_0)));
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 24, tmp_0);

  register_0 = _mm512_loadu_si512(in + 25);
  tmp_0 = _mm512_set1_epi32(_mm_cvtsi128_si32(_mm512_castsi512_si128(tmp_0)));
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 25, tmp_0);

  register_0 = _mm512_loadu_si512(in + 26);
  tmp_0 = _mm512_set1_epi32(_mm_cvtsi128_si32(_mm512_castsi512_si128(tmp_0)));
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 26, tmp_0);

  register_0 = _mm512_loadu_si512(in + 27);
  tmp_0 = _mm512_set1_epi32(_mm_cvtsi128_si32(_mm512_castsi512_si128(tmp_0)));
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 27, tmp_0);

  register_0 = _mm512_loadu_si512(in + 28);
  tmp_0 = _mm512_set1_epi32(_mm_cvtsi128_si32(_mm512_castsi512_si128(tmp_0)));
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 28, tmp_0);

  register_0 = _mm512_loadu_si512(in + 29);
  tmp_0 = _mm512_set1_epi32(_mm_cvtsi128_si32(_mm512_castsi512_si128(tmp_0)));
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 29, tmp_0);

  register_0 = _mm512_loadu_si512(in + 30);
  tmp_0 = _mm512_set1_epi32(_mm_cvtsi128_si32(_mm512_castsi512_si128(tmp_0)));
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 30, tmp_0);

  register_0 = _mm512_loadu_si512(in + 31);
  tmp_0 = _mm512_set1_epi32(_mm_cvtsi128_si32(_mm512_castsi512_si128(tmp_0)));
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 31, tmp_0);

  register_0 = _mm512_loadu_si512(in + 32);
  tmp_0 = _mm512_set1_epi32(_mm_cvtsi128_si32(_mm512_castsi512_si128(tmp_0)));
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 32, tmp_0);

  register_0 = _mm512_loadu_si512(in + 33);
  tmp_0 = _mm512_set1_epi32(_mm_cvtsi128_si32(_mm512_castsi512_si128(tmp_0)));
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 33, tmp_0);

  register_0 = _mm512_loadu_si512(in + 34);
  tmp_0 = _mm512_set1_epi32(_mm_cvtsi128_si32(_mm512_castsi512_si128(tmp_0)));
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 34, tmp_0);

  register_0 = _mm512_loadu_si512(in + 35);
  tmp_0 = _mm512_set1_epi32(_mm_cvtsi128_si32(_mm512_castsi512_si128(tmp_0)));
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 35, tmp_0);

  register_0 = _mm512_loadu_si512(in + 36);
  tmp_0 = _mm512_set1_epi32(_mm_cvtsi128_si32(_mm512_castsi512_si128(tmp_0)));
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 36, tmp_0);

  register_0 = _mm512_loadu_si512(in + 37);
  tmp_0 = _mm512_set1_epi32(_mm_cvtsi128_si32(_mm512_castsi512_si128(tmp_0)));
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 37, tmp_0);

  register_0 = _mm512_loadu_si512(in + 38);
  tmp_0 = _mm512_set1_epi32(_mm_cvtsi128_si32(_mm512_castsi512_si128(tmp_0)));
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 38, tmp_0);

  register_0 = _mm512_loadu_si512(in + 39);
  tmp_0 = _mm512_set1_epi32(_mm_cvtsi128_si32(_mm512_castsi512_si128(tmp_0)));
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 39, tmp_0);

  register_0 = _mm512_loadu_si512(in + 40);
  tmp_0 = _mm512_set1_epi32(_mm_cvtsi128_si32(_mm512_castsi512_si128(tmp_0)));
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 40, tmp_0);

  register_0 = _mm512_loadu_si512(in + 41);
  tmp_0 = _mm512_set1_epi32(_mm_cvtsi128_si32(_mm512_castsi512_si128(tmp_0)));
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 41, tmp_0);

  register_0 = _mm512_loadu_si512(in + 42);
  tmp_0 = _mm512_set1_epi32(_mm_cvtsi128_si32(_mm512_castsi512_si128(tmp_0)));
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 42, tmp_0);

  register_0 = _mm512_loadu_si512(in + 43);
  tmp_0 = _mm512_set1_epi32(_mm_cvtsi128_si32(_mm512_castsi512_si128(tmp_0)));
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 43, tmp_0);

  register_0 = _mm512_loadu_si512(in + 44);
  tmp_0 = _mm512_set1_epi32(_mm_cvtsi128_si32(_mm512_castsi512_si128(tmp_0)));
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 44, tmp_0);

  register_0 = _mm512_loadu_si512(in + 45);
  tmp_0 = _mm512_set1_epi32(_mm_cvtsi128_si32(_mm512_castsi512_si128(tmp_0)));
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 45, tmp_0);

  register_0 = _mm512_loadu_si512(in + 46);
  tmp_0 = _mm512_set1_epi32(_mm_cvtsi128_si32(_mm512_castsi512_si128(tmp_0)));
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 46, tmp_0);

  register_0 = _mm512_loadu_si512(in + 47);
  tmp_0 = _mm512_set1_epi32(_mm_cvtsi128_si32(_mm512_castsi512_si128(tmp_0)));
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 47, tmp_0);

  register_0 = _mm512_loadu_si512(in + 48);
  tmp_0 = _mm512_set1_epi32(_mm_cvtsi128_si32(_mm512_castsi512_si128(tmp_0)));
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 48, tmp_0);

  register_0 = _mm512_loadu_si512(in + 49);
  tmp_0 = _mm512_set1_epi32(_mm_cvtsi128_si32(_mm512_castsi512_si128(tmp_0)));
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 49, tmp_0);

  register_0 = _mm512_loadu_si512(in + 50);
  tmp_0 = _mm512_set1_epi32(_mm_cvtsi128_si32(_mm512_castsi512_si128(tmp_0)));
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 50, tmp_0);

  register_0 = _mm512_loadu_si512(in + 51);
  tmp_0 = _mm512_set1_epi32(_mm_cvtsi128_si32(_mm512_castsi512_si128(tmp_0)));
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 51, tmp_0);

  register_0 = _mm512_loadu_si512(in + 52);
  tmp_0 = _mm512_set1_epi32(_mm_cvtsi128_si32(_mm512_castsi512_si128(tmp_0)));
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 52, tmp_0);

  register_0 = _mm512_loadu_si512(in + 53);
  tmp_0 = _mm512_set1_epi32(_mm_cvtsi128_si32(_mm512_castsi512_si128(tmp_0)));
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 53, tmp_0);

  register_0 = _mm512_loadu_si512(in + 54);
  tmp_0 = _mm512_set1_epi32(_mm_cvtsi128_si32(_mm512_castsi512_si128(tmp_0)));
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 54, tmp_0);

  register_0 = _mm512_loadu_si512(in + 55);
  tmp_0 = _mm512_set1_epi32(_mm_cvtsi128_si32(_mm512_castsi512_si128(tmp_0)));
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 55, tmp_0);

  register_0 = _mm512_loadu_si512(in + 56);
  tmp_0 = _mm512_set1_epi32(_mm_cvtsi128_si32(_mm512_castsi512_si128(tmp_0)));
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 56, tmp_0);

  register_0 = _mm512_loadu_si512(in + 57);
  tmp_0 = _mm512_set1_epi32(_mm_cvtsi128_si32(_mm512_castsi512_si128(tmp_0)));
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 57, tmp_0);

  register_0 = _mm512_loadu_si512(in + 58);
  tmp_0 = _mm512_set1_epi32(_mm_cvtsi128_si32(_mm512_castsi512_si128(tmp_0)));
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 58, tmp_0);

  register_0 = _mm512_loadu_si512(in + 59);
  tmp_0 = _mm512_set1_epi32(_mm_cvtsi128_si32(_mm512_castsi512_si128(tmp_0)));
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 59, tmp_0);

  register_0 = _mm512_loadu_si512(in + 60);
  tmp_0 = _mm512_set1_epi32(_mm_cvtsi128_si32(_mm512_castsi512_si128(tmp_0)));
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 60, tmp_0);

  register_0 = _mm512_loadu_si512(in + 61);
  tmp_0 = _mm512_set1_epi32(_mm_cvtsi128_si32(_mm512_castsi512_si128(tmp_0)));
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 61, tmp_0);

  register_0 = _mm512_loadu_si512(in + 62);
  tmp_0 = _mm512_set1_epi32(_mm_cvtsi128_si32(_mm512_castsi512_si128(tmp_0)));
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 62, tmp_0);

  register_0 = _mm512_loadu_si512(in + 63);
  tmp_0 = _mm512_set1_epi32(_mm_cvtsi128_si32(_mm512_castsi512_si128(tmp_0)));
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 63, tmp_0);
}
} // namespace avx512bw_dm

namespace avx512bw_d4 {
void rsum(const uint32_t *__restrict a_in_p, uint32_t *__restrict a_out_p,
          const uint32_t *__restrict a_base_p) {
  [[maybe_unused]] auto out = reinterpret_cast<__m512i *>(a_out_p);
  [[maybe_unused]] const auto in = reinterpret_cast<const __m512i *>(a_in_p);
  [[maybe_unused]] const auto base =
      reinterpret_cast<const __m512i *>(a_base_p);
  [[maybe_unused]] __m512i register_0;
  [[maybe_unused]] __m512i tmp_0;
  [[maybe_unused]] __m512i base_0;

  register_0 = _mm512_loadu_si512(in);
  tmp_0 = _mm512_loadu_si512(base);
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out, tmp_0);

  register_0 = _mm512_loadu_si512(in + 1);
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 1, tmp_0);

  register_0 = _mm512_loadu_si512(in + 2);
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 2, tmp_0);

  register_0 = _mm512_loadu_si512(in + 3);
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 3, tmp_0);

  register_0 = _mm512_loadu_si512(in + 4);
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 4, tmp_0);

  register_0 = _mm512_loadu_si512(in + 5);
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 5, tmp_0);

  register_0 = _mm512_loadu_si512(in + 6);
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 6, tmp_0);

  register_0 = _mm512_loadu_si512(in + 7);
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 7, tmp_0);

  register_0 = _mm512_loadu_si512(in + 8);
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 8, tmp_0);

  register_0 = _mm512_loadu_si512(in + 9);
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 9, tmp_0);

  register_0 = _mm512_loadu_si512(in + 10);
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 10, tmp_0);

  register_0 = _mm512_loadu_si512(in + 11);
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 11, tmp_0);

  register_0 = _mm512_loadu_si512(in + 12);
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 12, tmp_0);

  register_0 = _mm512_loadu_si512(in + 13);
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 13, tmp_0);

  register_0 = _mm512_loadu_si512(in + 14);
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 14, tmp_0);

  register_0 = _mm512_loadu_si512(in + 15);
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 15, tmp_0);

  register_0 = _mm512_loadu_si512(in + 16);
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 16, tmp_0);

  register_0 = _mm512_loadu_si512(in + 17);
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 17, tmp_0);

  register_0 = _mm512_loadu_si512(in + 18);
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 18, tmp_0);

  register_0 = _mm512_loadu_si512(in + 19);
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 19, tmp_0);

  register_0 = _mm512_loadu_si512(in + 20);
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 20, tmp_0);

  register_0 = _mm512_loadu_si512(in + 21);
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 21, tmp_0);

  register_0 = _mm512_loadu_si512(in + 22);
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 22, tmp_0);

  register_0 = _mm512_loadu_si512(in + 23);
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 23, tmp_0);

  register_0 = _mm512_loadu_si512(in + 24);
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 24, tmp_0);

  register_0 = _mm512_loadu_si512(in + 25);
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 25, tmp_0);

  register_0 = _mm512_loadu_si512(in + 26);
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 26, tmp_0);

  register_0 = _mm512_loadu_si512(in + 27);
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 27, tmp_0);

  register_0 = _mm512_loadu_si512(in + 28);
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 28, tmp_0);

  register_0 = _mm512_loadu_si512(in + 29);
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 29, tmp_0);

  register_0 = _mm512_loadu_si512(in + 30);
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 30, tmp_0);

  register_0 = _mm512_loadu_si512(in + 31);
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 31, tmp_0);

  register_0 = _mm512_loadu_si512(in + 32);
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 32, tmp_0);

  register_0 = _mm512_loadu_si512(in + 33);
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 33, tmp_0);

  register_0 = _mm512_loadu_si512(in + 34);
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 34, tmp_0);

  register_0 = _mm512_loadu_si512(in + 35);
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 35, tmp_0);

  register_0 = _mm512_loadu_si512(in + 36);
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 36, tmp_0);

  register_0 = _mm512_loadu_si512(in + 37);
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 37, tmp_0);

  register_0 = _mm512_loadu_si512(in + 38);
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 38, tmp_0);

  register_0 = _mm512_loadu_si512(in + 39);
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 39, tmp_0);

  register_0 = _mm512_loadu_si512(in + 40);
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 40, tmp_0);

  register_0 = _mm512_loadu_si512(in + 41);
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 41, tmp_0);

  register_0 = _mm512_loadu_si512(in + 42);
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 42, tmp_0);

  register_0 = _mm512_loadu_si512(in + 43);
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 43, tmp_0);

  register_0 = _mm512_loadu_si512(in + 44);
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 44, tmp_0);

  register_0 = _mm512_loadu_si512(in + 45);
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 45, tmp_0);

  register_0 = _mm512_loadu_si512(in + 46);
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 46, tmp_0);

  register_0 = _mm512_loadu_si512(in + 47);
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 47, tmp_0);

  register_0 = _mm512_loadu_si512(in + 48);
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 48, tmp_0);

  register_0 = _mm512_loadu_si512(in + 49);
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 49, tmp_0);

  register_0 = _mm512_loadu_si512(in + 50);
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 50, tmp_0);

  register_0 = _mm512_loadu_si512(in + 51);
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 51, tmp_0);

  register_0 = _mm512_loadu_si512(in + 52);
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 52, tmp_0);

  register_0 = _mm512_loadu_si512(in + 53);
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 53, tmp_0);

  register_0 = _mm512_loadu_si512(in + 54);
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 54, tmp_0);

  register_0 = _mm512_loadu_si512(in + 55);
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 55, tmp_0);

  register_0 = _mm512_loadu_si512(in + 56);
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 56, tmp_0);

  register_0 = _mm512_loadu_si512(in + 57);
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 57, tmp_0);

  register_0 = _mm512_loadu_si512(in + 58);
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 58, tmp_0);

  register_0 = _mm512_loadu_si512(in + 59);
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 59, tmp_0);

  register_0 = _mm512_loadu_si512(in + 60);
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 60, tmp_0);

  register_0 = _mm512_loadu_si512(in + 61);
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 61, tmp_0);

  register_0 = _mm512_loadu_si512(in + 62);
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 62, tmp_0);

  register_0 = _mm512_loadu_si512(in + 63);
  tmp_0 = tmp_0 + register_0;
  _mm512_storeu_si512(out + 63, tmp_0);
}
} // namespace avx512bw_d4
} // namespace rsum::x86_64
} // namespace generated
