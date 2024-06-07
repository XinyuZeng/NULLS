#include "spaced_expand_fused.cpp"

namespace null_revisit {
template int32_t spacedExpandFused<xsimd::avx2, true>(
    const uint32_t *__restrict dense_in, const uint64_t *bits,
    int32_t bits_length, uint32_t *__restrict spaced_out, const xsimd::avx2 &);
template int32_t spacedExpandFusedMiniblocks<xsimd::avx2, true>(
    const uint32_t *__restrict dense_in, const uint64_t *bits,
    int32_t bits_length, uint32_t *__restrict spaced_out, const xsimd::avx2 &);
INSTANTIATE_ALL(xsimd::avx2, true)
} // namespace null_revisit