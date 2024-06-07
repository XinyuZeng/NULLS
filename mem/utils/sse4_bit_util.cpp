#include "spaced_expand_fused.cpp"

namespace null_revisit {
template int32_t spacedExpandFusedMiniblocks<xsimd::sse4_2, true>(
    const uint32_t *__restrict dense_in, const uint64_t *bits,
    int32_t bits_length, uint32_t *__restrict spaced_out,
    const xsimd::sse4_2 &);
template int32_t spacedExpandFusedMiniblocks<xsimd::sse4_2, false>(
    const uint32_t *__restrict dense_in, const uint64_t *bits,
    int32_t bits_length, uint32_t *__restrict spaced_out,
    const xsimd::sse4_2 &);
INSTANTIATE_ALL(xsimd::sse4_2, true)

template int32_t spacedExpandFused<xsimd::sse4_2, true>(
    const uint32_t *__restrict dense_in, const uint64_t *bits,
    int32_t bits_length, uint32_t *__restrict spaced_out,
    const xsimd::sse4_2 &);
template int32_t spacedExpandFused<xsimd::sse4_2, false>(
    const uint32_t *__restrict dense_in, const uint64_t *bits,
    int32_t bits_length, uint32_t *__restrict spaced_out,
    const xsimd::sse4_2 &);
} // namespace null_rep