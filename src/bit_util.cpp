#include "bit_util.hpp"

#include "xsimd/types/xsimd_avx2_register.hpp"

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

template <typename A, SpacedExpandSIMDMode simd_mode>
int32_t spacedExpandSSE(const uint32_t *__restrict dense_in,
                        const uint64_t *bits, int32_t bits_length,
                        uint32_t *__restrict spaced_out, const A &) {
    if (bits_length <= 0) {
        return 0;
    }
    int32_t row = 0;
    int32_t cur_dense_idx = 0;
    int32_t endWord = bits::roundUp(bits_length, 64) / 64;
    constexpr int batch_block_size = 1024;
    // Use blocks of size batch_block_size, in which we have miniblocks of size
    // 64
    int32_t index_buf[batch_block_size];
    constexpr int num_miniblocks = batch_block_size / 64;
    for (auto wordIndex = 0; wordIndex < endWord;) {
        auto index_buf_ptr = index_buf;
        bool is_simd = (simd_mode == SpacedExpandSIMDMode::ALWAYS_SIMD) ||
                       (simd_mode == SpacedExpandSIMDMode::ADAPTIVE &&
                        (cur_dense_idx >= (row >> 1)));
        // Expect this loop to be unrolled
        for (int miniblock_i = 0; miniblock_i < num_miniblocks;
             miniblock_i++, wordIndex++) {
            uint64_t word = bits[wordIndex];
            if (!word) {
                // Optimization: skip empty words
                row += 64;
                continue;
            }
            if (wordIndex == endWord - 1) {
                // Correctness: mask out bits after end
                int32_t lastBits = bits_length - (endWord - 1) * 64;
                if (lastBits < 64) {
                    word &= bits::lowMask(lastBits);
                    if (!word) {
                        break;
                    }
                }
                // Terminate the loop after this miniblock
                miniblock_i = num_miniblocks;
            }
            if (!is_simd) {
                // Heuristic: if we have less than 32 elements per word
                // (estimated), use scalar
                do {
                    spaced_out[__builtin_ctzll(word) + row] =
                        dense_in[cur_dense_idx++];
                    word = word & (word - 1);
                } while (word);
                row += 64;
            } else {
                // Expect this to be unrolled
                for (auto byteCnt = 0; byteCnt < 8; ++byteCnt) {
                    uint8_t byte = word;
                    word = word >> 8;
                    using Batch = xsimd::batch<int32_t, A>;
                    auto indices = byteSetBits(byte);
                    if constexpr (Batch::size == 8) {
                        (Batch::load_aligned(indices) + row)
                            .store_unaligned(index_buf_ptr);
                    } else {
                        static_assert(Batch::size == 4);
                        (Batch::load_aligned(indices) + row)
                            .store_unaligned(index_buf_ptr);
                        auto lo = byte & ((1 << 4) - 1);
                        int lo_pop = __builtin_popcount(lo);
                        (Batch::load_unaligned(indices + lo_pop) + row)
                            .store_unaligned(index_buf_ptr + lo_pop);
                    }
                    index_buf_ptr += __builtin_popcount(byte);
                    row += 8;
                }
            }
        }
        if (is_simd) {
            for (auto copy_ptr = index_buf; copy_ptr != index_buf_ptr;
                 copy_ptr++) {
                spaced_out[*copy_ptr] = dense_in[cur_dense_idx++];
            }
        }
    }
    return cur_dense_idx;
}

// Instantiate the SSE, adaptive and scalar versions
template int32_t
spacedExpandSSE<xsimd::sse4_2, SpacedExpandSIMDMode::ALWAYS_SCALAR>(
    const uint32_t *__restrict dense_in, const uint64_t *bits,
    int32_t bits_length, uint32_t *__restrict spaced_out,
    const xsimd::sse4_2 &);
template int32_t
spacedExpandSSE<xsimd::sse4_2, SpacedExpandSIMDMode::ALWAYS_SIMD>(
    const uint32_t *__restrict dense_in, const uint64_t *bits,
    int32_t bits_length, uint32_t *__restrict spaced_out,
    const xsimd::sse4_2 &);
template int32_t spacedExpandSSE<xsimd::sse4_2, SpacedExpandSIMDMode::ADAPTIVE>(
    const uint32_t *__restrict dense_in, const uint64_t *bits,
    int32_t bits_length, uint32_t *__restrict spaced_out,
    const xsimd::sse4_2 &);

template <typename A, bool is_simd>
int32_t bmToSV(const uint64_t *bits, int32_t bits_length,
               uint32_t *__restrict index_out, const A &) {
    if (bits_length <= 0) {
        return 0;
    }
    const auto orig_index_out = index_out;
    int32_t row = 0;
    int32_t endWord = bits::roundUp(bits_length, 64) / 64;
    for (auto wordIndex = 0; wordIndex < endWord; ++wordIndex) {
        uint64_t word = bits[wordIndex];
        if (!word) {
            // Optimization: skip empty words
            row += 64;
            continue;
        }
        if (wordIndex == endWord - 1) {
            // Correctness: mask out bits after end
            int32_t lastBits = bits_length - (endWord - 1) * 64;
            if (lastBits < 64) {
                word &= bits::lowMask(lastBits);
                if (!word) {
                    break;
                }
            }
        }
        if (!is_simd) {
            do {
                *index_out++ = __builtin_ctzll(word) + row;
                word = word & (word - 1);
            } while (word);
            row += 64;
        } else {
            for (auto byteCnt = 0; byteCnt < 8; ++byteCnt) {
                uint8_t byte = word;
                word = word >> 8;
                using Batch = xsimd::batch<int32_t, A>;
                auto indices = byteSetBits(byte);
                if constexpr (Batch::size == 8) {
                    (Batch::load_aligned(indices) + row)
                        .store_unaligned(index_out);
                } else {
                    static_assert(Batch::size == 4);
                    (Batch::load_aligned(indices) + row)
                        .store_unaligned(index_out);
                    auto lo = byte & ((1 << 4) - 1);
                    int lo_pop = __builtin_popcount(lo);
                    (Batch::load_unaligned(indices + lo_pop) + row)
                        .store_unaligned(index_out + lo_pop);
                }
                index_out += __builtin_popcount(byte);
                row += 8;
            }
        }
    }
    return index_out - orig_index_out;
}

template int32_t bmToSV<xsimd::sse4_2, true>(const uint64_t *bits,
                                             int32_t bits_length,
                                             uint32_t *__restrict index_out,
                                             const xsimd::sse4_2 &);
template int32_t bmToSV<xsimd::sse4_2, false>(const uint64_t *bits,
                                              int32_t bits_length,
                                              uint32_t *__restrict index_out,
                                              const xsimd::sse4_2 &);