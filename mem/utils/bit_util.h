#pragma once
#include <arrow/buffer.h>
#include <arrow/testing/uniform_real.h>
#include <arrow/util/bit_run_reader.h>
#include <arrow/util/bit_util.h>
#include <arrow/util/pcg_random.h>
#include <immintrin.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>
#include <xsimd/xsimd.hpp>

#include "macros.hpp"

namespace null_revisit {
/// Code vendored from Arrow
/// \brief Relocate values in buffer into positions of non-null values as
/// indicated by a validity bitmap.
///
/// \param[in] buffer the input buffer
/// \param[out] out the output buffer
/// \param[in] num_values total size of buffer including null slots
/// \param[in] null_count number of null slots
/// \param[in] valid_bits bitmap data indicating position of valid slots
/// \param[in] valid_bits_offset offset into valid_bits
/// \return The number of values expanded, including nulls.
template <typename T>
inline int SpacedScatter(T *buffer, T *out, int num_values, int null_count,
                         const uint8_t *valid_bits, int64_t valid_bits_offset) {
    // Point to end as we add the spacing from the back.
    int idx_decode = num_values - null_count;

    if (idx_decode == 0) {
        // All nulls, nothing more to do
        return num_values;
    }

    arrow::internal::ReverseSetBitRunReader reader(
        valid_bits, valid_bits_offset, num_values);
    while (true) {
        const auto run = reader.NextRun();
        if (run.length == 0) {
            break;
        }
        idx_decode -= static_cast<int32_t>(run.length);
        assert(idx_decode >= 0);
        std::memcpy(out + run.position, buffer + idx_decode,
                    run.length * sizeof(T));
    }

    // Otherwise caller gave an incorrect null_count
    assert(idx_decode == 0);
    return num_values;
}
inline std::shared_ptr<arrow::Buffer> GetSVBufferFromBM(const uint8_t *bits,
                                                        int length) {
    auto output = *arrow::AllocateResizableBuffer(length * sizeof(row_id_type));
    auto output_ptr = reinterpret_cast<row_id_type *>(output->mutable_data());
    arrow::internal::SetBitRunReader reader(bits, 0, length);

    int non_null_cnt = 0;
    while (true) {
        const auto run = reader.NextRun();
        if (run.length == 0) {
            break;
        }
        std::memcpy(output_ptr + non_null_cnt,
                    row_ids_prealloc.begin() + run.position,
                    run.length * sizeof(row_id_type));
        non_null_cnt += static_cast<int32_t>(run.length);
    }
    // for (int i = 0; i < length; ++i) {
    //   if (arrow::bit_util::GetBit(bits, i)) {
    //     output_ptr[non_null_i++] = i;
    //   }
    // }
    ARROW_UNUSED(output->Resize(non_null_cnt * sizeof(row_id_type)));
    return output;
}

inline SelVector GetSVFromBM(const uint8_t *bits, int length) {
    auto output = SelVector(length);
    auto output_ptr = reinterpret_cast<row_id_type *>(output.data());
    arrow::internal::SetBitRunReader reader(bits, 0, length);

    int non_null_cnt = 0;
    while (true) {
        const auto run = reader.NextRun();
        if (run.length == 0) {
            break;
        }
        std::memcpy(output_ptr + non_null_cnt,
                    row_ids_prealloc.begin() + run.position,
                    run.length * sizeof(row_id_type));
        non_null_cnt += static_cast<int32_t>(run.length);
    }
    output.resize(non_null_cnt);
    return output;
}

inline void GetSVFromBM(const uint8_t *bits, int length, SelVector &output,
                        SelVector &true_sel) {
    auto output_ptr = reinterpret_cast<row_id_type *>(output.data());
    arrow::internal::SetBitRunReader reader(bits, 0, length);

    int non_null_cnt = 0;
    while (true) {
        const auto run = reader.NextRun();
        if (run.length == 0) {
            break;
        }
        std::memcpy(output_ptr + non_null_cnt, &true_sel[run.position],
                    run.length * sizeof(row_id_type));
        non_null_cnt += static_cast<int32_t>(run.length);
    }
}

inline std::shared_ptr<arrow::Buffer> GetBMFromSV(const default_type *sv,
                                                  int length) {
    auto output = *arrow::AllocateEmptyBitmap(kVecSize);
    auto output_ptr = output->mutable_data();
    for (int i = 0; i < length; ++i) {
        arrow::bit_util::SetBit(output_ptr, sv[i]);
    }
    return output;
}

/// \param[in] buffer the input buffer. Aligned on AVX512
/// \param[in] indices the input row indices. Aligned on AVX512
/// \param[out] out the output buffer. Aligned on AVX512
inline int SpacedScatterAVX512(default_type *buffer, default_type *out,
                               int num_values, int null_count,
                               const row_id_type *indices) {
    // Point to end as we add the spacing from the back.
    int idx_decode = num_values - null_count;

    if (idx_decode == 0) {
        // All nulls, nothing more to do
        return num_values;
    }
    int i = 0;
    int num_avx = idx_decode / kBatchSize;
    for (; i < num_avx; ++i) {
        auto a = _mm512_load_si512(buffer + i * kBatchSize);
        auto index = _mm512_load_si512(indices + i * kBatchSize);
        _mm512_i32scatter_epi32(out, index, a, 4);
    }
    int num_scalar = idx_decode % kBatchSize;
    for (int j = 0; j < num_scalar; ++j) {
        out[indices[i * kBatchSize + j]] = buffer[i * kBatchSize + j];
    }
    return num_values;
}

inline int SpacedScatterAVX2(default_type *buffer, default_type *out,
                             int num_values, int null_count,
                             const row_id_type *indices) {
    // Point to end as we add the spacing from the back.
    int idx_decode = num_values - null_count;

    if (idx_decode == 0) {
        // All nulls, nothing more to do
        return num_values;
    }
    int i = 0;
    auto avx2_batch_size = kBatchSize / 2;
    int num_avx = idx_decode / avx2_batch_size;
    for (; i < num_avx; ++i) {
        auto a =
            _mm256_load_si256((const __m256i *)(buffer + i * avx2_batch_size));
        auto index =
            _mm256_load_si256((const __m256i *)(indices + i * avx2_batch_size));
        _mm256_i32scatter_epi32(out, index, a, 4);
    }
    int num_scalar = idx_decode % avx2_batch_size;
    for (int j = 0; j < num_scalar; ++j) {
        out[indices[i * avx2_batch_size + j]] = buffer[i * avx2_batch_size + j];
    }
    return num_values;
}

/// \param[in] buffer the input buffer. Aligned on AVX512
/// \param[in] indices the input row indices. Aligned on AVX512
/// \param[out] out the output buffer. Aligned on AVX512
inline int SpacedExpandAVX(default_type *buffer, default_type *out,
                           int num_values, int null_count,
                           const uint8_t *valid_bits) {
    assert(num_values % 512 == 0);
    // Point to end as we add the spacing from the back.
    int idx_decode = num_values - null_count;

    if (idx_decode == 0) {
        // All nulls, nothing more to do
        return num_values;
    }
    auto temp_buf = new (std::align_val_t{64}) uint8_t[64];
    int16_t *a_popcnt = (int16_t *)temp_buf;
    auto a_bm = (__mmask16 *)valid_bits;
    for (int i = 0; i < num_values; i += 512) {
        auto a_bm_512 = _mm512_load_si512(a_bm);
        _mm512_store_si512(a_popcnt, _mm512_popcnt_epi16(a_bm_512));
        for (int j = 0; j < 32; ++j) {
            __m512i a_expanded =
                _mm512_maskz_expand_epi32(a_bm[j], _mm512_loadu_si512(buffer));
            _mm512_storeu_si512(out, a_expanded);
            out += 16;
            buffer += a_popcnt[j];
        }
        a_bm += 32;
    }
    ::operator delete[](temp_buf, std::align_val_t(64));
    return num_values;
}

inline int SpacedExpandAVXInplace(default_type *buffer, int num_values,
                                  int null_count, const uint8_t *valid_bits) {
    // FIXME: can easily handle the modulo with regular scalar code from Arrow.
    assert(num_values % 512 == 0);
    // Point to end as we add the spacing from the back.
    int idx_decode = num_values - null_count;

    if (idx_decode == 0) {
        // All nulls, nothing more to do
        return num_values;
    }
    auto temp_buf = new (std::align_val_t{64}) uint8_t[64];
    int16_t *a_popcnt = (int16_t *)temp_buf;
    auto a_bm = (__mmask16 *)(valid_bits + num_values / 8 - 64);
    auto out = buffer + num_values - 16;
    buffer += idx_decode;
    for (int i = num_values; i > 0; i -= 512) {
        auto a_bm_512 = _mm512_load_si512(a_bm);
        _mm512_store_si512(a_popcnt, _mm512_popcnt_epi16(a_bm_512));
        for (int j = 31; j >= 0; --j) {
            buffer -= a_popcnt[j];
            __m512i a_expanded =
                _mm512_maskz_expand_epi32(a_bm[j], _mm512_loadu_si512(buffer));
            _mm512_storeu_si512(out, a_expanded);
            out -= 16;
        }
        a_bm -= 32;
    }
    ::operator delete[](temp_buf, std::align_val_t(64));
    return num_values;
}

#ifdef __AVX512F__
// https://github.com/WojciechMula/toys/blob/master/avx512/avx512f-alignr.cpp
inline __m512i _mm512_alignrvar_epi32(const __m512i hi, const __m512i lo,
                                      int shift) {
    switch (shift) {
    case 0:
        return lo;
        break;
    case 1:
        return _mm512_alignr_epi32(hi, lo, 1);
        break;
    case 2:
        return _mm512_alignr_epi32(hi, lo, 2);
        break;
    case 3:
        return _mm512_alignr_epi32(hi, lo, 3);
        break;
    case 4:
        return _mm512_alignr_epi32(hi, lo, 4);
        break;
    case 5:
        return _mm512_alignr_epi32(hi, lo, 5);
        break;
    case 6:
        return _mm512_alignr_epi32(hi, lo, 6);
        break;
    case 7:
        return _mm512_alignr_epi32(hi, lo, 7);
        break;
    case 8:
        return _mm512_alignr_epi32(hi, lo, 8);
        break;
    case 9:
        return _mm512_alignr_epi32(hi, lo, 9);
        break;
    case 10:
        return _mm512_alignr_epi32(hi, lo, 10);
        break;
    case 11:
        return _mm512_alignr_epi32(hi, lo, 11);
        break;
    case 12:
        return _mm512_alignr_epi32(hi, lo, 12);
        break;
    case 13:
        return _mm512_alignr_epi32(hi, lo, 13);
        break;
    case 14:
        return _mm512_alignr_epi32(hi, lo, 14);
        break;
    case 15:
        return _mm512_alignr_epi32(hi, lo, 15);
        break;
    case 16:
        return hi;
        break;
    default:
        throw std::runtime_error("Invalid shift");
    }
}
#endif

// https://github.com/Light-Dedup/Light-Dedup/blob/master/nova.h#L118
static inline void prefetcht0(const void *x) {
    asm volatile("prefetcht0 %0" : : "m"(*(const char *)x));
}

/// Code vendored from Arrow
static void GenerateBitmap(uint8_t *buffer, size_t n, int64_t *null_count,
                           double probability, int32_t seed) {
    int64_t count = 0;
    arrow_vendored::pcg32_fast rng(seed);
    ::arrow::random::bernoulli_distribution dist(1.0 - probability);

    for (size_t i = 0; i < n; i++) {
        if (dist(rng)) {
            arrow::bit_util::SetBit(buffer, i);
        } else {
            count++;
        }
    }

    if (null_count != nullptr)
        *null_count = count;
}

class LinearSelVecChecker {
  public:
    LinearSelVecChecker(SelVector *sel_vec) { sel_vec_ = sel_vec; }

    bool Check(row_id_type row_id) {
        assert(row_id >= 0);
#ifndef NDEBUG
        if (row_id < prev_idx_) {
            assert(false);
        }
        prev_idx_ = row_id;
#endif
        while (true) {
            if (cur_idx_ >= sel_vec_->size()) {
                return false;
            }
            if (row_id == (*sel_vec_)[cur_idx_]) {
                ++cur_idx_;
                return true;
            }
            if (row_id > (*sel_vec_)[cur_idx_]) {
                ++cur_idx_;
                continue;
            }
            return false;
        }
    }

  private:
    SelVector *sel_vec_;
    int cur_idx_ = 0;
    int prev_idx_ = -1;
};

static inline int GetNonNullPos(const uint8_t *bm,
                                const prefix_sum_type *prefix_sums, int pos) {
    int chunkIdx = pos / kJacobson_c;
    int chunkOffset = pos % kJacobson_c;

    return prefix_sums[chunkIdx] +
           jacobson_map[*((bit_string_type *)bm + chunkIdx)][chunkOffset];
}

/// Code in namespace bits and detail vendored from Velox
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

namespace detail {
extern int32_t byteSetBits[256][8];
}
inline const int32_t *byteSetBits(uint8_t byte) {
    return detail::byteSetBits[byte];
}

/// Code vendored from Velox
/// https://github.com/facebookincubator/velox/blob/02ca9b0b4f554868b533d2f6526a480ea1e7d035/velox/common/base/SimdUtil-inl.h#L179
/// Apache License
template <typename A, bool scalar = false>
int32_t indicesOfSetBits(const uint64_t *bits, int32_t begin, int32_t end,
                         int32_t *result, const A &) {
    if (end <= begin) {
        return 0;
    }
    int32_t row = 0;
    auto originalResult = result;
    int32_t endWord = bits::roundUp(end, 64) / 64;
    auto firstWord = begin / 64;
    for (auto wordIndex = firstWord; wordIndex < endWord; ++wordIndex) {
        uint64_t word = bits[wordIndex];
        if (!word) {
            row += 64;
            continue;
        }
        if (wordIndex == endWord - 1) {
            int32_t lastBits = end - (endWord - 1) * 64;
            if (lastBits < 64) {
                word &= bits::lowMask(lastBits);
                if (!word) {
                    break;
                }
            }
        }
        if constexpr (scalar) {
            do {
                *result++ = __builtin_ctzll(word) + row;
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
                        .store_unaligned(result);
                } else {
                    static_assert(Batch::size == 4);
                    (Batch::load_aligned(indices) + row)
                        .store_unaligned(result);
                    auto lo = byte & ((1 << 4) - 1);
                    int lo_pop = __builtin_popcount(lo);
                    (Batch::load_unaligned(indices + lo_pop) + row)
                        .store_unaligned(result + lo_pop);
                }
                result += __builtin_popcount(byte);
                row += 8;
            }
        }
    }
    return result - originalResult;
}

// The definition has been moved to spaced_expand_fused.cpp
template <typename A, bool is_simd, int miniblock_size = 1024>
extern int32_t
spacedExpandFusedMiniblocks(const uint32_t *__restrict dense_in,
                            const uint64_t *bits, int32_t bits_length,
                            uint32_t *__restrict spaced_out, const A &);

template <typename A, bool is_simd>
extern int32_t spacedExpandFused(const uint32_t *__restrict dense_in,
                                 const uint64_t *bits, int32_t bits_length,
                                 uint32_t *__restrict spaced_out, const A &);

} // namespace null_revisit