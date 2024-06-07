#include <xsimd/xsimd.hpp>

#include "bit_util.h"

namespace null_revisit {

template <typename A, bool is_simd, int miniblock_size>
int32_t spacedExpandFusedMiniblocks(const uint32_t* __restrict dense_in, const uint64_t* bits, int32_t bits_length,
                                    uint32_t* __restrict spaced_out, const A&) {
  if (bits_length <= 0) {
    return 0;
  }
  static_assert(miniblock_size >= 64 && miniblock_size % 64 == 0);
  int32_t row = 0;
  int32_t cur_dense_idx = 0;
  int32_t endWord = bits::roundUp(bits_length, 64) / 64;
  // Use blocks of size miniblock_size, in which we have miniblocks of size 64
  static int32_t index_buf[miniblock_size];
  constexpr int num_miniblocks = miniblock_size / 64;
  for (auto wordIndex = 0; wordIndex < endWord;) {
    auto index_buf_ptr = index_buf;
    // Expect this loop to be unrolled
    for (int miniblock_i = 0; miniblock_i < num_miniblocks; miniblock_i++, wordIndex++) {
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
      if constexpr (!is_simd) {
        do {
          spaced_out[__builtin_ctzll(word) + row] = dense_in[cur_dense_idx++];
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
            (Batch::load_aligned(indices) + row).store_unaligned(index_buf_ptr);
          } else {
            static_assert(Batch::size == 4);
            (Batch::load_aligned(indices) + row).store_unaligned(index_buf_ptr);
            auto lo = byte & ((1 << 4) - 1);
            int lo_pop = __builtin_popcount(lo);
            (Batch::load_unaligned(indices + lo_pop) + row).store_unaligned(index_buf_ptr + lo_pop);
          }
          index_buf_ptr += __builtin_popcount(byte);
          row += 8;
        }
      }
    }
    if constexpr (is_simd) {
      for (auto copy_ptr = index_buf; copy_ptr != index_buf_ptr; copy_ptr++) {
        spaced_out[*copy_ptr] = dense_in[cur_dense_idx++];
      }
    }
  }
  return cur_dense_idx;
}

template <typename A, bool is_simd>
int32_t spacedExpandFused(const uint32_t* __restrict dense_in, const uint64_t* bits, int32_t bits_length,
                          uint32_t* __restrict spaced_out, const A&) {
  if (bits_length <= 0) {
    return 0;
  }
  int32_t row = 0;
  int32_t index_buf[64];  // One batch of a word
  int32_t cur_dense_idx = 0;
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
    if constexpr (!is_simd) {
      do {
        spaced_out[__builtin_ctzll(word) + row] = dense_in[cur_dense_idx++];
        word = word & (word - 1);
      } while (word);
      row += 64;
    } else {
      auto index_buf_ptr = index_buf;
      // Expect this to be unrolled
      for (auto byteCnt = 0; byteCnt < 8; ++byteCnt) {
        uint8_t byte = word;
        word = word >> 8;
        using Batch = xsimd::batch<int32_t, A>;
        auto indices = byteSetBits(byte);
        if constexpr (Batch::size == 8) {
          (Batch::load_aligned(indices) + row).store_unaligned(index_buf_ptr);
        } else {
          static_assert(Batch::size == 4);
          (Batch::load_aligned(indices) + row).store_unaligned(index_buf_ptr);
          auto lo = byte & ((1 << 4) - 1);
          int lo_pop = __builtin_popcount(lo);
          (Batch::load_unaligned(indices + lo_pop) + row).store_unaligned(index_buf_ptr + lo_pop);
        }
        index_buf_ptr += __builtin_popcount(byte);
        row += 8;
      }
      if constexpr (std::is_same_v<A, xsimd::avx2>) {
        int32_t i = 0;
        using Batch = xsimd::batch<int32_t, A>;
        constexpr auto avx2_batch_size = Batch::size;
        int num_avx = (index_buf_ptr - index_buf) / avx2_batch_size;
        for (; i < num_avx; ++i) {
          auto a = _mm256_loadu_si256((const __m256i*)(dense_in + i * avx2_batch_size));
          auto index = _mm256_loadu_si256((const __m256i*)(index_buf + i * avx2_batch_size));
          _mm256_i32scatter_epi32(spaced_out, index, a, 4);
        }
        int num_scalar = (index_buf_ptr - index_buf) % avx2_batch_size;
        for (int j = 0; j < num_scalar; ++j) {
          spaced_out[index_buf[i * avx2_batch_size + j]] = dense_in[i * avx2_batch_size + j];
        }
        cur_dense_idx += (index_buf_ptr - index_buf);
      } else {
        for (auto copy_ptr = index_buf; copy_ptr != index_buf_ptr; copy_ptr++) {
          spaced_out[*copy_ptr] = dense_in[cur_dense_idx++];
        }
      }
    }
  }
  return cur_dense_idx;
}

#define INSTANTIATE_MINIBLOCKS(A, is_simd, val)                                                                       \
  template int32_t spacedExpandFusedMiniblocks<A, is_simd, val>(const uint32_t* __restrict, const uint64_t*, int32_t, \
                                                                uint32_t* __restrict, const A&);

// From 64 to 8M (except for the default 1024)
#define INSTANTIATE_ALL(A, is_simd)                                                                               \
  INSTANTIATE_MINIBLOCKS(A, is_simd, 64)                                                                          \
  INSTANTIATE_MINIBLOCKS(A, is_simd, 128)                                                                         \
  INSTANTIATE_MINIBLOCKS(A, is_simd, 256) INSTANTIATE_MINIBLOCKS(A, is_simd, 512)                                 \
      INSTANTIATE_MINIBLOCKS(A, is_simd, 2048) INSTANTIATE_MINIBLOCKS(A, is_simd, 4096)                           \
          INSTANTIATE_MINIBLOCKS(A, is_simd, 8192) INSTANTIATE_MINIBLOCKS(A, is_simd, 16384)                      \
              INSTANTIATE_MINIBLOCKS(A, is_simd, 32768) INSTANTIATE_MINIBLOCKS(A, is_simd, 65536)                 \
                  INSTANTIATE_MINIBLOCKS(A, is_simd, 131072) INSTANTIATE_MINIBLOCKS(A, is_simd, 262144)           \
                      INSTANTIATE_MINIBLOCKS(A, is_simd, 524288) INSTANTIATE_MINIBLOCKS(A, is_simd, 1048576)      \
                          INSTANTIATE_MINIBLOCKS(A, is_simd, 2097152) INSTANTIATE_MINIBLOCKS(A, is_simd, 4194304) \
                              INSTANTIATE_MINIBLOCKS(A, is_simd, 8388608)

};  // namespace null_rep