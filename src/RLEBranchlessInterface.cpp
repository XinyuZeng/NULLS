#include "RLEBranchlessInterface.hpp"
#ifdef __SSE4_2__
#include <xsimd/xsimd.hpp>
#endif
// #include <iostream>

namespace null_revisit {

extern const uint32_t prefix_sum_table[256][8];
size_t writeBitmap(const std::vector<uint32_t> &bitmap, uint8_t *dst,
                   size_t dst_size);

RLEBranchlessInterface::RLEBranchlessInterface(const std::string &codec_name)
    : CompressInterface(), word_size_(4), bitpacker_() {}

size_t RLEBranchlessInterface::compress(std::string_view src,
                                        arrow::ResizableBuffer *dst,
                                        const std::vector<bool> &nulls) {
    size_t n = src.size() / word_size_;
    std::vector<uint32_t> rle_values, rle_lengths;
    rle_values.reserve(n);
    rle_lengths.reserve(n);
    auto src_ptr = (const uint32_t *)src.data();
    uint32_t last = (*src_ptr++);
    rle_lengths.push_back(1);
    for (size_t i = 1; i < n; i++) {
        // Read 4 bytes as a uint32_t
        uint32_t val = *src_ptr++;
        if (val != last) {
            rle_values.push_back(last);
            rle_lengths.push_back(1);
            last = val;
        } else {
            rle_lengths.push_back(0);
        }
    }
    rle_values.push_back(last);
    // Compress rle_values and rle_lengths using Bitpacking
    // Write n into dst
    *(reinterpret_cast<uint32_t *>(dst->mutable_data())) = n;
    // Compress rle_lengths into dst
    const auto lengths_size = writeBitmap(
        rle_lengths, (uint8_t *)dst->mutable_data() + sizeof(uint32_t),
        dst->size() - sizeof(uint32_t));
    // Compress rle_values into dst
    const auto values_size = bitpacker_.compress_no_null(
        std::string_view((const char *)rle_values.data(),
                         rle_values.size() * sizeof(uint32_t)),
        (char *)dst->mutable_data() + sizeof(uint32_t) + lengths_size,
        dst->size() - sizeof(uint32_t) - lengths_size);
    const auto dst_buf_size = sizeof(uint32_t) + values_size + lengths_size;
    dst->Resize(dst_buf_size);
    return dst_buf_size;
}

size_t
RLEBranchlessInterface::compress_with_null(std::string_view src,
                                           arrow::ResizableBuffer *dst,
                                           const std::vector<bool> &nulls) {
    size_t n = src.size() / word_size_;
    std::vector<uint32_t> rle_values, rle_lengths;
    rle_values.reserve(n);
    rle_lengths.reserve(n);
    auto src_ptr = (const uint32_t *)src.data();
    uint32_t last = (*src_ptr++);
    rle_lengths.push_back(1);
    for (size_t i = 1; i < n; i++) {
        // Read 4 bytes as a uint32_t
        uint32_t val = *src_ptr++;
        if (!nulls[i] && val != last) {
            rle_values.push_back(last);
            rle_lengths.push_back(1);
            last = val;
        } else {
            rle_lengths.push_back(0);
        }
    }
    rle_values.push_back(last);
    // Compress rle_values and rle_lengths using Bitpacking
    // Write n into dst
    *(reinterpret_cast<uint32_t *>(dst->mutable_data())) = n;
    // Compress rle_lengths into dst
    const auto lengths_size = writeBitmap(
        rle_lengths, (uint8_t *)dst->mutable_data() + sizeof(uint32_t),
        dst->size() - sizeof(uint32_t));
    // Compress rle_values into dst
    const auto values_size = bitpacker_.compress_no_null(
        std::string_view((const char *)rle_values.data(),
                         rle_values.size() * sizeof(uint32_t)),
        (char *)dst->mutable_data() + sizeof(uint32_t) + lengths_size,
        dst->size() - sizeof(uint32_t) - lengths_size);
    const auto dst_buf_size = sizeof(uint32_t) + values_size + lengths_size;
    dst->Resize(dst_buf_size);
    return dst_buf_size;
}

size_t RLEBranchlessInterface::decompress(std::string_view src, char *const dst,
                                          size_t dst_size) {
    char *rle_values_buf_start = dst + (dst_size / 2);
    const auto buf_size = dst_size / 4;
    auto src_ptr = reinterpret_cast<const uint32_t *>(src.data());
    const auto n = *src_ptr++;
    const size_t lengths_size = (n + 31) / 32 * 4; // Align to 32 bits
    const size_t values_size = src.size() - sizeof(uint32_t) - lengths_size;
    // Decompress using Bitpacking
    const auto values_buf_size = bitpacker_.decompress(
        std::string_view((const char *)src_ptr + lengths_size, values_size),
        rle_values_buf_start, buf_size);
    // Decompress rle_lengths
    auto cnt_ptr =
        reinterpret_cast<const uint8_t *>(src.data() + sizeof(uint32_t));
    const size_t block_n = n / 8 * 8;
    uint32_t last_index = 0;
    auto dst_ptr = std::assume_aligned<16>((uint32_t *)dst);
    auto val_ptr = reinterpret_cast<const uint32_t *>(rle_values_buf_start);
    val_ptr--; // Prepare for the addition, as indices start from 1
    // dst_ptr[i] = val_ptr[indices_ptr[i]], for i in [0, n)
#ifndef __SSE4_2__
    // Scalar version
    for (size_t i = 0; i < block_n; i += 8) {
        uint8_t block_lengths = *cnt_ptr++;
        for (size_t j = 0; j < 8; j++) {
            dst_ptr[i + j] =
                val_ptr[last_index + prefix_sum_table[block_lengths][j]];
        }
        last_index += __builtin_popcount(block_lengths);
    }
#else
    // Vectorized version
    using Batch = xsimd::batch<uint32_t, xsimd::sse4_2>;
    static_assert(Batch::size == 4);
    for (size_t i = 0; i < block_n; i += 8) {
        uint8_t block_lengths = *cnt_ptr++;
        const auto lo_table = prefix_sum_table[block_lengths];
        uint32_t lo_indices[8], hi_indices[8];
        // const auto lo_indices = (Batch::load_aligned(lo_table) + last_index);
        // const auto hi_indices = (Batch::load_aligned(lo_table + 4) +
        // last_index); (Batch::gather(val_ptr,
        // lo_indices)).store_aligned(dst_ptr + i); (Batch::gather(val_ptr,
        // hi_indices)).store_aligned(dst_ptr + i + 4);
        (Batch::load_aligned(lo_table) + last_index).store_aligned(lo_indices);
        (Batch::load_aligned(lo_table + 4) + last_index)
            .store_aligned(hi_indices);
        for (size_t j = 0; j < 4; j++) {
            dst_ptr[i + j] = val_ptr[lo_indices[j]];
            dst_ptr[i + j + 4] = val_ptr[hi_indices[j]];
        }
        last_index += __builtin_popcount(block_lengths);
    }
#endif
    uint8_t last_block_lengths = *cnt_ptr;
    for (size_t i = block_n; i < n; i++) {
        last_index += (last_block_lengths & 1);
        dst_ptr[i] = val_ptr[last_index];
        last_block_lengths >>= 1;
    }
    return n * sizeof(uint32_t);
}

size_t writeBitmap(const std::vector<uint32_t> &bitmap, uint8_t *dst,
                   size_t dst_size) {
    const auto bitmap_size = (bitmap.size() + 31) / 32 * 4; // Align to 32 bits
    if (dst_size < bitmap_size) {
        return 0;
    }
    for (size_t i = 0; i < bitmap.size(); i++) {
        dst[i / 8] |= (bitmap[i] << (i % 8));
    }
    return bitmap_size;
}

const uint32_t prefix_sum_table[256][8] = {
    {0, 0, 0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1, 1, 1, 1},
    {0, 1, 1, 1, 1, 1, 1, 1}, {1, 2, 2, 2, 2, 2, 2, 2},
    {0, 0, 1, 1, 1, 1, 1, 1}, {1, 1, 2, 2, 2, 2, 2, 2},
    {0, 1, 2, 2, 2, 2, 2, 2}, {1, 2, 3, 3, 3, 3, 3, 3},
    {0, 0, 0, 1, 1, 1, 1, 1}, {1, 1, 1, 2, 2, 2, 2, 2},
    {0, 1, 1, 2, 2, 2, 2, 2}, {1, 2, 2, 3, 3, 3, 3, 3},
    {0, 0, 1, 2, 2, 2, 2, 2}, {1, 1, 2, 3, 3, 3, 3, 3},
    {0, 1, 2, 3, 3, 3, 3, 3}, {1, 2, 3, 4, 4, 4, 4, 4},
    {0, 0, 0, 0, 1, 1, 1, 1}, {1, 1, 1, 1, 2, 2, 2, 2},
    {0, 1, 1, 1, 2, 2, 2, 2}, {1, 2, 2, 2, 3, 3, 3, 3},
    {0, 0, 1, 1, 2, 2, 2, 2}, {1, 1, 2, 2, 3, 3, 3, 3},
    {0, 1, 2, 2, 3, 3, 3, 3}, {1, 2, 3, 3, 4, 4, 4, 4},
    {0, 0, 0, 1, 2, 2, 2, 2}, {1, 1, 1, 2, 3, 3, 3, 3},
    {0, 1, 1, 2, 3, 3, 3, 3}, {1, 2, 2, 3, 4, 4, 4, 4},
    {0, 0, 1, 2, 3, 3, 3, 3}, {1, 1, 2, 3, 4, 4, 4, 4},
    {0, 1, 2, 3, 4, 4, 4, 4}, {1, 2, 3, 4, 5, 5, 5, 5},
    {0, 0, 0, 0, 0, 1, 1, 1}, {1, 1, 1, 1, 1, 2, 2, 2},
    {0, 1, 1, 1, 1, 2, 2, 2}, {1, 2, 2, 2, 2, 3, 3, 3},
    {0, 0, 1, 1, 1, 2, 2, 2}, {1, 1, 2, 2, 2, 3, 3, 3},
    {0, 1, 2, 2, 2, 3, 3, 3}, {1, 2, 3, 3, 3, 4, 4, 4},
    {0, 0, 0, 1, 1, 2, 2, 2}, {1, 1, 1, 2, 2, 3, 3, 3},
    {0, 1, 1, 2, 2, 3, 3, 3}, {1, 2, 2, 3, 3, 4, 4, 4},
    {0, 0, 1, 2, 2, 3, 3, 3}, {1, 1, 2, 3, 3, 4, 4, 4},
    {0, 1, 2, 3, 3, 4, 4, 4}, {1, 2, 3, 4, 4, 5, 5, 5},
    {0, 0, 0, 0, 1, 2, 2, 2}, {1, 1, 1, 1, 2, 3, 3, 3},
    {0, 1, 1, 1, 2, 3, 3, 3}, {1, 2, 2, 2, 3, 4, 4, 4},
    {0, 0, 1, 1, 2, 3, 3, 3}, {1, 1, 2, 2, 3, 4, 4, 4},
    {0, 1, 2, 2, 3, 4, 4, 4}, {1, 2, 3, 3, 4, 5, 5, 5},
    {0, 0, 0, 1, 2, 3, 3, 3}, {1, 1, 1, 2, 3, 4, 4, 4},
    {0, 1, 1, 2, 3, 4, 4, 4}, {1, 2, 2, 3, 4, 5, 5, 5},
    {0, 0, 1, 2, 3, 4, 4, 4}, {1, 1, 2, 3, 4, 5, 5, 5},
    {0, 1, 2, 3, 4, 5, 5, 5}, {1, 2, 3, 4, 5, 6, 6, 6},
    {0, 0, 0, 0, 0, 0, 1, 1}, {1, 1, 1, 1, 1, 1, 2, 2},
    {0, 1, 1, 1, 1, 1, 2, 2}, {1, 2, 2, 2, 2, 2, 3, 3},
    {0, 0, 1, 1, 1, 1, 2, 2}, {1, 1, 2, 2, 2, 2, 3, 3},
    {0, 1, 2, 2, 2, 2, 3, 3}, {1, 2, 3, 3, 3, 3, 4, 4},
    {0, 0, 0, 1, 1, 1, 2, 2}, {1, 1, 1, 2, 2, 2, 3, 3},
    {0, 1, 1, 2, 2, 2, 3, 3}, {1, 2, 2, 3, 3, 3, 4, 4},
    {0, 0, 1, 2, 2, 2, 3, 3}, {1, 1, 2, 3, 3, 3, 4, 4},
    {0, 1, 2, 3, 3, 3, 4, 4}, {1, 2, 3, 4, 4, 4, 5, 5},
    {0, 0, 0, 0, 1, 1, 2, 2}, {1, 1, 1, 1, 2, 2, 3, 3},
    {0, 1, 1, 1, 2, 2, 3, 3}, {1, 2, 2, 2, 3, 3, 4, 4},
    {0, 0, 1, 1, 2, 2, 3, 3}, {1, 1, 2, 2, 3, 3, 4, 4},
    {0, 1, 2, 2, 3, 3, 4, 4}, {1, 2, 3, 3, 4, 4, 5, 5},
    {0, 0, 0, 1, 2, 2, 3, 3}, {1, 1, 1, 2, 3, 3, 4, 4},
    {0, 1, 1, 2, 3, 3, 4, 4}, {1, 2, 2, 3, 4, 4, 5, 5},
    {0, 0, 1, 2, 3, 3, 4, 4}, {1, 1, 2, 3, 4, 4, 5, 5},
    {0, 1, 2, 3, 4, 4, 5, 5}, {1, 2, 3, 4, 5, 5, 6, 6},
    {0, 0, 0, 0, 0, 1, 2, 2}, {1, 1, 1, 1, 1, 2, 3, 3},
    {0, 1, 1, 1, 1, 2, 3, 3}, {1, 2, 2, 2, 2, 3, 4, 4},
    {0, 0, 1, 1, 1, 2, 3, 3}, {1, 1, 2, 2, 2, 3, 4, 4},
    {0, 1, 2, 2, 2, 3, 4, 4}, {1, 2, 3, 3, 3, 4, 5, 5},
    {0, 0, 0, 1, 1, 2, 3, 3}, {1, 1, 1, 2, 2, 3, 4, 4},
    {0, 1, 1, 2, 2, 3, 4, 4}, {1, 2, 2, 3, 3, 4, 5, 5},
    {0, 0, 1, 2, 2, 3, 4, 4}, {1, 1, 2, 3, 3, 4, 5, 5},
    {0, 1, 2, 3, 3, 4, 5, 5}, {1, 2, 3, 4, 4, 5, 6, 6},
    {0, 0, 0, 0, 1, 2, 3, 3}, {1, 1, 1, 1, 2, 3, 4, 4},
    {0, 1, 1, 1, 2, 3, 4, 4}, {1, 2, 2, 2, 3, 4, 5, 5},
    {0, 0, 1, 1, 2, 3, 4, 4}, {1, 1, 2, 2, 3, 4, 5, 5},
    {0, 1, 2, 2, 3, 4, 5, 5}, {1, 2, 3, 3, 4, 5, 6, 6},
    {0, 0, 0, 1, 2, 3, 4, 4}, {1, 1, 1, 2, 3, 4, 5, 5},
    {0, 1, 1, 2, 3, 4, 5, 5}, {1, 2, 2, 3, 4, 5, 6, 6},
    {0, 0, 1, 2, 3, 4, 5, 5}, {1, 1, 2, 3, 4, 5, 6, 6},
    {0, 1, 2, 3, 4, 5, 6, 6}, {1, 2, 3, 4, 5, 6, 7, 7},
    {0, 0, 0, 0, 0, 0, 0, 1}, {1, 1, 1, 1, 1, 1, 1, 2},
    {0, 1, 1, 1, 1, 1, 1, 2}, {1, 2, 2, 2, 2, 2, 2, 3},
    {0, 0, 1, 1, 1, 1, 1, 2}, {1, 1, 2, 2, 2, 2, 2, 3},
    {0, 1, 2, 2, 2, 2, 2, 3}, {1, 2, 3, 3, 3, 3, 3, 4},
    {0, 0, 0, 1, 1, 1, 1, 2}, {1, 1, 1, 2, 2, 2, 2, 3},
    {0, 1, 1, 2, 2, 2, 2, 3}, {1, 2, 2, 3, 3, 3, 3, 4},
    {0, 0, 1, 2, 2, 2, 2, 3}, {1, 1, 2, 3, 3, 3, 3, 4},
    {0, 1, 2, 3, 3, 3, 3, 4}, {1, 2, 3, 4, 4, 4, 4, 5},
    {0, 0, 0, 0, 1, 1, 1, 2}, {1, 1, 1, 1, 2, 2, 2, 3},
    {0, 1, 1, 1, 2, 2, 2, 3}, {1, 2, 2, 2, 3, 3, 3, 4},
    {0, 0, 1, 1, 2, 2, 2, 3}, {1, 1, 2, 2, 3, 3, 3, 4},
    {0, 1, 2, 2, 3, 3, 3, 4}, {1, 2, 3, 3, 4, 4, 4, 5},
    {0, 0, 0, 1, 2, 2, 2, 3}, {1, 1, 1, 2, 3, 3, 3, 4},
    {0, 1, 1, 2, 3, 3, 3, 4}, {1, 2, 2, 3, 4, 4, 4, 5},
    {0, 0, 1, 2, 3, 3, 3, 4}, {1, 1, 2, 3, 4, 4, 4, 5},
    {0, 1, 2, 3, 4, 4, 4, 5}, {1, 2, 3, 4, 5, 5, 5, 6},
    {0, 0, 0, 0, 0, 1, 1, 2}, {1, 1, 1, 1, 1, 2, 2, 3},
    {0, 1, 1, 1, 1, 2, 2, 3}, {1, 2, 2, 2, 2, 3, 3, 4},
    {0, 0, 1, 1, 1, 2, 2, 3}, {1, 1, 2, 2, 2, 3, 3, 4},
    {0, 1, 2, 2, 2, 3, 3, 4}, {1, 2, 3, 3, 3, 4, 4, 5},
    {0, 0, 0, 1, 1, 2, 2, 3}, {1, 1, 1, 2, 2, 3, 3, 4},
    {0, 1, 1, 2, 2, 3, 3, 4}, {1, 2, 2, 3, 3, 4, 4, 5},
    {0, 0, 1, 2, 2, 3, 3, 4}, {1, 1, 2, 3, 3, 4, 4, 5},
    {0, 1, 2, 3, 3, 4, 4, 5}, {1, 2, 3, 4, 4, 5, 5, 6},
    {0, 0, 0, 0, 1, 2, 2, 3}, {1, 1, 1, 1, 2, 3, 3, 4},
    {0, 1, 1, 1, 2, 3, 3, 4}, {1, 2, 2, 2, 3, 4, 4, 5},
    {0, 0, 1, 1, 2, 3, 3, 4}, {1, 1, 2, 2, 3, 4, 4, 5},
    {0, 1, 2, 2, 3, 4, 4, 5}, {1, 2, 3, 3, 4, 5, 5, 6},
    {0, 0, 0, 1, 2, 3, 3, 4}, {1, 1, 1, 2, 3, 4, 4, 5},
    {0, 1, 1, 2, 3, 4, 4, 5}, {1, 2, 2, 3, 4, 5, 5, 6},
    {0, 0, 1, 2, 3, 4, 4, 5}, {1, 1, 2, 3, 4, 5, 5, 6},
    {0, 1, 2, 3, 4, 5, 5, 6}, {1, 2, 3, 4, 5, 6, 6, 7},
    {0, 0, 0, 0, 0, 0, 1, 2}, {1, 1, 1, 1, 1, 1, 2, 3},
    {0, 1, 1, 1, 1, 1, 2, 3}, {1, 2, 2, 2, 2, 2, 3, 4},
    {0, 0, 1, 1, 1, 1, 2, 3}, {1, 1, 2, 2, 2, 2, 3, 4},
    {0, 1, 2, 2, 2, 2, 3, 4}, {1, 2, 3, 3, 3, 3, 4, 5},
    {0, 0, 0, 1, 1, 1, 2, 3}, {1, 1, 1, 2, 2, 2, 3, 4},
    {0, 1, 1, 2, 2, 2, 3, 4}, {1, 2, 2, 3, 3, 3, 4, 5},
    {0, 0, 1, 2, 2, 2, 3, 4}, {1, 1, 2, 3, 3, 3, 4, 5},
    {0, 1, 2, 3, 3, 3, 4, 5}, {1, 2, 3, 4, 4, 4, 5, 6},
    {0, 0, 0, 0, 1, 1, 2, 3}, {1, 1, 1, 1, 2, 2, 3, 4},
    {0, 1, 1, 1, 2, 2, 3, 4}, {1, 2, 2, 2, 3, 3, 4, 5},
    {0, 0, 1, 1, 2, 2, 3, 4}, {1, 1, 2, 2, 3, 3, 4, 5},
    {0, 1, 2, 2, 3, 3, 4, 5}, {1, 2, 3, 3, 4, 4, 5, 6},
    {0, 0, 0, 1, 2, 2, 3, 4}, {1, 1, 1, 2, 3, 3, 4, 5},
    {0, 1, 1, 2, 3, 3, 4, 5}, {1, 2, 2, 3, 4, 4, 5, 6},
    {0, 0, 1, 2, 3, 3, 4, 5}, {1, 1, 2, 3, 4, 4, 5, 6},
    {0, 1, 2, 3, 4, 4, 5, 6}, {1, 2, 3, 4, 5, 5, 6, 7},
    {0, 0, 0, 0, 0, 1, 2, 3}, {1, 1, 1, 1, 1, 2, 3, 4},
    {0, 1, 1, 1, 1, 2, 3, 4}, {1, 2, 2, 2, 2, 3, 4, 5},
    {0, 0, 1, 1, 1, 2, 3, 4}, {1, 1, 2, 2, 2, 3, 4, 5},
    {0, 1, 2, 2, 2, 3, 4, 5}, {1, 2, 3, 3, 3, 4, 5, 6},
    {0, 0, 0, 1, 1, 2, 3, 4}, {1, 1, 1, 2, 2, 3, 4, 5},
    {0, 1, 1, 2, 2, 3, 4, 5}, {1, 2, 2, 3, 3, 4, 5, 6},
    {0, 0, 1, 2, 2, 3, 4, 5}, {1, 1, 2, 3, 3, 4, 5, 6},
    {0, 1, 2, 3, 3, 4, 5, 6}, {1, 2, 3, 4, 4, 5, 6, 7},
    {0, 0, 0, 0, 1, 2, 3, 4}, {1, 1, 1, 1, 2, 3, 4, 5},
    {0, 1, 1, 1, 2, 3, 4, 5}, {1, 2, 2, 2, 3, 4, 5, 6},
    {0, 0, 1, 1, 2, 3, 4, 5}, {1, 1, 2, 2, 3, 4, 5, 6},
    {0, 1, 2, 2, 3, 4, 5, 6}, {1, 2, 3, 3, 4, 5, 6, 7},
    {0, 0, 0, 1, 2, 3, 4, 5}, {1, 1, 1, 2, 3, 4, 5, 6},
    {0, 1, 1, 2, 3, 4, 5, 6}, {1, 2, 2, 3, 4, 5, 6, 7},
    {0, 0, 1, 2, 3, 4, 5, 6}, {1, 1, 2, 3, 4, 5, 6, 7},
    {0, 1, 2, 3, 4, 5, 6, 7}, {1, 2, 3, 4, 5, 6, 7, 8}};

} // namespace null_revisit
