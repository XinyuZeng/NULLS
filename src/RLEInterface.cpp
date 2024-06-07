#include "RLEInterface.hpp"
#ifdef __SSE4_2__
#include <xsimd/xsimd.hpp>
#endif

namespace null_revisit {
RLEInterface::RLEInterface(const std::string &codec_name)
    : CompressInterface(), word_size_(4), bitpacker_() {}

size_t RLEInterface::compress(std::string_view src, arrow::ResizableBuffer *dst,
                              const std::vector<bool> &nulls) {
    size_t n = src.size() / word_size_;
    std::vector<uint32_t> rle_values, rle_lengths;
    rle_values.reserve(n);
    rle_lengths.reserve(n);
    auto src_ptr = (const uint32_t *)src.data();
    uint32_t last = (*src_ptr++);
    uint32_t last_cnt = 1;
    for (size_t i = 1; i < n; i++) {
        // Read 4 bytes as a uint32_t
        uint32_t val = *src_ptr++;
        if (val != last) {
            rle_values.push_back(last);
            rle_lengths.push_back(last_cnt);
            last = val, last_cnt = 0;
        }
        last_cnt++;
    }
    rle_values.push_back(last);
    rle_lengths.push_back(last_cnt);
    // Compress rle_values and rle_lengths using Bitpacking
    const auto values_size = bitpacker_.compress_no_null(
        std::string_view((const char *)rle_values.data(),
                         rle_values.size() * sizeof(uint32_t)),
        (char *)dst->mutable_data() + sizeof(uint32_t),
        dst->size() - sizeof(uint32_t));
    // Write values_size into dst
    *(reinterpret_cast<uint32_t *>(dst->mutable_data())) = values_size;
    // Compress rle_lengths into dst
    const auto lengths_size = bitpacker_.compress_no_null(
        std::string_view((const char *)rle_lengths.data(),
                         rle_lengths.size() * sizeof(uint32_t)),
        (char *)dst->mutable_data() + sizeof(uint32_t) + values_size,
        dst->size() - values_size - sizeof(uint32_t));
    const auto dst_buf_size = sizeof(uint32_t) + values_size + lengths_size;
    dst->Resize(dst_buf_size);
    return dst_buf_size;
}

size_t RLEInterface::compress_with_null(std::string_view src,
                                        arrow::ResizableBuffer *dst,
                                        const std::vector<bool> &nulls) {
    size_t n = src.size() / word_size_;
    std::vector<uint32_t> rle_values, rle_lengths;
    rle_values.reserve(n);
    rle_lengths.reserve(n);
    auto src_ptr = (const uint32_t *)src.data();
    uint32_t last = (*src_ptr++);
    uint32_t last_cnt = 1;
    for (size_t i = 1; i < n; i++) {
        // Read 4 bytes as a uint32_t
        uint32_t val = *src_ptr++;
        if (!nulls[i] && val != last) {
            rle_values.push_back(last);
            rle_lengths.push_back(last_cnt);
            last = val, last_cnt = 0;
        }
        last_cnt++;
    }
    rle_values.push_back(last);
    rle_lengths.push_back(last_cnt);
    // Compress rle_values and rle_lengths using Bitpacking
    const auto values_size = bitpacker_.compress_no_null(
        std::string_view((const char *)rle_values.data(),
                         rle_values.size() * sizeof(uint32_t)),
        (char *)dst->mutable_data() + sizeof(uint32_t),
        dst->size() - sizeof(uint32_t));
    // Write values_size into dst
    *(reinterpret_cast<uint32_t *>(dst->mutable_data())) = values_size;
    // Compress rle_lengths into dst
    const auto lengths_size = bitpacker_.compress_no_null(
        std::string_view((const char *)rle_lengths.data(),
                         rle_lengths.size() * sizeof(uint32_t)),
        (char *)dst->mutable_data() + sizeof(uint32_t) + values_size,
        dst->size() - values_size - sizeof(uint32_t));
    const auto dst_buf_size = sizeof(uint32_t) + values_size + lengths_size;
    dst->Resize(dst_buf_size);
    return dst_buf_size;
}

size_t RLEInterface::decompress(std::string_view src, char *const dst,
                                size_t dst_size) {
    char *rle_values_buf_start = dst + (dst_size / 2);
    char *rle_lengths_buf_start = rle_values_buf_start + (dst_size / 4);
    const auto buf_size = dst_size / 4;
    auto src_ptr = reinterpret_cast<const uint32_t *>(src.data());
    const auto values_size = *src_ptr++;
    // Decompress using Bitpacking
    const auto values_buf_size = bitpacker_.decompress(
        std::string_view((const char *)src_ptr, values_size),
        rle_values_buf_start, buf_size);
    src_ptr += values_size / sizeof(uint32_t);
    // Decompress using Bitpacking
    const auto lengths_buf_size = bitpacker_.decompress(
        std::string_view((const char *)src_ptr,
                         src.size() - values_size - sizeof(uint32_t)),
        rle_lengths_buf_start, buf_size);
    // Decode rle_values and rle_lengths
    auto dst_ptr = (uint32_t *)dst;
#ifdef __SSE4_2__
    const auto run_count = values_buf_size / sizeof(uint32_t);
    for (size_t i = 0; i < run_count; i++) {
        const auto target = dst_ptr + reinterpret_cast<const uint32_t *>(
                                          rle_lengths_buf_start)[i];
        const auto val =
            reinterpret_cast<const uint32_t *>(rle_values_buf_start)[i];
        const auto vec_val = xsimd::batch<uint32_t, xsimd::sse4_2>(val);
        for (; dst_ptr < target; dst_ptr += decltype(vec_val)::size) {
            vec_val.store_unaligned(dst_ptr);
        }
        dst_ptr = target;
    }
#else
    for (size_t i = 0; i < values_buf_size / sizeof(uint32_t); ++i) {
        uint32_t val =
            reinterpret_cast<const uint32_t *>(rle_values_buf_start)[i];
        uint32_t cnt =
            reinterpret_cast<const uint32_t *>(rle_lengths_buf_start)[i];
        std::fill(dst_ptr, dst_ptr + cnt, val);
        dst_ptr += cnt;
    }
#endif
    return reinterpret_cast<char *>(dst_ptr) - dst;
}

} // namespace null_revisit
