#include "DeltaInterface.hpp"
#include "VarInts.hpp"
// #include <iostream>
#include <memory>
#ifdef __SSE4_2__
#include <xsimd/xsimd.hpp>
#endif

namespace null_revisit {
DeltaInterface::DeltaInterface(const std::string &codec_name)
    : CompressInterface(), word_size_(4) {}

size_t DeltaInterface::compress(std::string_view src,
                                arrow::ResizableBuffer *dst,
                                const std::vector<bool> &nulls) {
    size_t n = src.size() / word_size_;
    std::vector<uint32_t> zigzag_values;
    zigzag_values.reserve(n);
    auto src_ptr = (const uint32_t *)src.data();
    uint32_t last = 0;
    for (size_t i = 0; i < n; ++i) {
        // Read 4 bytes as a uint32_t
        uint32_t val = *src_ptr++;
        // Write the delta
        zigzag_values.push_back(
            encodeZigZagValInt32(int32_t(val) - int32_t(last)));
        last = val;
    }
    // Compress zigzag_values using Bitpacking
    const auto dst_buf_size = bitpacker_.compress_no_null(
        std::string_view((const char *)zigzag_values.data(),
                         zigzag_values.size() * sizeof(uint32_t)),
        (char *)dst->mutable_data(), dst->size());
    dst->Resize(dst_buf_size);
    return dst_buf_size;
}

size_t DeltaInterface::compress_with_null(std::string_view src,
                                          arrow::ResizableBuffer *dst,
                                          const std::vector<bool> &nulls) {
    if (compress_option_.heuristic == NullHeuristic::ZERO ||
        compress_option_.heuristic == NullHeuristic::SMART)
        return compress(src, dst, nulls);
    size_t n = src.size() / word_size_;
    std::vector<uint32_t> zigzag_values;
    zigzag_values.reserve(n);
    auto src_ptr = (const uint32_t *)src.data();
    uint32_t last = 0;
    for (size_t i = 0; i < n; ++i) {
        // Read 4 bytes as a uint32_t
        uint32_t val = *src_ptr++;
        if (nulls[i])
            val = last;
        zigzag_values.push_back(
            encodeZigZagValInt32(int32_t(val) - int32_t(last)));
        last = val;
    }
    // Compress zigzag_values using Bitpacking
    const auto dst_buf_size = bitpacker_.compress_no_null(
        std::string_view((const char *)zigzag_values.data(),
                         zigzag_values.size() * sizeof(uint32_t)),
        (char *)dst->mutable_data(), dst->size());
    dst->Resize(dst_buf_size);
    return dst_buf_size;
}

size_t
DeltaInterface::compress_with_null_linear(std::string_view src,
                                          arrow::ResizableBuffer *dst,
                                          const std::vector<bool> &nulls) {
    size_t n = src.size() / word_size_;
    std::vector<uint32_t> zigzag_values;
    zigzag_values.reserve(n);
    auto src_ptr = (const uint32_t *)src.data();
    auto src_start = src_ptr;
    // last: the last non-null value with index < i
    // next: first non-null value with index >= i
    // i-1 null? no update
    // i-1 non-null? i-1 -> last, find next
    // last_(next_)non_null values are from input;
    // last_encoded may contains interpolated null values
    uint32_t last_non_null = 0, next_non_null = 0, last_encoded = 0;
    int last_idx = -1, next_idx = 0;
    // Find next for i-1 == -1
    while (next_idx < n && nulls[next_idx])
        next_idx++;
    next_non_null = (next_idx < n) ? (src_start[next_idx]) : (0);

    // Ready for i == 0
    for (size_t i = 0; i < n; ++i) {
        // Read 4 bytes as a uint32_t
        if (nulls[i]) {
            const int32_t delta =
                int32_t(next_non_null) - int32_t(last_non_null);
            const int32_t delta_idx = next_idx - last_idx;
            const int32_t cur_delta_idx = i - last_idx;
            const int32_t cur_delta = delta * cur_delta_idx / delta_idx;
            zigzag_values.push_back(encodeZigZagValInt32(
                int32_t(last_non_null) + cur_delta - int32_t(last_encoded)));
            last_encoded = last_non_null + cur_delta;
            src_ptr++;
        } else {
            uint32_t val = *src_ptr++;
            zigzag_values.push_back(
                encodeZigZagValInt32(int32_t(val) - int32_t(last_encoded)));
            last_encoded = last_non_null = val, last_idx = i, next_idx = i + 1;
            while (next_idx < n && nulls[next_idx])
                next_idx++;
            next_non_null = (next_idx < n) ? (src_start[next_idx]) : (0);
        }
        // Ready for i+1
    }
    // Compress zigzag_values using Bitpacking
    const auto dst_buf_size = bitpacker_.compress_no_null(
        std::string_view((const char *)zigzag_values.data(),
                         zigzag_values.size() * sizeof(uint32_t)),
        (char *)dst->mutable_data(), dst->size());
    dst->Resize(dst_buf_size);
    return dst_buf_size;
}

size_t DeltaInterface::decompress(std::string_view src, char *dst,
                                  size_t dst_size) {
    // Decompress using Bitpacking
    char *zigzag_values_buf = dst + (dst_size / 2);
    const auto zigzag_buf_size = dst_size / 2;
    const auto zigzag_values_size =
        bitpacker_.decompress(src, zigzag_values_buf, zigzag_buf_size);
    // Decode zigzag values
    auto dst_ptr = std::assume_aligned<16>(reinterpret_cast<uint32_t *>(dst));
    auto zigzag_values = std::assume_aligned<16>(
        reinterpret_cast<const uint32_t *>(zigzag_values_buf));
    uint32_t last = 0;
    const auto n = zigzag_values_size / sizeof(uint32_t);
#ifndef __SSE4_2__
    // Scalar version
    for (size_t i = 0; i < n; ++i) {
        auto val = decodeZigZagValInt32(zigzag_values[i]);
        *dst_ptr++ = val + last;
        last = val + last;
    }
#else
    // From https://github.com/lemire/FastDifferentialCoding/
    using Batch = xsimd::batch<uint32_t, xsimd::sse4_2>;
    static_assert(Batch::size == 4);
    auto prev = Batch(0);
    const size_t block_n = n / 4 * 4;
    for (int i = 0; i < block_n; i += 4) {
        auto val = Batch::load_unaligned(zigzag_values + i);
        // Decode zigzag val: x -> (x >> 1) ^ (-(x & 1))
        val = (val >> 1) ^ (-(val & 1));
        const auto dis_2_sum = xsimd::slide_left<8>(val) + val;
        const auto dis_1_sum = xsimd::slide_left<4>(dis_2_sum) + dis_2_sum;
        prev = dis_1_sum +
               xsimd::swizzle(prev, xsimd::batch_constant<Batch, 3, 3, 3, 3>());
        prev.store_aligned(dst_ptr);
        dst_ptr += 4;
    }
    last = prev.get(3);
    for (size_t i = block_n; i < n; ++i) {
        auto val = decodeZigZagValInt32(zigzag_values[i]);
        *dst_ptr++ = val + last;
        last = val + last;
    }
#endif
    return n * sizeof(uint32_t);
}

} // namespace null_revisit
