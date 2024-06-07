#include "D4RLEInterface.hpp"
#include "utils.hpp"
#include <arrow/util/bit_util.h>
#include <cstdint>

namespace null_revisit {
D4RLEInterface::D4RLEInterface()
    : FLSInterface(), word_size_(4), bitpacker_() {}

size_t D4RLEInterface::compress(std::string_view src,
                                arrow::ResizableBuffer *dst,
                                const std::vector<bool> &nulls) {
    return compress_template<false>(src, dst, nulls);
}

size_t D4RLEInterface::compress_dense(std::string_view src,
                                      arrow::ResizableBuffer *dst) {
    return compress_template<true>(src, dst, {});
}

template <bool dense>
size_t D4RLEInterface::compress_template(std::string_view src,
                                         arrow::ResizableBuffer *dst,
                                         const std::vector<bool> &nulls) {
    size_t n = src.size() / word_size_;
    std::vector<uint32_t> rle_values, rle_idxs;
    uint32_t idx = 0;
    rle_values.reserve(n);
    rle_idxs.reserve(n);
    auto src_ptr = (const uint32_t *)src.data();
    uint32_t last = src_ptr[0];
    rle_idxs.push_back(idx);
    rle_values.push_back(last);
    for (size_t i = 1; i < n; i++) {
        // Read 4 bytes as a uint32_t
        uint32_t val = src_ptr[i];
        bool flag = true;
        if constexpr (!dense) {
            flag = nulls[i] == false;
        }
        if (val != last && flag) {
            last = val;
            rle_values.push_back(last);
            idx++;
        }
        rle_idxs.push_back(idx);
    }
    // Compress rle_values and rle_lengths using Bitpacking
    const auto values_size = bitpacker_.compress_no_resize(
        std::string_view((const char *)rle_values.data(),
                         rle_values.size() * sizeof(uint32_t)),
        (char *)dst->mutable_data() + sizeof(uint32_t));
    // Write values_size into dst
    *(reinterpret_cast<uint32_t *>(dst->mutable_data())) = values_size;
    auto bw =
        *(uint8_t *)(dst->mutable_data() + values_size - 1 + sizeof(uint32_t));
    // Compress rle_idxs into dst
    const auto idxs_size = delta_.compress_no_resize(
        std::string_view((const char *)rle_idxs.data(),
                         rle_idxs.size() * sizeof(uint32_t)),
        (char *)dst->mutable_data() + sizeof(uint32_t) + values_size);
    const auto dst_buf_size = sizeof(uint32_t) + values_size + idxs_size;
    dst->Resize(dst_buf_size);
    // TODO: hacky way to make compression ratio correct.
    return sizeof(uint32_t) +
           arrow::bit_util::BytesForBits(bw * rle_values.size()) + idxs_size;
}

size_t D4RLEInterface::compress_with_null(std::string_view src,
                                          arrow::ResizableBuffer *dst,
                                          const std::vector<bool> &nulls) {
    throw std::logic_error(
        "D4RLEInterface::compress_with_null Not implemented");
}

size_t D4RLEInterface::decompress(std::string_view src, char *const dst,
                                  size_t dst_size) {
    return decompress_template<false>(src, dst, dst_size);
}

template <bool untrans>
size_t D4RLEInterface::decompress_template(std::string_view src, char *dst,
                                           size_t dst_size) {
    auto src_ptr = reinterpret_cast<const uint8_t *>(src.data());
    const auto values_size = *(reinterpret_cast<const uint32_t *>(src_ptr));
    src_ptr += sizeof(uint32_t);
    // Decompress values using Bitpacking
    bitpacker_.decompress(std::string_view((const char *)src_ptr, values_size),
                          (char *)rle_values, 0);
    src_ptr += (values_size);
    // Decompress idxes using Delta
    size_t idxs_buf_size;
    if constexpr (untrans) { // will not call. D4 does not need to
                             // untranspose.
        throw std::logic_error(
            "D4RLEInterface::decompress_template untrans not implemented");
        idxs_buf_size = delta_.decompress_untranspose(
            std::string_view((const char *)src_ptr,
                             src.size() - values_size - sizeof(uint32_t)),
            (char *)rle_idxs, 0);
    } else {
        idxs_buf_size = delta_.decompress(
            std::string_view((const char *)src_ptr,
                             src.size() - values_size - sizeof(uint32_t)),
            (char *)rle_idxs, 0);
    }
    // Decode rle_values and rle_idxs
    uint32_t *__restrict dst_ptr = std::assume_aligned<64>((uint32_t *)dst);
    auto bound = idxs_buf_size / sizeof(uint32_t);
    size_t i = 0;
    for (; i <= bound - 512; i += 512) {
        UnrolledRLEDecoding512Values(dst_ptr + i, rle_values, rle_idxs + i);
    }
    for (; i <= bound - 16; i += 16) {
        UnrolledRLEDecoding16Values(dst_ptr + i, rle_values, rle_idxs + i);
    }
    for (; i < bound; i++) {
        dst_ptr[i] = rle_values[rle_idxs[i]];
    }
    /*
    assert(bound % 16 == 0);
    for (size_t i = 0; i < bound; i += 16) {
        // Load 16 indices using masked gather if bound is not a multiple
    of 16.
        // __m512i idxs = _mm512_maskz_loadu_epi32(
        //     _cvtu32_mask16(((1ULL << (bound - i)) - 1) & 0xFFFF),
        //     &rle_idxs[i]);
        __m512i idxs = _mm512_load_si512(&rle_idxs[i]);
        __m512i values =
            _mm512_i32gather_epi32(idxs, rle_values, sizeof(uint32_t));
        // Store the values in the destination array.
        // _mm512_mask_storeu_epi32(
        //     &dst_ptr[i], _cvtu32_mask16(((1ULL << (bound - i)) - 1) &
        //     0xFFFF), values);
        _mm512_store_si512(&dst_ptr[i], values);
    }
    */
    return 1024 * sizeof(uint32_t); // TODO: not hardcode FLS vector size
}

} // namespace null_revisit
