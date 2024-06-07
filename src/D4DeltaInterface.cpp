#include "D4DeltaInterface.hpp"
#include "VarInts.hpp"
#include "my_helper.hpp"
#include "pack.hpp"
#include "rsum.hpp"
#include "unpack.hpp"
#include <arrow/util/bit_util.h>
#include <cstdint>
#include <memory>

namespace null_revisit {
D4DeltaInterface::D4DeltaInterface() : FLSInterface(), word_size_(4) {}

size_t D4DeltaInterface::compress(std::string_view src,
                                  arrow::ResizableBuffer *dst,
                                  const std::vector<bool> &nulls) {
    return compress_template<false>(src, dst, nulls);
}

size_t D4DeltaInterface::compress_dense(std::string_view src,
                                        arrow::ResizableBuffer *dst) {
    return compress_template<true>(src, dst, {});
}
template <bool dense>
size_t D4DeltaInterface::compress_template(std::string_view src,
                                           arrow::ResizableBuffer *dst,
                                           const std::vector<bool> &nulls) {
    size_t dst_buf_size;
    if (!dense) {
        auto filled = fillNullsLinear(src, nulls);
        dst_buf_size = compress_no_resize(
            std::string_view((const char *)filled.data(), src.size()),
            (char *)dst->mutable_data());
    } else {
        dst_buf_size = compress_no_resize(src, (char *)dst->mutable_data());
    }
    dst->Resize(dst_buf_size);
    return dst_buf_size;
}

size_t D4DeltaInterface::compress_no_resize(std::string_view src, char *dst) {
    size_t n = src.size() / word_size_;
    assert(n <= 1024); // FLS api only supports 1024 values
    auto input = (const uint32_t *)src.data();
    auto output = reinterpret_cast<uint32_t *>(dst);
    auto base = output;

    uint32_t cur[16] = {0};
    // For AVX512, 16 32-bit values per 512-bit vector.
    for (size_t i = 0; i < 16; ++i) {
        base[i] = input[i];
        cur[i] = input[i];
    }

    uint32_t max_delta = 0;
    for (size_t i = 0; i < n / 16; ++i) {
        for (size_t j = 0; j < 16; ++j) {
            auto delta = input[i * 16 + j] - cur[j];
            deltas[i * 16 + j] = delta;
            if (delta > max_delta) {
                max_delta = delta;
            }
            cur[j] = input[i * 16 + j];
        }
    }
    auto bw = helper::NumRequiredBits(max_delta);
    output += 16; // 16 base values
    generated::pack::helper::scalar::pack(deltas,
                                          const_cast<uint32_t *>(output), bw);
    auto dst_buf_size = arrow::bit_util::BytesForBits(bw * 1024);
    dst[dst_buf_size + 16 * 4] = bw;
    return dst_buf_size + 16 * 4 + 1;
}

size_t D4DeltaInterface::compress_with_null(std::string_view src,
                                            arrow::ResizableBuffer *dst,
                                            const std::vector<bool> &nulls) {
    throw std::logic_error(
        "FLSDeltaInterface::compress_with_null Not implemented");
}

size_t D4DeltaInterface::decompress(std::string_view src, char *dst,
                                    size_t dst_size) {
    auto bw = src.data()[src.size() - 1]; // last byte is the bit width
    auto base = reinterpret_cast<const uint32_t *>(src.data());
    auto input = base + 16;
    auto output = reinterpret_cast<uint32_t *>(dst);
    generated::unpack::x86_64::avx512bw::unpack(
        input, const_cast<uint32_t *>(unpacked32), bw);
    generated::rsum::x86_64::avx512bw_d4::rsum(unpacked32, output, base);
    return 1024 * sizeof(uint32_t);
}

} // namespace null_revisit
