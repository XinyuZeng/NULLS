#include "FLSDeltaInterface.hpp"
#include "VarInts.hpp"
#include "my_helper.hpp"
#include "pack.hpp"
#include "rsum.hpp"
#include "unpack.hpp"
#include <arrow/util/bit_util.h>
#include <cstdint>
#include <memory>

namespace null_revisit {
FLSDeltaInterface::FLSDeltaInterface() : FLSInterface(), word_size_(4) {}

size_t FLSDeltaInterface::compress(std::string_view src,
                                   arrow::ResizableBuffer *dst,
                                   const std::vector<bool> &nulls) {
    return compress_template<false>(src, dst, nulls);
}

size_t FLSDeltaInterface::compress_dense(std::string_view src,
                                         arrow::ResizableBuffer *dst) {
    return compress_template<true>(src, dst, {});
}
template <bool dense>
size_t FLSDeltaInterface::compress_template(std::string_view src,
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

size_t FLSDeltaInterface::compress_no_resize(std::string_view src, char *dst) {
    size_t n = src.size() / word_size_;
    assert(n <= 1024); // FLS api only supports 1024 values
    auto src_ptr = (const uint32_t *)src.data();
    auto output = reinterpret_cast<uint32_t *>(dst);

    helper::fls_transpose(src_ptr, input_transposed);
    auto bw = helper::fls_delta(input_transposed, deltas, output);
    output += 32; // 32 base values
    generated::pack::helper::scalar::pack(deltas,
                                          const_cast<uint32_t *>(output), bw);
    auto dst_buf_size = arrow::bit_util::BytesForBits(bw * 1024);
    dst[dst_buf_size + 32 * 4] = bw;
    return dst_buf_size + 32 * 4 + 1;
}

size_t FLSDeltaInterface::compress_with_null(std::string_view src,
                                             arrow::ResizableBuffer *dst,
                                             const std::vector<bool> &nulls) {
    throw std::logic_error(
        "FLSDeltaInterface::compress_with_null Not implemented");
}

size_t FLSDeltaInterface::decompress(std::string_view src, char *dst,
                                     size_t dst_size) {
    return decompress_template<false>(src, dst, dst_size);
}

size_t FLSDeltaInterface::decompress_untranspose(std::string_view src,
                                                 char *dst, size_t dst_size) {
    return decompress_template<true>(src, dst, dst_size);
}

template <bool untrans>
size_t FLSDeltaInterface::decompress_template(std::string_view src, char *dst,
                                              size_t dst_size) {
    auto bw = src.data()[src.size() - 1]; // last byte is the bit width
    auto base = reinterpret_cast<const uint32_t *>(src.data());
    auto input = base + 32;
    auto output = reinterpret_cast<uint32_t *>(dst);
    generated::unpack::x86_64::avx512bw::unpack(
        input, const_cast<uint32_t *>(unpacked32), bw);
    if constexpr (untrans) {
        generated::rsum::x86_64::avx512bw::rsum(unpacked32, input_transposed,
                                                base); // reuse input_transposed
        helper::fls_untranspose_generated(input_transposed, output);
    } else {
        generated::rsum::x86_64::avx512bw::rsum(unpacked32, output, base);
    }

    return 1024 * sizeof(uint32_t);
}

} // namespace null_revisit
