#include "FLSBitpackingInterface.hpp"
#include "helper.hpp"
#include "my_helper.hpp"
#include "pack.hpp"
#include "rsum.hpp"
#include "unpack.hpp"
#include <arrow/util/bit_util.h>

#include <memory>

namespace null_revisit {
FLSBitpackingInterface::FLSBitpackingInterface()
    : FLSInterface(), word_size_(1) {}

size_t FLSBitpackingInterface::compress(std::string_view src,
                                        arrow::ResizableBuffer *dst,
                                        const std::vector<bool> &nulls) {
    return compress_dense(src, dst);
}

size_t FLSBitpackingInterface::compress_dense(std::string_view src,
                                              arrow::ResizableBuffer *dst) {
    auto dst_buf_size = compress_no_resize(src, (char *)dst->mutable_data());
    dst->Resize(dst_buf_size);
    return dst_buf_size;
}

size_t
FLSBitpackingInterface::compress_with_null(std::string_view src,
                                           arrow::ResizableBuffer *dst,
                                           const std::vector<bool> &nulls) {
    return compress(src, dst, nulls);
}

size_t FLSBitpackingInterface::compress_no_resize(std::string_view src,
                                                  char *dst) {
    const uint32_t n = src.size() / sizeof(uint32_t);
    assert(n <= 1024); // FLS api only supports 1024 values
    auto in_ptr = (const uint32_t *)src.data();
    auto bw = maxBits(in_ptr, n);
    generated::pack::helper::scalar::pack(
        in_ptr, reinterpret_cast<uint32_t *>(dst), bw);
    auto dst_buf_size = arrow::bit_util::BytesForBits(
        bw * 1024); // FLS API only supports 1024 values at a time.
    dst[dst_buf_size] = bw;
    return dst_buf_size + 1;
}

size_t FLSBitpackingInterface::decompress(std::string_view src, char *dst,
                                          size_t dst_size) {
    auto bw = src.data()[src.size() - 1]; // last byte is the bit width
    generated::unpack::x86_64::avx512bw::unpack(
        (const uint32_t *)src.data(), reinterpret_cast<uint32_t *>(dst), bw);
    return 1024 * sizeof(uint32_t);
}

uint8_t FLSBitpackingInterface::maxBits(const uint32_t *in, size_t in_length) {
    uint32_t accumulate_val = 0;
    for (size_t i = 0; i < in_length; i++) {
        accumulate_val |= in[i];
    }
    return (accumulate_val == 0) ? 0 : (32 - __builtin_clz(accumulate_val));
}

} // namespace null_revisit
