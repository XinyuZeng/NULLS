#pragma once

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>

#include "FLSBitpackingInterface.hpp"
#include "FLSInterface.hpp"

namespace null_revisit {
class FLSDeltaInterface : public FLSInterface {
  public:
    /**
     * FLSDelta currently only works with increasing values.
     */
    FLSDeltaInterface();

    ~FLSDeltaInterface() override {
        ::operator delete[](deltas, std::align_val_t(64));
        ::operator delete[](input_transposed, std::align_val_t(64));
        ::operator delete[](unpacked32, std::align_val_t(64));
    }

    /**
     * Initialize the compressor with the word size, useless for FastPFor
     *
     * @param word_size: the word size in bytes, default is 1
     */
    void init(size_t word_size = 1) override {
        switch (word_size) {
        case 1:
        case 2:
        case 4:
        case 8:
            word_size_ = word_size;
            break;
        default:
            throw std::logic_error("Unsupported word size");
        }
    };

    size_t compressBufSize(size_t src_size) override {
        return (src_size + 1192) / 16 * 16;
    }

    size_t decompressBufSize(size_t src_size) override { return src_size; }

    using CompressInterface::compressBufSize;

    /**
     * Compress the input string
     *
     * @param src: the input string
     * @param dst: the output string, which should have size >= src + 1024
     * @return: the size of the compressed string
     */
    size_t compress(std::string_view src, arrow::ResizableBuffer *dst,
                    const std::vector<bool> &nulls) override;
    size_t compress_dense(std::string_view src,
                          arrow::ResizableBuffer *dst) override;
    size_t compress_with_null(std::string_view src, arrow::ResizableBuffer *dst,
                              const std::vector<bool> &nulls) override;
    size_t compress_no_resize(std::string_view src, char *dst);
    /**
     * Decompress the input string
     *
     * @param src: the input string
     * @param dst: the output char pointer, which should have size >= src + 1024
     * @return: the size of the decompressed string
     */
    size_t decompress(std::string_view src, char *dst,
                      size_t dst_size) override;
    size_t decompress_untranspose(std::string_view src, char *dst,
                                  size_t dst_size) override;

    // Use the base class's compress and decompress template functions
    using CompressInterface::compress;
    using CompressInterface::decompress;

  private:
    FLSBitpackingInterface bitpacker_;
    size_t word_size_;
    uint32_t *input_transposed = new (std::align_val_t{64}) uint32_t[1024];
    uint32_t *deltas = new (std::align_val_t{64}) uint32_t[1024];
    uint32_t *unpacked32 = new (std::align_val_t{64}) uint32_t[1024];

    template <bool dense = false>
    size_t compress_template(std::string_view src, arrow::ResizableBuffer *dst,
                             const std::vector<bool> &nulls);
    template <bool untrans = false>
    size_t decompress_template(std::string_view src, char *dst,
                               size_t dst_size);
};
} // namespace null_revisit
