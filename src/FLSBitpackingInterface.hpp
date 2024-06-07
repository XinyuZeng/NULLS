#pragma once

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>

#include "FLSInterface.hpp"
#include "blockpacking.h"
#include "compositecodec.h"
#include "variablebyte.h"
#include <codecfactory.h>
#include <deltautil.h>
#include <fastpfor.h>

namespace null_revisit {
class FLSBitpackingInterface : public FLSInterface {
  public:
    /**
     * Construct a BitpackingInterface object
     *
     * @param codec_name: the name of the codec, default is
     * "fastbinarypacking32"
     */
    FLSBitpackingInterface();

    ~FLSBitpackingInterface() override = default;

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

    size_t compressBufSize(size_t src_size) override { return src_size + 1024; }

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

    // Use the base class's compress and decompress template functions
    using CompressInterface::compress;
    using CompressInterface::decompress;

    static constexpr size_t MINI_BLOCK_SIZE = 32;
    static constexpr uint8_t NUM_BITS_IN_BYTE = 8;
    static constexpr uint8_t NUM_BITS_IN_WORD = 32;

  private:
    size_t word_size_;
    uint8_t maxBits(const uint32_t *in, size_t in_length);
};
} // namespace null_revisit
