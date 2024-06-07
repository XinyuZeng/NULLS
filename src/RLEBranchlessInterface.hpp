#pragma once

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>

#include "BitpackingInterface.hpp"
#include "CompressInterface.hpp"

namespace null_revisit {
class RLEBranchlessInterface : public CompressInterface {
  public:
    /**
     * Construct a RLEBranchlessInterface object
     *
     * @param codec_name: the name of the codec, default is
     * "simple9_rle"
     */
    RLEBranchlessInterface(const std::string &codec_name = "simple9_rle");

    ~RLEBranchlessInterface() override = default;

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
        return (src_size * 4 + 4192) / 16 * 16;
    }

    size_t decompressBufSize(size_t src_size) override {
        return (src_size * 4 + 4192) / 16 * 16;
    }

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
    size_t compress_with_null(std::string_view src, arrow::ResizableBuffer *dst,
                              const std::vector<bool> &nulls) override;

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

  private:
    size_t word_size_;
    BitpackingInterface bitpacker_;
};
} // namespace null_revisit
