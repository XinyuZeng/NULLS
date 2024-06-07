#pragma once

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>

#include "D4DeltaInterface.hpp"
#include "FLSBitpackingInterface.hpp"
#include "FLSInterface.hpp"

namespace null_revisit {
/*
    This is just a copy of FLSRLEInterface with the delta changed to
   D4DeltaInterface. This is ugly and should be refactored.
*/
class D4RLEInterface : public FLSInterface {
  public:
    /**
     * Construct a RLEBranchlessInterface object
     *
     * @param codec_name: the name of the codec, default is
     * "simple9_rle"
     */
    D4RLEInterface();

    ~D4RLEInterface() override {
        ::operator delete[](rle_idxs, std::align_val_t(64));
        ::operator delete[](rle_values, std::align_val_t(64));
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
        return (src_size * 4 + 4192) / 16 * 16;
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
    FLSBitpackingInterface bitpacker_;
    D4DeltaInterface delta_;

    uint32_t *__restrict rle_values = new (std::align_val_t{64}) uint32_t[1024];
    uint32_t *__restrict rle_idxs = new (std::align_val_t{64}) uint32_t[1024];

    template <bool dense = false>
    size_t compress_template(std::string_view src, arrow::ResizableBuffer *dst,
                             const std::vector<bool> &nulls);

    template <bool untrans = false>
    size_t decompress_template(std::string_view src, char *dst,
                               size_t dst_size);
};
} // namespace null_revisit
