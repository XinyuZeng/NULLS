#pragma once

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>

#include "BitpackingStat.hpp"
#include "CompressInterface.hpp"

namespace null_revisit {
class SampleCompressInterface : public CompressInterface {
  public:
    enum CompressScheme { BITPACKING_C, RLE_C, DELTA_C, NUM_C_SCHEMES };
    /**
     * Construct a SampleCompressInterface object
     */
    SampleCompressInterface(const std::string &codec_name = "");

    ~SampleCompressInterface() override;

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

    void setOptions(const CompressOption &compress_option) override {
        compress_option_ = compress_option;
        interfaces[BITPACKING_C]->compress_option_ = compress_option_;
        interfaces[RLE_C]->compress_option_ = compress_option_;
        interfaces[DELTA_C]->compress_option_ = compress_option_;
    }

  private:
    static const uint32_t SAMPLE_SIZE = 64, SAMPLE_UNIT_SIZE = 4096;
    uint32_t sample_segments_ = 4;
    size_t word_size_;
    CompressInterface *interfaces[NUM_C_SCHEMES];
    SampleCompressInterface::CompressScheme sampleStat(const uint32_t *src_arr,
                                                       uint32_t n);

    struct SampleStat {
        uint32_t num_elements, num_runs;
        uint32_t last;
        BitpackingStat bp_stat;
        SampleStat(uint32_t num_elements, uint32_t last)
            : num_elements(num_elements), num_runs(0), last(last) {}
        SampleStat operator+(const SampleStat &rhs) {
            SampleStat ret(*this);
            ret += rhs;
            return ret;
        }
        SampleStat &operator+=(const SampleStat &rhs) {
            num_elements += rhs.num_elements;
            num_runs += rhs.num_runs;
            last = rhs.last;
            bp_stat += rhs.bp_stat;
            return *this;
        }
        CompressScheme compute() {
            bp_stat.finish();
            const double avg_val_bits =
                static_cast<double>(bp_stat.value_tot_bits) / bp_stat.block_cnt;
            const double avg_delta_bits =
                static_cast<double>(bp_stat.delta_tot_bits) / bp_stat.block_cnt;
            const double bp_bytes =
                avg_val_bits / 8 * num_elements + bp_stat.block_cnt;
            const double rle_bytes =
                bp_bytes / num_elements * num_runs + num_elements / 8;
            const double delta_bytes =
                avg_delta_bits / 8 * num_elements + num_elements / 8;
            return (bp_bytes <= rle_bytes)
                       ? ((bp_bytes <= delta_bytes) ? BITPACKING_C : DELTA_C)
                       : ((rle_bytes <= delta_bytes) ? RLE_C : DELTA_C);
        }
        void incData(uint32_t cur) {
            bp_stat.incData(cur, cur - last);
            num_runs += (cur != last), last = cur;
        }
    };
    void incEncodingCounter(CompressScheme scheme) {
        switch (scheme) {
        case BITPACKING_C:
            encoding_counter_.num_bitpacking++;
            break;
        case RLE_C:
            encoding_counter_.num_rle++;
            break;
        case DELTA_C:
            encoding_counter_.num_delta++;
            break;
        default:
            throw std::logic_error("Unsupported compress scheme");
        }
    }
};
} // namespace null_revisit
