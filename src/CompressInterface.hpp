#pragma once

#include <arrow/buffer.h>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace null_revisit {
class CompressInterface {
  public:
    CompressInterface(){};
    virtual ~CompressInterface() = default;

    /**
     * Initialize the compressor with the word size
     *
     * @param word_size: the word size in bytes
     */
    virtual void init(size_t word_size) = 0;

    /**
     * Get the advised buffer size for compression
     *
     * @param src_size: the size of the input string
     * @return: the advised compress buffer size
     */
    virtual size_t compressBufSize(size_t src_size) = 0;

    /**
     * Get the advised buffer size for decompression
     *
     * @param src_size: the size of the input string
     * @return: the advised decompress buffer size
     */
    virtual size_t decompressBufSize(size_t src_size) = 0;

    template <typename T> size_t compressBufSize(const std::vector<T> &src) {
        return compressBufSize(src.size() * sizeof(T));
    }
    size_t compressBufSize(std::string_view src) {
        return compressBufSize(src.size());
    }

    /**
     * Compress the input string. Nulls may or may not be used depending on the
     * codec. e.g., Bitpacking/RLE/Delta does not use nulls at all, but
     * SmartNull does, because SmartNull internally calls other schemes'
     * compress_with_null.
     *
     * Note that FLS and D4 automatically have smart null handling. So nulls are
     * used in their settings.
     *
     * @param src: the input string
     * @param dst: the output string, which should have size >= src + 1024
     * @param nulls: the null vector
     * @return: the size of the compressed string
     */
    virtual size_t compress(std::string_view src, arrow::ResizableBuffer *dst,
                            const std::vector<bool> &nulls) = 0;

    /**
     * Compress the input string. No nulls considered at all.
     */
    virtual size_t compress_dense(std::string_view src,
                                  arrow::ResizableBuffer *dst) {
        throw std::runtime_error("compress_dense Not implemented");
    }

    /**
     * Compress the input string. Nulls must be used.
     */
    virtual size_t compress_with_null(std::string_view src,
                                      arrow::ResizableBuffer *dst,
                                      const std::vector<bool> &nulls) = 0;

    /**
     * Compress the input vector
     *
     * @param src: the input vector
     * @param dst: the output string, which should have size >= src + 1024
     * @return: the size of the compressed string
     */
    template <typename T>
    size_t compress(const std::vector<T> &src, arrow::ResizableBuffer *dst,
                    const std::vector<bool> &nulls) {
        return compress(
            std::string_view(reinterpret_cast<const char *>(src.data()),
                             src.size() * sizeof(T)),
            dst, nulls);
    }

    /**
     * Decompress the input string
     *
     * @param src: the input string
     * @param dst: the output string, which should have size >= src + 1024
     * @return: the size of the decompressed string
     */
    virtual size_t decompress(std::string_view src, char *dst,
                              size_t dst_size) = 0;

    /**
     * Decompress the input string into a vector
     *
     * @param src: the input string
     * @param dst: the output vector
     * @return: the length of the decompressed vector
     */
    template <typename T>
    size_t decompress(std::string_view src, std::vector<T> &dst) {
        size_t dst_size = dst.size() * sizeof(T);
        if (dst_size < src.size() + 1024)
            dst.resize((src.size() + 1025) / sizeof(T)),
                dst_size = dst.size() * sizeof(T);
        dst_size =
            decompress(src, reinterpret_cast<char *>(dst.data()), dst_size);
        dst_size /= sizeof(T);
        dst.resize(dst_size);
        return dst_size;
    }

    /**
     * Decompress the input string into a string
     *
     * @param src: the input string
     * @param dst: the output string
     * @return: the length of the decompressed string
     */
    size_t decompress(std::string_view src, arrow::ResizableBuffer *dst) {
        size_t dst_size = dst->size();
        if (dst_size < src.size() + 1024)
            dst->Resize(src.size() + 1024), dst_size = dst->size();
        dst_size = decompress(
            src, reinterpret_cast<char *>(dst->mutable_data()), dst_size);
        dst->Resize(dst_size);
        return dst_size;
    }

    struct ExtraTimer {
        int64_t sample_time = 0;
        ExtraTimer() = default;
        ExtraTimer(double sample_time) : sample_time(sample_time) {}
        ExtraTimer operator+(const ExtraTimer &rhs) {
            return ExtraTimer(sample_time + rhs.sample_time);
        }
        ExtraTimer &operator+=(const ExtraTimer &rhs) {
            sample_time += rhs.sample_time;
            return *this;
        }
        void reset() { sample_time = 0; }
    };

    virtual ExtraTimer getExtraTimer() { return extra_timer_; }
    ExtraTimer extra_timer_;

    enum NullHeuristic {
        ZERO = 0,
        LAST = 1,
        FREQUENT = 2,
        LINEAR = 3,
        RANDOM = 4,
        SMART
    };
    struct CompressOption {
        NullHeuristic heuristic = SMART;
        bool always_delta = false;
    };

    virtual void setOptions(const CompressOption &compress_option) {
        compress_option_ = compress_option;
    }
    CompressOption compress_option_;

    struct EncodingCounter {
        uint32_t num_bitpacking = 0, num_rle = 0, num_delta = 0;
        void reset() {
            num_bitpacking = 0;
            num_rle = 0;
            num_delta = 0;
        }
    };
    EncodingCounter encoding_counter_;
    virtual EncodingCounter getEncodingCounter() { return encoding_counter_; }

  protected:
    auto fillNullsLinear(std::string_view src, const std::vector<bool> &nulls)
        -> std::vector<uint32_t> {
        size_t n = src.size() / sizeof(uint32_t);
        std::vector<uint32_t> filled_values;
        filled_values.reserve(n);
        auto src_ptr = (const uint32_t *)src.data();
        auto src_start = src_ptr;
        // last: the last non-null value with index < i
        // next: first non-null value with index >= i
        // i-1 null? no update
        // i-1 non-null? i-1 -> last, find next
        // last_(next_)non_null values are from input;
        uint32_t last_non_null = 0, next_non_null = 0;
        int last_idx = -1, next_idx = 0;
        // Find next for i-1 == -1
        while (next_idx < n && nulls[next_idx])
            next_idx++;
        bool no_non_null_left = false;
        if (next_idx < n) {
            next_non_null = src_start[next_idx];
            last_non_null = next_non_null;
        } else {
            no_non_null_left = true;
        }

        // Ready for i == 0
        for (size_t i = 0; i < n; ++i) {
            // Read 4 bytes as a uint32_t
            if (nulls[i]) {
                if (no_non_null_left) {
                    filled_values.push_back(
                        filled_values.empty() ? 0 : filled_values.back());
                } else {
                    const int32_t delta =
                        int32_t(next_non_null) - int32_t(last_non_null);
                    const int32_t delta_idx = next_idx - last_idx;
                    const int32_t cur_delta_idx = i - last_idx;
                    const int32_t cur_delta = delta * cur_delta_idx / delta_idx;
                    filled_values.push_back(int32_t(last_non_null) + cur_delta);
                }
                src_ptr++;
            } else {
                uint32_t val = *src_ptr++;
                filled_values.push_back(val);
                last_non_null = val, last_idx = i, next_idx = i + 1;
                while (next_idx < n && nulls[next_idx])
                    next_idx++;
                if (next_idx < n) {
                    next_non_null = src_start[next_idx];
                } else {
                    no_non_null_left = true;
                }
            }
            // Ready for i+1
        }
        return filled_values;
    }
};

inline CompressInterface::NullHeuristic
GetHeuristicFromString(const std::string &str) {
    if (str == "zero")
        return CompressInterface::NullHeuristic::ZERO;
    if (str == "last")
        return CompressInterface::NullHeuristic::LAST;
    if (str == "frequent")
        return CompressInterface::NullHeuristic::FREQUENT;
    if (str == "linear")
        return CompressInterface::NullHeuristic::LINEAR;
    if (str == "random")
        return CompressInterface::NullHeuristic::RANDOM;
    if (str == "smart")
        return CompressInterface::NullHeuristic::SMART;
    throw std::invalid_argument("Unknown null heuristic: " + str);
}
} // namespace null_revisit
