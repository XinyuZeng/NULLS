#include "BitpackingInterface.hpp"
#include <memory>

namespace null_revisit {
BitpackingInterface::BitpackingInterface(const std::string &codec_name)
    : CompressInterface(), factory_(), codec_(factory_.getFromName(codec_name)),
      word_size_(1) {
}

size_t BitpackingInterface::compress(std::string_view src,
                                     arrow::ResizableBuffer *dst,
                                     const std::vector<bool> &nulls) {
    const uint32_t n = src.size() / sizeof(uint32_t);
    size_t dst_buf_size = dst->size() / sizeof(uint32_t);
    codec_->encodeArray(
        reinterpret_cast<const uint32_t *>(src.data()), n,
        // slowEncodeArray(reinterpret_cast<const uint32_t *>(src.data()), n,
        reinterpret_cast<uint32_t *>(dst->mutable_data()), dst_buf_size);
    dst_buf_size *= sizeof(uint32_t);
    dst->Resize(dst_buf_size);
    return dst_buf_size;
}

size_t BitpackingInterface::compress_with_null(std::string_view src,
                                               arrow::ResizableBuffer *dst,
                                               const std::vector<bool> &nulls) {
    return compress(src, dst, nulls);
}

size_t BitpackingInterface::compress_no_null(std::string_view src, char *dst,
                                             size_t dst_size) {
    // This function is called by RLE and Delta
    size_t dst_buf_size = dst_size / sizeof(uint32_t);
    codec_->encodeArray(
        reinterpret_cast<const uint32_t *>(src.data()),
        src.size() / sizeof(uint32_t), reinterpret_cast<uint32_t *>(dst),
        dst_buf_size);
    dst_buf_size *= sizeof(uint32_t);
    return dst_buf_size;
}

size_t BitpackingInterface::decompress(std::string_view src, char *dst,
                                       size_t dst_size) {
    dst_size = dst_size / sizeof(uint32_t);
    codec_->decodeArray(
        reinterpret_cast<const uint32_t *>(src.data()),
        src.size() / sizeof(uint32_t), reinterpret_cast<uint32_t *>(dst),
        dst_size);
    dst_size *= sizeof(uint32_t);
    return dst_size;
}

void BitpackingInterface::slowEncodeArray(const uint32_t *in, size_t in_length,
                                          uint32_t *out, size_t &out_length) {
    uint8_t *out_ptr = reinterpret_cast<uint8_t *>(out);
    size_t out_pos = 0;
    for (size_t in_pos = 0; in_pos < in_length; in_pos += MINI_BLOCK_SIZE) {
        size_t in_end = in_pos + MINI_BLOCK_SIZE;
        if (in_end > in_length) {
            in_end = in_length;
        }
        size_t num_vals = in_end - in_pos;
        uint8_t max_bits = maxBits(in + in_pos, in_end - in_pos);
        if (num_vals != MINI_BLOCK_SIZE) {
            out_ptr[out_pos++] = max_bits + NUM_BITS_IN_WORD + 1;
            out_ptr[out_pos++] = num_vals;
        } else {
            out_ptr[out_pos++] = max_bits;
        }
        uint8_t remaining_output_bits = NUM_BITS_IN_BYTE;
        uint8_t remaining_input_bits = max_bits;
        uint32_t input_val = in[in_pos];
        out_ptr[out_pos] = 0;
        for (size_t i = in_pos;;) {
            uint8_t num_bits =
                std::min(remaining_input_bits, remaining_output_bits);
            uint32_t mask = (1 << num_bits) - 1;
            out_ptr[out_pos] |=
                ((input_val & mask) << (remaining_output_bits - num_bits));
            input_val >>= num_bits;
            remaining_input_bits -= num_bits;
            remaining_output_bits -= num_bits;
            if (remaining_output_bits == 0) {
                out_pos++;
                out_ptr[out_pos] = 0;
                remaining_output_bits = NUM_BITS_IN_BYTE;
            }
            if (remaining_input_bits == 0) {
                i++;
                if (i == in_end) {
                    break;
                }
                input_val = in[i];
                remaining_input_bits = max_bits;
            }
        }
        if (remaining_output_bits != NUM_BITS_IN_BYTE) {
            out_pos++;
            out_ptr[out_pos] = 0;
        }
    }
    out_length = (out_pos + sizeof(uint32_t) - 1) / sizeof(uint32_t);
    for (size_t i = out_pos; i < out_length * sizeof(uint32_t); i++) {
        out_ptr[i] = NUM_BITS_IN_WORD * 2 + 2;
    }
}

void BitpackingInterface::slowEncodeArrayWithNull(
    const uint32_t *in, size_t in_length, uint32_t *out, size_t &out_length,
    const std::vector<bool> &nulls) {
    uint8_t *out_ptr = reinterpret_cast<uint8_t *>(out);
    size_t out_pos = 0;
    for (size_t in_pos = 0; in_pos < in_length; in_pos += MINI_BLOCK_SIZE) {
        uint32_t last = 0;
        size_t in_end = in_pos + MINI_BLOCK_SIZE;
        if (in_end > in_length) {
            in_end = in_length;
        }
        size_t num_vals = in_end - in_pos;
        uint8_t max_bits = maxBits(in + in_pos, in_end - in_pos);
        if (num_vals != MINI_BLOCK_SIZE) {
            out_ptr[out_pos++] = max_bits + NUM_BITS_IN_WORD + 1;
            out_ptr[out_pos++] = num_vals;
        } else {
            out_ptr[out_pos++] = max_bits;
        }
        uint8_t remaining_output_bits = NUM_BITS_IN_BYTE;
        uint8_t remaining_input_bits = max_bits;
        uint32_t input_val = in[in_pos];
        (nulls[in_pos]) ? (input_val = last) : (last = input_val);
        out_ptr[out_pos] = 0;
        for (size_t i = in_pos;;) {
            uint8_t num_bits =
                std::min(remaining_input_bits, remaining_output_bits);
            uint32_t mask = (1 << num_bits) - 1;
            out_ptr[out_pos] |=
                ((input_val & mask) << (remaining_output_bits - num_bits));
            input_val >>= num_bits;
            remaining_input_bits -= num_bits;
            remaining_output_bits -= num_bits;
            if (remaining_output_bits == 0) {
                out_pos++;
                out_ptr[out_pos] = 0;
                remaining_output_bits = NUM_BITS_IN_BYTE;
            }
            if (remaining_input_bits == 0) {
                i++;
                if (i == in_end) {
                    break;
                }
                input_val = in[i];
                (nulls[i]) ? (input_val = last) : (last = input_val);
                remaining_input_bits = max_bits;
            }
        }
        if (remaining_output_bits != NUM_BITS_IN_BYTE) {
            out_pos++;
            out_ptr[out_pos] = 0;
        }
    }
    out_length = (out_pos + sizeof(uint32_t) - 1) / sizeof(uint32_t);
    for (size_t i = out_pos; i < out_length * sizeof(uint32_t); i++) {
        out_ptr[i] = NUM_BITS_IN_WORD * 2 + 2;
    }
}

void BitpackingInterface::slowDecodeArray(const uint32_t *in, size_t in_length,
                                          uint32_t *out, size_t &out_length) {
    const uint8_t *in_ptr = reinterpret_cast<const uint8_t *>(in);
    size_t in_length_bytes = in_length * sizeof(uint32_t);
    size_t in_pos = 0;
    size_t out_pos = 0;
    while (in_pos < in_length_bytes) {
        uint8_t max_bits = in_ptr[in_pos++];
        uint8_t num_vals = MINI_BLOCK_SIZE;
        if (max_bits > NUM_BITS_IN_WORD) {
            num_vals = in_ptr[in_pos++];
            max_bits -= NUM_BITS_IN_WORD + 1;
        }
        if (max_bits > NUM_BITS_IN_WORD)
            continue;
        uint8_t remaining_input_bits = NUM_BITS_IN_BYTE;
        uint8_t remaining_output_bits = max_bits;
        uint32_t output_val = 0;
        for (size_t i = 0; i < num_vals;) {
            uint8_t num_bits =
                std::min(remaining_input_bits, remaining_output_bits);
            uint32_t mask = (1 << num_bits) - 1;
            output_val |=
                ((in_ptr[in_pos] >> (remaining_input_bits - num_bits)) & mask)
                << (max_bits - remaining_output_bits);
            remaining_input_bits -= num_bits;
            remaining_output_bits -= num_bits;
            if (remaining_input_bits == 0) {
                in_pos++;
                remaining_input_bits = NUM_BITS_IN_BYTE;
            }
            if (remaining_output_bits == 0) {
                out[out_pos++] = output_val;
                output_val = 0;
                remaining_output_bits = max_bits;
                i++;
            }
        }
        if (remaining_input_bits != NUM_BITS_IN_BYTE) {
            in_pos++;
        }
    }
    out_length = out_pos;
}

uint8_t BitpackingInterface::maxBits(const uint32_t *in, size_t in_length) {
    uint32_t accumulate_val = 0;
    for (size_t i = 0; i < in_length; i++) {
        accumulate_val |= in[i];
    }
    return (accumulate_val == 0) ? 0 : (32 - __builtin_clz(accumulate_val));
}

} // namespace null_revisit
