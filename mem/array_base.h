#pragma once
#include <arrow/array/array_base.h>
#include <arrow/buffer.h>
#include <bits/stdint-intn.h>
#include <immintrin.h>

#include <cassert>
#include <cstdint>
#include <memory>
#include <new>
#include <vector>

#include "utils/macros.hpp"
namespace null_revisit {
using arrow::Buffer;

enum class ArrayType { Dense = 0, Spaced, SpecialVal, Invalid };

class BaseArray {
  public:
    // TODO: refactor those two constructors
    BaseArray() { random_array = new (std::align_val_t(64)) int32_t[length]; }
    BaseArray(const std::shared_ptr<arrow::Array> &array) {
        this->length = array->length();
        this->null_count = array->null_count();
        this->buffers = array->data()->buffers;
        if (buffers[0] == nullptr) {
            buffers[0] = *arrow::AllocateBitmap(length);
            memset(buffers[0]->mutable_data(), 1, buffers[0]->size());
        }
        random_array = new (std::align_val_t(64)) int32_t[length];
    }
    virtual ~BaseArray() {
        ::operator delete[](random_array, std::align_val_t(64));
    }

    // For all the following functions, the output buffer is assumed to be
    // pre-allocated. This is for in-cache benchmark purpose.
    virtual void CompareWithCol(const BaseArray &other,
                                std::shared_ptr<Buffer> output) const = 0;
    virtual void CompareWithConst(const BaseArray &other,
                                  std::shared_ptr<Buffer> output) const = 0;
    virtual void SumByGroupBM(const std::vector<int32_t> &group_ids,
                              std::vector<int64_t> &sum) const = 0;
    virtual void SumByGroupSV(const std::vector<int32_t> &group_ids,
                              std::vector<int64_t> &sum) const = 0;

    // I decided to use Arrow's Buffer class for memory management for its
    // convenience. buffers[0] for null bitmap (can be in different format),
    // buffers[1] for data. buffers[2:] for custom uses (optional).
    std::vector<std::shared_ptr<Buffer>> buffers;
    // Members are all public for convenience and self-use.
    int64_t length = 0;
    int64_t null_count = 0;
    ArrayType type = ArrayType::Invalid;
    int32_t *__restrict random_array;

  protected:
    template <bool five_op = false>
    inline void SpacedCompareAVX(const uint8_t *a, const uint8_t *b,
                                 const uint8_t *a_bm, const uint8_t *b_bm,
                                 std::shared_ptr<Buffer> output) const {
        __m512i *a_ptr = (__m512i *)a;
        __m512i *b_ptr = (__m512i *)b;
        auto output_ptr = reinterpret_cast<uint16_t *>(output->mutable_data());

        auto bm_a = (__m512i *)a_bm;
        auto bm_b = (__m512i *)b_bm;
        // TODO: loop unrolling.
        __m512i number = _mm512_load_epi32(random_array);
        for (int i = 0; i < length; i += 512) {
            for (int j = 0; j < 32; ++j) {
                __m512i a = _mm512_load_si512(a_ptr++);
                __m512i b = _mm512_load_si512(b_ptr++);
                if constexpr (five_op) {
                    a = _mm512_mul_epi32(a, number);
                    a = _mm512_add_epi32(a, number);
                    a = _mm512_mul_epi32(a, number);
                    a = _mm512_add_epi32(a, number);
                    a = _mm512_mul_epi32(a, number);
                    // a = _mm512_mul_epi32(a, number);
                    // a = _mm512_mul_epi32(a, number);
                    // a = _mm512_mul_epi32(a, number);
                    // a = _mm512_mul_epi32(a, number);

                    b = _mm512_mul_epi32(b, number);
                    b = _mm512_add_epi32(b, number);
                    b = _mm512_mul_epi32(b, number);
                    b = _mm512_add_epi32(b, number);
                    b = _mm512_mul_epi32(b, number);
                }
                output_ptr[j] = _mm512_cmplt_epi32_mask(
                    a, b); // TODO: it is possible to store the mask in
                           // registers, not in memory.
            }
            _mm512_store_si512(
                output_ptr,
                _mm512_and_si512(_mm512_load_si512(output_ptr),
                                 _mm512_and_si512(*bm_a++, *bm_b++)));
            output_ptr += 32;
        }
    }
};
} // namespace null_revisit