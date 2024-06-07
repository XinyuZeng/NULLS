#include "special_value_array.h"

#include <arrow/util/bit_block_counter.h>
#include <arrow/util/bit_util.h>
#include <assert.h>
#include <bits/stdint-uintn.h>

#include "utils/bit_util.h"
#include "utils/macros.hpp"

using namespace null_revisit;

template <bool five_op>
void SpecialValArray::CompareWithCol(const SpecialValArray &other,
                                     std::shared_ptr<Buffer> output) const {
    // input size corretness check
    assert(this->length == kVecSize);
    assert(this->length == other.length);
    assert(output->size() >= arrow::bit_util::BytesForBits(this->length));
    assert(this->length * kDefaultTypeSize % 512 == 0);
    // alignment check
    assert(reinterpret_cast<uint64_t>(this->buffers[1]->data()) % 64 == 0);
    assert(reinterpret_cast<uint64_t>(other.buffers[1]->data()) % 64 == 0);
    assert(reinterpret_cast<uint64_t>(output->data()) % 64 == 0);

    // SpacedCompareAVX<five_op>(this->buffers[1]->data(),
    // other.buffers[1]->data(), this->buffers[0]->data(),
    //                           other.buffers[0]->data(), output);
    __m512i *a_ptr = (__m512i *)this->buffers[1]->data();
    __m512i *b_ptr = (__m512i *)other.buffers[1]->data();
    auto output_ptr = reinterpret_cast<uint16_t *>(output->mutable_data());
    auto special_value = _mm512_set1_epi32(kSpecialValue);
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
            output_ptr[j] = _mm512_cmplt_epi32_mask(a, b) &
                            _mm512_cmpneq_epi32_mask(a, special_value) &
                            _mm512_cmpneq_epi32_mask(special_value, b);
        }
        // _mm512_store_si512(output_ptr,
        // _mm512_and_si512(_mm512_load_si512(output_ptr),
        // _mm512_and_si512(*bm_a++, *bm_b++)));
        output_ptr += 32;
    }
}
template void
SpecialValArray::CompareWithCol<true>(const SpecialValArray &,
                                      std::shared_ptr<Buffer>) const;
template void
SpecialValArray::CompareWithCol<false>(const SpecialValArray &,
                                       std::shared_ptr<Buffer>) const;

int SpecialValArray::CompareWithColSVPartial(const SpecialValArray &other,
                                             SelVector &output,
                                             SelVector &sel_vec) const {
    auto a_ptr = (default_type *)this->buffers[1]->data();
    auto b_ptr = (default_type *)other.buffers[1]->data();
    int output_idx = 0;
    for (size_t i = 0; i < sel_vec.size(); ++i) {
        auto row_id = sel_vec[i];
        output[output_idx] = row_id;
        // TODO: duckdb flat :
        // 1. AND the null bm first. 2. do evaluation, partial only according to
        // null 3. combine this bool vector with original SV duckdb generic:
        // 1. do evaluation according to each BM. 2. combine this bool with SV.
        bool a = a_ptr[row_id] != kSpecialValue;
        bool b = b_ptr[row_id] != kSpecialValue;
        bool c = a_ptr[row_id] < b_ptr[row_id];
        output_idx += a & b & c;
    }
    return output_idx;
}
template <bool eval_filter_first>
int SpecialValArray::CompareWithColSVPartialBranch(const SpecialValArray &other,
                                                   SelVector &output,
                                                   SelVector &sel_vec) const {
    auto a_ptr = (default_type *)this->buffers[1]->data();
    auto b_ptr = (default_type *)other.buffers[1]->data();
    int output_idx = 0;
    for (size_t i = 0; i < sel_vec.size(); ++i) {
        auto row_id = sel_vec[i];
        if (eval_filter_first) {
            if (a_ptr[row_id] < b_ptr[row_id] &&
                a_ptr[row_id] != kSpecialValue &&
                b_ptr[row_id] != kSpecialValue) {
                output[output_idx++] = row_id;
            }
        } else {
            if (a_ptr[row_id] != kSpecialValue &&
                b_ptr[row_id] != kSpecialValue &&
                a_ptr[row_id] < b_ptr[row_id]) {
                output[output_idx++] = row_id;
            }
        }
    }
    return output_idx;
}
template int SpecialValArray::CompareWithColSVPartialBranch<false>(
    const SpecialValArray &, SelVector &output, SelVector &sel_vec) const;
template int SpecialValArray::CompareWithColSVPartialBranch<true>(
    const SpecialValArray &, SelVector &output, SelVector &sel_vec) const;