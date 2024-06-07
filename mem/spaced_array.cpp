#include "spaced_array.h"

#include <arrow/util/bit_block_counter.h>
#include <arrow/util/bit_util.h>
#include <assert.h>
#include <bits/stdint-uintn.h>

#include "utils/bit_util.h"
#include "utils/macros.hpp"

using namespace null_revisit;

template <bool five_op>
void SpacedArray::CompareWithCol(const SpacedArray &other,
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

    SpacedCompareAVX<five_op>(
        this->buffers[1]->data(), other.buffers[1]->data(),
        this->buffers[0]->data(), other.buffers[0]->data(), output);
}
template void SpacedArray::CompareWithCol<true>(const SpacedArray &,
                                                std::shared_ptr<Buffer>) const;
template void SpacedArray::CompareWithCol<false>(const SpacedArray &,
                                                 std::shared_ptr<Buffer>) const;

void SpacedArray::CompareWithColScalar(const SpacedArray &other,
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
    // Load 512 bits (16 int32_t values) from a and b into AVX-512 registers
    auto a_ptr = (default_type *)this->buffers[1]->data();
    auto b_ptr = (default_type *)other.buffers[1]->data();
    auto output_ptr = output->mutable_data();
    static constexpr int kBMBatchSize = 32;

    uint32_t temp_output[kBMBatchSize];
    int num_batches = this->length / kBMBatchSize;
    // TODO: loop unrolling.
    for (int i = 0; i < num_batches; ++i) {
        for (int j = 0; j < kBMBatchSize; ++j) {
            temp_output[j] = *a_ptr++ < *b_ptr++;
        }
        arrow::bit_util::PackBits<kBMBatchSize>(temp_output, output_ptr);
        output_ptr += kBMBatchSize / 8;
    }
    auto output_ptr_512 = (__m512i *)output->data();
    auto bm_a = (__m512i *)this->buffers[0]->data();
    auto bm_b = (__m512i *)other.buffers[0]->data();
    assert(length % 512 == 0);
    for (int i = 0; i < length / 512; ++i) {
        *output_ptr_512 = _mm512_and_si512(*bm_a++, *output_ptr_512);
        *output_ptr_512 = _mm512_and_si512(*bm_b++, *output_ptr_512);
        output_ptr_512++;
    }
}

void SpacedArray::SumByGroupBM(const std::vector<int32_t> &group_ids,
                               std::vector<int64_t> &sum) const {
    auto a_ptr = (default_type *)this->buffers[1]->data();
    int idx = 0;
    arrow::internal::VisitBitBlocksVoid(
        buffers[0]->data(), 0, length, // NOLINT
        [&](int64_t i) {
            sum[group_ids[idx]] += a_ptr[idx];
            idx++;
        }, // NOLINT
        [&]() { idx++; });
}

void SpacedArray::SumByGroupSV(const std::vector<int32_t> &group_ids,
                               std::vector<int64_t> &sum) const {
    auto a_ptr = (default_type *)this->buffers[1]->data();
    auto a_sv = (row_id_type *)this->buffers[2]->data();
    auto idx_decode_a = length - null_count;
    for (int i = 0; i < idx_decode_a; ++i) {
        sum[group_ids[a_sv[i]]] += a_ptr[a_sv[i]];
    }
}

int SpacedArray::CompareWithColSVPartial(const SpacedArray &other,
                                         SelVector &output,
                                         SelVector &sel_vec) const {
    auto a_bm = this->buffers[0]->data();
    auto b_bm = other.buffers[0]->data();
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
        bool a = arrow::bit_util::GetBit(a_bm, row_id);
        bool b = arrow::bit_util::GetBit(b_bm, row_id);
        bool c = a_ptr[row_id] < b_ptr[row_id];
        output_idx += a & b & c;
    }
    return output_idx;
}
template <bool eval_filter_first>
int SpacedArray::CompareWithColSVPartialBranch(const SpacedArray &other,
                                               SelVector &output,
                                               SelVector &sel_vec) const {
    auto a_bm = this->buffers[0]->data();
    auto b_bm = other.buffers[0]->data();
    auto a_ptr = (default_type *)this->buffers[1]->data();
    auto b_ptr = (default_type *)other.buffers[1]->data();
    int output_idx = 0;
    for (size_t i = 0; i < sel_vec.size(); ++i) {
        auto row_id = sel_vec[i];
        if (eval_filter_first) {
            if (a_ptr[row_id] < b_ptr[row_id] &&
                arrow::bit_util::GetBit(a_bm, row_id) &&
                arrow::bit_util::GetBit(b_bm, row_id)) {
                output[output_idx++] = row_id;
            }
        } else {
            if (arrow::bit_util::GetBit(a_bm, row_id) &&
                arrow::bit_util::GetBit(b_bm, row_id) &&
                a_ptr[row_id] < b_ptr[row_id]) {
                output[output_idx++] = row_id;
            }
        }
    }
    return output_idx;
}
template int SpacedArray::CompareWithColSVPartialBranch<false>(
    const SpacedArray &, SelVector &output, SelVector &sel_vec) const;
template int SpacedArray::CompareWithColSVPartialBranch<true>(
    const SpacedArray &, SelVector &output, SelVector &sel_vec) const;

void SpacedArray::CompareWithColSVPartialManual(const SpacedArray &other,
                                                SelVector &output,
                                                SelVector &sel_vec) const {
    auto a_bm = this->buffers[0]->data();
    auto b_bm = other.buffers[0]->data();
    auto a_ptr = (default_type *)this->buffers[1]->data();
    auto b_ptr = (default_type *)other.buffers[1]->data();
    SelVector true_sel(sel_vec.size());
    int sel_idx = 0;
    for (size_t i = 0; i < sel_vec.size(); ++i) {
        auto row_id = sel_vec[i];
        true_sel[sel_idx] = row_id;
        // TODO: duckdb
        bool a = arrow::bit_util::GetBit(a_bm, row_id);
        bool b = arrow::bit_util::GetBit(b_bm, row_id);
        sel_idx += a && b;
    }
    if (true_sel.size() % 16 != 0) {
        true_sel.resize(true_sel.size() + 16 - true_sel.size() % 16);
    }
    auto res_buffer = *arrow::AllocateResizableBuffer(
        arrow::bit_util::BytesForBits(true_sel.size()));
    __mmask16 *res = (__mmask16 *)(res_buffer->mutable_data());
    int i = 0;
    for (; i < true_sel.size(); i += kBatchSize) {
        auto a_data =
            _mm512_i32gather_epi32(_mm512_loadu_si512(&true_sel[i]), a_ptr, 4);
        auto b_data =
            _mm512_i32gather_epi32(_mm512_loadu_si512(&true_sel[i]), b_ptr, 4);
        res[i / kBatchSize] = _mm512_cmplt_epi32_mask(a_data, b_data);
    }
    GetSVFromBM(res_buffer->data(), sel_idx, output, true_sel);
}