#include "dense_array.h"

#include <arrow/buffer.h>
#include <arrow/util/bit_block_counter.h>
#include <arrow/util/bit_util.h>
#include <arrow/util/macros.h>
#include <arrow/util/spaced.h>
#include <assert.h>
#include <bits/stdint-uintn.h>
#include <immintrin.h>

#include <memory>

#include "spaced_array.h"
#include "utils/bit_util.h"
#include "utils/macros.hpp"

using namespace null_revisit;

/// \param[out] output Require to be an empty bitmap
void DenseArray::CompareWithColNaiveScatter(
    const DenseArray &other, std::shared_ptr<Buffer> output) const {
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
    auto a_dense = (default_type *)this->buffers[1]->data();
    auto b_dense = (default_type *)other.buffers[1]->data();

    auto a_spaced = *arrow::AllocateBuffer(length * sizeof(default_type));
    auto b_spaced = *arrow::AllocateBuffer(length * sizeof(default_type));

    SpacedScatter<default_type>(
        a_dense, reinterpret_cast<default_type *>(a_spaced->mutable_data()),
        length, null_count, buffers[0]->data(), 0);
    SpacedScatter<default_type>(
        b_dense, reinterpret_cast<default_type *>(b_spaced->mutable_data()),
        length, other.null_count, other.buffers[0]->data(), 0);

    SpacedCompareAVX(a_spaced->data(), b_spaced->data(),
                     this->buffers[0]->data(), other.buffers[0]->data(),
                     output);
}

/// \param[out] output Require to be an empty bitmap
void DenseArray::CompareWithColAVXScatter(
    const DenseArray &other, std::shared_ptr<Buffer> output) const {
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
    auto a_dense = (default_type *)this->buffers[1]->data();
    auto b_dense = (default_type *)other.buffers[1]->data();

    auto a_spaced = *arrow::AllocateBuffer(length * sizeof(default_type));
    auto b_spaced = *arrow::AllocateBuffer(length * sizeof(default_type));

    SpacedScatterAVX512(
        a_dense, reinterpret_cast<default_type *>(a_spaced->mutable_data()),
        length, null_count, (const row_id_type *)(buffers[2]->data()));
    SpacedScatterAVX512(
        b_dense, reinterpret_cast<default_type *>(b_spaced->mutable_data()),
        length, other.null_count,
        (const row_id_type *)(other.buffers[2]->data()));

    SpacedCompareAVX(a_spaced->data(), b_spaced->data(),
                     this->buffers[0]->data(), other.buffers[0]->data(),
                     output);
}

// FIXME: only work for default_type=int32_t
template <bool five_op>
void DenseArray::CompareWithColAVXExpand(const DenseArray &other,
                                         std::shared_ptr<Buffer> output) const {
    // Load 512 bits (16 int32_t values) from a and b into AVX-512 registers
    auto a_ptr = this->buffers[1]->data();
    auto b_ptr = other.buffers[1]->data();
    auto a_bm = (__mmask16 *)this->buffers[0]->data();
    auto b_bm = (__mmask16 *)other.buffers[0]->data();
    auto output_ptr = output->mutable_data();
    auto temp_buf = new (std::align_val_t{64}) uint8_t[192];
    int16_t *a_popcnt = (int16_t *)temp_buf,
            *b_popcnt = (int16_t *)(temp_buf + 64);
    __mmask16 *res = (__mmask16 *)(temp_buf + 128);
    // int16_t a_popcnt[32], b_popcnt[32];
    // __mmask16 res[32];
    // TODO: loop unrolling.
    __m512i number = _mm512_load_epi32(random_array);
    for (int i = 0; i < length; i += 512) {
        auto a_bm_512 = _mm512_load_si512(a_bm);
        auto b_bm_512 = _mm512_load_si512(b_bm);
        _mm512_store_si512(a_popcnt, _mm512_popcnt_epi16(a_bm_512));
        _mm512_store_si512(b_popcnt, _mm512_popcnt_epi16(b_bm_512));
        for (int j = 0; j < 32; ++j) {
            __m512i a_expanded =
                _mm512_maskz_expand_epi32(a_bm[j], _mm512_loadu_si512(a_ptr));
            __m512i b_expanded =
                _mm512_maskz_expand_epi32(b_bm[j], _mm512_loadu_si512(b_ptr));
            if constexpr (five_op) {
                a_expanded = _mm512_mul_epi32(a_expanded, number);
                a_expanded = _mm512_add_epi32(a_expanded, number);
                a_expanded = _mm512_mul_epi32(a_expanded, number);
                a_expanded = _mm512_add_epi32(a_expanded, number);
                a_expanded = _mm512_mul_epi32(a_expanded, number);
                // a_expanded = _mm512_mul_epi32(a_expanded, const_three);
                // a_expanded = _mm512_mul_epi32(a_expanded, const_three);
                // a_expanded = _mm512_mul_epi32(a_expanded, const_three);
                // a_expanded = _mm512_mul_epi32(a_expanded, const_three);

                b_expanded = _mm512_mul_epi32(b_expanded, number);
                b_expanded = _mm512_add_epi32(b_expanded, number);
                b_expanded = _mm512_mul_epi32(b_expanded, number);
                b_expanded = _mm512_add_epi32(b_expanded, number);
                b_expanded = _mm512_mul_epi32(b_expanded, number);
            }
            res[j] = _mm512_cmplt_epi32_mask(a_expanded, b_expanded);
            a_ptr += a_popcnt[j] * 4;
            b_ptr += b_popcnt[j] * 4;
        }
        a_bm += 32;
        b_bm += 32;
        _mm512_store_si512(
            output_ptr, _mm512_and_si512(_mm512_load_si512(res),
                                         _mm512_and_si512(a_bm_512, b_bm_512)));
        output_ptr += (512 >> 3);
    }
    ::operator delete[](temp_buf, std::align_val_t(64));
}

template void
DenseArray::CompareWithColAVXExpand<false>(const DenseArray &,
                                           std::shared_ptr<Buffer>) const;
template void
DenseArray::CompareWithColAVXExpand<true>(const DenseArray &,
                                          std::shared_ptr<Buffer>) const;

void DenseArray::CompareWithColAVXExpandShift(
    const DenseArray &other, std::shared_ptr<Buffer> output) const {
    // Load 512 bits (16 int32_t values) from a and b into AVX-512 registers
    auto a_ptr = (__m512i *)this->buffers[1]->data();
    auto b_ptr = (__m512i *)other.buffers[1]->data();
    auto a_bm = (__mmask16 *)this->buffers[0]->data();
    auto b_bm = (__mmask16 *)other.buffers[0]->data();
    auto output_ptr = output->mutable_data();
    auto temp_buf = new (std::align_val_t{64}) uint8_t[192];
    int16_t *a_popcnt = (int16_t *)temp_buf,
            *b_popcnt = (int16_t *)(temp_buf + 64);
    __mmask16 *res = (__mmask16 *)(temp_buf + 128);
    __m512i a_shifted = _mm512_load_si512(a_ptr),
            b_shifted = _mm512_load_si512(b_ptr);
    __m512i a_cur = _mm512_load_si512(a_ptr++),
            b_cur = _mm512_load_si512(b_ptr++);
    __m512i a_next = _mm512_load_si512(a_ptr++),
            b_next = _mm512_load_si512(b_ptr++);
    int16_t a_popcnt_rsum = 0, b_popcnt_rsum = 0;
    // int16_t a_popcnt[32], b_popcnt[32];
    // __mmask16 res[32];
    // TODO: loop unrolling.
    for (int i = 0; i < length; i += 512) {
        auto a_bm_512 = _mm512_load_si512(a_bm);
        auto b_bm_512 = _mm512_load_si512(b_bm);
        _mm512_store_si512(a_popcnt, _mm512_popcnt_epi16(a_bm_512));
        _mm512_store_si512(b_popcnt, _mm512_popcnt_epi16(b_bm_512));
        for (int j = 0; j < 32; ++j) {
            __m512i a_expanded = _mm512_maskz_expand_epi32(a_bm[j], a_shifted);
            __m512i b_expanded = _mm512_maskz_expand_epi32(b_bm[j], b_shifted);
            res[j] = _mm512_cmplt_epi32_mask(a_expanded, b_expanded);
            a_popcnt_rsum += a_popcnt[j];
            b_popcnt_rsum += b_popcnt[j];
            // prefetcht0(a_ptr + 1);
            // prefetcht0(b_ptr + 1);
            if (ARROW_PREDICT_FALSE(a_popcnt_rsum >= 16)) {
                a_cur = a_next;
                a_next = _mm512_load_si512(a_ptr++);
                a_popcnt_rsum -= 16;
            }
            if (ARROW_PREDICT_FALSE(b_popcnt_rsum >= 16)) {
                b_cur = b_next;
                b_next = _mm512_load_si512(b_ptr++);
                b_popcnt_rsum -= 16;
            }
            // const uint8_t tmp = const_cast<uint8_t&>(a_popcnt_rsum);
            // a_shifted = _mm512_alignr_epi32(a_next, a_cur, tmp);
            a_shifted = _mm512_alignrvar_epi32(a_next, a_cur, a_popcnt_rsum);
            // const uint8_t tmp2 = const_cast<uint8_t&>(a_popcnt_rsum);
            // b_shifted = _mm512_alignr_epi32(b_next, b_cur, tmp2);
            b_shifted = _mm512_alignrvar_epi32(b_next, b_cur, b_popcnt_rsum);
        }
        a_bm += 32;
        b_bm += 32;
        _mm512_store_si512(
            output_ptr, _mm512_and_si512(_mm512_load_si512(res),
                                         _mm512_and_si512(a_bm_512, b_bm_512)));
        output_ptr += (512 >> 3);
    }
    ::operator delete[](temp_buf, std::align_val_t(64));
}

// TODO: SV and BM should be abstracted as classes instead of using Buffer.
void DenseArray::CompareWithColDirect(const DenseArray &other,
                                      std::shared_ptr<Buffer> output) const {
    auto bm = output->mutable_data();
    auto sv_a = (row_id_type *)this->buffers[2]->data();
    auto sv_b = (row_id_type *)other.buffers[2]->data();

    auto a_ptr = (default_type *)this->buffers[1]->data();
    auto b_ptr = (default_type *)other.buffers[1]->data();
    auto idx_decode_a = length - null_count;
    auto idx_decode_b = other.length - other.null_count;
    for (int i = 0, j = 0; i < idx_decode_a && j < idx_decode_b;) {
        if (sv_a[i] < sv_b[j]) {
            ++i;
        } else if (sv_a[i] > sv_b[j]) {
            ++j;
        } else {
            arrow::bit_util::SetBitTo(bm, sv_a[i], a_ptr[i] < b_ptr[j]);
            ++i;
            ++j;
        }
    }
}

void DenseArray::CompareWithColAVXExpand(const SpacedArray &other,
                                         std::shared_ptr<Buffer> output) const {
    auto a_ptr = this->buffers[1]->data();
    auto b_ptr = (__m512i *)other.buffers[1]->data();
    auto a_bm = (__mmask16 *)this->buffers[0]->data();
    auto b_bm = (__mmask16 *)other.buffers[0]->data();
    auto output_ptr = output->mutable_data();
    auto temp_buf = new (std::align_val_t{64}) uint8_t[192];
    int16_t *a_popcnt = (int16_t *)temp_buf,
            *b_popcnt = (int16_t *)(temp_buf + 64);
    __mmask16 *res = (__mmask16 *)(temp_buf + 128);
    // int16_t a_popcnt[32], b_popcnt[32];
    // __mmask16 res[32];
    // TODO: loop unrolling.
    for (int i = 0; i < length; i += 512) {
        auto a_bm_512 = _mm512_load_si512(a_bm);
        auto b_bm_512 = _mm512_load_si512(b_bm);
        _mm512_store_si512(a_popcnt, _mm512_popcnt_epi16(a_bm_512));
        for (int j = 0; j < 32; ++j) {
            __m512i a_expanded =
                _mm512_maskz_expand_epi32(a_bm[j], _mm512_loadu_si512(a_ptr));
            res[j] = _mm512_cmplt_epi32_mask(a_expanded, *b_ptr++);
            a_ptr += a_popcnt[j] * 4;
        }
        a_bm += 32;
        b_bm += 32;
        _mm512_store_si512(
            output_ptr, _mm512_and_si512(_mm512_load_si512(res),
                                         _mm512_and_si512(a_bm_512, b_bm_512)));
        output_ptr += (512 >> 3);
    }
    ::operator delete[](temp_buf, std::align_val_t(64));
}

void DenseArray::CompareWithColAVXScatter(
    const SpacedArray &other, std::shared_ptr<Buffer> output) const {
    auto a_dense = (default_type *)this->buffers[1]->data();

    auto a_spaced = *arrow::AllocateBuffer(length * sizeof(default_type));

    SpacedScatterAVX512(
        a_dense, reinterpret_cast<default_type *>(a_spaced->mutable_data()),
        length, null_count, (const row_id_type *)(buffers[2]->data()));
    SpacedCompareAVX(a_spaced->data(), other.buffers[1]->data(),
                     this->buffers[0]->data(), other.buffers[0]->data(),
                     output);
}

void DenseArray::CompareWithColDirect(const SpacedArray &other,
                                      std::shared_ptr<Buffer> output) const {
    auto bm = output->mutable_data();
    auto sv_a = (row_id_type *)this->buffers[2]->data();
    auto bm_b = other.buffers[0]->data();
    auto a_ptr = (default_type *)this->buffers[1]->data();
    auto b_ptr = (default_type *)other.buffers[1]->data();
    auto idx_decode_a = length - null_count;
    for (int i = 0; i < idx_decode_a; ++i) {
        // TODO: gather first, then AVX
        arrow::bit_util::SetBitTo(bm, sv_a[i],
                                  a_ptr[i] < b_ptr[sv_a[i]] &&
                                      arrow::bit_util::GetBit(bm_b, sv_a[i]));
    }
}

void DenseArray::SumByGroupBM(const std::vector<int32_t> &group_ids,
                              std::vector<int64_t> &sum) const {
    auto a_ptr = (default_type *)this->buffers[1]->data();
    int idx = 0;
    arrow::internal::VisitBitBlocksVoid(
        buffers[0]->data(), 0, length,                         // NOLINT
        [&](int64_t i) { sum[group_ids[idx++]] += *a_ptr++; }, // NOLINT
        [&]() { idx++; });
}

void DenseArray::SumByGroupSV(const std::vector<int32_t> &group_ids,
                              std::vector<int64_t> &sum) const {
    auto a_ptr = (default_type *)this->buffers[1]->data();
    auto a_sv = (row_id_type *)this->buffers[2]->data();
    auto idx_decode_a = length - null_count;
    for (int i = 0; i < idx_decode_a; ++i) {
        sum[group_ids[a_sv[i]]] += a_ptr[i];
    }
}

void DenseArray::CompareWithColSVPartial(const SpacedArray &other,
                                         SelVector &output,
                                         SelVector &sel_vec) const {
    auto a_bm = this->buffers[0]->data();
    auto b_bm = other.buffers[0]->data();
    auto a_ptr = (default_type *)this->buffers[1]->data();
    auto b_ptr = (default_type *)other.buffers[1]->data();
    int true_idx = 0;
    int a_idx = 0;
    int output_idx = 0;
    auto bit_run_reader = arrow::internal::SetBitRunReader(a_bm, 0, length);
    auto sel_vec_checker = LinearSelVecChecker(&sel_vec);
    while (true) {
        auto run = bit_run_reader.NextRun();
        if (run.length == 0) {
            break;
        }
        true_idx = run.position;
        for (int i = 0; i < run.length; ++i) {
            if (arrow::bit_util::GetBit(b_bm, true_idx) &&
                sel_vec_checker.Check(true_idx) &&
                a_ptr[a_idx + i] < b_ptr[true_idx]) {
                output[output_idx++] = true_idx;
            }
            true_idx++;
        }
        a_idx += run.length;
    }
}

int DenseArray::CompareWithColSVPartial(const DenseArray &other,
                                        SelVector &output,
                                        SelVector &sel_vec) const {
    auto a_bm = this->buffers[0]->data();
    auto b_bm = other.buffers[0]->data();
    auto a_ptr = (default_type *)this->buffers[1]->data();
    auto b_ptr = (default_type *)other.buffers[1]->data();
    auto a_rsum = (prefix_sum_type *)this->buffers[3]->data();
    auto b_rsum = (prefix_sum_type *)other.buffers[3]->data();
    int output_idx = 0;
    for (size_t i = 0; i < sel_vec.size(); ++i) {
        auto row_id = sel_vec[i];
        output[output_idx] = row_id;
        bool a = arrow::bit_util::GetBit(a_bm, row_id);
        bool b = arrow::bit_util::GetBit(b_bm, row_id);
        bool c = a_ptr[GetNonNullPos(a_bm, a_rsum, row_id)] <
                 b_ptr[GetNonNullPos(b_bm, b_rsum, row_id)];
        output_idx += a & b & c;
    }
    return output_idx;
}
template <bool eval_filter_first>
int DenseArray::CompareWithColSVPartialBranch(const DenseArray &other,
                                              SelVector &output,
                                              SelVector &sel_vec) const {
    auto a_bm = this->buffers[0]->data();
    auto b_bm = other.buffers[0]->data();
    auto a_ptr = (default_type *)this->buffers[1]->data();
    auto b_ptr = (default_type *)other.buffers[1]->data();
    auto a_rsum = (prefix_sum_type *)this->buffers[3]->data();
    auto b_rsum = (prefix_sum_type *)other.buffers[3]->data();
    int output_idx = 0;
    for (size_t i = 0; i < sel_vec.size(); ++i) {
        auto row_id = sel_vec[i];
        if constexpr (eval_filter_first) {
            if (a_ptr[GetNonNullPos(a_bm, a_rsum, row_id)] <
                    b_ptr[GetNonNullPos(b_bm, b_rsum, row_id)] &&
                arrow::bit_util::GetBit(a_bm, row_id) &&
                arrow::bit_util::GetBit(b_bm, row_id)) {
                output[output_idx++] = row_id;
            }
        } else {
            if (arrow::bit_util::GetBit(a_bm, row_id) &&
                arrow::bit_util::GetBit(b_bm, row_id) &&
                a_ptr[GetNonNullPos(a_bm, a_rsum, row_id)] <
                    b_ptr[GetNonNullPos(b_bm, b_rsum, row_id)]) {
                output[output_idx++] = row_id;
            }
        }
    }
    return output_idx;
}
template int DenseArray::CompareWithColSVPartialBranch<false>(
    const DenseArray &, SelVector &output, SelVector &sel_vec) const;
template int DenseArray::CompareWithColSVPartialBranch<true>(
    const DenseArray &, SelVector &output, SelVector &sel_vec) const;

int DenseArray::CompareWithColSVPartialFlat(DenseArray &other,
                                            SelVector &output,
                                            SelVector &sel_vec) {
    auto a_bm = this->buffers[0]->data();
    auto b_bm = other.buffers[0]->data();
    auto a_ptr = (default_type *)this->buffers[1]->data();
    auto b_ptr = (default_type *)other.buffers[1]->data();
    // analogy: DuckDB's ToUnifiedFormat
    auto a_flat = *arrow::AllocateBuffer(length * sizeof(default_type));
    auto b_flat = *arrow::AllocateBuffer(length * sizeof(default_type));
    auto a_flat_ptr = reinterpret_cast<default_type *>(a_flat->mutable_data());
    auto b_flat_ptr = reinterpret_cast<default_type *>(b_flat->mutable_data());
    this->buffers.push_back(std::move(a_flat));
    other.buffers.push_back(std::move(b_flat));
    SpacedScatter(a_ptr, a_flat_ptr, length, null_count, a_bm, 0);
    SpacedScatter(b_ptr, b_flat_ptr, length, other.null_count, b_bm, 0);
    int output_idx = 0;
    for (auto row_id : sel_vec) {
        output[output_idx] = row_id;
        bool a = arrow::bit_util::GetBit(a_bm, row_id);
        bool b = arrow::bit_util::GetBit(b_bm, row_id);
        bool c = a_flat_ptr[row_id] < b_flat_ptr[row_id];
        output_idx += a && b && c;
    }
    return output_idx;
}

int DenseArray::CompareWithColSVPartialFlatBranch(DenseArray &other,
                                                  SelVector &output,
                                                  SelVector &sel_vec) {
    auto a_bm = this->buffers[0]->data();
    auto b_bm = other.buffers[0]->data();
    auto a_ptr = (default_type *)this->buffers[1]->data();
    auto b_ptr = (default_type *)other.buffers[1]->data();
    // analogy: DuckDB's ToUnifiedFormat
    auto a_flat = *arrow::AllocateBuffer(length * sizeof(default_type));
    auto b_flat = *arrow::AllocateBuffer(length * sizeof(default_type));
    auto a_flat_ptr = reinterpret_cast<default_type *>(a_flat->mutable_data());
    auto b_flat_ptr = reinterpret_cast<default_type *>(b_flat->mutable_data());
    this->buffers.push_back(std::move(a_flat));
    other.buffers.push_back(std::move(b_flat));
    SpacedScatter(a_ptr, a_flat_ptr, length, null_count, a_bm, 0);
    SpacedScatter(b_ptr, b_flat_ptr, length, other.null_count, b_bm, 0);
    int output_idx = 0;
    for (auto row_id : sel_vec) {
        output[output_idx] = row_id;
        output_idx += arrow::bit_util::GetBit(a_bm, row_id) &&
                      arrow::bit_util::GetBit(b_bm, row_id) &&
                      a_flat_ptr[row_id] < b_flat_ptr[row_id];
    }
    return output_idx;
}

void DenseArray::CompareWithColSVManual(const DenseArray &other,
                                        SelVector &output,
                                        SelVector &sel_vec) const {
    auto a_bm = this->buffers[0]->data();
    auto b_bm = other.buffers[0]->data();
    auto a_ptr = (default_type *)this->buffers[1]->data();
    auto b_ptr = (default_type *)other.buffers[1]->data();
    auto a_rsum = (prefix_sum_type *)this->buffers[3]->data();
    auto b_rsum = (prefix_sum_type *)other.buffers[3]->data();
    int sel_idx = 0;
    SelVector true_sel(sel_vec.size());
    SelVector true_sel_in_a_dense(sel_vec.size());
    SelVector true_sel_in_b_dense(sel_vec.size());
    for (size_t i = 0; i < sel_vec.size(); ++i) {
        auto row_id = sel_vec[i];
        true_sel[sel_idx] = row_id;
        true_sel_in_a_dense[sel_idx] = GetNonNullPos(a_bm, a_rsum, row_id);
        true_sel_in_b_dense[sel_idx] = GetNonNullPos(b_bm, b_rsum, row_id);
        bool a = arrow::bit_util::GetBit(a_bm, row_id);
        bool b = arrow::bit_util::GetBit(b_bm, row_id);
        sel_idx += a && b;
    }
    if (true_sel.size() % 16 != 0) {
        true_sel.resize(true_sel.size() + 16 - true_sel.size() % 16);
        true_sel_in_a_dense.resize(true_sel.size() + 16 - true_sel.size() % 16);
        true_sel_in_b_dense.resize(true_sel.size() + 16 - true_sel.size() % 16);
    }
    auto res_buffer = *arrow::AllocateResizableBuffer(
        arrow::bit_util::BytesForBits(true_sel.size()));
    __mmask16 *res = (__mmask16 *)(res_buffer->mutable_data());
    int i = 0;
    for (; i < true_sel.size(); i += kBatchSize) {
        auto a_data = _mm512_i32gather_epi32(
            _mm512_loadu_si512(&true_sel_in_a_dense[i]), a_ptr, 4);
        auto b_data = _mm512_i32gather_epi32(
            _mm512_loadu_si512(&true_sel_in_b_dense[i]), b_ptr, 4);
        res[i / kBatchSize] = _mm512_cmplt_epi32_mask(a_data, b_data);
    }
    GetSVFromBM(res_buffer->data(), sel_idx, output, true_sel);
}