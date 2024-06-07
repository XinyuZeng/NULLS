#include <arrow/api.h>
#include <arrow/buffer.h>
#include <arrow/compute/api.h>
#include <arrow/testing/random.h>
#include <arrow/util/bit_util.h>
#include <fmt/core.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>

#include "dense_array.h"
#include "spaced_array.h"
#include "utils/bit_util.h"
#include "utils/macros.hpp"
#include "utils/test_util.h"

using namespace null_revisit;

TEST(SpacedArray, CompareWithCol) {
    SpacedArray array_a(rand_arr_a, a_null_bm, kVecSize);
    SpacedArray array_b(rand_arr_b, b_null_bm, kVecSize);
    std::shared_ptr<Buffer> output = *arrow::AllocateEmptyBitmap(kVecSize);

    array_a.CompareWithCol(array_b, output);
    for (int i = 0; i < kVecSize; ++i) {
        EXPECT_EQ(arrow::bit_util::GetBit(output->data(), i), less_result[i]);
    }
    output = *arrow::AllocateEmptyBitmap(kVecSize);

    array_a.CompareWithColScalar(array_b, output);
    for (int i = 0; i < kVecSize; ++i) {
        EXPECT_EQ(arrow::bit_util::GetBit(output->data(), i), less_result[i]);
    }
}

TEST(DenseArray, CompareWithColNaiveScatter) {
    DenseArray array_a(rand_arr_a, a_null_bm, kVecSize);
    DenseArray array_b(rand_arr_b, b_null_bm, kVecSize);
    std::shared_ptr<Buffer> output = *arrow::AllocateEmptyBitmap(kVecSize);

    array_a.CompareWithColNaiveScatter(array_b, output);
    for (int i = 0; i < kVecSize; ++i) {
        EXPECT_EQ(arrow::bit_util::GetBit(output->data(), i), less_result[i]);
    }
}

TEST(DenseArray, CompareWithColAVXScatter) {
    DenseArray array_a(rand_arr_a, a_null_bm, kVecSize);
    DenseArray array_b(rand_arr_b, b_null_bm, kVecSize);
    std::shared_ptr<Buffer> output = *arrow::AllocateEmptyBitmap(kVecSize);

    array_a.CompareWithColAVXScatter(array_b, output);
    for (int i = 0; i < kVecSize; ++i) {
        EXPECT_EQ(arrow::bit_util::GetBit(output->data(), i), less_result[i]);
        // if (arrow::bit_util::GetBit(output->data(), i) != less_result[i]) {
        //   fmt::print("i: {}, a: {}, b: {}, res: {}, arrow: {}\n", i,
        //   rand_arr_a[i], rand_arr_b[i], less_result[i],
        //              arrow::bit_util::GetBit(output->data(), i));
        // }
    }
}
TEST(DenseArray, CompareWithColDirect) {
    DenseArray array_a(rand_arr_a, a_null_bm, kVecSize);
    DenseArray array_b(rand_arr_b, b_null_bm, kVecSize);
    std::shared_ptr<Buffer> output = *arrow::AllocateEmptyBitmap(kVecSize);

    array_a.CompareWithColDirect(array_b, output);
    for (int i = 0; i < kVecSize; ++i) {
        EXPECT_EQ(arrow::bit_util::GetBit(output->data(), i), less_result[i]);
    }
}

TEST(Array, ArrowInit) {
    auto rand = arrow::random::RandomArrayGenerator(1923);
    auto array1 = rand.Numeric<arrow::Int32Type>(null_revisit::kVecSize, 0,
                                                 4096 - 1, 0.2);
    auto array2 = rand.Numeric<arrow::Int32Type>(null_revisit::kVecSize, 0,
                                                 4096 - 1, 0.2);
    auto result_boolean_datum =
        *arrow::compute::CallFunction("less", {array1, array2});
    auto result_boolean_array = std::static_pointer_cast<arrow::BooleanArray>(
        result_boolean_datum.make_array());
    auto numeric_array1 =
        std::static_pointer_cast<arrow::NumericArray<arrow::Int32Type>>(array1);
    auto numeric_array2 =
        std::static_pointer_cast<arrow::NumericArray<arrow::Int32Type>>(array2);

    SpacedArray array_a(array1);
    SpacedArray array_b(array2);
    std::shared_ptr<Buffer> output = *arrow::AllocateEmptyBitmap(kVecSize);

    array_a.CompareWithCol(array_b, output);
    for (int i = 0; i < kVecSize; ++i) {
        EXPECT_EQ(arrow::bit_util::GetBit(output->data(), i),
                  (*result_boolean_array)[i].value_or(0));
    }
    arrow::bit_util::ClearBitmap(output->mutable_data(), 0, kVecSize);

    array_a.CompareWithColScalar(array_b, output);
    for (int i = 0; i < kVecSize; ++i) {
        EXPECT_EQ(arrow::bit_util::GetBit(output->data(), i),
                  (*result_boolean_array)[i].value_or(0));
    }

    DenseArray array_aa(array1);
    DenseArray array_bb(array2);
    arrow::bit_util::ClearBitmap(output->mutable_data(), 0, kVecSize);

    array_aa.CompareWithColAVXExpand(array_bb, output);
    // array_aa.CompareWithColAVXScatter(array_bb, output);
    for (int i = 0; i < kVecSize; ++i) {
        EXPECT_EQ(arrow::bit_util::GetBit(output->data(), i),
                  (*result_boolean_array)[i].value_or(0));
    }
    arrow::bit_util::ClearBitmap(output->mutable_data(), 0, kVecSize);

    array_aa.CompareWithColDirect(array_bb, output);
    for (int i = 0; i < kVecSize; ++i) {
        EXPECT_EQ(arrow::bit_util::GetBit(output->data(), i),
                  (*result_boolean_array)[i].value_or(0));
    }
}

TEST(SpacedDense, AVXAndDirect) {
    DenseArray array_a(rand_arr_a, a_null_bm, kVecSize);
    SpacedArray array_b(rand_arr_b, b_null_bm, kVecSize);
    std::shared_ptr<Buffer> output = *arrow::AllocateEmptyBitmap(kVecSize);

    array_a.CompareWithColAVXScatter(array_b, output);
    for (int i = 0; i < kVecSize; ++i) {
        EXPECT_EQ(arrow::bit_util::GetBit(output->data(), i), less_result[i]);
    }
    arrow::bit_util::ClearBitmap(output->mutable_data(), 0, kVecSize);

    array_a.CompareWithColAVXExpand(array_b, output);
    for (int i = 0; i < kVecSize; ++i) {
        EXPECT_EQ(arrow::bit_util::GetBit(output->data(), i), less_result[i]);
    }
    arrow::bit_util::ClearBitmap(output->mutable_data(), 0, kVecSize);

    array_a.CompareWithColDirect(array_b, output);
    for (int i = 0; i < kVecSize; ++i) {
        EXPECT_EQ(arrow::bit_util::GetBit(output->data(), i), less_result[i]);
    }
}

TEST(DenseArray, CompareWithColAVXExpand) {
    DenseArray array_a(rand_arr_a, a_null_bm, kVecSize);
    DenseArray array_b(rand_arr_b, b_null_bm, kVecSize);
    std::shared_ptr<Buffer> output = *arrow::AllocateEmptyBitmap(kVecSize);

    array_a.CompareWithColAVXExpand(array_b, output);
    for (int i = 0; i < kVecSize; ++i) {
        EXPECT_EQ(arrow::bit_util::GetBit(output->data(), i), less_result[i]);
        if (arrow::bit_util::GetBit(output->data(), i) != less_result[i]) {
            fmt::print("i: {}, a: {}, b: {}, res: {}, arrow: {}\n", i,
                       rand_arr_a[i], rand_arr_b[i], less_result[i],
                       arrow::bit_util::GetBit(output->data(), i));
        }
    }
}

TEST(DenseArray, CompareWithColAVXExpandShift) {
    DenseArray array_a(rand_arr_a, a_null_bm, kVecSize);
    DenseArray array_b(rand_arr_b, b_null_bm, kVecSize);
    std::shared_ptr<Buffer> output = *arrow::AllocateEmptyBitmap(kVecSize);

    array_a.CompareWithColAVXExpandShift(array_b, output);
    for (int i = 0; i < kVecSize; ++i) {
        EXPECT_EQ(arrow::bit_util::GetBit(output->data(), i), less_result[i]);
        if (arrow::bit_util::GetBit(output->data(), i) != less_result[i]) {
            fmt::print("i: {}, a: {}, b: {}, res: {}, arrow: {}\n", i,
                       rand_arr_a[i], rand_arr_b[i], less_result[i],
                       arrow::bit_util::GetBit(output->data(), i));
        }
    }
}

TEST(DenseSpaced, CompareWithColSVPartial) {
    DenseArray array_a(rand_arr_a, a_null_bm, kVecSize);
    SpacedArray array_b(rand_arr_b, b_null_bm, kVecSize);
    SelVector output(kVecSize);
    std::shared_ptr<Buffer> sel_bm = *arrow::AllocateEmptyBitmap(kVecSize);
    int64_t null_count = 0;
    GenerateBitmap(sel_bm->mutable_data(), kVecSize, &null_count, 0.5,
                   01011527);
    auto sel_vec = GetSVFromBM(sel_bm->mutable_data(), kVecSize);
    array_a.CompareWithColSVPartial(array_b, output, sel_vec);
    auto output_bm = GetBMFromSV(output.data(), kVecSize);
    for (int i = 0; i < kVecSize; ++i) {
        EXPECT_EQ(arrow::bit_util::GetBit(output_bm->data(), i),
                  less_result[i] && arrow::bit_util::GetBit(sel_bm->data(), i));
        // fmt::println("i: {}, a: {}, b: {}, res: {}, my: {}", i,
        // rand_arr_a[i], rand_arr_b[i], less_result[i],
        //              arrow::bit_util::GetBit(output_bm->data(), i));
    }
}

TEST(Spaced, CompareWithColSVPartial) {
    SpacedArray array_a(rand_arr_a, a_null_bm, kVecSize);
    SpacedArray array_b(rand_arr_b, b_null_bm, kVecSize);
    SelVector output(kVecSize);
    std::shared_ptr<Buffer> sel_bm = *arrow::AllocateEmptyBitmap(kVecSize);
    int64_t null_count = 0;
    GenerateBitmap(sel_bm->mutable_data(), kVecSize, &null_count, 0.5,
                   01011527);
    auto sel_vec = GetSVFromBM(sel_bm->mutable_data(), kVecSize);
    auto selected_size =
        array_a.CompareWithColSVPartial(array_b, output, sel_vec);

    auto output_bm = GetBMFromSV(output.data(), kVecSize);
    for (int i = 0; i < selected_size; ++i) {
        EXPECT_EQ(arrow::bit_util::GetBit(output_bm->data(), i),
                  less_result[i] && arrow::bit_util::GetBit(sel_bm->data(), i));
    }
}

TEST(Spaced, CompareWithColSVPartialManual) {
    SpacedArray array_a(rand_arr_a, a_null_bm, kVecSize);
    SpacedArray array_b(rand_arr_b, b_null_bm, kVecSize);
    SelVector output(kVecSize);
    std::shared_ptr<Buffer> sel_bm = *arrow::AllocateEmptyBitmap(kVecSize);
    int64_t null_count = 0;
    GenerateBitmap(sel_bm->mutable_data(), kVecSize, &null_count, 0.5,
                   01011527);
    auto sel_vec = GetSVFromBM(sel_bm->mutable_data(), kVecSize);
    array_a.CompareWithColSVPartialManual(array_b, output, sel_vec);

    // for (auto rowid : output) {
    //   std::cout << rowid << " ";
    // }
    // std::cout << std::endl;
    auto output_bm = GetBMFromSV(output.data(), kVecSize);
    for (int i = 0; i < kVecSize; ++i) {
        EXPECT_EQ(arrow::bit_util::GetBit(output_bm->data(), i),
                  less_result[i] && arrow::bit_util::GetBit(sel_bm->data(), i));
    }
}

TEST(Dense, CompareWithColSVPartial) {
    DenseArray array_a(rand_arr_a, a_null_bm, kVecSize);
    DenseArray array_b(rand_arr_b, b_null_bm, kVecSize);
    SelVector output(kVecSize);
    std::shared_ptr<Buffer> sel_bm = *arrow::AllocateEmptyBitmap(kVecSize);
    int64_t null_count = 0;
    GenerateBitmap(sel_bm->mutable_data(), kVecSize, &null_count, 0.5,
                   01011527);
    auto sel_vec = GetSVFromBM(sel_bm->mutable_data(), kVecSize);
    auto selected_size =
        array_a.CompareWithColSVPartial(array_b, output, sel_vec);
    // for (auto rowid : output) {
    //   std::cout << rowid << " ";
    // }
    // std::cout << std::endl;
    auto output_bm = GetBMFromSV(output.data(), kVecSize);
    for (int i = 0; i < selected_size; ++i) {
        EXPECT_EQ(arrow::bit_util::GetBit(output_bm->data(), i),
                  less_result[i] && arrow::bit_util::GetBit(sel_bm->data(), i));
    }
}

TEST(Dense, CompareWithColSVPartialManual) {
    DenseArray array_a(rand_arr_a, a_null_bm, kVecSize);
    DenseArray array_b(rand_arr_b, b_null_bm, kVecSize);
    SelVector output(kVecSize);
    std::shared_ptr<Buffer> sel_bm = *arrow::AllocateEmptyBitmap(kVecSize);
    int64_t null_count = 0;
    GenerateBitmap(sel_bm->mutable_data(), kVecSize, &null_count, 0.5,
                   01011527);
    auto sel_vec = GetSVFromBM(sel_bm->mutable_data(), kVecSize);
    array_a.CompareWithColSVManual(array_b, output, sel_vec);
    // for (auto rowid : output) {
    //   std::cout << rowid << " ";
    // }
    // std::cout << std::endl;
    auto output_bm = GetBMFromSV(output.data(), kVecSize);
    for (int i = 0; i < kVecSize; ++i) {
        EXPECT_EQ(arrow::bit_util::GetBit(output_bm->data(), i),
                  less_result[i] && arrow::bit_util::GetBit(sel_bm->data(), i));
    }
}

constexpr int GROUP_CNT = 32;
auto SumByGroupSetup()
    -> std::tuple<std::vector<int32_t>, std::vector<int64_t>> {
    // generate kVecSize random numbers, range from 0 to 15
    std::vector<int32_t> group_ids(kVecSize);
    std::mt19937 rng(12251126); // Use a fixed seed for deterministic output
    std::uniform_int_distribution<int32_t> dist(0, GROUP_CNT - 1);
    for (int i = 0; i < kVecSize; ++i) {
        group_ids[i] = dist(rng);
    }
    std::vector<int64_t> sum(GROUP_CNT, 0);
    for (int i = 0; i < kVecSize; ++i) {
        sum[group_ids[i]] += a_null_bm[i] ? rand_arr_a[i] : 0;
    }
    return {group_ids, sum};
}

TEST(Array, SumByGroup) {
    DenseArray array_a(rand_arr_a, a_null_bm, kVecSize);
    SpacedArray spaced_array_a(rand_arr_a, a_null_bm, kVecSize);
    std::vector<int64_t> sum(GROUP_CNT, 0);
    auto [group_ids, res_sum] = SumByGroupSetup();

    array_a.SumByGroupBM(group_ids, sum);
    for (int i = 0; i < GROUP_CNT; ++i) {
        EXPECT_EQ(sum[i], res_sum[i]);
    }
    memset(sum.data(), 0, GROUP_CNT * sizeof(int64_t));
    spaced_array_a.SumByGroupBM(group_ids, sum);
    for (int i = 0; i < GROUP_CNT; ++i) {
        EXPECT_EQ(sum[i], res_sum[i]);
    }
    memset(sum.data(), 0, GROUP_CNT * sizeof(int64_t));
    array_a.SumByGroupSV(group_ids, sum);
    for (int i = 0; i < GROUP_CNT; ++i) {
        EXPECT_EQ(sum[i], res_sum[i]);
    }
    memset(sum.data(), 0, GROUP_CNT * sizeof(int64_t));
    spaced_array_a.SumByGroupSV(group_ids, sum);
    for (int i = 0; i < GROUP_CNT; ++i) {
        EXPECT_EQ(sum[i], res_sum[i]);
    }
}

TEST(Dense, DenseToSpaced) {
    DenseArray array_a(rand_arr_a, a_null_bm, kVecSize);
    std::shared_ptr<Buffer> output =
        *arrow::AllocateResizableBuffer(kVecSize * sizeof(int32_t));
    SpacedScatter((int32_t *)array_a.buffers[1]->data(),
                  (int32_t *)output->mutable_data(), array_a.length,
                  array_a.null_count, array_a.buffers[0]->data(), 0);
    for (int i = 0; i < kVecSize; ++i) {
        if (a_null_bm[i]) {
            EXPECT_EQ(rand_arr_a[i], ((int32_t *)output->data())[i]);
        }
    }
    output = *arrow::AllocateResizableBuffer(kVecSize * sizeof(int32_t));
    SpacedScatterAVX512((int32_t *)array_a.buffers[1]->data(),
                        (int32_t *)output->mutable_data(), array_a.length,
                        array_a.null_count,
                        (row_id_type *)array_a.buffers[2]->data());
    for (int i = 0; i < kVecSize; ++i) {
        if (a_null_bm[i]) {
            EXPECT_EQ(rand_arr_a[i], ((int32_t *)output->data())[i]);
        }
    }
    output = *arrow::AllocateResizableBuffer(kVecSize * sizeof(int32_t));
    SpacedExpandAVX((int32_t *)array_a.buffers[1]->data(),
                    (int32_t *)output->mutable_data(), array_a.length,
                    array_a.null_count, array_a.buffers[0]->data());
    for (int i = 0; i < kVecSize; ++i) {
        if (a_null_bm[i]) {
            EXPECT_EQ(rand_arr_a[i], ((int32_t *)output->data())[i]);
        }
    }
    output = *arrow::AllocateResizableBuffer(kVecSize * sizeof(int32_t));
    memcpy(output->mutable_data(), array_a.buffers[1]->data(),
           array_a.buffers[1]->size());
    SpacedExpandAVXInplace((int32_t *)output->mutable_data(), array_a.length,
                           array_a.null_count, array_a.buffers[0]->data());
    for (int i = 0; i < kVecSize; ++i) {
        if (a_null_bm[i]) {
            EXPECT_EQ(rand_arr_a[i], ((int32_t *)output->data())[i]);
        }
    }
}