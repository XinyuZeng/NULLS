#pragma once
#include <arrow/array.h>
#include <arrow/buffer.h>
#include <arrow/type_fwd.h>
#include <benchmark/benchmark.h>
#include <stdlib.h> /* srand, rand */

#include <cstdint>
#include <limits>
#include <memory>

#include "arrow/testing/random.h"
#include "parquet/encoding.h"
#include "utils/macros.hpp"

enum class NULLStorage { DENSE, SPACED };

static void BM_Args(benchmark::internal::Benchmark *bench) {
    std::vector<int64_t> null_percent = {1,  5,  10, 20, 30, 40, 50,
                                         60, 70, 80, 90, 95, 100};
    std::vector<int64_t> type = {int(NULLStorage::DENSE),
                                 int(NULLStorage::SPACED)};
    std::vector<int64_t> dist = {1, 2, 3, 4, 5}; // control the skewness of null
    std::vector<int64_t> num_ops = {1, 2, 3, 4, 5};
    bench->ArgsProduct({null_percent, type});
}

template <bool diff_type = false>
static void BM_MEM_Args(benchmark::internal::Benchmark *bench) {
    constexpr auto million = 1024 * 1024;
    std::vector<int64_t> null_percent = {1,  5,  10, 20, 30, 40, 50, 60, 70,
                                         80, 90, 92, 95, 96, 97, 98, 99};
    std::vector<int64_t> data_size = {2048, million, 2 * million, 4 * million,
                                      8 * million};
    std::vector<int64_t> type = {int(NULLStorage::DENSE),
                                 int(NULLStorage::SPACED)};
    std::vector<int64_t> dist = {1, 2, 3, 4, 5}; // control the skewness of null
    std::vector<int64_t> num_ops = {1, 2, 3, 4, 5};
    if constexpr (diff_type) {
        bench->ArgsProduct({null_percent, type});
    } else {
        bench->ArgsProduct({null_percent, data_size});
    }
}

static void BM_BLOCKSIZE_Args(benchmark::internal::Benchmark *bench) {
    constexpr auto million = 1024 * 1024;
    std::vector<int64_t> null_percent = {1, 10, 40, 50, 60, 90, 99};
    std::vector<int64_t> data_size = {8 * million};
    bench->ArgsProduct({null_percent, data_size});
}

#define UPDATE_COUNTERS                                                        \
    do {                                                                       \
        state.counters["null_percent"] = state.range(0);                       \
        state.counters["data_size"] = state.range(1);                          \
    } while (0);

class SetupBenchmarks {
  public:
    using ArrowType = arrow::Int32Type;
    using ArrayType = typename arrow::TypeTraits<ArrowType>::ArrayType;
    using ParquetType = parquet::Int32Type;
    static constexpr auto GROUP_CNT = 64;

    explicit SetupBenchmarks(benchmark::State &_state) : state_(_state) {
        auto null_percent = static_cast<double>(state_.range(0)) / 100.0;
        auto rand = arrow::random::RandomArrayGenerator(1923);

        array1 = rand.Numeric<ArrowType>(null_revisit::kVecSize, 0, 4096 - 1,
                                         null_percent);
        array2 = rand.Numeric<ArrowType>(null_revisit::kVecSize, 0, 4096 - 1,
                                         null_percent);
        valid_bits1 = array1->null_bitmap_data();
        null_count1 = static_cast<int>(array1->null_count());

        auto array_actual =
            arrow::internal::checked_pointer_cast<ArrayType>(array1);
        raw_values1 = array_actual->raw_values();

        auto encoder =
            parquet::MakeTypedEncoder<ParquetType>(parquet::Encoding::PLAIN);
        encoder->Put(*array1);
        std::shared_ptr<arrow::Buffer> buf = encoder->FlushValues();

        dense_array1 =
            std::make_shared<ArrayType>(array1->length() - null_count1, buf);

        group_ids = std::vector<int32_t>(array1->length());
        std::mt19937 rng(12251126); // Use a fixed seed for deterministic output
        std::uniform_int_distribution<int32_t> dist(0, GROUP_CNT - 1);
        for (int i = 0; i < array1->length(); ++i) {
            group_ids[i] = dist(rng);
        }
    }

    std::shared_ptr<arrow::Array> array1;
    const uint8_t *valid_bits1;
    int null_count1;
    const int32_t *raw_values1;
    std::shared_ptr<ArrayType> dense_array1;
    std::shared_ptr<arrow::Scalar> increment;

    std::shared_ptr<arrow::Array> array2;

    std::vector<int32_t> group_ids;

  private:
    benchmark::State &state_;
};

// w/o Arrow dependency, and vary NULL distribution
class SetupBenchmarksV2 {
  public:
    using CType = uint32_t;

    explicit SetupBenchmarksV2(benchmark::State &_state) : state_(_state) {
        auto null_percent = static_cast<double>(state_.range(0)) / 100.0;
        bm_buffer_ = *arrow::AllocateEmptyBitmap(null_revisit::kVecSize,
                                                 arrow::kDefaultBufferAlignment,
                                                 arrow::default_memory_pool());
        // gen_bitmap
        spaced_buffer_ = *arrow::AllocateBuffer(
            null_revisit::kVecSize * sizeof(CType),
            arrow::kDefaultBufferAlignment, arrow::default_memory_pool());
        // put values in spaced_buffer_
        dense_buffer_ = *arrow::AllocateBuffer(
            (null_revisit::kVecSize - null_count) * sizeof(CType),
            arrow::kDefaultBufferAlignment, arrow::default_memory_pool());
        // put values in dense_buffer_
    }
    const uint8_t *valid_bits;
    int null_count;
    const int32_t *spaced_values;
    const int32_t *dense_values;

  private:
    benchmark::State &state_;
    std::shared_ptr<arrow::Buffer> bm_buffer_;
    std::shared_ptr<arrow::Buffer> spaced_buffer_;
    std::shared_ptr<arrow::Buffer> dense_buffer_;
};

template <typename ArrayTypeA, typename ArrayTypeB> class SetupBenchmarksBatch {
  public:
    using ArrowType = arrow::Int32Type;
    using ArrayType = typename arrow::TypeTraits<ArrowType>::ArrayType;
    using ParquetType = parquet::Int32Type;
    static constexpr auto GROUP_CNT = 64;

    explicit SetupBenchmarksBatch(benchmark::State &_state) : state_(_state) {
        auto null_percent = static_cast<double>(state_.range(0)) / 100.0;
        // auto data_size = 1024 * 1024;  // FIXME: hardcode max.
        auto data_size = state_.range(1);
        assert(data_size % null_revisit::kVecSize == 0);
        // auto rd = arrow::random::RandomArrayGenerator(1923);
        std::mt19937 g(0x20240500);
        std::uniform_int_distribution<int> uniform;

        for (int i = 0; i < data_size / null_revisit::kVecSize; ++i) {
            auto rd = arrow::random::RandomArrayGenerator(uniform(g));
            array_a.push_back(
                std::make_shared<ArrayTypeA>(rd.Numeric<ArrowType>(
                    null_revisit::kVecSize, 0, 4096 - 1, null_percent)));
            rd = arrow::random::RandomArrayGenerator(uniform(g));
            array_b.push_back(
                std::make_shared<ArrayTypeB>(rd.Numeric<ArrowType>(
                    null_revisit::kVecSize, 0, 4096 - 1, null_percent)));
        }

        group_ids = std::vector<std::vector<int32_t>>(array_a.size());
        for (int i = 0; i < array_a.size(); ++i) {
            group_ids[i] = std::vector<int32_t>(array_a[i]->length);
        }
        std::mt19937 rng(12251126); // Use a fixed seed for deterministic output
        std::uniform_int_distribution<int32_t> dist(0, GROUP_CNT - 1);
        for (int i = 0; i < array_a.size(); ++i) {
            for (int j = 0; j < array_a[i]->length; ++j) {
                group_ids[i][j] = dist(rng);
            }
        }
    }

    std::vector<std::shared_ptr<ArrayTypeA>> array_a;

    std::vector<std::shared_ptr<ArrayTypeB>> array_b;

    std::vector<std::vector<int32_t>> group_ids;

  private:
    benchmark::State &state_;
};