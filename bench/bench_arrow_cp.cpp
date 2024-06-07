#include <arrow/type.h>
#include <benchmark/benchmark.h>
#include <bits/stdint-intn.h>
#include <bits/stdint-uintn.h>
#include <x86intrin.h>

#include <vector>

#include "arrow/array.h"
#include "arrow/compute/api.h"
#include "arrow/testing/random.h"
#include "bench_helper.h"
#include "parquet/encoding.h"
#include "parquet/types.h"
#include "utils/macros.hpp"

#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__GNUC__) || defined(__llvm__)
#include <immintrin.h>
#endif

static void BM_AGG(benchmark::State& state) {
  SetupBenchmarks setup(state);
  auto type = static_cast<NULLStorage>(state.range(1));
  std::shared_ptr<arrow::Scalar> increment = std::make_shared<arrow::Int32Scalar>(1);
  arrow::Datum incremented_datum;
  if (type == NULLStorage::SPACED) {
    for (auto _ : state) {
      benchmark::DoNotOptimize(incremented_datum = arrow::compute::Sum(setup.array1).ValueOrDie());
    }
  } else {
    for (auto _ : state) {
      benchmark::DoNotOptimize(incremented_datum = arrow::compute::Sum(setup.dense_array1).ValueOrDie());
    }
  }
}

BENCHMARK(BM_AGG)->Apply(BM_Args);

static void BM_ADD(benchmark::State& state) {
  SetupBenchmarks setup(state);
  auto type = static_cast<NULLStorage>(state.range(1));
  std::shared_ptr<arrow::Scalar> increment = std::make_shared<arrow::Int32Scalar>(1);
  arrow::Datum incremented_datum;
  if (type == NULLStorage::SPACED) {
    for (auto _ : state) {
      incremented_datum = arrow::compute::Add(setup.array1, increment).ValueOrDie();
    }
  } else {
    for (auto _ : state) {
      incremented_datum = arrow::compute::Add(setup.dense_array1, increment).ValueOrDie();
    }
  }
}

BENCHMARK(BM_ADD)->Apply(BM_Args);

// not correctbench_arrow_cp
static void BM_FILTER(benchmark::State& state) {
  SetupBenchmarks setup(state);
  auto type = static_cast<NULLStorage>(state.range(1));
  arrow::Datum result_boolean_datum;
  std::shared_ptr<arrow::Scalar> scalar_val = arrow::MakeScalar(arrow::int32(), 5).ValueOrDie();

  if (type == NULLStorage::SPACED) {
    for (auto _ : state) {
      result_boolean_datum = arrow::compute::CallFunction("less", {setup.array1, scalar_val}).ValueOrDie();
    }
  } else {
    for (auto _ : state) {
      result_boolean_datum = arrow::compute::CallFunction("less", {setup.dense_array1, scalar_val}).ValueOrDie();
      if (setup.valid_bits1 != nullptr) {
        auto result_boolean_array = std::static_pointer_cast<arrow::BooleanArray>(result_boolean_datum.make_array());
        // pext
        auto dense_bm = result_boolean_array->data()->buffers[1]->data();
        auto dense_bm_size = result_boolean_array->data()->buffers[1]->size();
        for (int i = 0; i < dense_bm_size; i += 8) {
          benchmark::DoNotOptimize(_pdep_u64(*reinterpret_cast<const uint64_t*>(dense_bm + i),
                                             *reinterpret_cast<const uint64_t*>(setup.valid_bits1 + i)));
        }
      }
    }
  }
}
BENCHMARK(BM_FILTER)->Apply(BM_Args);

BENCHMARK_MAIN();