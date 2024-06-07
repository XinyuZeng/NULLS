
#include <benchmark/benchmark.h>

#include <cstdint>
#include <memory>
#include <random>

#include "array_base.h"
#include "bench_helper.h"
#include "dense_array.h"
#include "spaced_array.h"
#include "special_value_array.h"
#include "utils/bit_util.h"
#include "utils/macros.hpp"
#include "utils/test_util.h"

using namespace null_revisit;

template <typename ArrayType, bool include_alloc_cost = false>
static void BM_Filter(benchmark::State &state) {
    SetupBenchmarksBatch<ArrayType, ArrayType> setup(state);
    if constexpr (include_alloc_cost) {
        for (auto _ : state) {
            std::shared_ptr<Buffer> output =
                *arrow::AllocateEmptyBitmap(kVecSize);
            for (int i = 0; i < setup.array_a.size(); ++i) {
                setup.array_a[i]->CompareWithCol(*setup.array_b[i], output);
            }
            benchmark::DoNotOptimize(output);
            benchmark::ClobberMemory();
        }
    } else {
        std::shared_ptr<Buffer> output = *arrow::AllocateEmptyBitmap(kVecSize);
        for (auto _ : state) {
            for (int i = 0; i < setup.array_a.size(); ++i) {
                setup.array_a[i]->CompareWithCol(*setup.array_b[i], output);
            }
            benchmark::ClobberMemory();
        }
    }
    UPDATE_COUNTERS
}
static void BM_Spaced_filter_AVX(benchmark::State &state) {
    BM_Filter<SpacedArray>(state);
}
static void BM_Dense_filter_memcpy(benchmark::State &state) {
    BM_Filter<DenseArray>(state);
}
static void BM_Spaced_filter_Alloc(benchmark::State &state) {
    BM_Filter<SpacedArray, true>(state);
}
static void BM_Dense_filter_Alloc(benchmark::State &state) {
    BM_Filter<DenseArray, true>(state);
}

template <bool include_alloc_cost = false, bool with_SV_convert = false>
static void BM_Dense_filter_AVXScatter(benchmark::State &state) {
    SetupBenchmarksBatch<DenseArray, DenseArray> setup(state);
    if constexpr (include_alloc_cost) {
        for (auto _ : state) {
            if constexpr (with_SV_convert) {
                for (int i = 0; i < setup.array_a.size(); ++i) {
                    benchmark::DoNotOptimize(GetSVBufferFromBM(
                        setup.array_a[i]->buffers[0]->data(), kVecSize));
                    benchmark::DoNotOptimize(GetSVBufferFromBM(
                        setup.array_b[i]->buffers[0]->data(), kVecSize));
                }
            }
            std::shared_ptr<Buffer> output =
                *arrow::AllocateEmptyBitmap(kVecSize);
            for (int i = 0; i < setup.array_a.size(); ++i) {
                setup.array_a[i]->CompareWithColAVXScatter(*setup.array_b[i],
                                                           output);
            }
            benchmark::DoNotOptimize(output);
            benchmark::ClobberMemory();
        }
    } else {
        std::shared_ptr<Buffer> output = *arrow::AllocateEmptyBitmap(kVecSize);
        for (auto _ : state) {
            if constexpr (with_SV_convert) {
                for (int i = 0; i < setup.array_a.size(); ++i) {
                    benchmark::DoNotOptimize(GetSVBufferFromBM(
                        setup.array_a[i]->buffers[0]->data(), kVecSize));
                    benchmark::DoNotOptimize(GetSVBufferFromBM(
                        setup.array_b[i]->buffers[0]->data(), kVecSize));
                }
            }
            for (int i = 0; i < setup.array_a.size(); ++i) {
                setup.array_a[i]->CompareWithColAVXScatter(*setup.array_b[i],
                                                           output);
            }
            benchmark::ClobberMemory();
        }
    }
    UPDATE_COUNTERS
}
static void BM_Dense_filter_AVXScatter_Alloc(benchmark::State &state) {
    BM_Dense_filter_AVXScatter<true>(state);
}
static void BM_Dense_filter_AVXScatter_BMtoSV(benchmark::State &state) {
    BM_Dense_filter_AVXScatter<false, true>(state);
}

template <typename ArrayTypeA, typename ArrayTypeB, typename Func>
static void BM_filter(benchmark::State &state, Func func) {
    SetupBenchmarksBatch<ArrayTypeA, ArrayTypeB> setup(state);
    std::shared_ptr<Buffer> output = *arrow::AllocateEmptyBitmap(kVecSize);
    for (auto _ : state) {
        for (int i = 0; i < setup.array_a.size(); ++i) {
            (*(setup.array_a[i]).*func)(*setup.array_b[i], output);
        }
        benchmark::ClobberMemory();
    }
    UPDATE_COUNTERS
}
static void BM_Spaced_filter_Scalar(benchmark::State &state) {
    BM_filter<SpacedArray, SpacedArray>(state,
                                        &SpacedArray::CompareWithColScalar);
}

static void BM_Dense_filter_direct(benchmark::State &state) {
    BM_filter<DenseArray, DenseArray,
              void (DenseArray::*)(const DenseArray &, std::shared_ptr<Buffer>)
                  const>(state, &DenseArray::CompareWithColDirect);
}

BENCHMARK(BM_Spaced_filter_AVX)->Apply(BM_MEM_Args);

static void BM_Spaced_filter_AVX_TenOp(benchmark::State &state) {
    BM_filter<SpacedArray, SpacedArray,
              void (SpacedArray::*)(const SpacedArray &,
                                    std::shared_ptr<Buffer>) const>(
        state, &SpacedArray::CompareWithCol<true>);
}
// BENCHMARK(BM_Spaced_filter_AVX_TenOp)->Apply(BM_MEM_Args);

// BENCHMARK(BM_Spaced_filter_Scalar)->Apply(BM_MEM_Args);
// BENCHMARK(BM_Dense_filter_memcpy)->Apply(BM_MEM_Args);
// BENCHMARK(BM_Spaced_filter_Alloc)->Apply(BM_MEM_Args);
// BENCHMARK(BM_Dense_filter_Alloc)->Apply(BM_MEM_Args);
// BENCHMARK(BM_Dense_filter_AVXScatter)->Apply(BM_MEM_Args);
// BENCHMARK(BM_Dense_filter_AVXScatter_BMtoSV)->Apply(BM_MEM_Args);
// BENCHMARK(BM_Dense_filter_AVXScatter_Alloc)->Apply(BM_MEM_Args);
// BENCHMARK(BM_Dense_filter_direct)->Apply(BM_MEM_Args);

static void BM_DNS_filter_AVXScatter(benchmark::State &state) {
    BM_filter<DenseArray, SpacedArray,
              void (DenseArray::*)(const SpacedArray &, std::shared_ptr<Buffer>)
                  const>(state, &DenseArray::CompareWithColAVXScatter);
}
static void BM_DNS_filter_AVXExpand(benchmark::State &state) {
    BM_filter<DenseArray, SpacedArray,
              void (DenseArray::*)(const SpacedArray &, std::shared_ptr<Buffer>)
                  const>(state, &DenseArray::CompareWithColAVXExpand);
}
static void BM_DNS_filter_direct(benchmark::State &state) {
    BM_filter<DenseArray, SpacedArray,
              void (DenseArray::*)(const SpacedArray &, std::shared_ptr<Buffer>)
                  const>(state, &DenseArray::CompareWithColDirect);
}
// BENCHMARK(BM_DNS_filter_AVXScatter)->Apply(BM_MEM_Args);
// BENCHMARK(BM_DNS_filter_AVXExpand)->Apply(BM_MEM_Args);
// BENCHMARK(BM_DNS_filter_direct)->Apply(BM_MEM_Args);

static void BM_Dense_filter_AVXExpandShift(benchmark::State &state) {
    BM_filter<DenseArray, DenseArray>(
        state, &DenseArray::CompareWithColAVXExpandShift);
}
// BENCHMARK(BM_Dense_filter_AVXExpandShift)->Apply(BM_MEM_Args);

static void BM_Dense_filter_AVXExpand(benchmark::State &state) {
    BM_filter<DenseArray, DenseArray,
              void (DenseArray::*)(const DenseArray &, std::shared_ptr<Buffer>)
                  const>(state, &DenseArray::CompareWithColAVXExpand);
}
BENCHMARK(BM_Dense_filter_AVXExpand)->Apply(BM_MEM_Args);
static void BM_Dense_filter_AVXExpand_TenOp(benchmark::State &state) {
    BM_filter<DenseArray, DenseArray,
              void (DenseArray::*)(const DenseArray &, std::shared_ptr<Buffer>)
                  const>(state, &DenseArray::CompareWithColAVXExpand<true>);
}
// BENCHMARK(BM_Dense_filter_AVXExpand_TenOp)->Apply(BM_MEM_Args);

template <typename ArrayTypeA, typename ArrayTypeB, typename Func>
static void BM_filter_partial(benchmark::State &state, Func func) {
    auto data_size = state.range(1);
    auto num_array_per_iter = data_size / kVecSize;
    int filter_selectivity = state.range(2);
    auto null_percent = static_cast<double>(state.range(0)) / 100.0;
    SetupBenchmarksBatch<ArrayTypeA, ArrayTypeB> setup(state);
    std::vector<SelVector> outputs;
    for (int i = 0; i < setup.array_a.size(); ++i) {
        outputs.push_back(SelVector(kVecSize));
    }
    // SelVector output(kVecSize);
    std::vector<SelVector> sel_vecs(setup.array_a.size());
    for (int i = 0; i < setup.array_a.size(); ++i) {
        std::shared_ptr<Buffer> sel_bm = *arrow::AllocateEmptyBitmap(kVecSize);
        int64_t null_count = 0;
        GenerateBitmap(sel_bm->mutable_data(), kVecSize, &null_count,
                       1 - filter_selectivity / 100.0, 01011527 + i);
        sel_vecs[i] = GetSVFromBM(sel_bm->mutable_data(), kVecSize);
    }
    // uint64_t cur_array_idx = 0;
    for (auto _ : state) {
        // for (int i = cur_array_idx; i < cur_array_idx + num_array_per_iter;
        // ++i) {
        for (int i = 0; i < setup.array_a.size(); ++i) {
            (*(setup.array_a[i]).*func)(*setup.array_b[i], outputs[i],
                                        sel_vecs[i]);
        }
        benchmark::ClobberMemory();
        // state.PauseTiming();
        // for (int i = 0; i < setup.array_a.size(); ++i) {
        //   GenerateBitmap(setup.array_a[i]->buffers[0]->mutable_data(),
        //   setup.array_a[i]->length,
        //                  &setup.array_a[i]->null_count, null_percent,
        //                  rand());
        //   GenerateBitmap(setup.array_b[i]->buffers[0]->mutable_data(),
        //   setup.array_b[i]->length,
        //                  &setup.array_b[i]->null_count, null_percent,
        //                  rand());
        // }
        // state.ResumeTiming();
        // cur_array_idx += num_array_per_iter;
        // cur_array_idx >= setup.array_a.size() ? cur_array_idx = 0 :
        // cur_array_idx;
    }
    UPDATE_COUNTERS
    state.counters["filter_selectivity"] = filter_selectivity;
}

// filter selectivity only includes 20 50 80.
static void BM_Limited2D_Args(benchmark::internal::Benchmark *bench) {
    std::vector<int64_t> null_percent = {1,  5,  10, 20, 30, 40, 50, 60, 70,
                                         80, 90, 92, 95, 96, 97, 98, 99};
    std::vector<int64_t> data_size = {2048, 1024 * 1024};
    std::vector<int64_t> filter_selectivity = {20, 50, 80};

    bench->ArgsProduct({null_percent, data_size});
}

static void BM_2D_Args(benchmark::internal::Benchmark *bench) {
    std::vector<int64_t> null_percent = {1,  5,  10, 20, 30, 40, 50, 60, 70,
                                         80, 90, 92, 95, 96, 97, 98, 99};
    // std::vector<int64_t> null_percent = {50};
    // std::vector<int64_t> null_percent{};
    // for (int i = 1; i <= 100; i += 4) {
    //   null_percent.push_back(i);
    // }
    // std::vector<int64_t> data_size{};
    // for (int i = 1024; i <= 32 * 1024; i += 1024) {
    //   data_size.push_back(i);
    // }
    std::vector<int64_t> data_size = {2048, 1024 * 1024};
    // std::vector<int64_t> data_size = {2048};
    std::vector<int64_t> filter_selectivity = {50};
    // std::vector<int64_t> filter_selectivity = {1, 10, 20, 30, 40, 50, 60, 70,
    // 80, 90, 99}; std::vector<int64_t> filter_selectivity{}; for (int i = 1; i
    // <= 100; i += 4) {
    //   filter_selectivity.push_back(i);
    // }
    std::vector<int64_t> eval_filter_first = {0};
    bench->ArgsProduct(
        {null_percent, data_size, filter_selectivity, eval_filter_first});
}

static void BM_Dense_filter_SVPartial(benchmark::State &state) {
    BM_filter_partial<DenseArray, DenseArray,
                      int (DenseArray::*)(const DenseArray &, SelVector &,
                                          SelVector &) const>(
        state, &DenseArray::CompareWithColSVPartial);
}
BENCHMARK(BM_Dense_filter_SVPartial)->Apply(BM_2D_Args);

static void BM_Dense_filter_SVPartialBranch(benchmark::State &state) {
    bool use_filter_first = state.range(3);
    state.counters["use_filter_first"] = use_filter_first;
    BM_filter_partial<DenseArray, DenseArray,
                      int (DenseArray::*)(const DenseArray &, SelVector &,
                                          SelVector &) const>(
        state, use_filter_first
                   ? &DenseArray::CompareWithColSVPartialBranch<true>
                   : &DenseArray::CompareWithColSVPartialBranch<false>);
}
BENCHMARK(BM_Dense_filter_SVPartialBranch)->Apply(BM_2D_Args);

static void BM_Dense_filter_SVPartialFlat(benchmark::State &state) {
    BM_filter_partial<DenseArray, DenseArray,
                      int (DenseArray::*)(DenseArray &, SelVector &,
                                          SelVector &)>(
        state, &DenseArray::CompareWithColSVPartialFlat);
}
// BENCHMARK(BM_Dense_filter_SVPartialFlat)->Apply(BM_Limited2D_Args);

static void BM_Dense_filter_SVPartialFlatBranch(benchmark::State &state) {
    BM_filter_partial<DenseArray, DenseArray,
                      int (DenseArray::*)(DenseArray &, SelVector &,
                                          SelVector &)>(
        state, &DenseArray::CompareWithColSVPartialFlatBranch);
}
// BENCHMARK(BM_Dense_filter_SVPartialFlatBranch)->Apply(BM_Limited2D_Args);

static void BM_Dense_filter_SVManual(benchmark::State &state) {
    BM_filter_partial<DenseArray, DenseArray,
                      void (DenseArray::*)(const DenseArray &, SelVector &,
                                           SelVector &) const>(
        state, &DenseArray::CompareWithColSVManual);
}
// BENCHMARK(BM_Dense_filter_SVManual)->Apply(BM_Limited2D_Args);

static void BM_Spaced_filter_SVPartial(benchmark::State &state) {
    BM_filter_partial<SpacedArray, SpacedArray,
                      int (SpacedArray::*)(const SpacedArray &, SelVector &,
                                           SelVector &) const>(
        state, &SpacedArray::CompareWithColSVPartial);
}
BENCHMARK(BM_Spaced_filter_SVPartial)->Apply(BM_2D_Args);

static void BM_Spaced_filter_SVPartialBranch(benchmark::State &state) {
    bool use_filter_first = state.range(3);
    state.counters["use_filter_first"] = use_filter_first;
    BM_filter_partial<SpacedArray, SpacedArray,
                      int (SpacedArray::*)(const SpacedArray &, SelVector &,
                                           SelVector &) const>(
        state, use_filter_first
                   ? &SpacedArray::CompareWithColSVPartialBranch<true>
                   : &SpacedArray::CompareWithColSVPartialBranch<false>);
}
BENCHMARK(BM_Spaced_filter_SVPartialBranch)->Apply(BM_2D_Args);

static void BM_Spaced_filter_SVManual(benchmark::State &state) {
    BM_filter_partial<SpacedArray, SpacedArray,
                      void (SpacedArray::*)(const SpacedArray &, SelVector &,
                                            SelVector &) const>(
        state, &SpacedArray::CompareWithColSVPartialManual);
}
// BENCHMARK(BM_Spaced_filter_SVManual)->Apply(BM_Limited2D_Args);

static void BM_Sum(benchmark::State &state) {
    SetupBenchmarks setup(state);
    std::unique_ptr<BaseArray> array_a;
    if (state.range(1) == int(NULLStorage::DENSE)) {
        array_a = std::make_unique<DenseArray>(setup.array1);
    } else {
        array_a = std::make_unique<SpacedArray>(setup.array1);
    }
    std::vector<int64_t> sum(SetupBenchmarks::GROUP_CNT, 0);
    for (auto _ : state) {
        array_a->SumByGroupSV(setup.group_ids, sum);
        benchmark::ClobberMemory();
    }
    UPDATE_COUNTERS
    state.counters["type"] = state.range(1);
}
// BENCHMARK(BM_Sum)->Apply(BM_MEM_Args</*diff_type=*/true>);

static void BM_SumBM(benchmark::State &state) {
    SetupBenchmarks setup(state);
    std::unique_ptr<BaseArray> array_a;
    if (state.range(1) == int(NULLStorage::DENSE)) {
        array_a = std::make_unique<DenseArray>(setup.array1);
    } else {
        array_a = std::make_unique<SpacedArray>(setup.array1);
    }
    std::vector<int64_t> sum(SetupBenchmarks::GROUP_CNT, 0);
    for (auto _ : state) {
        array_a->SumByGroupBM(setup.group_ids, sum);
        benchmark::ClobberMemory();
    }
    UPDATE_COUNTERS
    state.counters["type"] = state.range(1);
}
// BENCHMARK(BM_SumBM)->Apply(BM_MEM_Args</*diff_type=*/true>);

static void BM_branch_Args(benchmark::internal::Benchmark *bench) {
    std::vector<int64_t> null_percent = {1,  5,  10, 20, 30, 40, 50, 60, 70,
                                         80, 90, 92, 95, 96, 97, 98, 99};
    std::vector<int64_t> branch = {1};
    std::vector<int64_t> size = {512, 1024, 2048, 384 * 1024};
    bench->ArgsProduct({null_percent, branch, size});
}
static void BM_branch(benchmark::State &state) {
    auto null_percent = static_cast<double>(state.range(0)) / 100.0;
    auto branch = state.range(1);
    auto size = state.range(2);
    std::vector<bool> validity_bm(size);
    std::bernoulli_distribution dist(1 - null_percent);
    std::mt19937 rng(
        202401181626UL); // Use a fixed seed for deterministic output
    for (int i = 0; i < size; ++i) {
        validity_bm[i] = dist(rng);
    }
    std::vector<int32_t> data1(size);
    std::vector<int32_t> data2(size);
    std::vector<uint64_t> out(size);
    for (auto _ : state) {
        int out_idx = 0;
        if (branch) {
            for (int i = 0; i < size; ++i) {
                if (validity_bm[i] && data1[i] < data2[i]) {
                    out[out_idx++] = i;
                }
            }
        } else {
            for (int i = 0; i < size; ++i) {
                out[out_idx] = i;
                out_idx += validity_bm[i] && data1[i] < data2[i];
            }
        }
    }
    UPDATE_COUNTERS
    state.counters["branch"] = state.range(1);
    state.counters["size"] = state.range(2);
}
BENCHMARK(BM_branch)->Apply(BM_branch_Args);

static void BM_Special_filter_AVX(benchmark::State &state) {
    BM_Filter<SpecialValArray>(state);
}
BENCHMARK(BM_Special_filter_AVX)->Apply(BM_MEM_Args);

static void BM_Specialfilter_SVPartial(benchmark::State &state) {
    BM_filter_partial<SpecialValArray, SpecialValArray,
                      int (SpecialValArray::*)(const SpecialValArray &,
                                               SelVector &, SelVector &) const>(
        state, &SpecialValArray::CompareWithColSVPartial);
}
BENCHMARK(BM_Specialfilter_SVPartial)->Apply(BM_2D_Args);

static void BM_Special_filter_SVPartialBranch(benchmark::State &state) {
    bool use_filter_first = state.range(3);
    state.counters["use_filter_first"] = use_filter_first;
    BM_filter_partial<SpecialValArray, SpecialValArray,
                      int (SpecialValArray::*)(const SpecialValArray &,
                                               SelVector &, SelVector &) const>(
        state, use_filter_first
                   ? &SpecialValArray::CompareWithColSVPartialBranch<true>
                   : &SpecialValArray::CompareWithColSVPartialBranch<false>);
}
BENCHMARK(BM_Special_filter_SVPartialBranch)->Apply(BM_2D_Args);

BENCHMARK_MAIN();