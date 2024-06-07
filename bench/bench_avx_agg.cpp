#include <benchmark/benchmark.h>
#include <bits/stdint-uintn.h>
#include <fmt/core.h>
#include <immintrin.h>

#include "utils/macros.hpp"

static void BM_SCALAR(benchmark::State &state) {
    uint64_t sum = 0;
    for (auto _ : state) {
        sum = 0;
        for (int i = 0; i < null_revisit::kVecSize; ++i) {
            benchmark::DoNotOptimize(sum += null_revisit::rand_uint32[i]);
        }
    }
    //   fmt::print("{}\n", sum);
}

static void BM_AVX512(benchmark::State &state) {
    uint64_t sum = 0;
    __m512i vec_sum = _mm512_set1_epi32(0);
    for (auto _ : state) {
        sum = 0;
        vec_sum = _mm512_set1_epi32(0);
        for (int i = 0; i < null_revisit::kVecSize; i += 16) {
            __m512i vec = _mm512_loadu_si512(&null_revisit::rand_uint32[i]);
            vec_sum = _mm512_add_epi32(vec, vec_sum);
        }
        benchmark::DoNotOptimize(sum = _mm512_reduce_add_epi32(vec_sum));
    }
    //   fmt::print("{}\n", sum);
}

static void BM_AVX512_REDUCE_ONLY(benchmark::State &state) {
    uint64_t sum = 0;
    __m512i vec_sum = _mm512_set1_epi32(0);
    for (auto _ : state) {
        sum = 0;
        vec_sum = _mm512_set1_epi32(0);
        for (int i = 0; i < null_revisit::kVecSize; i += 16) {
            __m512i vec = _mm512_loadu_si512(&null_revisit::rand_uint32[i]);
            benchmark::DoNotOptimize(sum += _mm512_reduce_add_epi32(vec));
        }
    }
    //   fmt::print("{}\n", sum);
}

static inline __m512i load_and_extend(const uint32_t *array) {
    __m256i tmp = _mm256_load_si256(
        (__m256i *)array);             // load 8 int32_t values to lower part
    return _mm512_cvtepi32_epi64(tmp); // extend to int64_t values
}

static void BM_AVX512_NO_OVERFLOW(benchmark::State &state) {
    uint64_t sum = 0;
    __m512i vec_sum = _mm512_set1_epi64(0);
    for (auto _ : state) {
        sum = 0;
        vec_sum = _mm512_set1_epi64(0);
        for (int i = 0; i < null_revisit::kVecSize; i += 8) {
            __m512i vec = load_and_extend(
                &null_revisit::rand_uint32[i]); // load and extend 8 elements
            vec_sum = _mm512_add_epi64(vec_sum, vec); // add vector to sum
        }

        benchmark::DoNotOptimize(
            sum = _mm512_reduce_add_epi64(vec_sum)); // reduce sum
    }
    //   fmt::print("{}\n", sum);
}

BENCHMARK(BM_SCALAR);
BENCHMARK(BM_AVX512);
BENCHMARK(BM_AVX512_REDUCE_ONLY);
BENCHMARK(BM_AVX512_NO_OVERFLOW);

BENCHMARK_MAIN();