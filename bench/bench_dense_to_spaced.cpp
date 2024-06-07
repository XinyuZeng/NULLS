
#include <benchmark/benchmark.h>

#include <cstdint>
#include <memory>
#include <random>

#include "array_base.h"
#include "bench_helper.h"
#include "dense_array.h"
#include "roaring/roaring.hh"
#include "spaced_array.h"
#include "utils/bit_util.h"
#include "utils/macros.hpp"
#include "utils/test_util.h"
#include "xsimd/types/xsimd_avx2_register.hpp"
#include "xsimd/types/xsimd_sse4_2_register.hpp"

using namespace null_revisit;

// Baseline: Arrow
static void BM_DenseToSpacedBase(benchmark::State &state) {
    SetupBenchmarksBatch<DenseArray, DenseArray> setup(state);
    std::vector<std::shared_ptr<Buffer>> outputs;
    for (int i = 0; i < setup.array_a.size(); ++i) {
        outputs.push_back(
            *arrow::AllocateResizableBuffer(kVecSize * sizeof(int32_t)));
    }
    // std::shared_ptr<Buffer> output = *arrow::AllocateResizableBuffer(kVecSize
    // * sizeof(int32_t));
    for (auto _ : state) {
        for (int i = 0; i < setup.array_a.size(); ++i) {
            SpacedScatter((int32_t *)setup.array_a[i]->buffers[1]->data(),
                          (int32_t *)outputs[i]->mutable_data(),
                          setup.array_a[i]->length,
                          setup.array_a[i]->null_count,
                          setup.array_a[i]->buffers[0]->data(), 0);
        }
        benchmark::ClobberMemory();
    }
    UPDATE_COUNTERS
}
BENCHMARK(BM_DenseToSpacedBase)->Apply(BM_MEM_Args);

// BM->SV by SSE4 (code from Velox), then scalar scatter
static void BM_DenseToSpacedSSE4Scatter(benchmark::State &state) {
    SetupBenchmarksBatch<DenseArray, DenseArray> setup(state);
    std::shared_ptr<Buffer> output =
        *arrow::AllocateResizableBuffer(kVecSize * sizeof(int32_t));
    SelVector sel(kVecSize);
    auto indices_buffer =
        *arrow::AllocateBuffer(sizeof(row_id_type) * kVecSize);
    for (auto _ : state) {
        for (int i = 0; i < setup.array_a.size(); ++i) {
            // the conversion is the same cost as using Arrow's BitRunReader
            auto indices = (row_id_type *)indices_buffer->mutable_data();
            auto bits = (const uint64_t *)setup.array_a[i]->buffers[0]->data();
            auto in = (int32_t *)setup.array_a[i]->buffers[1]->data();
            auto out = (int32_t *)output->mutable_data();
            indicesOfSetBits<xsimd::sse4_2>(bits, 0, setup.array_a[i]->length,
                                            indices, {});
            auto num_nonnull =
                setup.array_a[i]->length - setup.array_a[i]->null_count;
            for (int j = 0; j < num_nonnull; ++j) {
                out[indices[j]] = in[j];
            }
        }
        benchmark::ClobberMemory();
    }
    UPDATE_COUNTERS
}
BENCHMARK(BM_DenseToSpacedSSE4Scatter)->Apply(BM_MEM_Args);

// BM->SV by AVX2 (code from Velox), then scatter using AVX2
static void BM_DenseToSpacedAVX2Scatter(benchmark::State &state) {
    SetupBenchmarksBatch<DenseArray, DenseArray> setup(state);
    std::shared_ptr<Buffer> output =
        *arrow::AllocateResizableBuffer(kVecSize * sizeof(int32_t));
    SelVector sel(kVecSize);
    auto indices_buffer =
        *arrow::AllocateBuffer(sizeof(row_id_type) * kVecSize);
    for (auto _ : state) {
        for (int i = 0; i < setup.array_a.size(); ++i) {
            // the conversion is the same cost as using Arrow's BitRunReader
            auto indices = (row_id_type *)indices_buffer->mutable_data();
            auto bits = (const uint64_t *)setup.array_a[i]->buffers[0]->data();
            indicesOfSetBits<xsimd::avx2>(bits, 0, setup.array_a[i]->length,
                                          indices, {});
            SpacedScatterAVX2((int32_t *)setup.array_a[i]->buffers[1]->data(),
                              (int32_t *)output->mutable_data(),
                              setup.array_a[i]->length,
                              setup.array_a[i]->null_count, indices);
        }
        benchmark::ClobberMemory();
    }
    UPDATE_COUNTERS
}
BENCHMARK(BM_DenseToSpacedAVX2Scatter)->Apply(BM_MEM_Args);

// BM->SV fused with scatter, by AVX2
static void BM_DenseToSpacedFusedAVX2Scatter(benchmark::State &state) {
    SetupBenchmarksBatch<DenseArray, DenseArray> setup(state);
    std::vector<std::shared_ptr<Buffer>> outputs;
    for (int i = 0; i < setup.array_a.size(); ++i) {
        outputs.push_back(
            *arrow::AllocateResizableBuffer(kVecSize * sizeof(int32_t)));
    }
    SelVector sel(kVecSize);
    auto indices_buffer =
        *arrow::AllocateBuffer(sizeof(row_id_type) * kVecSize);
    for (auto _ : state) {
        for (int i = 0; i < setup.array_a.size(); ++i) {
            auto indices = (row_id_type *)indices_buffer->mutable_data();
            auto bits = (const uint64_t *)setup.array_a[i]->buffers[0]->data();
            spacedExpandFused<xsimd::avx2, true>(
                (const uint32_t *)setup.array_a[i]->buffers[1]->data(), bits,
                setup.array_a[i]->length,
                (uint32_t *)outputs[i]->mutable_data(), {});
        }
        benchmark::ClobberMemory();
    }
    UPDATE_COUNTERS
}
BENCHMARK(BM_DenseToSpacedFusedAVX2Scatter)->Apply(BM_MEM_Args);

// BM->SV by AVX2 fused with scalar scatter using miniblocks
static void
BM_DenseToSpacedFusedAVX2MiniblocksScatter(benchmark::State &state) {
    SetupBenchmarksBatch<DenseArray, DenseArray> setup(state);
    std::vector<std::shared_ptr<Buffer>> outputs;
    for (int i = 0; i < setup.array_a.size(); ++i) {
        outputs.push_back(
            *arrow::AllocateResizableBuffer(kVecSize * sizeof(int32_t)));
    }
    SelVector sel(kVecSize);
    auto indices_buffer =
        *arrow::AllocateBuffer(sizeof(row_id_type) * kVecSize);
    for (auto _ : state) {
        for (int i = 0; i < setup.array_a.size(); ++i) {
            auto indices = (row_id_type *)indices_buffer->mutable_data();
            auto bits = (const uint64_t *)setup.array_a[i]->buffers[0]->data();
            spacedExpandFusedMiniblocks<xsimd::avx2, true>(
                (const uint32_t *)setup.array_a[i]->buffers[1]->data(), bits,
                setup.array_a[i]->length,
                (uint32_t *)outputs[i]->mutable_data(), {});
        }
        benchmark::ClobberMemory();
    }
    UPDATE_COUNTERS
}
BENCHMARK(BM_DenseToSpacedFusedAVX2MiniblocksScatter)->Apply(BM_MEM_Args);

// BM->SV by SSE4 fused with scalar scatter
static void BM_DenseToSpacedFusedSSEScatter(benchmark::State &state) {
    SetupBenchmarksBatch<DenseArray, DenseArray> setup(state);
    std::vector<std::shared_ptr<Buffer>> outputs;
    for (int i = 0; i < setup.array_a.size(); ++i) {
        outputs.push_back(
            *arrow::AllocateResizableBuffer(kVecSize * sizeof(int32_t)));
    }
    SelVector sel(kVecSize);
    for (auto _ : state) {
        for (int i = 0; i < setup.array_a.size(); ++i) {
            auto bits = (const uint64_t *)setup.array_a[i]->buffers[0]->data();
            spacedExpandFused<xsimd::sse4_2, true>(
                (const uint32_t *)setup.array_a[i]->buffers[1]->data(), bits,
                setup.array_a[i]->length,
                (uint32_t *)outputs[i]->mutable_data(), {});
        }
        benchmark::ClobberMemory();
    }
    UPDATE_COUNTERS
}
BENCHMARK(BM_DenseToSpacedFusedSSEScatter)->Apply(BM_MEM_Args);

// BM->SV by SSE4 fused with scalar scatter using miniblocks
static void BM_DenseToSpacedFusedSSEMiniblocksScatter(benchmark::State &state) {
    SetupBenchmarksBatch<DenseArray, DenseArray> setup(state);
    std::shared_ptr<Buffer> output =
        *arrow::AllocateResizableBuffer(kVecSize * sizeof(int32_t));
    SelVector sel(kVecSize);
    for (auto _ : state) {
        for (int i = 0; i < setup.array_a.size(); ++i) {
            auto bits = (const uint64_t *)setup.array_a[i]->buffers[0]->data();
            spacedExpandFusedMiniblocks<xsimd::sse4_2, true>(
                (const uint32_t *)setup.array_a[i]->buffers[1]->data(), bits,
                setup.array_a[i]->length, (uint32_t *)output->mutable_data(),
                {});
        }
        benchmark::ClobberMemory();
    }
    UPDATE_COUNTERS
}
BENCHMARK(BM_DenseToSpacedFusedSSEMiniblocksScatter)->Apply(BM_MEM_Args);

// BM->SV by AVX2 (code from Velox), then scalar scatter
static void BM_DenseToSpacedScalarScatter(benchmark::State &state) {
    SetupBenchmarksBatch<DenseArray, DenseArray> setup(state);
    std::shared_ptr<Buffer> output =
        *arrow::AllocateResizableBuffer(kVecSize * sizeof(int32_t));
    SelVector sel(kVecSize);
    auto indices_buffer =
        *arrow::AllocateBuffer(sizeof(row_id_type) * kVecSize);
    for (auto _ : state) {
        for (int i = 0; i < setup.array_a.size(); ++i) {
            // the conversion is the same cost as using Arrow's BitRunReader
            auto in = (int32_t *)setup.array_a[i]->buffers[1]->data();
            auto out = (int32_t *)output->mutable_data();
            auto num_nonnull =
                setup.array_a[i]->length - setup.array_a[i]->null_count;
            auto indices = (row_id_type *)indices_buffer->mutable_data();
            auto bits = (const uint64_t *)setup.array_a[i]->buffers[0]->data();
            indicesOfSetBits<xsimd::avx2, true>(
                bits, 0, setup.array_a[i]->length, indices, {});
            for (int j = 0; j < num_nonnull; ++j) {
                out[indices[j]] = in[j];
            }
        }
        benchmark::ClobberMemory();
    }
    UPDATE_COUNTERS
}
BENCHMARK(BM_DenseToSpacedScalarScatter)->Apply(BM_MEM_Args);

// Scalar BM->SV + scatter fused
static void BM_DenseToSpacedScalarScatterFused(benchmark::State &state) {
    SetupBenchmarksBatch<DenseArray, DenseArray> setup(state);
    std::vector<std::shared_ptr<Buffer>> outputs;
    for (int i = 0; i < setup.array_a.size(); ++i) {
        outputs.push_back(
            *arrow::AllocateResizableBuffer(kVecSize * sizeof(int32_t)));
    }

    for (auto _ : state) {
        for (int i = 0; i < setup.array_a.size(); ++i) {
            auto bits = (const uint64_t *)setup.array_a[i]->buffers[0]->data();
            spacedExpandFused<xsimd::sse4_2, false>(
                (const uint32_t *)setup.array_a[i]->buffers[1]->data(), bits,
                setup.array_a[i]->length,
                (uint32_t *)outputs[i]->mutable_data(), {});
        }
        benchmark::ClobberMemory();
    }
    UPDATE_COUNTERS
}
BENCHMARK(BM_DenseToSpacedScalarScatterFused)->Apply(BM_MEM_Args);

// Dense to spaced using AVX512 expand
static void BM_DenseToSpacedAVXExpand(benchmark::State &state) {
    SetupBenchmarksBatch<DenseArray, DenseArray> setup(state);
    std::vector<std::shared_ptr<Buffer>> outputs;
    for (int i = 0; i < setup.array_a.size(); ++i) {
        outputs.push_back(
            *arrow::AllocateResizableBuffer(kVecSize * sizeof(int32_t)));
    }
    for (auto _ : state) {
        for (int i = 0; i < setup.array_a.size(); ++i) {
            SpacedExpandAVX((int32_t *)setup.array_a[i]->buffers[1]->data(),
                            (int32_t *)outputs[i]->mutable_data(),
                            setup.array_a[i]->length,
                            setup.array_a[i]->null_count,
                            setup.array_a[i]->buffers[0]->data());
        }
        benchmark::ClobberMemory();
    }
    UPDATE_COUNTERS
}
BENCHMARK(BM_DenseToSpacedAVXExpand)->Apply(BM_MEM_Args);

// Dense to spaced using Roaring bitmap
static void BM_DenseToSpacedCRoaring(benchmark::State &state) {
    SetupBenchmarksBatch<DenseArray, DenseArray> setup(state);
    std::shared_ptr<Buffer> output =
        *arrow::AllocateResizableBuffer(kVecSize * sizeof(int32_t));
    std::vector<roaring::Roaring> r_bm_s(setup.array_a.size());
    for (int i = 0; i < setup.array_a.size(); ++i) {
        for (int j = 0; j < setup.array_a[i]->length; ++j) {
            if (arrow::bit_util::GetBit(setup.array_a[i]->buffers[0]->data(),
                                        j)) {
                r_bm_s[i].add(j);
            }
        }
    }
    for (auto _ : state) {
        for (int i = 0; i < setup.array_a.size(); ++i) {
            std::pair<int32_t *, int32_t *> param =
                std::make_pair((int32_t *)output->mutable_data(),
                               (int32_t *)setup.array_a[i]->buffers[1]->data());
            r_bm_s[i].iterate(
                [](uint32_t value, void *param) {
                    auto p =
                        reinterpret_cast<std::pair<int32_t *, int32_t *> *>(
                            param);
                    // dest[value] = *exceptions_ptr++;
                    p->first[value] = *(p->second);
                    p->second++;
                    return true;
                },
                &param);
        }
        benchmark::ClobberMemory();
    }
    UPDATE_COUNTERS
}
BENCHMARK(BM_DenseToSpacedCRoaring)->Apply(BM_MEM_Args);

// Test performance of different miniblock sizes
template <typename A, int block_size>
static void BM_MiniblockSizeDenseToSpaced(benchmark::State &state) {
    SetupBenchmarksBatch<DenseArray, DenseArray> setup(state);
    std::shared_ptr<Buffer> output =
        *arrow::AllocateResizableBuffer(kVecSize * sizeof(int32_t));
    SelVector sel(kVecSize);
    for (auto _ : state) {
        for (int i = 0; i < setup.array_a.size(); ++i) {
            auto bits = (const uint64_t *)setup.array_a[i]->buffers[0]->data();
            spacedExpandFusedMiniblocks<A, true, block_size>(
                (const uint32_t *)setup.array_a[i]->buffers[1]->data(), bits,
                setup.array_a[i]->length, (uint32_t *)output->mutable_data(),
                {});
        }
        benchmark::ClobberMemory();
    }
    UPDATE_COUNTERS
    state.counters["block_size"] = block_size;
    state.counters["arch"] = (A::name())[0];
}

#define BENCHMARK_BLOCK_SIZE(arch, block_size)                                 \
    BENCHMARK_TEMPLATE(BM_MiniblockSizeDenseToSpaced, arch, block_size)        \
        ->Apply(BM_BLOCKSIZE_Args)
BENCHMARK_BLOCK_SIZE(xsimd::sse4_2, 64);
BENCHMARK_BLOCK_SIZE(xsimd::sse4_2, 128);
BENCHMARK_BLOCK_SIZE(xsimd::sse4_2, 256);
BENCHMARK_BLOCK_SIZE(xsimd::sse4_2, 512);
BENCHMARK_BLOCK_SIZE(xsimd::sse4_2, 1024);
BENCHMARK_BLOCK_SIZE(xsimd::sse4_2, 2048);
BENCHMARK_BLOCK_SIZE(xsimd::sse4_2, 4096);
BENCHMARK_BLOCK_SIZE(xsimd::sse4_2, 8192);
BENCHMARK_BLOCK_SIZE(xsimd::sse4_2, 16384);
BENCHMARK_BLOCK_SIZE(xsimd::sse4_2, 32768);
BENCHMARK_BLOCK_SIZE(xsimd::sse4_2, 65536);
BENCHMARK_BLOCK_SIZE(xsimd::sse4_2, 131072);
BENCHMARK_BLOCK_SIZE(xsimd::sse4_2, 262144);
BENCHMARK_BLOCK_SIZE(xsimd::sse4_2, 524288);
BENCHMARK_BLOCK_SIZE(xsimd::sse4_2, 1048576);
BENCHMARK_BLOCK_SIZE(xsimd::sse4_2, 2097152);
BENCHMARK_BLOCK_SIZE(xsimd::sse4_2, 4194304);
BENCHMARK_BLOCK_SIZE(xsimd::sse4_2, 8388608);

BENCHMARK_BLOCK_SIZE(xsimd::avx2, 64);
BENCHMARK_BLOCK_SIZE(xsimd::avx2, 128);
BENCHMARK_BLOCK_SIZE(xsimd::avx2, 256);
BENCHMARK_BLOCK_SIZE(xsimd::avx2, 512);
BENCHMARK_BLOCK_SIZE(xsimd::avx2, 1024);
BENCHMARK_BLOCK_SIZE(xsimd::avx2, 2048);
BENCHMARK_BLOCK_SIZE(xsimd::avx2, 4096);
BENCHMARK_BLOCK_SIZE(xsimd::avx2, 8192);
BENCHMARK_BLOCK_SIZE(xsimd::avx2, 16384);
BENCHMARK_BLOCK_SIZE(xsimd::avx2, 32768);
BENCHMARK_BLOCK_SIZE(xsimd::avx2, 65536);
BENCHMARK_BLOCK_SIZE(xsimd::avx2, 131072);
BENCHMARK_BLOCK_SIZE(xsimd::avx2, 262144);
BENCHMARK_BLOCK_SIZE(xsimd::avx2, 524288);
BENCHMARK_BLOCK_SIZE(xsimd::avx2, 1048576);
BENCHMARK_BLOCK_SIZE(xsimd::avx2, 2097152);
BENCHMARK_BLOCK_SIZE(xsimd::avx2, 4194304);
BENCHMARK_BLOCK_SIZE(xsimd::avx2, 8388608);
#undef BENCHMARK_BLOCK_SIZE

BENCHMARK_MAIN();
