#include "BitpackingInterface.hpp"
#include "D4DeltaInterface.hpp"
#include "D4RLEInterface.hpp"
#include "DeltaInterface.hpp"
#include "FLSBitpackingInterface.hpp"
#include "FLSDeltaInterface.hpp"
#include "FLSInterface.hpp"
#include "FLSRLEInterface.hpp"
#include "RLEBranchlessInterface.hpp"
#include "RLEInterface.hpp"
#include "SampleCompressInterface.hpp"
#include "SampleSmartNullCompressInterface.hpp"
#include "bit_util.hpp"
#include "cycleclock.hpp"
#include "my_helper.hpp"
#include "nullgen.hpp"
#include "roaring/roaring.hh"
#include "utils.hpp"
#include "xsimd/types/xsimd_sse4_2_register.hpp"
#include <arrow/buffer.h>
#include <arrow/memory_pool.h>
#include <benchmark/benchmark.h>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

constexpr auto DECOMP_REPETITION = 10;

template <typename T, bool is_dense>
    requires std::is_base_of_v<null_revisit::CompressInterface, T>
constexpr auto compressor_name = "unknown";

template <bool is_dense>
constexpr auto compressor_name<null_revisit::BitpackingInterface, is_dense> =
    is_dense ? "bitpacking_dense" : "bitpacking";

template <bool is_dense>
constexpr auto
    compressor_name<null_revisit::SampleSmartNullCompressInterface, is_dense> =
        is_dense ? "sample_smart_null_dense" : "sample_smart_null";

template <bool is_dense>
constexpr auto
    compressor_name<null_revisit::SampleCompressInterface, is_dense> =
        is_dense ? "sample_dense" : "sample";

template <bool is_dense>
constexpr auto compressor_name<null_revisit::RLEInterface, is_dense> =
    is_dense ? "rle_dense" : "rle";

template <bool is_dense>
constexpr auto compressor_name<null_revisit::RLEBranchlessInterface, is_dense> =
    is_dense ? "rle_branchless_dense" : "rle_branchless";

template <bool is_dense>
constexpr auto compressor_name<null_revisit::DeltaInterface, is_dense> =
    is_dense ? "delta_dense" : "delta";

template <bool is_dense>
constexpr auto compressor_name<null_revisit::FLSBitpackingInterface, is_dense> =
    is_dense ? "flsbp_dense" : "flsbp";

template <bool is_dense>
constexpr auto compressor_name<null_revisit::FLSDeltaInterface, is_dense> =
    is_dense ? "flsdelta_dense" : "flsdelta";

template <bool is_dense>
constexpr auto compressor_name<null_revisit::FLSRLEInterface, is_dense> =
    is_dense ? "flsrle_dense" : "flsrle";

template <bool is_dense>
constexpr auto compressor_name<null_revisit::D4DeltaInterface, is_dense> =
    is_dense ? "d4delta_dense" : "d4delta";

template <bool is_dense>
constexpr auto compressor_name<null_revisit::D4RLEInterface, is_dense> =
    is_dense ? "d4rle_dense" : "d4rle";

template <typename T, bool dense_flag = false, bool simd_flag = false,
          bool simd_fuse_flag = true, bool special_val = false>
    requires std::is_base_of_v<null_revisit::CompressInterface, T>
Stat compress(const std::vector<uint32_t> &values,
              const std::vector<bool> &nulls, const uint32_t try_count,
              const size_t block_size,
              const null_revisit::CompressInterface::CompressOption &options) {
    T compressor;
    compressor.setOptions(options);
    int64_t total_comp_time = 0, total_decomp_time = 0, total_to_dense_time = 0,
            total_sample_time = 0;
    uint32_t total_size = 0;
    std::vector<std::unique_ptr<arrow::ResizableBuffer>> comp_outs;
    std::vector<std::unique_ptr<arrow::ResizableBuffer>> decomp_outs;
    for (uint32_t i = 0; i < try_count; i++) {
        auto buf = *arrow::AllocateResizableBuffer(
            compressor.compressBufSize(block_size * sizeof(uint32_t)));
        memset(buf->mutable_data(), 0, static_cast<size_t>(buf->size()));
        comp_outs.push_back(std::move(buf));
        buf = *arrow::AllocateResizableBuffer(
            compressor.decompressBufSize(block_size * sizeof(uint32_t)));
        memset(buf->mutable_data(), 0, static_cast<size_t>(buf->size()));
        decomp_outs.push_back(std::move(buf));
    }
    int64_t start = 0;
    std::vector<uint32_t> dense;
    if constexpr (dense_flag) {
        dense.reserve(values.size());
        start = benchmark::cycleclock::Now();
        for (size_t i = 0; i < values.size(); ++i) {
            if (!nulls[i]) {
                dense.push_back(values[i]);
            }
        }
        total_comp_time += benchmark::cycleclock::Now() - start;
        total_to_dense_time += total_comp_time;
    }
    auto comp_func = [&](const std::vector<uint32_t> &input, size_t i,
                         size_t num_elements, size_t val_index) {
        std::string_view block((const char *)input.data() +
                                   val_index * sizeof(uint32_t),
                               num_elements * sizeof(uint32_t));
        std::vector<bool> block_nulls(num_elements, false);
        if constexpr (!special_val) {
            block_nulls.assign(nulls.begin() + val_index,
                               nulls.begin() + val_index + num_elements);
        }
        auto start = benchmark::cycleclock::Now();
        if constexpr (dense_flag &&
                      std::is_base_of_v<null_revisit::FLSInterface, T>) {
            total_size += compressor.compress_dense(block, comp_outs[i].get());
        } else {
            total_size +=
                compressor.compress(block, comp_outs[i].get(), block_nulls);
        }
        total_comp_time += benchmark::cycleclock::Now() - start;
        // total_sample_time += compressor.getExtraTimer().sample_time;
    };
    auto decomp_func = [&](const std::vector<uint32_t> &input, size_t i,
                           size_t num_elements, size_t val_index) {
        std::string_view block((const char *)input.data() +
                                   val_index * sizeof(uint32_t),
                               num_elements * sizeof(uint32_t));
        std::vector<bool> block_nulls(nulls.begin() + val_index,
                                      nulls.begin() + val_index +
                                          num_elements); // useless; not match
        // auto aligned_decomp_out_size = decomp_outs[i].size();
        // void *decomp_out_data = decomp_outs[i].data();
        // auto aligned_decomp_out = reinterpret_cast<char *>(
        //     std::align(16, 16, decomp_out_data, aligned_decomp_out_size));
        // aligned_decomp_out_size = aligned_decomp_out_size / 64 * 64;
        int64_t decomp_time_sum = 0;
        int64_t to_dense_sum = 0;
        for (int rep = 0; rep < DECOMP_REPETITION; ++rep) {
            start = benchmark::cycleclock::Now();
            if constexpr (dense_flag &&
                          (std::is_same_v<T, null_revisit::FLSDeltaInterface> ||
                           std::is_same_v<T, null_revisit::FLSRLEInterface>)) {
                // downcast compressor to FLSInterface and call
                // decompress_untranspose
                auto fls_compressor =
                    dynamic_cast<null_revisit::FLSInterface *>(&compressor);
                fls_compressor->decompress_untranspose(
                    std::string_view(*comp_outs[i]),
                    (char *)decomp_outs[i]->mutable_data(),
                    decomp_outs[i]->size());
            } else {
                compressor.decompress(std::string_view(*comp_outs[i]),
                                      (char *)decomp_outs[i]->mutable_data(),
                                      decomp_outs[i]->size());
            }
            decomp_time_sum += benchmark::cycleclock::Now() - start;
        }
        total_decomp_time += decomp_time_sum / DECOMP_REPETITION;
        total_to_dense_time += to_dense_sum / DECOMP_REPETITION;
#if !NDEBUG
        // Check correctness
        if constexpr (!(std::is_same_v<T, null_revisit::FLSDeltaInterface> ||
                        std::is_same_v<T, null_revisit::FLSRLEInterface> ||
                        std::is_same_v<T, null_revisit::D4DeltaInterface>)) {
            // if (true) {
            // FLSDelta reorder tuples, so ignore. I manually checked the
            // output. As long as the block size is 1024 this should be fine.
            if constexpr (dense_flag) {
                if (memcmp(block.data(), decomp_outs[i]->data(),
                           block.size()) != 0) {
                    std::cerr << "Decompress error with "
                              << compressor_name<T, dense_flag> << std::endl;
                    throw std::runtime_error("Decompress error");
                }
            } else {
                for (size_t j = 0; j < block_size; ++j) {
                    if (!block_nulls[j]) {
                        if (memcmp(block.data() + j * sizeof(uint32_t),
                                   decomp_outs[i]->data() +
                                       j * sizeof(uint32_t),
                                   sizeof(uint32_t)) != 0) {
                            std::cerr << "Decompress error with "
                                      << compressor_name<T, false> << std::endl;
                            std::cerr << reinterpret_cast<const uint32_t *>(
                                             block.data())[j]
                                      << " "
                                      << reinterpret_cast<const uint32_t *>(
                                             decomp_outs[i]->data())[j]
                                      << std::endl;
                            std::cerr << i << " " << j << " " << val_index
                                      << std::endl;
                            for (size_t k = 0; k < block_size; ++k) {
                                std::cerr << reinterpret_cast<const uint32_t *>(
                                                 block.data())[k]
                                          << ",";
                            }
                            std::cerr << std::endl;
                            for (size_t k = 0; k < block_size; ++k) {
                                std::cerr << reinterpret_cast<const uint32_t *>(
                                                 decomp_outs[i]->data())[k]
                                          << ",";
                            }
                            std::cerr << std::endl;
                            exit(1);
                        }
                    }
                }
            }
        }
#endif
    };
    auto exp_func = [&](const std::vector<uint32_t> &input) {
        if constexpr (!dense_flag) {
            assert(input.size() % block_size == 0);
        }
        uint32_t i = 0;
        for (uint32_t val_index = 0; i < input.size() / block_size;
             i++, val_index += block_size) {
            comp_func(input, i, block_size, val_index);
        }
        assert(i <= try_count);
        // TODO: let us simply ignore last block now for FLS compatibility.
        // if (input.size() % block_size != 0) {
        //     comp_func(input, i, input.size() % block_size,
        //               input.size() - input.size() % block_size);
        // }
        i = 0;
        for (uint32_t val_index = 0; i < input.size() / block_size;
             i++, val_index += block_size) {
            decomp_func(input, i, block_size, val_index);
        }
        if constexpr (dense_flag) {
            auto null_count = values.size() - dense.size();
            // Dense to spaced
            std::vector<uint32_t> spaced(dense);
            const auto bitmap = toNullBitmap(nulls);
            spaced.resize(values.size());
            std::vector<uint32_t> index_buf(values.size());
            start = benchmark::cycleclock::Now();
            for (int rep = 0; rep < DECOMP_REPETITION; ++rep) {
                if constexpr (std::is_base_of_v<null_revisit::FLSInterface,
                                                T>) {
                    SpacedExpandAVX(dense.data(), spaced.data(), values.size(),
                                    null_count, bitmap->data());
                } else {
                    if constexpr (simd_fuse_flag) {
                        benchmark::DoNotOptimize(
                            spacedExpandSSE<xsimd::sse4_2,
                                            SpacedExpandSIMDMode::ADAPTIVE>(
                                dense.data(), (const uint64_t *)bitmap->data(),
                                (int32_t)values.size(), spaced.data(), {}));
                        benchmark::ClobberMemory();
                    } else {
                        const auto nonnull_count = bmToSV<xsimd::sse4_2, true>(
                            (const uint64_t *)bitmap->data(),
                            (int32_t)values.size(), index_buf.data(), {});
                        for (int j = 0; j < nonnull_count; ++j) {
                            spaced[index_buf[j]] = dense[j];
                        }
                    }
                }
            }
            const auto dense_to_spaced_time =
                (benchmark::cycleclock::Now() - start) / DECOMP_REPETITION;
            total_decomp_time += dense_to_spaced_time;
            total_sample_time += dense_to_spaced_time;
            total_to_dense_time += dense_to_spaced_time;

#if !NDEBUG
            for (size_t j = 0; j < values.size(); ++j) {
                if (!nulls[j] && spaced[j] != values[j]) {
                    std::cerr << "Decompress error with spaced to dense"
                              << std::endl;
                    std::cerr << j << " " << spaced[j] << " " << values[j]
                              << std::endl;
                    throw std::runtime_error("Decompress error");
                }
            }
#endif
        }
        // if (input.size() % block_size != 0) {
        //     decomp_func(input, i, input.size() % block_size,
        //                 input.size() - input.size() % block_size);
        // }
    };
    if constexpr (dense_flag) {
        exp_func(dense);
    } else if constexpr (special_val) {
        uint32_t i = 0;
        std::vector<uint32_t> values_special(values.size());
        for (uint32_t val_index = 0; i < values.size() / block_size;
             i++, val_index += block_size) {
            uint32_t max_val = std::numeric_limits<uint32_t>::min();
            for (uint32_t j = 0; j < block_size; ++j) {
                if (!nulls[val_index + j] && values[val_index + j] > max_val) {
                    max_val = values[val_index + j];
                }
            }
            max_val += 1;
            for (uint32_t j = 0; j < block_size; ++j) {
                if (nulls[val_index + j]) {
                    values_special[val_index + j] = max_val;
                } else {
                    values_special[val_index + j] = values[val_index + j];
                }
            }
        }
        exp_func(values_special);
    } else {
        exp_func(values);
    }
    static_assert(!(special_val && simd_flag),
                  "special_val and simd_flag cannot be both true");
    const auto comp_name =
        special_val ? std::string(compressor_name<T, dense_flag>) + "_special"
        : (simd_flag)
            ? (std::string(compressor_name<T, dense_flag>) +
               ((simd_fuse_flag) ? ("_simd_fused") : ("_simd_unfused")))
            : std::string(compressor_name<T, dense_flag>);
    auto ret = Stat(comp_name, try_count, total_comp_time, total_decomp_time,
                    total_sample_time, total_size,
                    compressor.getEncodingCounter().num_rle,
                    compressor.getEncodingCounter().num_delta,
                    compressor.getEncodingCounter().num_bitpacking);
    ret.spaced_to_dense_time = total_to_dense_time;
    return ret;
}

int main(int argc, char **argv) {
    if (argc != 8 && argc != 9) {
        std::cerr << "Usage: " << argv[0]
                  << " <null_rate> <repeat_rate> <range_max> "
                     "<try_count> <block_size> <output_file> [uniform, "
                     "gentle_zipf, hotspot, linear] [[zero, last, linear, "
                     "frequent, random, smart]]"
                  << std::endl;
        return 1;
    }
    null_revisit::CompressInterface::CompressOption options;
    bool isCompareHeuristic = (argc >= 9);
    const float null_rate = std::stof(argv[1]);
    const float repeat_rate = std::stof(argv[2]);
    const uint32_t range_max = std::stoul(argv[3]);
    const uint32_t try_count = std::stoul(argv[4]);
    const uint32_t block_size = std::stoul(argv[5]);
    const std::string output_file = argv[6];
    const null_revisit::Distribution dist =
        null_revisit::GetDistFromString(argv[7]);
    if (argc >= 9) {
        options.heuristic = null_revisit::GetHeuristicFromString(argv[8]);
    }
    const null_revisit::NullGenerator nullgen(null_rate, repeat_rate, range_max,
                                              dist);
    std::vector<Stat> stats;
    const auto [values, nulls] = nullgen.generate(try_count * block_size);
    if (!isCompareHeuristic) {
        stats.push_back(
            compress<null_revisit::SampleCompressInterface>( // spaced
                values, nulls, try_count, block_size, options));
        stats.push_back(compress<null_revisit::SampleCompressInterface,
                                 true>( // dense optimized scalar
            values, nulls, try_count, block_size, options));
        stats.push_back(compress<null_revisit::SampleCompressInterface, true,
                                 true>( // dense simd fused
            values, nulls, try_count, block_size, options));
        stats.push_back(
            compress<null_revisit::SampleCompressInterface, true, true,
                     false>( // dense simd not fused
                values, nulls, try_count, block_size, options));
        stats.push_back(
            compress<null_revisit::SampleCompressInterface, false, false, true,
                     true>( // special val
                values, nulls, try_count, block_size, options));

        if (block_size == 1024) {
            stats.push_back(compress<null_revisit::FLSBitpackingInterface>(
                values, nulls, try_count, block_size, options));
            stats.push_back(
                compress<null_revisit::FLSBitpackingInterface, true>(
                    values, nulls, try_count, block_size, options));

            stats.push_back(compress<null_revisit::FLSRLEInterface>(
                values, nulls, try_count, block_size, options));
            stats.push_back(compress<null_revisit::FLSRLEInterface, true>(
                values, nulls, try_count, block_size, options));
            stats.push_back(compress<null_revisit::D4RLEInterface, false>(
                values, nulls, try_count, block_size, options));
            stats.push_back(compress<null_revisit::D4RLEInterface, true>(
                values, nulls, try_count, block_size, options));

            // deltas
            stats.push_back(compress<null_revisit::FLSDeltaInterface>(
                values, nulls, try_count, block_size, options));
            stats.push_back(compress<null_revisit::FLSDeltaInterface, true>(
                values, nulls, try_count, block_size, options));
            stats.push_back(compress<null_revisit::D4DeltaInterface, false>(
                values, nulls, try_count, block_size, options));
            stats.push_back(compress<null_revisit::D4DeltaInterface, true>(
                values, nulls, try_count, block_size, options));
        }
    }
    stats.push_back(compress<null_revisit::SampleSmartNullCompressInterface>(
        values, nulls, try_count, block_size, options)); // spaced smart null

    std::ofstream out(output_file);
    // Output as JSON
    out << "[\n";
    for (uint32_t i = 0; i < stats.size(); i++) {
        out << stats[i].to_json();
        if (i != stats.size() - 1) {
            out << ",\n";
        }
    }
    out << "]\n";
    out.close();
    return 0;
}
