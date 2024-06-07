#include "SampleSmartNullCompressInterface.hpp"
#include "BitpackingInterface.hpp"
#include "BitpackingStat.hpp"
#include "DeltaInterface.hpp"
#include "RLEBranchlessInterface.hpp"
#include "VarInts.hpp"
#include "cycleclock.hpp"
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <random>

namespace null_revisit {
SampleSmartNullCompressInterface::SampleSmartNullCompressInterface(
    const std::string &codec_name)
    : CompressInterface(), word_size_(4) {
    interfaces[BITPACKING_C] = new BitpackingInterface();
    interfaces[BITPACKING_C]->compress_option_ = compress_option_;
    interfaces[RLE_C] = new RLEBranchlessInterface();
    interfaces[RLE_C]->compress_option_ = compress_option_;
    interfaces[DELTA_C] = new DeltaInterface();
    interfaces[DELTA_C]->compress_option_ = compress_option_;
}

SampleSmartNullCompressInterface::~SampleSmartNullCompressInterface() {
    delete interfaces[BITPACKING_C];
    delete interfaces[RLE_C];
    delete interfaces[DELTA_C];
}

size_t
SampleSmartNullCompressInterface::compress(std::string_view src,
                                           arrow::ResizableBuffer *dst,
                                           const std::vector<bool> &nulls) {
    extra_timer_.reset();
    auto sample_begin = benchmark::cycleclock::Now();
    std::unique_ptr<std::vector<uint32_t>> temp;
    switch (compress_option_.heuristic) {
    case NullHeuristic::LAST:
        // Fill nulls with last
        temp =
            std::make_unique<std::vector<uint32_t>>(fillNullsLast(src, nulls));
        src = std::string_view((const char *)temp->data(),
                               temp->size() * sizeof(uint32_t));
        break;
    case NullHeuristic::LINEAR:
        // Fill nulls with linear
        temp = std::make_unique<std::vector<uint32_t>>(
            fillNullsLinear(src, nulls));
        src = std::string_view((const char *)temp->data(),
                               temp->size() * sizeof(uint32_t));
        break;
    case NullHeuristic::ZERO:
        // Fill nulls with zero (do nothing)
        break;
    case NullHeuristic::FREQUENT:
        // Fill nulls with most frequent
        temp = std::make_unique<std::vector<uint32_t>>(
            fillNullsMostFreq(src, nulls));
        src = std::string_view((const char *)temp->data(),
                               temp->size() * sizeof(uint32_t));
        break;
    case NullHeuristic::RANDOM:
        // Fill nulls with random
        temp = std::make_unique<std::vector<uint32_t>>(
            fillNullsRandom(src, nulls));
        src = std::string_view((const char *)temp->data(),
                               temp->size() * sizeof(uint32_t));
        break;
    }

    const uint32_t n = src.size() / word_size_;
    // Stat
    // Choose the best scheme
    sample_segments_ = std::max(n / SAMPLE_UNIT_SIZE, 4u);
    auto compress_scheme = sampleStat((const uint32_t *)src.data(), nulls, n);
    if (compress_option_.heuristic != NullHeuristic::SMART) {
        // All nulls have been filled, so create a null vector with all false
        const auto temp_nulls = std::vector<bool>(nulls.size(), false);
        compress_scheme =
            sampleStat((const uint32_t *)src.data(), temp_nulls, n);
    }
    if (compress_option_.always_delta)
        compress_scheme = DELTA_C;
    extra_timer_.sample_time += benchmark::cycleclock::Now() - sample_begin;
    incEncodingCounter(compress_scheme);
    if (compress_scheme == DELTA_C &&
        compress_option_.heuristic == NullHeuristic::SMART) {
        // First fill with linear
        temp = std::make_unique<std::vector<uint32_t>>(
            fillNullsLinear(src, nulls));
        src = std::string_view((const char *)temp->data(),
                               temp->size() * sizeof(uint32_t));
    }
    // Compress
    const auto compress_size =
        (compress_option_.heuristic == NullHeuristic::SMART)
            ? interfaces[compress_scheme]->compress_with_null(src, dst, nulls)
            : interfaces[compress_scheme]->compress(src, dst, nulls);
    dst->Resize(compress_size + 1);
    // Write the scheme
    dst->mutable_data()[compress_size] = compress_scheme;
    return dst->size();
}

size_t SampleSmartNullCompressInterface::compress_with_null(
    std::string_view src, arrow::ResizableBuffer *dst,
    const std::vector<bool> &nulls) {
    return compress(src, dst, nulls);
}

size_t SampleSmartNullCompressInterface::decompress(std::string_view src,
                                                    char *dst,
                                                    size_t dst_size) {
    // Read the scheme
    const auto compress_scheme = src.back();
    src.remove_suffix(1);
    // Decompress
    return interfaces[compress_scheme]->decompress(src, dst, dst_size);
}

SampleSmartNullCompressInterface::CompressScheme
SampleSmartNullCompressInterface::sampleStat(const uint32_t *src_arr,
                                             const std::vector<bool> &src_null,
                                             uint32_t n) {
    auto filled_values = fillNullsLinear(
        std::string_view((const char *)src_arr, n * sizeof(uint32_t)),
        src_null);
    src_arr = (const uint32_t *)filled_values.data();
    SampleStat stat(sample_segments_ * SAMPLE_SIZE, src_arr[0]);
    if (n < SAMPLE_SIZE * sample_segments_) {
        for (uint32_t i = 0; i < n; i++) {
            stat.incData(src_arr[i],
                         src_null[i] && (compress_option_.heuristic ==
                                         NullHeuristic::SMART));
        }
    } else {
        const auto sample_skip = n / sample_segments_;
        for (uint32_t i = 0, src_idx = 0; i < sample_segments_;
             i++, src_idx += sample_skip) {
            const auto src_start_idx =
                src_idx + std::rand() % (sample_skip - SAMPLE_SIZE);
            const auto last_idx = (src_start_idx > 0) ? (src_start_idx - 1) : 0;
            stat.setLast(src_arr[last_idx]);
            for (uint32_t j = 0; j < SAMPLE_SIZE; j++) {
                stat.incData(
                    src_arr[src_start_idx + j],
                    src_null[src_start_idx + j] &&
                        (compress_option_.heuristic == NullHeuristic::SMART));
            }
        }
    }
    return stat.compute();
}

std::vector<uint32_t> SampleSmartNullCompressInterface::fillNullsLast(
    std::string_view src, const std::vector<bool> &nulls) {
    const uint32_t n = src.size() / word_size_;
    std::vector<uint32_t> filled_values;
    filled_values.reserve(n);
    auto src_ptr = (const uint32_t *)src.data();
    uint32_t last = 0;
    for (size_t i = 0; i < n; ++i) {
        // Read 4 bytes as a uint32_t
        uint32_t val = *src_ptr++;
        if (nulls[i])
            val = last;
        filled_values.push_back(val);
        last = val;
    }
    return filled_values;
}

std::vector<uint32_t> SampleSmartNullCompressInterface::fillNullsMostFreq(
    std::string_view src, const std::vector<bool> &nulls) {
    const uint32_t n = src.size() / word_size_;
    std::vector<uint32_t> filled_values;
    filled_values.reserve(n);
    auto src_ptr = (const uint32_t *)src.data();
    // Use Misra-Gries algorithm to count the most frequent value
    // Choose values with count > n / k, where k = 32
    constexpr uint32_t k = 32;
    std::array<uint32_t, k> most_freq, most_freq_count;
    for (size_t i = 0; i < k; ++i) {
        most_freq[i] = 0;
        most_freq_count[i] = 0;
    }
    for (size_t i = 0; i < n; ++i) {
        // Read 4 bytes as a uint32_t
        uint32_t val = *src_ptr++;
        if (nulls[i])
            continue;
        bool found = false;
        for (size_t j = 0; j < k; ++j) {
            if (most_freq[j] == val) {
                most_freq_count[j]++;
                found = true;
                break;
            }
        }
        if (!found) {
            for (size_t j = 0; j < k; ++j) {
                if (most_freq_count[j] == 0) {
                    most_freq[j] = val;
                    most_freq_count[j] = 1;
                    found = true;
                    break;
                }
            }
        }
        if (!found) {
            for (size_t j = 0; j < k; ++j)
                most_freq_count[j]--;
        }
    }
    // Find the most frequent value
    uint32_t most_freq_val = 0, most_freq_val_count = 0;
    for (size_t i = 0; i < k; ++i) {
        if (most_freq_count[i] > most_freq_val_count) {
            most_freq_val = most_freq[i];
            most_freq_val_count = most_freq_count[i];
        }
    }
    src_ptr = (const uint32_t *)src.data();
    // Fill nulls with the most frequent value
    for (size_t i = 0; i < n; ++i) {
        // Read 4 bytes as a uint32_t
        uint32_t val = *src_ptr++;
        if (nulls[i])
            val = most_freq_val;
        filled_values.push_back(val);
    }
    return filled_values;
}

std::vector<uint32_t> SampleSmartNullCompressInterface::fillNullsRandom(
    std::string_view src, const std::vector<bool> &nulls) {
    const uint32_t n = src.size() / word_size_;
    std::vector<uint32_t> filled_values;
    filled_values.reserve(n);
    auto src_ptr = (const uint32_t *)src.data();
    // Scan once to find the range
    uint32_t min_val = *src_ptr, max_val = *src_ptr;
    for (size_t i = 0; i < n; ++i) {
        // Read 4 bytes as a uint32_t
        uint32_t val = *src_ptr++;
        if (nulls[i])
            continue;
        if (val < min_val)
            min_val = val;
        if (val > max_val)
            max_val = val;
    }
    src_ptr = (const uint32_t *)src.data();
    std::mt19937_64 gen(0x20240303);
    std::uniform_int_distribution<> uniform(min_val, max_val);
    // Fill nulls with random values in the range
    for (size_t i = 0; i < n; ++i) {
        // Read 4 bytes as a uint32_t
        uint32_t val = *src_ptr++;
        if (nulls[i])
            val = uniform(gen);
        filled_values.push_back(val);
    }
    return filled_values;
}

} // namespace null_revisit
