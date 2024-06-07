#include "SampleCompressInterface.hpp"
#include "BitpackingInterface.hpp"
#include "DeltaInterface.hpp"
#include "RLEBranchlessInterface.hpp"
#include "VarInts.hpp"
#include "cycleclock.hpp"
#include <chrono>
#include <cstdio>

namespace null_revisit {
SampleCompressInterface::SampleCompressInterface(const std::string &codec_name)
    : CompressInterface(), word_size_(4) {
    interfaces[BITPACKING_C] = new BitpackingInterface();
    interfaces[BITPACKING_C]->compress_option_ = compress_option_;
    interfaces[RLE_C] = new RLEBranchlessInterface();
    interfaces[RLE_C]->compress_option_ = compress_option_;
    interfaces[DELTA_C] = new DeltaInterface();
    interfaces[DELTA_C]->compress_option_ = compress_option_;
}

SampleCompressInterface::~SampleCompressInterface() {
    delete interfaces[BITPACKING_C];
    delete interfaces[RLE_C];
    delete interfaces[DELTA_C];
}

size_t SampleCompressInterface::compress(std::string_view src,
                                         arrow::ResizableBuffer *dst,
                                         const std::vector<bool> &nulls) {
    extra_timer_.reset();
    auto sample_begin = benchmark::cycleclock::Now();
    const uint32_t n = src.size() / word_size_;
    // Stat
    // Choose the best scheme
    sample_segments_ = std::max(n / SAMPLE_UNIT_SIZE, 4u);
    auto compress_scheme = sampleStat((const uint32_t *)src.data(), n);
    if (compress_option_.always_delta)
        compress_scheme = DELTA_C;
    extra_timer_.sample_time += benchmark::cycleclock::Now() - sample_begin;
    incEncodingCounter(compress_scheme);
    // Compress
    const auto compress_size =
        interfaces[compress_scheme]->compress(src, dst, nulls);
    dst->Resize(compress_size + 1);
    // Write the scheme
    dst->mutable_data()[compress_size] = compress_scheme;
    return dst->size();
}

size_t
SampleCompressInterface::compress_with_null(std::string_view src,
                                            arrow::ResizableBuffer *dst,
                                            const std::vector<bool> &nulls) {
    return compress(src, dst, nulls);
}

size_t SampleCompressInterface::decompress(std::string_view src, char *dst,
                                           size_t dst_size) {
    // Read the scheme
    const auto compress_scheme = src.back();
    src.remove_suffix(1);
    // Decompress
    return interfaces[(uint8_t)(compress_scheme)]->decompress(src, dst,
                                                              dst_size);
}

SampleCompressInterface::CompressScheme
SampleCompressInterface::sampleStat(const uint32_t *src_arr, uint32_t n) {
    if (n <= SAMPLE_SIZE * sample_segments_) {
        SampleStat stat(n, src_arr[0]);
        for (uint32_t i = 1; i < n; i++) {
            uint32_t cur = src_arr[i];
            stat.incData(cur);
        }
        return stat.compute();
    }
    SampleStat stat(sample_segments_ * SAMPLE_SIZE, src_arr[0]);
    const auto sample_skip = n / sample_segments_;
    for (uint32_t i = 0, src_idx = 0; i < sample_segments_;
         i++, src_idx += sample_skip) {
        const auto src_start_idx =
            src_idx + std::rand() % (sample_skip - SAMPLE_SIZE);
        for (uint32_t j = 0; j < SAMPLE_SIZE; j++)
            stat.incData(src_arr[src_start_idx + j]);
    }
    return stat.compute();
}

} // namespace null_revisit
