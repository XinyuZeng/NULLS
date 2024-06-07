#pragma once

#include <arrow/api.h>
#include <arrow/util/bitmap_builders.h>
#include <arrow/util/spaced.h>
#include <benchmark/benchmark.h>
#include <chrono>
#include <cstdint>
#include <immintrin.h>
#include <new>
#include <string>
#include <vector>

class Stat {
  public:
    Stat(std::string name, uint32_t try_count, int64_t comp_time,
         int64_t decomp_time, int64_t extra_sampling_time, size_t size,
         uint32_t rle_count, uint32_t delta_count, uint32_t bitpacking_count)
        : name(name), try_count(try_count), comp_time(comp_time),
          decomp_time(decomp_time), extra_sampling_time(extra_sampling_time),
          size(size), rle_count(rle_count), delta_count(delta_count),
          bitpacking_count(bitpacking_count) {}
    std::string to_json() const {
        std::string json = "{";
        json += "\"name\": \"" + name + "\",";
        json += "\"try_count\": " + std::to_string(try_count) + ",";
        json +=
            "\"comp_time\": " + std::to_string(cycles_to_seconds(comp_time)) +
            ",";
        json += "\"decomp_time\": " +
                std::to_string(cycles_to_seconds(decomp_time)) + ",";
        json += "\"extra_sampling_time\": " +
                std::to_string(cycles_to_seconds(extra_sampling_time)) + ",";
        json += "\"spaced_to_dense_time\": " +
                std::to_string(cycles_to_seconds(spaced_to_dense_time)) + ",";
        json += "\"size\": " + std::to_string(size) + ",";
        json += "\"rle_count\": " + std::to_string(rle_count) + ",";
        json += "\"delta_count\": " + std::to_string(delta_count) + ",";
        json += "\"bitpacking_count\": " + std::to_string(bitpacking_count);
        json += "\n}";
        return json;
    }
    static Stat best(const std::vector<Stat> &stats) {
        Stat best_stat = stats[0];
        auto best_size = best_stat.size;
        for (const auto &stat : stats) {
            if (stat.size < best_size) {
                best_stat = stat;
                best_size = stat.size;
            }
        }
        best_stat.name = "best";
        return best_stat;
    }

    std::string name;
    uint32_t try_count;
    int64_t comp_time, decomp_time;
    int64_t extra_sampling_time = 0;
    int64_t spaced_to_dense_time = 0;
    uint32_t size;
    uint32_t rle_count, delta_count, bitpacking_count;

  private:
    int64_t cycles_to_seconds(int64_t cycles) const {
        // return cycles / benchmark::BenchmarkReporter::Context()
        //                     .cpu_info.cycles_per_second;
        return cycles;
    }
};

static inline auto toNullBitmap(const std::vector<bool> &isNull) {
    std::vector<uint8_t> valid_bytes;
    valid_bytes.reserve(isNull.size());
    for (const auto null : isNull) {
        // In Arrow and DuckDB's validity bitmap, bit '1' means non-null, bit
        // '0' means null
        valid_bytes.push_back(!null);
    }
    return *arrow::internal::BytesToBits(valid_bytes);
}

static inline int SpacedExpandAVXInplace(uint32_t *buffer, int num_values,
                                         int null_count,
                                         const uint8_t *valid_bits) {
    // FIXME: can easily handle the modulo with regular scalar code from Arrow.
    assert(num_values % 512 == 0);
    // Point to end as we add the spacing from the back.
    int idx_decode = num_values - null_count;

    if (idx_decode == 0) {
        // All nulls, nothing more to do
        return num_values;
    }
    auto temp_buf = new (std::align_val_t{64}) uint8_t[64];
    int16_t *a_popcnt = (int16_t *)temp_buf;
    auto a_bm = (__mmask16 *)(valid_bits + num_values / 8 - 64);
    auto out = buffer + num_values - 16;
    buffer += idx_decode;
    for (int i = num_values; i > 0; i -= 512) {
        auto a_bm_512 = _mm512_load_si512(a_bm);
        _mm512_store_si512(a_popcnt, _mm512_popcnt_epi16(a_bm_512));
        for (int j = 31; j >= 0; --j) {
            buffer -= a_popcnt[j];
            __m512i a_expanded =
                _mm512_maskz_expand_epi32(a_bm[j], _mm512_loadu_si512(buffer));
            _mm512_storeu_si512(out, a_expanded);
            out -= 16;
        }
        a_bm -= 32;
    }
    ::operator delete[](temp_buf, std::align_val_t{64});
    return num_values;
}

static inline int SpacedExpandAVX(uint32_t *in_ptr, uint32_t *out_ptr,
                                  int num_values, int null_count,
                                  const uint8_t *valid_bits) {
    // FIXME: can easily handle the modulo with regular scalar code from Arrow.
    assert(num_values % 512 == 0);
    // Point to end as we add the spacing from the back.
    int idx_decode = num_values - null_count;

    if (idx_decode == 0) {
        // All nulls, nothing more to do
        return num_values;
    }
    auto temp_buf = new (std::align_val_t{64}) uint8_t[64];
    int16_t *a_popcnt = (int16_t *)temp_buf;
    auto a_bm = (__mmask16 *)(valid_bits + num_values / 8 - 64);
    auto out = out_ptr + num_values - 16;
    in_ptr += idx_decode;
    for (int i = num_values; i > 0; i -= 512) {
        auto a_bm_512 = _mm512_load_si512(a_bm);
        _mm512_store_si512(a_popcnt, _mm512_popcnt_epi16(a_bm_512));
        for (int j = 31; j >= 0; --j) {
            in_ptr -= a_popcnt[j];
            __m512i a_expanded =
                _mm512_maskz_expand_epi32(a_bm[j], _mm512_loadu_si512(in_ptr));
            _mm512_storeu_si512(out, a_expanded);
            out -= 16;
        }
        a_bm -= 32;
    }
    ::operator delete[](temp_buf, std::align_val_t{64});
    return num_values;
}

static inline void
UnrolledRLEDecoding16Values(uint32_t *__restrict dst_ptr,
                            const uint32_t *__restrict rle_values,
                            const uint32_t *__restrict rle_idxs) {
    dst_ptr[0] = rle_values[rle_idxs[0]];
    dst_ptr[1] = rle_values[rle_idxs[1]];
    dst_ptr[2] = rle_values[rle_idxs[2]];
    dst_ptr[3] = rle_values[rle_idxs[3]];
    dst_ptr[4] = rle_values[rle_idxs[4]];
    dst_ptr[5] = rle_values[rle_idxs[5]];
    dst_ptr[6] = rle_values[rle_idxs[6]];
    dst_ptr[7] = rle_values[rle_idxs[7]];
    dst_ptr[8] = rle_values[rle_idxs[8]];
    dst_ptr[9] = rle_values[rle_idxs[9]];
    dst_ptr[10] = rle_values[rle_idxs[10]];
    dst_ptr[11] = rle_values[rle_idxs[11]];
    dst_ptr[12] = rle_values[rle_idxs[12]];
    dst_ptr[13] = rle_values[rle_idxs[13]];
    dst_ptr[14] = rle_values[rle_idxs[14]];
    dst_ptr[15] = rle_values[rle_idxs[15]];
}
static inline void
UnrolledRLEDecoding512Values(uint32_t *__restrict dst_ptr,
                             const uint32_t *__restrict rle_values,
                             const uint32_t *__restrict rle_idxs) {
    dst_ptr[0] = rle_values[rle_idxs[0]];
    dst_ptr[1] = rle_values[rle_idxs[1]];
    dst_ptr[2] = rle_values[rle_idxs[2]];
    dst_ptr[3] = rle_values[rle_idxs[3]];
    dst_ptr[4] = rle_values[rle_idxs[4]];
    dst_ptr[5] = rle_values[rle_idxs[5]];
    dst_ptr[6] = rle_values[rle_idxs[6]];
    dst_ptr[7] = rle_values[rle_idxs[7]];
    dst_ptr[8] = rle_values[rle_idxs[8]];
    dst_ptr[9] = rle_values[rle_idxs[9]];
    dst_ptr[10] = rle_values[rle_idxs[10]];
    dst_ptr[11] = rle_values[rle_idxs[11]];
    dst_ptr[12] = rle_values[rle_idxs[12]];
    dst_ptr[13] = rle_values[rle_idxs[13]];
    dst_ptr[14] = rle_values[rle_idxs[14]];
    dst_ptr[15] = rle_values[rle_idxs[15]];
    dst_ptr[16] = rle_values[rle_idxs[16]];
    dst_ptr[17] = rle_values[rle_idxs[17]];
    dst_ptr[18] = rle_values[rle_idxs[18]];
    dst_ptr[19] = rle_values[rle_idxs[19]];
    dst_ptr[20] = rle_values[rle_idxs[20]];
    dst_ptr[21] = rle_values[rle_idxs[21]];
    dst_ptr[22] = rle_values[rle_idxs[22]];
    dst_ptr[23] = rle_values[rle_idxs[23]];
    dst_ptr[24] = rle_values[rle_idxs[24]];
    dst_ptr[25] = rle_values[rle_idxs[25]];
    dst_ptr[26] = rle_values[rle_idxs[26]];
    dst_ptr[27] = rle_values[rle_idxs[27]];
    dst_ptr[28] = rle_values[rle_idxs[28]];
    dst_ptr[29] = rle_values[rle_idxs[29]];
    dst_ptr[30] = rle_values[rle_idxs[30]];
    dst_ptr[31] = rle_values[rle_idxs[31]];
    dst_ptr[32] = rle_values[rle_idxs[32]];
    dst_ptr[33] = rle_values[rle_idxs[33]];
    dst_ptr[34] = rle_values[rle_idxs[34]];
    dst_ptr[35] = rle_values[rle_idxs[35]];
    dst_ptr[36] = rle_values[rle_idxs[36]];
    dst_ptr[37] = rle_values[rle_idxs[37]];
    dst_ptr[38] = rle_values[rle_idxs[38]];
    dst_ptr[39] = rle_values[rle_idxs[39]];
    dst_ptr[40] = rle_values[rle_idxs[40]];
    dst_ptr[41] = rle_values[rle_idxs[41]];
    dst_ptr[42] = rle_values[rle_idxs[42]];
    dst_ptr[43] = rle_values[rle_idxs[43]];
    dst_ptr[44] = rle_values[rle_idxs[44]];
    dst_ptr[45] = rle_values[rle_idxs[45]];
    dst_ptr[46] = rle_values[rle_idxs[46]];
    dst_ptr[47] = rle_values[rle_idxs[47]];
    dst_ptr[48] = rle_values[rle_idxs[48]];
    dst_ptr[49] = rle_values[rle_idxs[49]];
    dst_ptr[50] = rle_values[rle_idxs[50]];
    dst_ptr[51] = rle_values[rle_idxs[51]];
    dst_ptr[52] = rle_values[rle_idxs[52]];
    dst_ptr[53] = rle_values[rle_idxs[53]];
    dst_ptr[54] = rle_values[rle_idxs[54]];
    dst_ptr[55] = rle_values[rle_idxs[55]];
    dst_ptr[56] = rle_values[rle_idxs[56]];
    dst_ptr[57] = rle_values[rle_idxs[57]];
    dst_ptr[58] = rle_values[rle_idxs[58]];
    dst_ptr[59] = rle_values[rle_idxs[59]];
    dst_ptr[60] = rle_values[rle_idxs[60]];
    dst_ptr[61] = rle_values[rle_idxs[61]];
    dst_ptr[62] = rle_values[rle_idxs[62]];
    dst_ptr[63] = rle_values[rle_idxs[63]];
    dst_ptr[64] = rle_values[rle_idxs[64]];
    dst_ptr[65] = rle_values[rle_idxs[65]];
    dst_ptr[66] = rle_values[rle_idxs[66]];
    dst_ptr[67] = rle_values[rle_idxs[67]];
    dst_ptr[68] = rle_values[rle_idxs[68]];
    dst_ptr[69] = rle_values[rle_idxs[69]];
    dst_ptr[70] = rle_values[rle_idxs[70]];
    dst_ptr[71] = rle_values[rle_idxs[71]];
    dst_ptr[72] = rle_values[rle_idxs[72]];
    dst_ptr[73] = rle_values[rle_idxs[73]];
    dst_ptr[74] = rle_values[rle_idxs[74]];
    dst_ptr[75] = rle_values[rle_idxs[75]];
    dst_ptr[76] = rle_values[rle_idxs[76]];
    dst_ptr[77] = rle_values[rle_idxs[77]];
    dst_ptr[78] = rle_values[rle_idxs[78]];
    dst_ptr[79] = rle_values[rle_idxs[79]];
    dst_ptr[80] = rle_values[rle_idxs[80]];
    dst_ptr[81] = rle_values[rle_idxs[81]];
    dst_ptr[82] = rle_values[rle_idxs[82]];
    dst_ptr[83] = rle_values[rle_idxs[83]];
    dst_ptr[84] = rle_values[rle_idxs[84]];
    dst_ptr[85] = rle_values[rle_idxs[85]];
    dst_ptr[86] = rle_values[rle_idxs[86]];
    dst_ptr[87] = rle_values[rle_idxs[87]];
    dst_ptr[88] = rle_values[rle_idxs[88]];
    dst_ptr[89] = rle_values[rle_idxs[89]];
    dst_ptr[90] = rle_values[rle_idxs[90]];
    dst_ptr[91] = rle_values[rle_idxs[91]];
    dst_ptr[92] = rle_values[rle_idxs[92]];
    dst_ptr[93] = rle_values[rle_idxs[93]];
    dst_ptr[94] = rle_values[rle_idxs[94]];
    dst_ptr[95] = rle_values[rle_idxs[95]];
    dst_ptr[96] = rle_values[rle_idxs[96]];
    dst_ptr[97] = rle_values[rle_idxs[97]];
    dst_ptr[98] = rle_values[rle_idxs[98]];
    dst_ptr[99] = rle_values[rle_idxs[99]];
    dst_ptr[100] = rle_values[rle_idxs[100]];
    dst_ptr[101] = rle_values[rle_idxs[101]];
    dst_ptr[102] = rle_values[rle_idxs[102]];
    dst_ptr[103] = rle_values[rle_idxs[103]];
    dst_ptr[104] = rle_values[rle_idxs[104]];
    dst_ptr[105] = rle_values[rle_idxs[105]];
    dst_ptr[106] = rle_values[rle_idxs[106]];
    dst_ptr[107] = rle_values[rle_idxs[107]];
    dst_ptr[108] = rle_values[rle_idxs[108]];
    dst_ptr[109] = rle_values[rle_idxs[109]];
    dst_ptr[110] = rle_values[rle_idxs[110]];
    dst_ptr[111] = rle_values[rle_idxs[111]];
    dst_ptr[112] = rle_values[rle_idxs[112]];
    dst_ptr[113] = rle_values[rle_idxs[113]];
    dst_ptr[114] = rle_values[rle_idxs[114]];
    dst_ptr[115] = rle_values[rle_idxs[115]];
    dst_ptr[116] = rle_values[rle_idxs[116]];
    dst_ptr[117] = rle_values[rle_idxs[117]];
    dst_ptr[118] = rle_values[rle_idxs[118]];
    dst_ptr[119] = rle_values[rle_idxs[119]];
    dst_ptr[120] = rle_values[rle_idxs[120]];
    dst_ptr[121] = rle_values[rle_idxs[121]];
    dst_ptr[122] = rle_values[rle_idxs[122]];
    dst_ptr[123] = rle_values[rle_idxs[123]];
    dst_ptr[124] = rle_values[rle_idxs[124]];
    dst_ptr[125] = rle_values[rle_idxs[125]];
    dst_ptr[126] = rle_values[rle_idxs[126]];
    dst_ptr[127] = rle_values[rle_idxs[127]];
    dst_ptr[128] = rle_values[rle_idxs[128]];
    dst_ptr[129] = rle_values[rle_idxs[129]];
    dst_ptr[130] = rle_values[rle_idxs[130]];
    dst_ptr[131] = rle_values[rle_idxs[131]];
    dst_ptr[132] = rle_values[rle_idxs[132]];
    dst_ptr[133] = rle_values[rle_idxs[133]];
    dst_ptr[134] = rle_values[rle_idxs[134]];
    dst_ptr[135] = rle_values[rle_idxs[135]];
    dst_ptr[136] = rle_values[rle_idxs[136]];
    dst_ptr[137] = rle_values[rle_idxs[137]];
    dst_ptr[138] = rle_values[rle_idxs[138]];
    dst_ptr[139] = rle_values[rle_idxs[139]];
    dst_ptr[140] = rle_values[rle_idxs[140]];
    dst_ptr[141] = rle_values[rle_idxs[141]];
    dst_ptr[142] = rle_values[rle_idxs[142]];
    dst_ptr[143] = rle_values[rle_idxs[143]];
    dst_ptr[144] = rle_values[rle_idxs[144]];
    dst_ptr[145] = rle_values[rle_idxs[145]];
    dst_ptr[146] = rle_values[rle_idxs[146]];
    dst_ptr[147] = rle_values[rle_idxs[147]];
    dst_ptr[148] = rle_values[rle_idxs[148]];
    dst_ptr[149] = rle_values[rle_idxs[149]];
    dst_ptr[150] = rle_values[rle_idxs[150]];
    dst_ptr[151] = rle_values[rle_idxs[151]];
    dst_ptr[152] = rle_values[rle_idxs[152]];
    dst_ptr[153] = rle_values[rle_idxs[153]];
    dst_ptr[154] = rle_values[rle_idxs[154]];
    dst_ptr[155] = rle_values[rle_idxs[155]];
    dst_ptr[156] = rle_values[rle_idxs[156]];
    dst_ptr[157] = rle_values[rle_idxs[157]];
    dst_ptr[158] = rle_values[rle_idxs[158]];
    dst_ptr[159] = rle_values[rle_idxs[159]];
    dst_ptr[160] = rle_values[rle_idxs[160]];
    dst_ptr[161] = rle_values[rle_idxs[161]];
    dst_ptr[162] = rle_values[rle_idxs[162]];
    dst_ptr[163] = rle_values[rle_idxs[163]];
    dst_ptr[164] = rle_values[rle_idxs[164]];
    dst_ptr[165] = rle_values[rle_idxs[165]];
    dst_ptr[166] = rle_values[rle_idxs[166]];
    dst_ptr[167] = rle_values[rle_idxs[167]];
    dst_ptr[168] = rle_values[rle_idxs[168]];
    dst_ptr[169] = rle_values[rle_idxs[169]];
    dst_ptr[170] = rle_values[rle_idxs[170]];
    dst_ptr[171] = rle_values[rle_idxs[171]];
    dst_ptr[172] = rle_values[rle_idxs[172]];
    dst_ptr[173] = rle_values[rle_idxs[173]];
    dst_ptr[174] = rle_values[rle_idxs[174]];
    dst_ptr[175] = rle_values[rle_idxs[175]];
    dst_ptr[176] = rle_values[rle_idxs[176]];
    dst_ptr[177] = rle_values[rle_idxs[177]];
    dst_ptr[178] = rle_values[rle_idxs[178]];
    dst_ptr[179] = rle_values[rle_idxs[179]];
    dst_ptr[180] = rle_values[rle_idxs[180]];
    dst_ptr[181] = rle_values[rle_idxs[181]];
    dst_ptr[182] = rle_values[rle_idxs[182]];
    dst_ptr[183] = rle_values[rle_idxs[183]];
    dst_ptr[184] = rle_values[rle_idxs[184]];
    dst_ptr[185] = rle_values[rle_idxs[185]];
    dst_ptr[186] = rle_values[rle_idxs[186]];
    dst_ptr[187] = rle_values[rle_idxs[187]];
    dst_ptr[188] = rle_values[rle_idxs[188]];
    dst_ptr[189] = rle_values[rle_idxs[189]];
    dst_ptr[190] = rle_values[rle_idxs[190]];
    dst_ptr[191] = rle_values[rle_idxs[191]];
    dst_ptr[192] = rle_values[rle_idxs[192]];
    dst_ptr[193] = rle_values[rle_idxs[193]];
    dst_ptr[194] = rle_values[rle_idxs[194]];
    dst_ptr[195] = rle_values[rle_idxs[195]];
    dst_ptr[196] = rle_values[rle_idxs[196]];
    dst_ptr[197] = rle_values[rle_idxs[197]];
    dst_ptr[198] = rle_values[rle_idxs[198]];
    dst_ptr[199] = rle_values[rle_idxs[199]];
    dst_ptr[200] = rle_values[rle_idxs[200]];
    dst_ptr[201] = rle_values[rle_idxs[201]];
    dst_ptr[202] = rle_values[rle_idxs[202]];
    dst_ptr[203] = rle_values[rle_idxs[203]];
    dst_ptr[204] = rle_values[rle_idxs[204]];
    dst_ptr[205] = rle_values[rle_idxs[205]];
    dst_ptr[206] = rle_values[rle_idxs[206]];
    dst_ptr[207] = rle_values[rle_idxs[207]];
    dst_ptr[208] = rle_values[rle_idxs[208]];
    dst_ptr[209] = rle_values[rle_idxs[209]];
    dst_ptr[210] = rle_values[rle_idxs[210]];
    dst_ptr[211] = rle_values[rle_idxs[211]];
    dst_ptr[212] = rle_values[rle_idxs[212]];
    dst_ptr[213] = rle_values[rle_idxs[213]];
    dst_ptr[214] = rle_values[rle_idxs[214]];
    dst_ptr[215] = rle_values[rle_idxs[215]];
    dst_ptr[216] = rle_values[rle_idxs[216]];
    dst_ptr[217] = rle_values[rle_idxs[217]];
    dst_ptr[218] = rle_values[rle_idxs[218]];
    dst_ptr[219] = rle_values[rle_idxs[219]];
    dst_ptr[220] = rle_values[rle_idxs[220]];
    dst_ptr[221] = rle_values[rle_idxs[221]];
    dst_ptr[222] = rle_values[rle_idxs[222]];
    dst_ptr[223] = rle_values[rle_idxs[223]];
    dst_ptr[224] = rle_values[rle_idxs[224]];
    dst_ptr[225] = rle_values[rle_idxs[225]];
    dst_ptr[226] = rle_values[rle_idxs[226]];
    dst_ptr[227] = rle_values[rle_idxs[227]];
    dst_ptr[228] = rle_values[rle_idxs[228]];
    dst_ptr[229] = rle_values[rle_idxs[229]];
    dst_ptr[230] = rle_values[rle_idxs[230]];
    dst_ptr[231] = rle_values[rle_idxs[231]];
    dst_ptr[232] = rle_values[rle_idxs[232]];
    dst_ptr[233] = rle_values[rle_idxs[233]];
    dst_ptr[234] = rle_values[rle_idxs[234]];
    dst_ptr[235] = rle_values[rle_idxs[235]];
    dst_ptr[236] = rle_values[rle_idxs[236]];
    dst_ptr[237] = rle_values[rle_idxs[237]];
    dst_ptr[238] = rle_values[rle_idxs[238]];
    dst_ptr[239] = rle_values[rle_idxs[239]];
    dst_ptr[240] = rle_values[rle_idxs[240]];
    dst_ptr[241] = rle_values[rle_idxs[241]];
    dst_ptr[242] = rle_values[rle_idxs[242]];
    dst_ptr[243] = rle_values[rle_idxs[243]];
    dst_ptr[244] = rle_values[rle_idxs[244]];
    dst_ptr[245] = rle_values[rle_idxs[245]];
    dst_ptr[246] = rle_values[rle_idxs[246]];
    dst_ptr[247] = rle_values[rle_idxs[247]];
    dst_ptr[248] = rle_values[rle_idxs[248]];
    dst_ptr[249] = rle_values[rle_idxs[249]];
    dst_ptr[250] = rle_values[rle_idxs[250]];
    dst_ptr[251] = rle_values[rle_idxs[251]];
    dst_ptr[252] = rle_values[rle_idxs[252]];
    dst_ptr[253] = rle_values[rle_idxs[253]];
    dst_ptr[254] = rle_values[rle_idxs[254]];
    dst_ptr[255] = rle_values[rle_idxs[255]];
    dst_ptr[256] = rle_values[rle_idxs[256]];
    dst_ptr[257] = rle_values[rle_idxs[257]];
    dst_ptr[258] = rle_values[rle_idxs[258]];
    dst_ptr[259] = rle_values[rle_idxs[259]];
    dst_ptr[260] = rle_values[rle_idxs[260]];
    dst_ptr[261] = rle_values[rle_idxs[261]];
    dst_ptr[262] = rle_values[rle_idxs[262]];
    dst_ptr[263] = rle_values[rle_idxs[263]];
    dst_ptr[264] = rle_values[rle_idxs[264]];
    dst_ptr[265] = rle_values[rle_idxs[265]];
    dst_ptr[266] = rle_values[rle_idxs[266]];
    dst_ptr[267] = rle_values[rle_idxs[267]];
    dst_ptr[268] = rle_values[rle_idxs[268]];
    dst_ptr[269] = rle_values[rle_idxs[269]];
    dst_ptr[270] = rle_values[rle_idxs[270]];
    dst_ptr[271] = rle_values[rle_idxs[271]];
    dst_ptr[272] = rle_values[rle_idxs[272]];
    dst_ptr[273] = rle_values[rle_idxs[273]];
    dst_ptr[274] = rle_values[rle_idxs[274]];
    dst_ptr[275] = rle_values[rle_idxs[275]];
    dst_ptr[276] = rle_values[rle_idxs[276]];
    dst_ptr[277] = rle_values[rle_idxs[277]];
    dst_ptr[278] = rle_values[rle_idxs[278]];
    dst_ptr[279] = rle_values[rle_idxs[279]];
    dst_ptr[280] = rle_values[rle_idxs[280]];
    dst_ptr[281] = rle_values[rle_idxs[281]];
    dst_ptr[282] = rle_values[rle_idxs[282]];
    dst_ptr[283] = rle_values[rle_idxs[283]];
    dst_ptr[284] = rle_values[rle_idxs[284]];
    dst_ptr[285] = rle_values[rle_idxs[285]];
    dst_ptr[286] = rle_values[rle_idxs[286]];
    dst_ptr[287] = rle_values[rle_idxs[287]];
    dst_ptr[288] = rle_values[rle_idxs[288]];
    dst_ptr[289] = rle_values[rle_idxs[289]];
    dst_ptr[290] = rle_values[rle_idxs[290]];
    dst_ptr[291] = rle_values[rle_idxs[291]];
    dst_ptr[292] = rle_values[rle_idxs[292]];
    dst_ptr[293] = rle_values[rle_idxs[293]];
    dst_ptr[294] = rle_values[rle_idxs[294]];
    dst_ptr[295] = rle_values[rle_idxs[295]];
    dst_ptr[296] = rle_values[rle_idxs[296]];
    dst_ptr[297] = rle_values[rle_idxs[297]];
    dst_ptr[298] = rle_values[rle_idxs[298]];
    dst_ptr[299] = rle_values[rle_idxs[299]];
    dst_ptr[300] = rle_values[rle_idxs[300]];
    dst_ptr[301] = rle_values[rle_idxs[301]];
    dst_ptr[302] = rle_values[rle_idxs[302]];
    dst_ptr[303] = rle_values[rle_idxs[303]];
    dst_ptr[304] = rle_values[rle_idxs[304]];
    dst_ptr[305] = rle_values[rle_idxs[305]];
    dst_ptr[306] = rle_values[rle_idxs[306]];
    dst_ptr[307] = rle_values[rle_idxs[307]];
    dst_ptr[308] = rle_values[rle_idxs[308]];
    dst_ptr[309] = rle_values[rle_idxs[309]];
    dst_ptr[310] = rle_values[rle_idxs[310]];
    dst_ptr[311] = rle_values[rle_idxs[311]];
    dst_ptr[312] = rle_values[rle_idxs[312]];
    dst_ptr[313] = rle_values[rle_idxs[313]];
    dst_ptr[314] = rle_values[rle_idxs[314]];
    dst_ptr[315] = rle_values[rle_idxs[315]];
    dst_ptr[316] = rle_values[rle_idxs[316]];
    dst_ptr[317] = rle_values[rle_idxs[317]];
    dst_ptr[318] = rle_values[rle_idxs[318]];
    dst_ptr[319] = rle_values[rle_idxs[319]];
    dst_ptr[320] = rle_values[rle_idxs[320]];
    dst_ptr[321] = rle_values[rle_idxs[321]];
    dst_ptr[322] = rle_values[rle_idxs[322]];
    dst_ptr[323] = rle_values[rle_idxs[323]];
    dst_ptr[324] = rle_values[rle_idxs[324]];
    dst_ptr[325] = rle_values[rle_idxs[325]];
    dst_ptr[326] = rle_values[rle_idxs[326]];
    dst_ptr[327] = rle_values[rle_idxs[327]];
    dst_ptr[328] = rle_values[rle_idxs[328]];
    dst_ptr[329] = rle_values[rle_idxs[329]];
    dst_ptr[330] = rle_values[rle_idxs[330]];
    dst_ptr[331] = rle_values[rle_idxs[331]];
    dst_ptr[332] = rle_values[rle_idxs[332]];
    dst_ptr[333] = rle_values[rle_idxs[333]];
    dst_ptr[334] = rle_values[rle_idxs[334]];
    dst_ptr[335] = rle_values[rle_idxs[335]];
    dst_ptr[336] = rle_values[rle_idxs[336]];
    dst_ptr[337] = rle_values[rle_idxs[337]];
    dst_ptr[338] = rle_values[rle_idxs[338]];
    dst_ptr[339] = rle_values[rle_idxs[339]];
    dst_ptr[340] = rle_values[rle_idxs[340]];
    dst_ptr[341] = rle_values[rle_idxs[341]];
    dst_ptr[342] = rle_values[rle_idxs[342]];
    dst_ptr[343] = rle_values[rle_idxs[343]];
    dst_ptr[344] = rle_values[rle_idxs[344]];
    dst_ptr[345] = rle_values[rle_idxs[345]];
    dst_ptr[346] = rle_values[rle_idxs[346]];
    dst_ptr[347] = rle_values[rle_idxs[347]];
    dst_ptr[348] = rle_values[rle_idxs[348]];
    dst_ptr[349] = rle_values[rle_idxs[349]];
    dst_ptr[350] = rle_values[rle_idxs[350]];
    dst_ptr[351] = rle_values[rle_idxs[351]];
    dst_ptr[352] = rle_values[rle_idxs[352]];
    dst_ptr[353] = rle_values[rle_idxs[353]];
    dst_ptr[354] = rle_values[rle_idxs[354]];
    dst_ptr[355] = rle_values[rle_idxs[355]];
    dst_ptr[356] = rle_values[rle_idxs[356]];
    dst_ptr[357] = rle_values[rle_idxs[357]];
    dst_ptr[358] = rle_values[rle_idxs[358]];
    dst_ptr[359] = rle_values[rle_idxs[359]];
    dst_ptr[360] = rle_values[rle_idxs[360]];
    dst_ptr[361] = rle_values[rle_idxs[361]];
    dst_ptr[362] = rle_values[rle_idxs[362]];
    dst_ptr[363] = rle_values[rle_idxs[363]];
    dst_ptr[364] = rle_values[rle_idxs[364]];
    dst_ptr[365] = rle_values[rle_idxs[365]];
    dst_ptr[366] = rle_values[rle_idxs[366]];
    dst_ptr[367] = rle_values[rle_idxs[367]];
    dst_ptr[368] = rle_values[rle_idxs[368]];
    dst_ptr[369] = rle_values[rle_idxs[369]];
    dst_ptr[370] = rle_values[rle_idxs[370]];
    dst_ptr[371] = rle_values[rle_idxs[371]];
    dst_ptr[372] = rle_values[rle_idxs[372]];
    dst_ptr[373] = rle_values[rle_idxs[373]];
    dst_ptr[374] = rle_values[rle_idxs[374]];
    dst_ptr[375] = rle_values[rle_idxs[375]];
    dst_ptr[376] = rle_values[rle_idxs[376]];
    dst_ptr[377] = rle_values[rle_idxs[377]];
    dst_ptr[378] = rle_values[rle_idxs[378]];
    dst_ptr[379] = rle_values[rle_idxs[379]];
    dst_ptr[380] = rle_values[rle_idxs[380]];
    dst_ptr[381] = rle_values[rle_idxs[381]];
    dst_ptr[382] = rle_values[rle_idxs[382]];
    dst_ptr[383] = rle_values[rle_idxs[383]];
    dst_ptr[384] = rle_values[rle_idxs[384]];
    dst_ptr[385] = rle_values[rle_idxs[385]];
    dst_ptr[386] = rle_values[rle_idxs[386]];
    dst_ptr[387] = rle_values[rle_idxs[387]];
    dst_ptr[388] = rle_values[rle_idxs[388]];
    dst_ptr[389] = rle_values[rle_idxs[389]];
    dst_ptr[390] = rle_values[rle_idxs[390]];
    dst_ptr[391] = rle_values[rle_idxs[391]];
    dst_ptr[392] = rle_values[rle_idxs[392]];
    dst_ptr[393] = rle_values[rle_idxs[393]];
    dst_ptr[394] = rle_values[rle_idxs[394]];
    dst_ptr[395] = rle_values[rle_idxs[395]];
    dst_ptr[396] = rle_values[rle_idxs[396]];
    dst_ptr[397] = rle_values[rle_idxs[397]];
    dst_ptr[398] = rle_values[rle_idxs[398]];
    dst_ptr[399] = rle_values[rle_idxs[399]];
    dst_ptr[400] = rle_values[rle_idxs[400]];
    dst_ptr[401] = rle_values[rle_idxs[401]];
    dst_ptr[402] = rle_values[rle_idxs[402]];
    dst_ptr[403] = rle_values[rle_idxs[403]];
    dst_ptr[404] = rle_values[rle_idxs[404]];
    dst_ptr[405] = rle_values[rle_idxs[405]];
    dst_ptr[406] = rle_values[rle_idxs[406]];
    dst_ptr[407] = rle_values[rle_idxs[407]];
    dst_ptr[408] = rle_values[rle_idxs[408]];
    dst_ptr[409] = rle_values[rle_idxs[409]];
    dst_ptr[410] = rle_values[rle_idxs[410]];
    dst_ptr[411] = rle_values[rle_idxs[411]];
    dst_ptr[412] = rle_values[rle_idxs[412]];
    dst_ptr[413] = rle_values[rle_idxs[413]];
    dst_ptr[414] = rle_values[rle_idxs[414]];
    dst_ptr[415] = rle_values[rle_idxs[415]];
    dst_ptr[416] = rle_values[rle_idxs[416]];
    dst_ptr[417] = rle_values[rle_idxs[417]];
    dst_ptr[418] = rle_values[rle_idxs[418]];
    dst_ptr[419] = rle_values[rle_idxs[419]];
    dst_ptr[420] = rle_values[rle_idxs[420]];
    dst_ptr[421] = rle_values[rle_idxs[421]];
    dst_ptr[422] = rle_values[rle_idxs[422]];
    dst_ptr[423] = rle_values[rle_idxs[423]];
    dst_ptr[424] = rle_values[rle_idxs[424]];
    dst_ptr[425] = rle_values[rle_idxs[425]];
    dst_ptr[426] = rle_values[rle_idxs[426]];
    dst_ptr[427] = rle_values[rle_idxs[427]];
    dst_ptr[428] = rle_values[rle_idxs[428]];
    dst_ptr[429] = rle_values[rle_idxs[429]];
    dst_ptr[430] = rle_values[rle_idxs[430]];
    dst_ptr[431] = rle_values[rle_idxs[431]];
    dst_ptr[432] = rle_values[rle_idxs[432]];
    dst_ptr[433] = rle_values[rle_idxs[433]];
    dst_ptr[434] = rle_values[rle_idxs[434]];
    dst_ptr[435] = rle_values[rle_idxs[435]];
    dst_ptr[436] = rle_values[rle_idxs[436]];
    dst_ptr[437] = rle_values[rle_idxs[437]];
    dst_ptr[438] = rle_values[rle_idxs[438]];
    dst_ptr[439] = rle_values[rle_idxs[439]];
    dst_ptr[440] = rle_values[rle_idxs[440]];
    dst_ptr[441] = rle_values[rle_idxs[441]];
    dst_ptr[442] = rle_values[rle_idxs[442]];
    dst_ptr[443] = rle_values[rle_idxs[443]];
    dst_ptr[444] = rle_values[rle_idxs[444]];
    dst_ptr[445] = rle_values[rle_idxs[445]];
    dst_ptr[446] = rle_values[rle_idxs[446]];
    dst_ptr[447] = rle_values[rle_idxs[447]];
    dst_ptr[448] = rle_values[rle_idxs[448]];
    dst_ptr[449] = rle_values[rle_idxs[449]];
    dst_ptr[450] = rle_values[rle_idxs[450]];
    dst_ptr[451] = rle_values[rle_idxs[451]];
    dst_ptr[452] = rle_values[rle_idxs[452]];
    dst_ptr[453] = rle_values[rle_idxs[453]];
    dst_ptr[454] = rle_values[rle_idxs[454]];
    dst_ptr[455] = rle_values[rle_idxs[455]];
    dst_ptr[456] = rle_values[rle_idxs[456]];
    dst_ptr[457] = rle_values[rle_idxs[457]];
    dst_ptr[458] = rle_values[rle_idxs[458]];
    dst_ptr[459] = rle_values[rle_idxs[459]];
    dst_ptr[460] = rle_values[rle_idxs[460]];
    dst_ptr[461] = rle_values[rle_idxs[461]];
    dst_ptr[462] = rle_values[rle_idxs[462]];
    dst_ptr[463] = rle_values[rle_idxs[463]];
    dst_ptr[464] = rle_values[rle_idxs[464]];
    dst_ptr[465] = rle_values[rle_idxs[465]];
    dst_ptr[466] = rle_values[rle_idxs[466]];
    dst_ptr[467] = rle_values[rle_idxs[467]];
    dst_ptr[468] = rle_values[rle_idxs[468]];
    dst_ptr[469] = rle_values[rle_idxs[469]];
    dst_ptr[470] = rle_values[rle_idxs[470]];
    dst_ptr[471] = rle_values[rle_idxs[471]];
    dst_ptr[472] = rle_values[rle_idxs[472]];
    dst_ptr[473] = rle_values[rle_idxs[473]];
    dst_ptr[474] = rle_values[rle_idxs[474]];
    dst_ptr[475] = rle_values[rle_idxs[475]];
    dst_ptr[476] = rle_values[rle_idxs[476]];
    dst_ptr[477] = rle_values[rle_idxs[477]];
    dst_ptr[478] = rle_values[rle_idxs[478]];
    dst_ptr[479] = rle_values[rle_idxs[479]];
    dst_ptr[480] = rle_values[rle_idxs[480]];
    dst_ptr[481] = rle_values[rle_idxs[481]];
    dst_ptr[482] = rle_values[rle_idxs[482]];
    dst_ptr[483] = rle_values[rle_idxs[483]];
    dst_ptr[484] = rle_values[rle_idxs[484]];
    dst_ptr[485] = rle_values[rle_idxs[485]];
    dst_ptr[486] = rle_values[rle_idxs[486]];
    dst_ptr[487] = rle_values[rle_idxs[487]];
    dst_ptr[488] = rle_values[rle_idxs[488]];
    dst_ptr[489] = rle_values[rle_idxs[489]];
    dst_ptr[490] = rle_values[rle_idxs[490]];
    dst_ptr[491] = rle_values[rle_idxs[491]];
    dst_ptr[492] = rle_values[rle_idxs[492]];
    dst_ptr[493] = rle_values[rle_idxs[493]];
    dst_ptr[494] = rle_values[rle_idxs[494]];
    dst_ptr[495] = rle_values[rle_idxs[495]];
    dst_ptr[496] = rle_values[rle_idxs[496]];
    dst_ptr[497] = rle_values[rle_idxs[497]];
    dst_ptr[498] = rle_values[rle_idxs[498]];
    dst_ptr[499] = rle_values[rle_idxs[499]];
    dst_ptr[500] = rle_values[rle_idxs[500]];
    dst_ptr[501] = rle_values[rle_idxs[501]];
    dst_ptr[502] = rle_values[rle_idxs[502]];
    dst_ptr[503] = rle_values[rle_idxs[503]];
    dst_ptr[504] = rle_values[rle_idxs[504]];
    dst_ptr[505] = rle_values[rle_idxs[505]];
    dst_ptr[506] = rle_values[rle_idxs[506]];
    dst_ptr[507] = rle_values[rle_idxs[507]];
    dst_ptr[508] = rle_values[rle_idxs[508]];
    dst_ptr[509] = rle_values[rle_idxs[509]];
    dst_ptr[510] = rle_values[rle_idxs[510]];
    dst_ptr[511] = rle_values[rle_idxs[511]];
}