#pragma once

#include "BitpackingInterface.hpp"

namespace null_revisit {
struct BitpackingStat {
    static const auto BLOCK_SIZE = BitpackingInterface::MINI_BLOCK_SIZE;

    BitpackingStat()
        : value_accu(0), delta_accu(0), value_tot_bits(0), delta_tot_bits(0),
          value_cnt(0), block_cnt(0) {}

    void incData(uint32_t value, uint32_t delta) {
        value_accu |= value;
        delta_accu |= delta;
        value_cnt++;
        if (value_cnt == BLOCK_SIZE) {
            value_tot_bits += bitWidth(value_accu);
            delta_tot_bits += bitWidth(delta_accu);
            block_cnt++;
            value_accu = 0;
            delta_accu = 0;
            value_cnt = 0;
        }
    }
    void finish() {
        if (value_cnt) {
            value_tot_bits += bitWidth(value_accu);
            delta_tot_bits += bitWidth(delta_accu);
            block_cnt++;
        }
        value_accu = 0;
        delta_accu = 0;
        value_cnt = 0;
    }
    void clear() {
        value_accu = 0;
        delta_accu = 0;
        value_tot_bits = 0;
        delta_tot_bits = 0;
        value_cnt = 0;
        block_cnt = 0;
    }
    BitpackingStat operator+(const BitpackingStat &rhs) {
        BitpackingStat ret(*this);
        ret += rhs;
        return ret;
    }
    BitpackingStat &operator+=(const BitpackingStat &rhs) {
        value_accu |= rhs.value_accu;
        delta_accu |= rhs.delta_accu;
        value_tot_bits += rhs.value_tot_bits;
        delta_tot_bits += rhs.delta_tot_bits;
        value_cnt += rhs.value_cnt;
        block_cnt += rhs.block_cnt;
        return *this;
    }

    uint32_t value_accu, delta_accu;
    uint32_t value_tot_bits, delta_tot_bits;
    uint32_t value_cnt, block_cnt;

  private:
    uint32_t bitWidth(uint32_t value) { return 32 - __builtin_clz(value); }
};

} // namespace null_revisit
