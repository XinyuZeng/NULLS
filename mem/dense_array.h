#pragma once
#include <arrow/buffer.h>
#include <arrow/util/bit_util.h>

#include <cassert>
#include <cstddef>
#include <memory>
#include <vector>

#include "array_base.h"
#include "utils/bit_util.h"
#include "utils/exceptions.hpp"
#include "utils/macros.hpp"

namespace null_revisit {
class SpacedArray;
class DenseArray : public BaseArray {
  public:
    DenseArray(const std::shared_ptr<arrow::Array> &array) : BaseArray() {
        auto visitor = [](const void *null_vector_ptr, size_t i) -> bool {
            return arrow::bit_util::GetBit((const uint8_t *)null_vector_ptr, i);
        };
        Init((const default_type *)array->data()->buffers[1]->data(),
             array->data()->buffers[0] ? array->data()->buffers[0]->data()
                                       : nullptr,
             array->length(), visitor);
    }

    DenseArray(const default_type *data, const bool *null_vector, size_t length)
        : BaseArray() {
        auto visitor = [](const void *null_vector_ptr, size_t i) -> bool {
            return ((const bool *)null_vector_ptr)[i];
        };
        Init(data, null_vector, length, visitor);
    }

    ~DenseArray() = default;

    void CompareWithCol(const BaseArray &other,
                        std::shared_ptr<Buffer> output) const override {
        Exception::NYI("polymorphism");
    }
    void CompareWithConst(const BaseArray &other,
                          std::shared_ptr<Buffer> output) const override {
        Exception::NYI("polymorphism");
    }
    void CompareWithColSVPartial(const SpacedArray &other, SelVector &output,
                                 SelVector &sel_vec) const;
    void CompareWithColSVManual(const SpacedArray &other, SelVector &output,
                                SelVector &sel_vec) const;
    int CompareWithColSVPartial(const DenseArray &other, SelVector &output,
                                SelVector &sel_vec) const;
    template <bool eval_filter_first = false>
    int CompareWithColSVPartialBranch(const DenseArray &other,
                                      SelVector &output,
                                      SelVector &sel_vec) const;
    int CompareWithColSVPartialFlat(DenseArray &other, SelVector &output,
                                    SelVector &sel_vec);
    int CompareWithColSVPartialFlatBranch(DenseArray &other, SelVector &output,
                                          SelVector &sel_vec);
    int CompareWithColSVPartialDuckDBFlat(const DenseArray &other,
                                          SelVector &output,
                                          SelVector &sel_vec) const;
    void CompareWithColSVManual(const DenseArray &other, SelVector &output,
                                SelVector &sel_vec) const;
    void SumByGroupBM(const std::vector<int32_t> &group_ids,
                      std::vector<int64_t> &sum) const override;
    void SumByGroupSV(const std::vector<int32_t> &group_ids,
                      std::vector<int64_t> &sum) const override;
    void CompareWithCol(const DenseArray &other,
                        std::shared_ptr<Buffer> output) const {
        CompareWithColNaiveScatter(other, output);
    }
    void CompareWithColNaiveScatter(const DenseArray &other,
                                    std::shared_ptr<Buffer> output) const;
    void CompareWithColAVXScatter(const DenseArray &other,
                                  std::shared_ptr<Buffer> output) const;
    template <bool five_op = false>
    void CompareWithColAVXExpand(const DenseArray &other,
                                 std::shared_ptr<Buffer> output) const;
    void CompareWithColAVXExpandShift(const DenseArray &other,
                                      std::shared_ptr<Buffer> output) const;
    void CompareWithColDirect(const DenseArray &other,
                              std::shared_ptr<Buffer> output) const;
    void CompareWithColAVXScatter(const SpacedArray &other,
                                  std::shared_ptr<Buffer> output) const;
    void CompareWithColAVXExpand(const SpacedArray &other,
                                 std::shared_ptr<Buffer> output) const;
    void CompareWithColDirect(const SpacedArray &other,
                              std::shared_ptr<Buffer> output) const;

  private:
    // null_values_ptr may be bit or byte stored
    template <class Visitor>
    void Init(const default_type *data, const void *null_values_ptr,
              size_t length, Visitor &&visit) {
        this->type = ArrayType::Dense;
        this->length = length;
        this->null_count = 0;
        this->buffers.push_back(*arrow::AllocateEmptyBitmap(length));
        for (size_t i = 0; i < length; ++i) {
            if (null_values_ptr != nullptr && visit(null_values_ptr, i) == 0) {
                this->null_count++;
            } else {
                arrow::bit_util::SetBit(this->buffers[0]->mutable_data(), i);
            }
        }
        this->buffers.push_back(*arrow::AllocateBuffer((length - null_count) *
                                                       sizeof(default_type)));

        auto data_ptr =
            reinterpret_cast<default_type *>(this->buffers[1]->mutable_data());
        for (size_t i = 0, j = 0; i < length; ++i) {
            if (null_values_ptr != nullptr && visit(null_values_ptr, i) != 0) {
                data_ptr[j++] = data[i];
            }
        }
        // buffers[2]: selection vector.
        this->buffers.push_back(GetSVBufferFromBM(buffers[0]->data(), length));
        // buffers[3]: prefix sum.
        auto prefix_sums_size_in_byte =
            length / kJacobson_c * sizeof(prefix_sum_type);
        auto buf = *arrow::AllocateResizableBuffer(prefix_sums_size_in_byte);
        auto bytes_for_bm = buffers[0]->data();
        auto prefix_sums = (prefix_sum_type *)buf->mutable_data();
        constexpr auto num_bytes = kJacobson_c / 8;
        static_assert(kJacobson_c % 8 == 0,
                      "kJacobson_c must be a multiple of 8");
        assert(length > 0);
        int prev_sum = 0;
        for (size_t i = 0; i < prefix_sums_size_in_byte;
             i += sizeof(prefix_sum_type)) {
            prefix_sums[i / sizeof(prefix_sum_type)] = prev_sum;
            for (size_t j = 0; j < num_bytes && (i + j) < length / 8; ++j) {
                prev_sum += arrow::bit_util::kBytePopcount[bytes_for_bm[i + j]];
            }
        }
        this->buffers.push_back(std::shared_ptr<Buffer>(std::move(buf)));
    }
};
} // namespace null_revisit