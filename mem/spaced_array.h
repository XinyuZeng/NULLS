#pragma once
#include <arrow/util/bit_util.h>

#include <cassert>
#include <cstddef>
#include <vector>

#include "array_base.h"
#include "utils/bit_util.h"
#include "utils/exceptions.hpp"
#include "utils/macros.hpp"

namespace null_revisit {
class SpacedArray : public BaseArray {
  public:
    SpacedArray(const default_type *data, const bool *null_vector,
                size_t length)
        : BaseArray() {
        this->type = ArrayType::Spaced;
        this->length = length;
        this->null_count = 0;
        this->buffers.push_back(*arrow::AllocateEmptyBitmap(length));
        this->buffers.push_back(
            *arrow::AllocateBuffer(length * sizeof(default_type)));
        memcpy(this->buffers[1]->mutable_data(), data,
               length * sizeof(default_type));
        for (size_t i = 0; i < length; ++i) {
            if (null_vector[i] == 0) {
                this->null_count++;
            } else {
                arrow::bit_util::SetBit(this->buffers[0]->mutable_data(), i);
            }
        } // buffers[2]: selection vector.
        this->buffers.push_back(GetSVBufferFromBM(buffers[0]->data(), length));
    }
    SpacedArray(const std::shared_ptr<arrow::Array> &array) : BaseArray(array) {
        this->type = ArrayType::Spaced;
        this->buffers.push_back(GetSVBufferFromBM(buffers[0]->data(), length));
    }

    ~SpacedArray() = default;

    void CompareWithCol(const BaseArray &other,
                        std::shared_ptr<Buffer> output) const override {
        Exception::NYI("polymorphism");
    }
    void CompareWithConst(const BaseArray &other,
                          std::shared_ptr<Buffer> output) const override {
        Exception::NYI("polymorphism");
    }
    void SumByGroupBM(const std::vector<int32_t> &group_ids,
                      std::vector<int64_t> &sum) const override;
    void SumByGroupSV(const std::vector<int32_t> &group_ids,
                      std::vector<int64_t> &sum) const override;

    template <bool five_op = false>
    void CompareWithCol(const SpacedArray &other,
                        std::shared_ptr<Buffer> output) const;
    void CompareWithColScalar(const SpacedArray &other,
                              std::shared_ptr<Buffer> output) const;
    int CompareWithColSVPartial(const SpacedArray &other, SelVector &output,
                                SelVector &sel_vec) const;
    template <bool eval_filter_first = false>
    int CompareWithColSVPartialBranch(const SpacedArray &other,
                                      SelVector &output,
                                      SelVector &sel_vec) const;
    void CompareWithColSVPartialManual(const SpacedArray &other,
                                       SelVector &output,
                                       SelVector &sel_vec) const;
};
} // namespace null_revisit