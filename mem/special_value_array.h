#pragma once
#include <arrow/util/bit_util.h>

#include <cassert>
#include <cstddef>
#include <limits>
#include <vector>

#include "array_base.h"
#include "utils/bit_util.h"
#include "utils/exceptions.hpp"
#include "utils/macros.hpp"

namespace null_revisit {
class SpecialValArray : public BaseArray {
  public:
    SpecialValArray(const default_type *data, const bool *null_vector,
                    size_t length)
        : BaseArray() {
        this->type = ArrayType::SpecialVal;
        this->length = length;
        this->null_count = 0;
        // FIXME: no need a bitmap for special value array.
        this->buffers.push_back(*arrow::AllocateEmptyBitmap(length));
        this->buffers.push_back(
            *arrow::AllocateBuffer(length * sizeof(default_type)));
        memcpy(this->buffers[1]->mutable_data(), data,
               length * sizeof(default_type));
        auto data_ptr =
            reinterpret_cast<default_type *>(this->buffers[1]->mutable_data());
        for (size_t i = 0; i < length; ++i) {
            if (null_vector[i] == 0) {
                this->null_count++;
                data_ptr[i] = kSpecialValue; // FIXME: hardcode max for now.
            } else {
                arrow::bit_util::SetBit(this->buffers[0]->mutable_data(), i);
            }
        } // buffers[2]: selection vector.
        this->buffers.push_back(GetSVBufferFromBM(buffers[0]->data(), length));
    }
    SpecialValArray(const std::shared_ptr<arrow::Array> &array)
        : BaseArray(array) {
        this->type = ArrayType::SpecialVal;
        this->buffers.push_back(GetSVBufferFromBM(buffers[0]->data(), length));
        auto data_ptr =
            reinterpret_cast<default_type *>(this->buffers[1]->mutable_data());
        for (size_t i = 0; i < length; ++i) {
            if (arrow::bit_util::GetBit(buffers[0]->data(), i) == 0) {
                data_ptr[i] = kSpecialValue; // FIXME: hardcode max for now.
            }
        } // buffers[2]: selection vector.
    }

    ~SpecialValArray() = default;

    void CompareWithCol(const BaseArray &other,
                        std::shared_ptr<Buffer> output) const override {
        Exception::NYI("polymorphism");
    }
    void CompareWithConst(const BaseArray &other,
                          std::shared_ptr<Buffer> output) const override {
        Exception::NYI("polymorphism");
    }
    void SumByGroupBM(const std::vector<int32_t> &group_ids,
                      std::vector<int64_t> &sum) const override {
        Exception::NYI("polymorphism");
    }
    void SumByGroupSV(const std::vector<int32_t> &group_ids,
                      std::vector<int64_t> &sum) const override {
        Exception::NYI("polymorphism");
    }
    template <bool five_op = false>
    void CompareWithCol(const SpecialValArray &other,
                        std::shared_ptr<Buffer> output) const;
    void CompareWithColScalar(const SpecialValArray &other,
                              std::shared_ptr<Buffer> output) const {
        Exception::NYI("polymorphism");
    }
    int CompareWithColSVPartial(const SpecialValArray &other, SelVector &output,
                                SelVector &sel_vec) const;
    template <bool eval_filter_first = false>
    int CompareWithColSVPartialBranch(const SpecialValArray &other,
                                      SelVector &output,
                                      SelVector &sel_vec) const;
    void CompareWithColSVPartialManual(const SpecialValArray &other,
                                       SelVector &output,
                                       SelVector &sel_vec) const {
        Exception::NYI("polymorphism");
    }
};
} // namespace null_revisit