#pragma once

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>

#include "CompressInterface.hpp"

namespace null_revisit {
class FLSInterface : public CompressInterface {
  public:
    /**
     * Construct a FLSInterface object
     */
    FLSInterface() = default;

    ~FLSInterface() override = default;
    virtual size_t decompress_untranspose(std::string_view src, char *dst,
                                          size_t dst_size) {
        throw std::logic_error(
            "FLSInterface::decompress_untranspose Not implemented");
    }
};
} // namespace null_revisit
