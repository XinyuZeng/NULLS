#pragma once
#include <arrow/util/string_builder.h>

#include <exception>
#include <string>

namespace null_revisit {
// vendored from ParquetException, in Apache License
class Exception : public std::exception {
  public:
    static void EofException(const std::string &msg = "") {
        static std::string prefix = "Unexpected end of stream";
        if (msg.empty()) {
            throw Exception(prefix);
        }
        throw Exception(prefix, ": ", msg);
    }

    static void NYI(const std::string &msg = "") {
        throw Exception("Not yet implemented: ", msg, ".");
    }

    template <typename... Args>
    explicit Exception(Args &&...args)
        : msg_(::arrow::util::StringBuilder(std::forward<Args>(args)...)) {}

    explicit Exception(std::string msg) : msg_(std::move(msg)) {}

    explicit Exception(const char *msg, const std::exception &) : msg_(msg) {}

    Exception(const Exception &) = default;
    Exception &operator=(const Exception &) = default;
    Exception(Exception &&) = default;
    Exception &operator=(Exception &&) = default;

    const char *what() const noexcept override { return msg_.c_str(); }

  private:
    std::string msg_;
};
} // namespace null_revisit