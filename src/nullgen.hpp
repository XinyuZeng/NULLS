#pragma once

#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace null_revisit {
// From Format Eval: Skew Pattern
enum class Distribution {
    UNIFORM = 0,
    GENTLE_ZIPF,
    HOTSPOT,
    LINEAR,
    // BINARY,
};

static inline Distribution GetDistFromString(const std::string &dist_str) {
    if (dist_str == "uniform") {
        return Distribution::UNIFORM;
    } else if (dist_str == "gentle_zipf") {
        return Distribution::GENTLE_ZIPF;
    } else if (dist_str == "hotspot") {
        return Distribution::HOTSPOT;
    } else if (dist_str == "linear") {
        return Distribution::LINEAR;
    } else {
        throw std::runtime_error("Unknown distribution: " + dist_str);
    }
}
class NullGenerator {
  public:
    NullGenerator(float null_rate, float repeat_rate, uint32_t range_max,
                  Distribution dist)
        : null_rate_(null_rate), rep_rate_(repeat_rate), range_max_(range_max),
          dist_(dist) {}
    std::pair<std::vector<uint32_t>, std::vector<bool>>
    generate(uint32_t size) const;

  private:
    float null_rate_;
    [[gnu::unused]] float rep_rate_;
    uint32_t range_max_;
    Distribution dist_;
    const std::string gentle_zipf_path =
        std::filesystem::path(__FILE__).parent_path().string() +
        "/../exp_data/gentle_zipf.txt";
    const std::string hotspot_path =
        std::filesystem::path(__FILE__).parent_path().string() +
        "/../exp_data/hotspot.txt";
    const std::string linear_path =
        std::filesystem::path(__FILE__).parent_path().string() +
        "/../exp_data/books_8M.txt";

    std::vector<uint32_t> createShuffledVector(size_t size) const;
};
} // namespace null_revisit
