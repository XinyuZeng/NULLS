#include "nullgen.hpp"
#include "ZipfGenerator.hpp"
#include "zipf.hpp"
#include <cassert>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <random>
#include <string>

namespace null_revisit {
template <typename T>
static void load_data_binary(std::vector<T> &out, const std::string &filename,
                             size_t target_size, bool print = true) {

    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "unable to open " << filename << std::endl;
        exit(EXIT_FAILURE);
    }
    out.resize(target_size);
    // Read size.
    uint64_t size;
    in.read(reinterpret_cast<char *>(&size), sizeof(uint64_t));
    if (target_size < size) {
        in.read(reinterpret_cast<char *>(out.data()), target_size * sizeof(T));
    } else {
        in.read(reinterpret_cast<char *>(out.data()), size * sizeof(T));
        for (size_t i = size; i < target_size;) {
            auto to_read =
                std::min(size, static_cast<uint64_t>(target_size - i));
            std::copy(out.begin(), out.begin() + to_read, out.begin() + i);
            i += to_read;
        }
    }
    // Read values.
    in.close();
}

template <typename T>
static void load_data(std::vector<T> &out, const std::string &filename,
                      size_t target_size, bool print = true) {
    std::ifstream srcFile(filename, std::ios::in);
    if (!srcFile) {
        std::cout << "error opening source file." << std::endl;
    }
    auto cnt = 0;
    while (srcFile.good() && cnt < target_size) {
        T next;
        srcFile >> next;
        if (!srcFile.good()) {
            break;
        }
        out[cnt++] = next;
    }
    srcFile.close();
    for (size_t i = cnt; i < target_size; i++) {
        out[i] = out[i % cnt];
    }
}

std::pair<std::vector<uint32_t>, std::vector<bool>>
NullGenerator::generate(uint32_t size) const {
    auto read_real_data = [](const std::string &path) {
        std::ifstream file(path); // Make sure this path is correct
        if (!file) {
            std::cerr << "Unable to open file" << std::endl;
            throw std::runtime_error("Unable to open file");
        }
        std::vector<uint32_t> values;
        std::string line;
        while (std::getline(file, line)) {
            try {
                auto number = std::stoul(line); // Convert string to integer
                values.push_back(number);
            } catch (const std::invalid_argument &e) {
                throw std::runtime_error("Invalid argument (not an integer)");
            } catch (const std::out_of_range &e) {
                throw std::runtime_error("Integer out_of_range");
            }
        }
        return values;
    };
    std::vector<uint32_t> values(size);
    std::vector<bool> nulls(size);
    std::mt19937_64 gen(0x20240121);
    switch (dist_) {
    case Distribution::UNIFORM: {
        std::uniform_int_distribution<> uniform(0, range_max_ - 1);
        for (uint32_t i = 0; i < size; i++) {
            values[i] = uniform(gen);
            assert(values[i] < range_max_);
        }
        break;
    }
    case Distribution::GENTLE_ZIPF: {
        auto real_data = read_real_data(gentle_zipf_path);
        for (uint32_t i = 0; i < size; i++) {
            values[i] = real_data[i % real_data.size()];
        }
        break;
    }
    case Distribution::HOTSPOT: {
        auto real_data = read_real_data(hotspot_path);
        for (uint32_t i = 0; i < size; i++) {
            values[i] = real_data[i % real_data.size()];
        }
        break;
    }
    case Distribution::LINEAR: {
        load_data(values, linear_path, size);
        break;
    }
    default:
        throw std::runtime_error("Unknown distribution");
    }

    std::bernoulli_distribution d(null_rate_);
    for (uint32_t i = 0; i < size; i++) {
        if (d(gen)) {
            values[i] = 0;
            nulls[i] = true;
        } else {
            nulls[i] = false;
        }
    }
    return std::make_pair(values, nulls);
}

std::vector<uint32_t> NullGenerator::createShuffledVector(size_t size) const {
    std::vector<uint32_t> v(size);

    for (size_t i = 0; i < size; ++i) {
        v[i] = i;
    }

    std::mt19937 g(0x20240308);

    // Shuffle every block of 16 elements
    for (size_t i = 0; i < size; i += 16) {
        // Calculate the end of the current block
        size_t end = std::min(i + 16, size);
        // Shuffle the elements from i to end
        std::shuffle(v.begin() + i, v.begin() + end, g);
    }

    return v;
}
} // namespace null_revisit
