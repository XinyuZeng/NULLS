// This file to be deleted.
#include <arrow/array.h>
#include <fmt/core.h>

#include <cstdint>
#include <iostream>
#include <random>

// useless test program. ignore it.
class SkewedNullDistribution {
 public:
  SkewedNullDistribution(double probability, double skewFactor) : probability_(probability), skewFactor_(skewFactor) {}

  bool operator()(std::default_random_engine& rng, size_t index) {
    // Calculate the probability of having a null at the given index
    double baseProbability = probability_;
    double skewedProbability =
        baseProbability + std::sin(index / skewFactor_) * std::min(baseProbability, 1 - baseProbability);

    // Ensure that the probability is within the valid range [0, 1]
    // skewedProbability = std::min(1.0, std::max(0.0, skewedProbability));

    // Generate a random number between 0 and 1
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    double randomValue = dist(rng);

    // Return true if the random value is less than the skewed probability
    return randomValue < skewedProbability;
  }

 private:
  double probability_;
  double skewFactor_;
};

void GenerateSkewedBitmap(uint8_t* buffer, size_t n, int64_t* null_count, double probability, double skewFactor) {
  int64_t count = 0;
  std::default_random_engine rng;
  SkewedNullDistribution dist(probability, skewFactor);

  for (size_t i = 0; i < n; i++) {
    if (!dist(rng, i)) {
      // Set the bit to represent a non null
      // For your implementation, use your own SetBit function
      // bit_util::SetBit(buffer, i);
    } else {
      std::cout << "Setting bit " << i << " to null\n";
      count++;
    }
  }

  if (null_count != nullptr) *null_count = count;
}

int main() {
  // auto arrow_array = arrow::MakeArray({1});
  fmt::print("/home/xinyu/NullRepresentation/data/bitmap_random_{:.1f}_{}_{}.txt", 0.2, 1024, 1.15);
  // Example usage
  const size_t n = 100;
  uint8_t buffer[n];
  int64_t null_count;

  // Adjust probability and skewFactor as needed
  double probability = 0.1;
  double skewFactor = 100.0;

  GenerateSkewedBitmap(buffer, n, &null_count, probability, skewFactor);

  // Output null count
  std::cout << "Null Count: " << null_count << std::endl;

  return 0;
}
