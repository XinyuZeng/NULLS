
#include <iostream>

#include "roaring/roaring.hh"

using namespace roaring;

int main() {
  auto block_size = 64 * 1024;
  for (double null_rate = 0; null_rate <= 1; null_rate += 0.025) {
    Roaring r;
    for (int i = 0; i < block_size; i++) {
      if (rand() % 100 < null_rate * 100) {  // means null
      } else {
        r.add(i);
      }
    }
    r.runOptimize();
    std::cout << r.getSizeInBytes() << std::endl;
  }
  return 0;
}