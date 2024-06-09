/* https://github.com/zxjcarrot/tabby-dbos/blob/master/tpcc/ZipfGenerator.hpp
MIT License.
*/
#pragma once
// -------------------------------------------------------------------------------------
// -------------------------------------------------------------------------------------

#include <cstdint>
#include <random>
#include <algorithm>
#include <atomic>
#include <cassert>

class MersenneTwister
{
  private:
   static const int NN = 312;
   static const int MM = 156;
   static const uint64_t MATRIX_A = 0xB5026F5AA96619E9ULL;
   static const uint64_t UM = 0xFFFFFFFF80000000ULL;
   static const uint64_t LM = 0x7FFFFFFFULL;
   uint64_t mt[NN];
   int mti;
   void init(uint64_t seed);

  public:
   MersenneTwister(uint64_t seed = 19650218ULL);
   uint64_t rnd();
};

static thread_local MersenneTwister mt_generator;

class RandomGenerator
{
  public:
   // ATTENTION: open interval [min, max)
   static uint64_t getRanduint64_t(uint64_t min, uint64_t max)
   {
      uint64_t rand = min + (mt_generator.rnd() % (max - min));
      assert(rand < max);
      assert(rand >= min);
      return rand;
   }
   static uint64_t getRanduint64_t() { return mt_generator.rnd(); }
   template <typename T>
   static inline T getRand(T min, T max)
   {
      uint64_t rand = getRanduint64_t(min, max);
      return static_cast<T>(rand);
   }
   static void getRandString(uint8_t* dst, uint64_t size);
};

static std::atomic<uint64_t> mt_counter = 0;
// -------------------------------------------------------------------------------------
MersenneTwister::MersenneTwister(uint64_t seed) : mti(NN + 1)
{
   init(seed + (mt_counter++));
}
// -------------------------------------------------------------------------------------
void MersenneTwister::init(uint64_t seed)
{
   mt[0] = seed;
   for (mti = 1; mti < NN; mti++)
      mt[mti] = (6364136223846793005ULL * (mt[mti - 1] ^ (mt[mti - 1] >> 62)) + mti);
}
// -------------------------------------------------------------------------------------
uint64_t MersenneTwister::rnd()
{
   int i;
   uint64_t x;
   static uint64_t mag01[2] = {0ULL, MATRIX_A};

   if (mti >= NN) { /* generate NN words at one time */

      for (i = 0; i < NN - MM; i++) {
         x = (mt[i] & UM) | (mt[i + 1] & LM);
         mt[i] = mt[i + MM] ^ (x >> 1) ^ mag01[(int)(x & 1ULL)];
      }
      for (; i < NN - 1; i++) {
         x = (mt[i] & UM) | (mt[i + 1] & LM);
         mt[i] = mt[i + (MM - NN)] ^ (x >> 1) ^ mag01[(int)(x & 1ULL)];
      }
      x = (mt[NN - 1] & UM) | (mt[0] & LM);
      mt[NN - 1] = mt[MM - 1] ^ (x >> 1) ^ mag01[(int)(x & 1ULL)];

      mti = 0;
   }

   x = mt[mti++];

   x ^= (x >> 29) & 0x5555555555555555ULL;
   x ^= (x << 17) & 0x71D67FFFEDA60000ULL;
   x ^= (x << 37) & 0xFFF7EEE000000000ULL;
   x ^= (x >> 43);

   return x;
}
// -------------------------------------------------------------------------------------
void RandomGenerator::getRandString(uint8_t* dst, uint64_t size)
{
   for (uint64_t t_i = 0; t_i < size; t_i++) {
      dst[t_i] = getRand(48, 123);
   }
}

using namespace std;
// -------------------------------------------------------------------------------------
// -------------------------------------------------------------------------------------
// A Zipf distributed random number generator
// Based on Jim Gray Algorithm as described in "Quickly Generating Billion-Record..."
// -------------------------------------------------------------------------------------
class ZipfGenerator
{
   // -------------------------------------------------------------------------------------
  private:
   uint64_t n;
   double theta;
   // -------------------------------------------------------------------------------------
   double alpha, zetan, eta;
   // -------------------------------------------------------------------------------------
   double zeta(uint64_t n, double theta);

  public:
   // [0, n)
   ZipfGenerator(uint64_t ex_n, double theta);
   // uint64_t rand(uint64_t new_n);
   uint64_t rand();
};

// -------------------------------------------------------------------------------------
// -------------------------------------------------------------------------------------
ZipfGenerator::ZipfGenerator(uint64_t ex_n, double theta) : n(ex_n - 1), theta(theta)
{
   alpha = 1.0 / (1.0 - theta);
   zetan = zeta(n, theta);
   eta = (1.0 - std::pow(2.0 / n, 1.0 - theta)) / (1.0 - zeta(2, theta) / zetan);
}
// -------------------------------------------------------------------------------------
double ZipfGenerator::zeta(uint64_t n, double theta)
{
   double ans = 0;
   for (uint64_t i = 1; i <= n; i++)
      ans += std::pow(1.0 / n, theta);
   return ans;
}
// -------------------------------------------------------------------------------------
uint64_t ZipfGenerator::rand()
{
   double constant = 1000000000000000000.0;
   uint64_t i = RandomGenerator::getRanduint64_t(0, 1000000000000000001);
   double u = static_cast<double>(i) / constant;
   // return (uint64_t)u;
   double uz = u * zetan;
   if (uz < 1) {
      return 1;
   }
   if (uz < (1 + std::pow(0.5, theta)))
      return 2;
   uint64_t ret = 1 + (long)(n * pow(eta * u - eta + 1, alpha));
   return ret;
}