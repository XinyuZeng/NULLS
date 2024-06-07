# use with https://github.com/casey/just
build:
  mkdir -p build/Release;\
  cd build/Release;\
  cmake ../..;\
  cmake --build . --parallel

bench_agg: set_kVecSize_1K build
  ./build/Release/bench_in_mem --benchmark_filter=BM_Sum --benchmark_format=csv > outputs/BM_agg.csv

bench_filter: set_kVecSize_1K build
  ./build/Release/bench_in_mem --benchmark_filter=BM_.+_filter --benchmark_format=csv > outputs/BM_mem.csv

bench_full: set_kVecSize_1K build
  ./build/Release/bench_in_mem --benchmark_filter=.AVX. --benchmark_format=csv > outputs/BM_full.csv

bench_sv: set_kVecSize_1K build
  ./build/Release/bench_in_mem --benchmark_filter=.SVPartial. --benchmark_format=csv > outputs/BM_partial.csv

bench_size_branch_miss: set_kVecSize_1K build
  ./build/Release/bench_in_mem --benchmark_filter="BM_branch"

bench_dense_to_spaced: set_kVecSize_1K build
  ./build/Release/bench_dense_to_spaced --benchmark_filter="BM_DenseToSpaced" --benchmark_format=csv  --benchmark_perf_counters=cycles > outputs/BM_dense_to_spaced.csv

bench_miniblock_size: set_kVecSize_8M build
  ./build/Release/bench_dense_to_spaced --benchmark_filter="BM_MiniblockSizeDenseToSpaced" --benchmark_format=csv  --benchmark_perf_counters=cycles > outputs/BM_dense_to_spaced_miniblock.csv

set_kVecSize_8M:
  python3 scripts/set_kVecSize.py "8*1024*1024"

set_kVecSize_1K:
  python3 scripts/set_kVecSize.py "1*1024"

motivate:
  ./build/Release/motivation --benchmark_format=csv > outputs/motivation.csv