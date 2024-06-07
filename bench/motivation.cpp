#include <arrow/api.h>
#include <arrow/array.h>
#include <arrow/buffer.h>
#include <arrow/io/api.h>
#include <arrow/type_fwd.h>
#include <benchmark/benchmark.h>

#include <iostream>
#include <random>

#include "arrow/testing/random.h"
#include "parquet/arrow/reader.h"
#include "parquet/arrow/writer.h"
#include "ZipfGenerator.hpp"

// Note: this file is for pure parquet files generation. Please refer to `read_pq.cpp` and `motivate.py` for the
// experiments on reading parquet files.

// To use this executable to generate files, run `just motivate` in the root directory of the project. Then run `python3
// scripts/motivate.py`

using ArrowType = arrow::Int32Type;
using parquet::ArrowWriterProperties;
using parquet::WriterProperties;

static void BM_Args(benchmark::internal::Benchmark* bench) {
  std::vector<int64_t> null_percent = {0, 1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 97, 99, 100};
  std::vector<int64_t> num_rows = {64 * 1024 * 1024};
  std::vector<int64_t> max = {8 * 1024 * 1024};
  std::vector<int64_t> num_cols = {20};
  bench->ArgsProduct({null_percent, num_rows, max, num_cols});
}

static void BM_parquet_arrow_scan(benchmark::State& state) {
  auto null_percent = static_cast<double>(state.range(0)) / 100.0;
  auto num_rows = state.range(1);
  auto max = state.range(2);
  auto num_cols = state.range(3);
  auto rand = arrow::random::RandomArrayGenerator(1923);

  std::mt19937_64 gen(0x202303061813);
  ZipfGenerator gentle_zipf(max + 1, 0.5);
  std::bernoulli_distribution bernoulli(null_percent);
  std::vector<std::shared_ptr<arrow::Array>> arrays;
  std::vector<std::shared_ptr<arrow::Field>> fields;
  for (int i = 0; i < num_cols; ++i) {
    fields.push_back(arrow::field("f" + std::to_string(i), arrow::int32()));
    auto array1_builder = std::make_shared<arrow::Int32Builder>();
    for (int j = 0; j < num_rows; ++j) {
      if (bernoulli(gen)) {
        [[maybe_unused]] auto result = array1_builder->AppendNull();
      } else {
        [[maybe_unused]] auto result = array1_builder->Append(gentle_zipf.rand() - 1);
      }
    }
    auto array1 = *array1_builder->Finish();
    arrays.push_back(array1);
  }
  auto table = arrow::Table::Make(arrow::schema(fields), arrays);
  // Choose compression, LZ4 is the fastest.
  std::shared_ptr<WriterProperties> props = WriterProperties::Builder().compression(arrow::Compression::LZ4)->build();

  // Opt to store Arrow schema for easier reads back into Arrow
  std::shared_ptr<ArrowWriterProperties> arrow_props = ArrowWriterProperties::Builder().store_schema()->build();

  std::shared_ptr<arrow::io::FileOutputStream> outfile;
  std::string filename = std::to_string(state.range(0)) + ".parquet";
  outfile = *arrow::io::FileOutputStream::Open(filename);

  auto status = parquet::arrow::WriteTable(*table.get(), arrow::default_memory_pool(), outfile,
                                           /*chunk_size=*/parquet::DEFAULT_MAX_ROW_GROUP_LENGTH, props, arrow_props);
  if (!status.ok()) {
    std::cout << status.message() << std::endl;
    return;
  }
  (void)outfile->Close();

  arrow::MemoryPool* pool = arrow::default_memory_pool();
  std::shared_ptr<arrow::io::RandomAccessFile> input;
  input = *arrow::io::ReadableFile::Open(filename);

  // Open Parquet file reader
  std::unique_ptr<parquet::arrow::FileReader> arrow_reader;
  status = parquet::arrow::OpenFile(input, pool, &arrow_reader);
  if (!status.ok()) {
    std::cout << status.message() << std::endl;
    return;
  }

  // Read entire file as a single Arrow table
  std::shared_ptr<arrow::Table> table_read;
  for (auto _ : state) {
    status = arrow_reader->ReadTable(&table_read);
    if (!status.ok()) {
      std::cout << status.message() << std::endl;
      return;
    }
    state.PauseTiming();
    table_read.reset();
    state.ResumeTiming();
    benchmark::DoNotOptimize(table_read);
  }
  state.counters["null_percent"] = null_percent;
}
BENCHMARK(BM_parquet_arrow_scan)->Apply(BM_Args);
BENCHMARK_MAIN();