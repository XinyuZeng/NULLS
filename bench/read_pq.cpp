#include <arrow/api.h>
#include <arrow/array.h>
#include <arrow/buffer.h>
#include <arrow/io/api.h>
#include <arrow/type_fwd.h>

#include <chrono>
#include <iostream>

#include "arrow/testing/random.h"
#include "parquet/arrow/reader.h"
#include "parquet/arrow/writer.h"

using ArrowType = arrow::Int32Type;

int main(int argc, char** argv) {
  std::string filename = argv[1];
  std::shared_ptr<arrow::io::FileOutputStream> outfile;
  arrow::MemoryPool* pool = arrow::default_memory_pool();
  auto begin = std::chrono::steady_clock::now();
  std::shared_ptr<arrow::io::RandomAccessFile> input;
  input = *arrow::io::ReadableFile::Open(filename);

  // Open Parquet file reader
  std::unique_ptr<parquet::arrow::FileReader> arrow_reader;
  auto status = parquet::arrow::OpenFile(input, pool, &arrow_reader);
  if (!status.ok()) {
    std::cout << status.message() << std::endl;
    return -1;
  }

  // Read entire file as a single Arrow table
  std::shared_ptr<arrow::Table> table_read;
  status = arrow_reader->ReadTable(&table_read);
  if (!status.ok()) {
    std::cout << status.message() << std::endl;
    return -1;
  }
  auto end = std::chrono::steady_clock::now();
  std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << std::endl;
  return 0;
}