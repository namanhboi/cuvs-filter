/* SPDX-License-Identifier: Apache-2.0 */

#include <favor.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

struct matrix {
  std::uint32_t rows{};
  std::uint32_t dimensions{};
  std::vector<float> values;
};

matrix load_fbin(std::string const &path, std::uint32_t requested_rows) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) {
    throw std::runtime_error("cannot open dataset: " + path);
  }
  auto const file_bytes = static_cast<std::uint64_t>(input.tellg());
  input.seekg(0);

  matrix result;
  input.read(reinterpret_cast<char *>(&result.rows), sizeof(result.rows));
  input.read(reinterpret_cast<char *>(&result.dimensions),
             sizeof(result.dimensions));
  auto const expected_bytes =
      sizeof(std::uint32_t) * 2 + static_cast<std::uint64_t>(result.rows) *
                                      result.dimensions * sizeof(float);
  if (file_bytes != expected_bytes || result.rows == 0 ||
      result.dimensions == 0) {
    throw std::runtime_error("invalid fbin dataset: " + path);
  }
  if (requested_rows != 0) {
    if (requested_rows > result.rows) {
      throw std::runtime_error("requested rows exceed dataset rows");
    }
    result.rows = requested_rows;
  }
  result.values.resize(static_cast<std::size_t>(result.rows) *
                       result.dimensions);
  input.read(
      reinterpret_cast<char *>(result.values.data()),
      static_cast<std::streamsize>(result.values.size() * sizeof(float)));
  if (!input) {
    throw std::runtime_error("short fbin dataset read: " + path);
  }
  return result;
}

template <typename Function>
void parallel_for(std::size_t end, std::size_t thread_count,
                  Function function) {
  std::atomic<std::size_t> next{0};
  std::exception_ptr failure;
  std::mutex failure_mutex;
  std::vector<std::thread> workers;
  workers.reserve(thread_count);
  for (std::size_t thread = 0; thread < thread_count; ++thread) {
    workers.emplace_back([&] {
      while (true) {
        auto const row = next.fetch_add(1);
        if (row >= end) {
          break;
        }
        try {
          function(row);
        } catch (...) {
          std::lock_guard lock(failure_mutex);
          if (!failure) {
            failure = std::current_exception();
          }
          next = end;
          break;
        }
      }
    });
  }
  for (auto &worker : workers) {
    worker.join();
  }
  if (failure) {
    std::rethrow_exception(failure);
  }
}

} // namespace

int main(int argc, char **argv) try {
  if (argc < 3 || argc > 5) {
    throw std::runtime_error("usage: FAVOR_REFERENCE_DELTA BASE.fbin "
                             "OUTPUT.json [ROWS=all] [THREADS=32]");
  }
  auto const requested_rows =
      argc >= 4 ? static_cast<std::uint32_t>(std::stoul(argv[3])) : 0u;
  auto const thread_count =
      argc >= 5 ? static_cast<std::size_t>(std::stoul(argv[4])) : 32u;
  if (thread_count == 0) {
    throw std::runtime_error("THREADS must be positive");
  }

  constexpr std::size_t m = 16;
  constexpr std::size_t ef_construction = 64;
  constexpr std::uint32_t alpha = 10;
  constexpr std::uint32_t beta = 64;
  auto dataset = load_fbin(argv[1], requested_rows);
  hnswlib::L2Space space(dataset.dimensions);
  favor::FAVOR<float> index(&space, dataset.rows, m, ef_construction, 1);

  auto const begin = std::chrono::steady_clock::now();
  parallel_for(dataset.rows, thread_count, [&](std::size_t row) {
    float attribute = 0.0f;
    index.addPoint(dataset.values.data() + row * dataset.dimensions, row,
                   &attribute);
  });
  auto const seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - begin)
          .count();

  std::ofstream output(argv[2]);
  if (!output) {
    throw std::runtime_error("cannot open output: " + std::string(argv[2]));
  }
  output << std::setprecision(10) << "{\n"
         << "  \"rows\": " << dataset.rows << ",\n"
         << "  \"dimensions\": " << dataset.dimensions << ",\n"
         << "  \"M\": " << m << ",\n"
         << "  \"max_base_degree\": " << 2 * m << ",\n"
         << "  \"ef_construction\": " << ef_construction << ",\n"
         << "  \"alpha\": " << alpha << ",\n"
         << "  \"beta\": " << beta << ",\n"
         << "  \"delta_d\": " << index.delta_d << ",\n"
         << "  \"build_seconds\": " << seconds << "\n"
         << "}\n";
  std::cout << "FAVOR delta_d=" << index.delta_d << " rows=" << dataset.rows
            << " dimensions=" << dataset.dimensions
            << " build_seconds=" << seconds << '\n';
  return 0;
} catch (std::exception const &error) {
  std::cerr << "FAVOR_REFERENCE_DELTA: " << error.what() << '\n';
  return 1;
}
