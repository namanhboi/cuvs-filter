/* SPDX-License-Identifier: Apache-2.0 */
#include <cuvs/neighbors/cagra.hpp>

#include <raft/core/host_mdarray.hpp>
#include <raft/core/resources.hpp>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <regex>
#include <stdexcept>
#include <string>

namespace {
double json_number(std::string const& json, std::string const& key)
{
  std::regex pattern("\\\"" + key + "\\\"\\s*:\\s*([-+0-9.eE]+)");
  std::smatch match;
  if (!std::regex_search(json, match, pattern))
    throw std::runtime_error("missing JSON key: " + key);
  return std::stod(match[1]);
}

auto load_fbin(std::string const& path)
{
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) throw std::runtime_error("cannot open dataset: " + path);
  auto bytes = static_cast<uint64_t>(input.tellg());
  input.seekg(0);
  uint32_t rows{}, dimensions{};
  input.read(reinterpret_cast<char*>(&rows), 4);
  input.read(reinterpret_cast<char*>(&dimensions), 4);
  if (bytes != 8ull + static_cast<uint64_t>(rows) * dimensions * sizeof(float))
    throw std::runtime_error("invalid .fbin file size");
  auto data = raft::make_host_matrix<float, int64_t>(rows, dimensions);
  input.read(reinterpret_cast<char*>(data.data_handle()), data.size() * sizeof(float));
  if (!input) throw std::runtime_error("short .fbin read");
  return data;
}
}  // namespace

int main(int argc, char** argv)
try {
  if (argc < 5) {
    throw std::runtime_error(
      "usage: CAGRA_FAVOR_COMPARE BASE.fbin CAGRA.index FAVOR.json OUTPUT.json [BFS_DEPTH]");
  }
  std::ifstream favor_input(argv[3]);
  std::string favor_json((std::istreambuf_iterator<char>(favor_input)), {});
  if (favor_json.empty()) throw std::runtime_error("cannot read FAVOR JSON");
  auto dataset = load_fbin(argv[1]);

  constexpr uint32_t alpha = 10, beta = 64;
  auto bfs_depth = argc > 5 ? static_cast<uint32_t>(std::stoul(argv[5])) : 2u;
  if (bfs_depth == 0) throw std::runtime_error("BFS_DEPTH must be positive");
  if (json_number(favor_json, "rows") != dataset.extent(0) ||
      json_number(favor_json, "dimensions") != dataset.extent(1) ||
      json_number(favor_json, "M") != 16 || json_number(favor_json, "max_base_degree") != 32 ||
      json_number(favor_json, "ef_construction") != 64 ||
      json_number(favor_json, "alpha") != alpha || json_number(favor_json, "beta") != beta) {
    throw std::runtime_error("FAVOR result is incompatible with the matched CAGRA experiment");
  }

  raft::resources res;
  cuvs::neighbors::cagra::index<float, uint32_t> index(res);
  cuvs::neighbors::cagra::deserialize(res, argv[2], &index);
  index.update_dataset(res, raft::make_const_mdspan(dataset.view()));
  if (index.graph_degree() != 32) throw std::runtime_error("CAGRA graph degree must be 32");

  auto begin = std::chrono::steady_clock::now();
  auto cagra_delta =
    cuvs::neighbors::cagra::compute_favor_delta_d(res, {alpha, beta, bfs_depth}, index);
  auto cagra_seconds =
    std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
  auto sidecar = std::string(argv[2]) + ".delta_d";
  cuvs::neighbors::cagra::save_favor_delta_d(
    res, sidecar, {alpha, beta, bfs_depth}, index, cagra_delta);
  // A startup path would deserialize the graph, attach the dataset as above, then load this value.
  auto loaded_delta =
    cuvs::neighbors::cagra::load_favor_delta_d(res, sidecar, {alpha, beta, bfs_depth}, index);
  if (std::memcmp(&loaded_delta, &cagra_delta, sizeof(float)) != 0) {
    throw std::runtime_error("delta-d sidecar did not preserve the scalar exactly");
  }
  auto favor_delta   = json_number(favor_json, "delta_d");
  auto favor_seconds = json_number(favor_json, "build_seconds");
  auto difference    = cagra_delta - favor_delta;
  auto absolute      = std::abs(difference);
  auto relative      = favor_delta == 0 ? 0 : 100.0 * absolute / std::abs(favor_delta);
  auto ratio         = favor_delta == 0 ? 0 : cagra_delta / favor_delta;

  std::ofstream output(argv[4]);
  if (!output) throw std::runtime_error("cannot open comparison output");
  output << std::setprecision(10) << "{\n"
         << "  \"rows\": " << dataset.extent(0) << ",\n"
         << "  \"dimensions\": " << dataset.extent(1) << ",\n"
         << "  \"M\": 16,\n  \"max_base_degree\": 32,\n  \"ef_construction\": 64,\n"
         << "  \"alpha\": 10,\n  \"beta\": 64,\n  \"bfs_depth\": " << bfs_depth << ",\n"
         << "  \"favor_delta_d\": " << favor_delta << ",\n"
         << "  \"cagra_delta_d\": " << cagra_delta << ",\n"
         << "  \"signed_difference\": " << difference << ",\n"
         << "  \"absolute_difference\": " << absolute << ",\n"
         << "  \"relative_percent_difference\": " << relative << ",\n"
         << "  \"ratio\": " << ratio << ",\n"
         << "  \"favor_build_seconds\": " << favor_seconds << ",\n"
         << "  \"cagra_compute_seconds\": " << cagra_seconds << "\n}\n";
  std::cout << "FAVOR=" << favor_delta << " CAGRA=" << cagra_delta << " difference=" << difference
            << " (" << relative << "%)"
            << " sidecar=" << sidecar << '\n';
  return 0;
} catch (std::exception const& e) {
  std::cerr << "CAGRA_FAVOR_COMPARE: " << e.what() << '\n';
  return 1;
}
