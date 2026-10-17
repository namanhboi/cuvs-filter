/* SPDX-License-Identifier: Apache-2.0 */
// Compile this file separately against each checkout. It uses only upstream
// public APIs.
#include <cuvs/core/bitmap.hpp>
#include <cuvs/neighbors/cagra.hpp>
#include <nlohmann/json.hpp>
#include <raft/core/copy.cuh>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <rmm/device_uvector.hpp>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

using json = nlohmann::json;

struct mapped_file {
  void *data{MAP_FAILED};
  std::size_t bytes{};
  explicit mapped_file(std::string const &path) {
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0)
      throw std::runtime_error("cannot open " + path);
    struct stat st{};
    if (::fstat(fd, &st) != 0) {
      ::close(fd);
      throw std::runtime_error("cannot stat " + path);
    }
    bytes = st.st_size;
    data = ::mmap(nullptr, bytes, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    if (data == MAP_FAILED)
      throw std::runtime_error("cannot map " + path);
  }
  mapped_file(mapped_file const &) = delete;
  ~mapped_file() {
    if (data != MAP_FAILED)
      ::munmap(data, bytes);
  }
  template <typename T>
  T const *matrix(std::int64_t &rows, std::int64_t &dim) const {
    if (bytes < 8)
      throw std::runtime_error("truncated matrix header");
    auto h = static_cast<std::uint32_t const *>(data);
    rows = h[0];
    dim = h[1];
    if (bytes != 8 + std::size_t(rows) * dim * sizeof(T))
      throw std::runtime_error("incorrect matrix size");
    return reinterpret_cast<T const *>(h + 2);
  }
};

template <typename T>
json run(json const &conf, std::string const &output_prefix) {
  auto const &ds = conf.at("dataset");
  auto const &idx_conf = conf.at("index").at(0);
  auto const &sp = idx_conf.at("search_params").at(0);
  auto k = conf.at("search_basic_param").at("k").get<std::int64_t>();
  mapped_file base(ds.at("base_file")), query(ds.at("query_file"));
  mapped_file gt(ds.at("groundtruth_neighbors_file"));
  std::int64_t rows, dim, nq, qdim, gtrows, gtdim;
  auto hbase = base.matrix<T>(rows, dim);
  auto hquery = query.matrix<T>(nq, qdim);
  auto hgt = gt.matrix<std::uint32_t>(gtrows, gtdim);
  if (nq != 2048 || qdim != dim || gtrows != nq || gtdim < k)
    throw std::runtime_error(
        "native control requires the frozen 2048-query shard");

  raft::device_resources res;
  auto stream = raft::resource::get_cuda_stream(res);
  auto owner = cuvs::neighbors::make_device_padded_dataset(
      res, raft::make_host_matrix_view(hbase, rows, dim));
  cuvs::neighbors::cagra::device_padded_index<T, std::uint32_t> index(res);
  cuvs::neighbors::cagra::deserialize(
      res, idx_conf.at("file").get<std::string>(), &index);
  if (index.size() != rows || index.source_indices().has_value())
    throw std::runtime_error(
        "native control requires the original identity-ID graph");
  index = cuvs::neighbors::cagra::update_dataset(res, std::move(index),
                                                 owner->as_dataset_view());
  auto queries = raft::make_device_matrix<T, std::int64_t>(res, nq, dim);
  raft::copy(queries.data_handle(), hquery, nq * dim, stream);

  mapped_file bitmap(ds.at("filter").at("file"));
  auto b = static_cast<std::uint8_t const *>(bitmap.data);
  std::uint32_t version, width;
  std::uint64_t brows, bcols, nwords;
  if (bitmap.bytes < 40 ||
      std::string(reinterpret_cast<char const *>(b), 8) != "CUVSBMAP")
    throw std::runtime_error("invalid bitmap header");
  std::memcpy(&version, b + 8, 4);
  std::memcpy(&width, b + 12, 4);
  std::memcpy(&brows, b + 16, 8);
  std::memcpy(&bcols, b + 24, 8);
  std::memcpy(&nwords, b + 32, 8);
  if (version != 1 || width != 32 || brows != std::uint64_t(nq) ||
      bcols != std::uint64_t(rows) || nwords != (brows * bcols + 31) / 32 ||
      bitmap.bytes != 40 + nwords * 4)
    throw std::runtime_error("incorrect bitmap shape");
  auto host_words = reinterpret_cast<std::uint32_t const *>(b + 40);
  rmm::device_uvector<std::uint32_t> words(nwords, stream);
  raft::copy(words.data(), host_words, nwords, stream);
  auto filter = cuvs::neighbors::filtering::bitmap_filter(
      cuvs::core::bitmap_view<std::uint32_t, std::int64_t>(words.data(), nq,
                                                           rows));

  cuvs::neighbors::cagra::search_params params;
  params.algo = cuvs::neighbors::cagra::search_algo::SINGLE_CTA;
  params.max_queries = 2048;
  params.itopk_size = sp.at("itopk");
  params.search_width = sp.at("search_width");
  params.max_iterations = sp.at("max_iterations");
  params.rand_xor_mask = sp.value("rand_xor_mask", std::uint64_t{0x128394});
  params.hashmap_max_fill_rate = sp.value("hashmap_max_fill_rate", 0.5f);
  params.filtering_rate = sp.value("filtering_rate", -1.0f);
  auto ids = raft::make_device_matrix<std::int64_t, std::int64_t>(res, nq, k);
  auto distances = raft::make_device_matrix<float, std::int64_t>(res, nq, k);
  auto search = [&] {
    cuvs::neighbors::cagra::search(res, params, index,
                                   raft::make_const_mdspan(queries.view()),
                                   ids.view(), distances.view(), filter);
    raft::resource::sync_stream(res);
  };
  search(); // JIT compilation and first-call setup are outside timing.
  auto warmup = std::chrono::steady_clock::now();
  do {
    search();
  } while (
      std::chrono::duration<double>(std::chrono::steady_clock::now() - warmup)
          .count() < 0.1);
  json repetitions = json::array();
  for (int rep = 0; rep < 3; ++rep) {
    int iterations = 0;
    double elapsed;
    auto start = std::chrono::steady_clock::now();
    do {
      search();
      ++iterations;
      elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                              start)
                    .count();
    } while (elapsed < 0.1);
    repetitions.push_back({{"repetition", rep},
                           {"iterations", iterations},
                           {"seconds", elapsed},
                           {"qps", double(nq) * iterations / elapsed}});
  }
  std::vector<std::int64_t> hids(ids.size());
  std::vector<float> hdist(distances.size());
  raft::copy(hids.data(), ids.data_handle(), ids.size(), stream);
  raft::copy(hdist.data(), distances.data_handle(), distances.size(), stream);
  raft::resource::sync_stream(res);
  std::uint64_t matching = 0, total = 0, violations = 0, duplicates = 0,
                invalid_ids = 0, legacy_sentinel_slots = 0;
  for (std::int64_t q = 0; q < nq; ++q) {
    std::set<std::uint32_t> truth;
    for (std::int64_t rank = 0; rank < k; ++rank) {
      auto id = hgt[q * gtdim + rank];
      if (id < rows)
        truth.insert(id);
    }
    total += truth.size();
    std::set<std::int64_t> seen;
    for (std::int64_t rank = 0; rank < k; ++rank) {
      auto id = hids[q * k + rank];
      if (id < 0 || id >= rows) {
        // Upstream clears the internal uint32 visited bit before widening output
        // IDs, including empty slots. Keep that raw encoding in the saved array.
        const bool legacy = id == std::numeric_limits<std::int32_t>::max();
        const bool canonical = id == std::numeric_limits<std::int64_t>::max();
        invalid_ids += !(legacy || canonical) ||
                       hdist[q * k + rank] != std::numeric_limits<float>::max();
        legacy_sentinel_slots += legacy;
        continue;
      }
      if (!seen.insert(id).second) {
        ++duplicates;
        continue;
      }
      auto bit = std::uint64_t(q) * rows + id;
      violations +=
          (host_words[bit / 32] & (std::uint32_t{1} << (bit % 32))) == 0;
      matching += truth.count(id);
    }
  }
  auto write = [&](std::string suffix, auto const &values) {
    std::ofstream out(output_prefix + suffix, std::ios::binary);
    out.write(reinterpret_cast<char const *>(values.data()),
              values.size() * sizeof(values[0]));
    if (!out)
      throw std::runtime_error("cannot save native outputs");
  };
  write(".ids", hids);
  write(".distances", hdist);
  if (violations || invalid_ids)
    throw std::runtime_error("native filtered output is invalid: violations=" +
                             std::to_string(violations) + ", invalid_ids=" +
                             std::to_string(invalid_ids));
  return {{"repetitions", repetitions},
          {"recall", total ? double(matching) / total : 0.0},
          {"n_queries", nq},
          {"n_rows", rows},
          {"k", k},
          {"matching", matching},
          {"valid_gt", total},
          {"filter_violations", violations},
          {"legacy_sentinel_slots", legacy_sentinel_slots},
          {"duplicate_results", duplicates}};
}

int main(int argc, char **argv) {
  try {
    if (argc != 3)
      throw std::runtime_error("usage: native_control CONFIG OUTPUT_PREFIX");
    json conf;
    std::ifstream(argv[1]) >> conf;
    auto dtype = conf.at("dataset").at("dtype").get<std::string>();
    json result;
    if (dtype == "float")
      result = run<float>(conf, argv[2]);
    else if (dtype == "uint8")
      result = run<std::uint8_t>(conf, argv[2]);
    else
      throw std::runtime_error("unsupported control dtype");
    std::ofstream out(std::string(argv[2]) + ".json");
    out << result.dump(2) << '\n';
    if (!out)
      throw std::runtime_error("cannot save native summary");
    return 0;
  } catch (std::exception const &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
