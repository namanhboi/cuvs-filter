/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/core/bitmap.hpp>
#include <cuvs/neighbors/ivf_flat.hpp>
#include <cuvs/neighbors/ivf_pq.hpp>
#include <cuvs/neighbors/ivf_sq.hpp>
#include <cuvs/neighbors/refine.hpp>

#include "../../src/neighbors/detail/jit_lto_kernels/filter_impl.cuh"
#include "../../src/neighbors/ivf_bitmap_filter.cuh"
#include <cuvs/neighbors/dynamic_batching.hpp>

#include <raft/core/copy.cuh>
#include <raft/core/device_resources.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <rmm/device_uvector.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <unordered_set>
#include <vector>

namespace cuvs::neighbors {
namespace {

constexpr int64_t rows = 769;  // Packed bitmap rows deliberately share words.
constexpr int64_t dim  = 32;
using bitmap_filter_t  = filtering::bitmap_filter<uint32_t, int64_t>;
using bitset_filter_t  = filtering::bitset_filter<uint32_t, int64_t>;
using data_view_t      = raft::device_matrix_view<const float, int64_t>;
using ids_view_t       = raft::device_matrix_view<int64_t, int64_t>;
using dists_view_t     = raft::device_matrix_view<float, int64_t>;

TEST(IvfBitmapRefinement, UnderfilledGpuCandidatePools)
{
  raft::device_resources res;
  auto stream          = raft::resource::get_cuda_stream(res);
  constexpr int64_t nq = 3;
  constexpr int64_t nc = 16;
  constexpr int64_t k  = 8;
  std::vector<float> host_data(rows * dim);
  std::vector<float> host_queries(nq * dim);
  for (int64_t i = 0; i < rows * dim; ++i) {
    host_data[i] = float((i * 17 + i / dim) % 101) / 10.0f;
  }
  std::copy_n(host_data.begin(), host_queries.size(), host_queries.begin());
  std::vector<int64_t> host_candidates(nq * nc, std::numeric_limits<int64_t>::max());
  for (int64_t i = 0; i < nc; ++i) {
    host_candidates[i] = i;
  }
  host_candidates[2 * nc]     = 17;
  host_candidates[2 * nc + 1] = 513;
  host_candidates[2 * nc + 2] = 768;
  rmm::device_uvector<float> data(host_data.size(), stream);
  rmm::device_uvector<float> queries(host_queries.size(), stream);
  rmm::device_uvector<int64_t> candidates(host_candidates.size(), stream);
  rmm::device_uvector<int64_t> ids(nq * k, stream);
  rmm::device_uvector<float> distances(nq * k, stream);
  raft::copy(data.data(), host_data.data(), data.size(), stream);
  raft::copy(queries.data(), host_queries.data(), queries.size(), stream);
  raft::copy(candidates.data(), host_candidates.data(), candidates.size(), stream);
  refine(res,
         raft::make_device_matrix_view<const float, int64_t>(data.data(), rows, dim),
         raft::make_device_matrix_view<const float, int64_t>(queries.data(), nq, dim),
         raft::make_device_matrix_view<const int64_t, int64_t>(candidates.data(), nq, nc),
         raft::make_device_matrix_view<int64_t, int64_t>(ids.data(), nq, k),
         raft::make_device_matrix_view<float, int64_t>(distances.data(), nq, k),
         distance::DistanceType::L2Unexpanded);
  std::vector<int64_t> host_ids(nq * k);
  std::vector<float> host_distances(nq * k);
  raft::copy(host_ids.data(), ids.data(), ids.size(), stream);
  raft::copy(host_distances.data(), distances.data(), distances.size(), stream);
  raft::resource::sync_stream(res);
  for (int64_t q = 0; q < nq; ++q) {
    std::unordered_set<int64_t> valid;
    for (int64_t rank = 0; rank < k; ++rank) {
      auto id = host_ids[q * k + rank];
      if (id < 0 || id >= rows) {
        EXPECT_GT(host_distances[q * k + rank], 1e20f);
        continue;
      }
      EXPECT_TRUE(valid.insert(id).second);
      EXPECT_NE(
        std::find(host_candidates.begin() + q * nc, host_candidates.begin() + (q + 1) * nc, id),
        host_candidates.begin() + (q + 1) * nc);
      float expected_distance = 0;
      for (int64_t d = 0; d < dim; ++d) {
        auto delta = host_queries[q * dim + d] - host_data[id * dim + d];
        expected_distance += delta * delta;
      }
      EXPECT_NEAR(host_distances[q * k + rank], expected_distance, 0.001f);
    }
    EXPECT_EQ(valid.size(), q == 0 ? 8 : q == 1 ? 0 : 3);
  }
}

struct FlatBackend {
  using params_t = ivf_flat::search_params;
  static auto build(raft::resources const& res, data_view_t data, distance::DistanceType metric)
  {
    ivf_flat::index_params params;
    params.n_lists = 4;
    params.metric  = metric;
    return ivf_flat::build(res, params, data);
  }
  static void search(raft::resources const& res,
                     params_t params,
                     ivf_flat::index<float, int64_t> const& index,
                     data_view_t queries,
                     ids_view_t ids,
                     dists_view_t distances,
                     filtering::base_filter const& filter)
  {
    ivf_flat::search(res, params, index, queries, ids, distances, filter);
  }
};

struct PqBackend {
  using params_t = ivf_pq::search_params;
  static auto build(raft::resources const& res, data_view_t data, distance::DistanceType metric)
  {
    ivf_pq::index_params params;
    params.n_lists = 4;
    params.pq_bits = 4;
    params.pq_dim  = 8;
    params.metric  = metric;
    return ivf_pq::build(res, params, data);
  }
  static void search(raft::resources const& res,
                     params_t params,
                     ivf_pq::index<int64_t> const& index,
                     data_view_t queries,
                     ids_view_t ids,
                     dists_view_t distances,
                     filtering::base_filter const& filter)
  {
    ivf_pq::search(res, params, index, queries, ids, distances, filter);
  }
};

struct SqBackend {
  using params_t = ivf_sq::search_params;
  static auto build(raft::resources const& res, data_view_t data, distance::DistanceType metric)
  {
    ivf_sq::index_params params;
    params.n_lists = 4;
    params.metric  = metric;
    return ivf_sq::build(res, params, data);
  }
  static void search(raft::resources const& res,
                     params_t params,
                     ivf_sq::index<uint8_t> const& index,
                     data_view_t queries,
                     ids_view_t ids,
                     dists_view_t distances,
                     filtering::base_filter const& filter)
  {
    ivf_sq::search(res, params, index, queries, ids, distances, filter);
  }
};

bool passes(int64_t row, int64_t col)
{
  switch (row % 7) {
    case 0: return true;
    case 1: return false;
    case 2: return col == 17 || col == 513 || col == 768;
    default: return (col * 7 + row * 11) % 19 < row % 7;
  }
}

std::vector<uint32_t> bitmap_words(int64_t query_rows)
{
  std::vector<uint32_t> result((query_rows * rows + 31) / 32, 0);
  for (int64_t q = 0; q < query_rows; ++q) {
    for (int64_t id = 0; id < rows; ++id) {
      if (passes(q, id)) { result[(q * rows + id) / 32] |= uint32_t{1} << ((q * rows + id) % 32); }
    }
  }
  // Padding bits must never be treated as dataset IDs.
  if ((query_rows * rows) % 32 != 0) {
    result.back() |= ~((uint32_t{1} << ((query_rows * rows) % 32)) - 1);
  }
  return result;
}

template <typename T>
void check_large_pq_candidate_pools()
{
  raft::device_resources res;
  auto stream                   = raft::resource::get_cuda_stream(res);
  constexpr int64_t test_rows   = 8193;
  constexpr int64_t query_count = 9;
  constexpr int64_t candidates  = 6400;
  constexpr int64_t final_k     = 100;
  std::vector<T> host_data(test_rows * dim);
  std::vector<T> host_queries(query_count * dim);
  for (int64_t i = 0; i < test_rows * dim; ++i) {
    host_data[i] = T((i * 17 + i / dim * 13) % 251);
  }
  std::copy_n(host_data.begin(), host_queries.size(), host_queries.begin());
  std::vector<uint32_t> host_bitmap((query_count * test_rows + 31) / 32, 0);
  for (int64_t q = 0; q < query_count; ++q) {
    for (int64_t id = 0; id < test_rows; ++id) {
      if (passes(q, id)) {
        auto bit = q * test_rows + id;
        host_bitmap[bit / 32] |= uint32_t{1} << (bit % 32);
      }
    }
  }
  rmm::device_uvector<T> data(host_data.size(), stream);
  rmm::device_uvector<T> queries(host_queries.size(), stream);
  rmm::device_uvector<uint32_t> bitmap(host_bitmap.size(), stream);
  raft::copy(data.data(), host_data.data(), data.size(), stream);
  raft::copy(queries.data(), host_queries.data(), queries.size(), stream);
  raft::copy(bitmap.data(), host_bitmap.data(), bitmap.size(), stream);
  bitmap_filter_t filter(
    core::bitmap_view<uint32_t, int64_t>(bitmap.data(), query_count, test_rows));
  ivf_pq::index_params build;
  build.n_lists                  = 64;
  build.pq_dim                   = 8;
  build.pq_bits                  = 4;
  build.kmeans_n_iters           = 5;
  build.kmeans_trainset_fraction = 1.0;
  auto index                     = ivf_pq::build(
    res, build, raft::make_device_matrix_view<const T, int64_t>(data.data(), test_rows, dim));
  for (const int64_t offset : {0, 2}) {
    const int64_t count = query_count - offset;
    for (const uint32_t probes : {1, 64}) {
      SCOPED_TRACE(::testing::Message() << "offset=" << offset << " nprobe=" << probes);
      rmm::device_uvector<int64_t> ids(count * candidates, stream);
      rmm::device_uvector<float> distances(count * candidates, stream);
      // Poison outputs with plausible IDs and distances to expose unwritten selection slots.
      RAFT_CUDA_TRY(cudaMemsetAsync(ids.data(), 0, ids.size() * sizeof(int64_t), stream.get()));
      RAFT_CUDA_TRY(
        cudaMemsetAsync(distances.data(), 0, distances.size() * sizeof(float), stream.get()));
      ivf_pq::search_params params;
      params.n_probes                = probes;
      params.filter_query_offset     = offset;
      params.max_internal_batch_size = 2;
      ivf_pq::search(
        res,
        params,
        index,
        raft::make_device_matrix_view<const T, int64_t>(queries.data() + offset * dim, count, dim),
        raft::make_device_matrix_view<int64_t, int64_t>(ids.data(), count, candidates),
        raft::make_device_matrix_view<float, int64_t>(distances.data(), count, candidates),
        filter);
      std::vector<int64_t> host_ids(ids.size());
      std::vector<float> host_distances(distances.size());
      raft::copy(host_ids.data(), ids.data(), ids.size(), stream);
      raft::copy(host_distances.data(), distances.data(), distances.size(), stream);
      raft::resource::sync_stream(res);
      std::vector<std::unordered_set<int64_t>> pools(count);
      for (int64_t q = 0; q < count; ++q) {
        for (int64_t rank = 0; rank < candidates; ++rank) {
          const auto id = host_ids[q * candidates + rank];
          if (id == INT64_MAX) {
            ASSERT_EQ(host_distances[q * candidates + rank], std::numeric_limits<float>::max());
            continue;
          }
          ASSERT_GE(id, 0);
          ASSERT_LT(id, test_rows);
          ASSERT_TRUE(passes(q + offset, id));
          ASSERT_TRUE(pools[q].insert(id).second);
        }
        if ((q + offset) % 7 == 0) { ASSERT_FALSE(pools[q].empty()); }
        if ((q + offset) % 7 == 1) { ASSERT_TRUE(pools[q].empty()); }
        if (probes == 64) {
          int64_t passing = 0;
          for (int64_t id = 0; id < test_rows; ++id) {
            passing += passes(q + offset, id);
          }
          ASSERT_EQ(pools[q].size(), std::min(passing, candidates));
        }
      }
      rmm::device_uvector<int64_t> final_ids(count * final_k, stream);
      rmm::device_uvector<float> final_distances(count * final_k, stream);
      // Exercise full refinement chunks and the final partial chunk with the same candidate rows.
      for (int64_t first = 0; first < count; first += 3) {
        auto chunk = std::min<int64_t>(3, count - first);
        refine(res,
               raft::make_device_matrix_view<const T, int64_t>(data.data(), test_rows, dim),
               raft::make_device_matrix_view<const T, int64_t>(
                 queries.data() + (offset + first) * dim, chunk, dim),
               raft::make_device_matrix_view<const int64_t, int64_t>(
                 ids.data() + first * candidates, chunk, candidates),
               raft::make_device_matrix_view<int64_t, int64_t>(
                 final_ids.data() + first * final_k, chunk, final_k),
               raft::make_device_matrix_view<float, int64_t>(
                 final_distances.data() + first * final_k, chunk, final_k),
               distance::DistanceType::L2Unexpanded);
      }
      host_ids.resize(final_ids.size());
      host_distances.resize(final_distances.size());
      raft::copy(host_ids.data(), final_ids.data(), final_ids.size(), stream);
      raft::copy(host_distances.data(), final_distances.data(), final_distances.size(), stream);
      raft::resource::sync_stream(res);
      for (int64_t q = 0; q < count; ++q) {
        std::unordered_set<int64_t> seen;
        for (int64_t rank = 0; rank < final_k; ++rank) {
          const auto id = host_ids[q * final_k + rank];
          if (id == INT64_MAX) {
            ASSERT_GT(host_distances[q * final_k + rank], 1e20f);
            continue;
          }
          ASSERT_TRUE(pools[q].contains(id));
          ASSERT_TRUE(seen.insert(id).second);
          float exact = 0;
          for (int64_t d = 0; d < dim; ++d) {
            auto delta =
              float(host_queries[(q + offset) * dim + d]) - float(host_data[id * dim + d]);
            exact += delta * delta;
          }
          ASSERT_NEAR(host_distances[q * final_k + rank], exact, 0.001f);
        }
        ASSERT_EQ(seen.size(), std::min<std::size_t>(final_k, pools[q].size()));
      }
    }
  }
}

TEST(IvfBitmapRefinement, LargeFloatPqPoolsWithUnderfilledProbes)
{
  check_large_pq_candidate_pools<float>();
}

TEST(IvfBitmapRefinement, LargeUint8PqPoolsWithUnderfilledProbes)
{
  check_large_pq_candidate_pools<uint8_t>();
}

__global__ void evaluate_bitmap_filter(const int64_t* const* list_ids,
                                       uint32_t* bitmap,
                                       uint8_t* output)
{
  const auto i = threadIdx.x;
  if (i < 64) {
    // Include a query row past the bitmap, as well as invalid source IDs in the fake list.
    output[i] = detail::sample_filter_bitmap<uint32_t, int64_t>(
      list_ids, 2 + i / 8, 0, i % 8, bitmap, 9, 9 * rows, rows);
  }
}

TEST(IvfBitmapFilterDevice, MapsSourceIdsAndCanonicalizesMissingResults)
{
  int count         = 0;
  const auto status = cudaGetDeviceCount(&count);
  if (status != cudaSuccess || count == 0) {
    GTEST_SKIP() << "CUDA device unavailable: " << cudaGetErrorString(status);
  }
  raft::resources res;
  auto stream = raft::resource::get_cuda_stream(res);
  const std::vector<int64_t> host_ids{0, 7, 17, 513, 768, -1, rows, INT64_MAX};
  auto words = bitmap_words(9);
  rmm::device_uvector<int64_t> ids(host_ids.size(), stream);
  rmm::device_uvector<const int64_t*> lists(1, stream);
  rmm::device_uvector<uint32_t> bitmap(words.size(), stream);
  rmm::device_uvector<uint8_t> output(64, stream);
  raft::copy(ids.data(), host_ids.data(), host_ids.size(), stream);
  const int64_t* list = ids.data();
  raft::copy(lists.data(), &list, 1, stream);
  raft::copy(bitmap.data(), words.data(), words.size(), stream);
  evaluate_bitmap_filter<<<1, 64, 0, stream.get()>>>(lists.data(), bitmap.data(), output.data());
  std::vector<uint8_t> actual(64);
  raft::copy(actual.data(), output.data(), actual.size(), stream);
  raft::resource::sync_stream(res);
  for (int64_t i = 0; i < 64; ++i) {
    const auto query = 2 + i / 8;
    const auto id    = host_ids[i % 8];
    EXPECT_EQ(bool(actual[i]), query < 9 && id >= 0 && id < rows && passes(query, id));
  }

  std::vector<int64_t> host_neighbors(7 * 8);
  std::vector<float> host_distances(host_neighbors.size(), 7.0f);
  host_distances[0] = std::numeric_limits<float>::max();
  host_distances[1] = std::numeric_limits<float>::infinity();
  host_distances[2] = std::numeric_limits<float>::lowest();
  for (std::size_t i = 0; i < host_neighbors.size(); ++i) {
    host_neighbors[i] = host_ids[i % 8];
  }
  rmm::device_uvector<int64_t> neighbors(host_neighbors.size(), stream);
  rmm::device_uvector<float> distances(host_distances.size(), stream);
  raft::copy(neighbors.data(), host_neighbors.data(), host_neighbors.size(), stream);
  raft::copy(distances.data(), host_distances.data(), host_distances.size(), stream);
  bitmap_filter_t filter(core::bitmap_view<uint32_t, int64_t>(bitmap.data(), 9, rows));
  filtering::normalize_ivf_bitmap_results(res, filter, neighbors.data(), distances.data(), 7, 8, 2);
  raft::copy(host_neighbors.data(), neighbors.data(), host_neighbors.size(), stream);
  raft::copy(host_distances.data(), distances.data(), host_distances.size(), stream);
  raft::resource::sync_stream(res);
  for (std::size_t i = 0; i < host_neighbors.size(); ++i) {
    const auto id    = host_ids[i % 8];
    const bool valid = i >= 3 && id >= 0 && id < rows && passes(2 + i / 8, id);
    EXPECT_EQ(host_neighbors[i], valid ? id : INT64_MAX);
    EXPECT_EQ(host_distances[i], valid ? 7.0f : std::numeric_limits<float>::max());
  }
}

struct search_result {
  std::vector<int64_t> ids;
  std::vector<float> distances;
};

template <typename Backend>
class IvfBitmapFilterTest : public ::testing::Test {
 protected:
  void SetUp() override
  {
    int count         = 0;
    const auto status = cudaGetDeviceCount(&count);
    if (status != cudaSuccess || count == 0) {
      GTEST_SKIP() << "CUDA device unavailable: " << cudaGetErrorString(status);
    }
  }

  template <typename Index>
  search_result search(Index const& index,
                       typename Backend::params_t params,
                       const float* queries,
                       int64_t n_queries,
                       int64_t k,
                       filtering::base_filter const& filter)
  {
    auto stream = raft::resource::get_cuda_stream(res);
    rmm::device_uvector<int64_t> ids(n_queries * k, stream);
    rmm::device_uvector<float> distances(n_queries * k, stream);
    Backend::search(res,
                    params,
                    index,
                    raft::make_device_matrix_view<const float, int64_t>(queries, n_queries, dim),
                    raft::make_device_matrix_view<int64_t, int64_t>(ids.data(), n_queries, k),
                    raft::make_device_matrix_view<float, int64_t>(distances.data(), n_queries, k),
                    filter);
    search_result result{std::vector<int64_t>(n_queries * k), std::vector<float>(n_queries * k)};
    raft::copy(result.ids.data(), ids.data(), result.ids.size(), stream);
    raft::copy(result.distances.data(), distances.data(), result.distances.size(), stream);
    raft::resource::sync_stream(res);
    return result;
  }

  std::vector<float> data(int64_t n)
  {
    std::vector<float> result(n * dim);
    for (int64_t i = 0; i < n * dim; ++i) {
      result[i] = std::sin(float(i * 17 % 997)) + std::cos(float(i * 13 % 991));
    }
    return result;
  }

  raft::resources res;
};

using Backends = ::testing::Types<FlatBackend, PqBackend, SqBackend>;
TYPED_TEST_SUITE(IvfBitmapFilterTest, Backends);

TYPED_TEST(IvfBitmapFilterTest, MatchesPerQueryBitsetsAndPreservesOffsets)
{
  auto stream       = raft::resource::get_cuda_stream(this->res);
  auto host_data    = this->data(rows);
  auto host_queries = this->data(7);
  auto host_bitmap  = bitmap_words(9);
  rmm::device_uvector<float> dataset(host_data.size(), stream);
  rmm::device_uvector<float> queries(host_queries.size(), stream);
  rmm::device_uvector<uint32_t> bitmap(host_bitmap.size(), stream);
  raft::copy(dataset.data(), host_data.data(), host_data.size(), stream);
  raft::copy(queries.data(), host_queries.data(), host_queries.size(), stream);
  raft::copy(bitmap.data(), host_bitmap.data(), host_bitmap.size(), stream);
  bitmap_filter_t filter(core::bitmap_view<uint32_t, int64_t>(bitmap.data(), 9, rows));

  for (const auto metric :
       {distance::DistanceType::L2Expanded, distance::DistanceType::InnerProduct}) {
    auto index = TypeParam::build(
      this->res,
      raft::make_device_matrix_view<const float, int64_t>(dataset.data(), rows, dim),
      metric);
    typename TypeParam::params_t params;
    params.n_probes = 4;  // Every eligible candidate is scanned.
    if constexpr (std::is_same_v<TypeParam, PqBackend>) { params.max_internal_batch_size = 2; }
    for (const int64_t k : {8, 100, 300}) {  // Fused top-k and materialized-distance paths.
      SCOPED_TRACE(::testing::Message() << "metric=" << int(metric) << " k=" << k);
      params.filter_query_offset = 2;
      auto actual                = this->search(index, params, queries.data(), 7, k, filter);
      for (int64_t q = 0; q < 7; ++q) {
        std::vector<uint32_t> words((rows + 31) / 32, 0);
        int64_t passing_count = 0;
        for (int64_t id = 0; id < rows; ++id) {
          if (passes(q + 2, id)) {
            words[id / 32] |= uint32_t{1} << (id % 32);
            ++passing_count;
          }
        }
        rmm::device_uvector<uint32_t> bitset(words.size(), stream);
        raft::copy(bitset.data(), words.data(), words.size(), stream);
        bitset_filter_t shared(core::bitset_view<uint32_t, int64_t>(bitset.data(), rows));
        params.filter_query_offset = 0;
        auto expected = this->search(index, params, queries.data() + q * dim, 1, k, shared);
        std::vector<float> actual_distances, expected_distances;
        std::unordered_set<int64_t> seen;
        for (int64_t rank = 0; rank < k; ++rank) {
          const auto id = actual.ids[q * k + rank];
          if (id == std::numeric_limits<int64_t>::max()) {
            EXPECT_EQ(actual.distances[q * k + rank], std::numeric_limits<float>::max());
          } else {
            ASSERT_GE(id, 0);
            ASSERT_LT(id, rows);
            ASSERT_TRUE(passes(q + 2, id));
            ASSERT_TRUE(seen.insert(id).second);
            actual_distances.push_back(actual.distances[q * k + rank]);
          }
          // Existing bitset materialization can return dummy IDs for missing results.
          const auto ref_id = expected.ids[rank];
          if (ref_id >= 0 && ref_id < rows && passes(q + 2, ref_id) &&
              std::abs(expected.distances[rank]) < std::numeric_limits<float>::max() / 2) {
            expected_distances.push_back(expected.distances[rank]);
          }
        }
        ASSERT_EQ(actual_distances.size(), std::min(k, passing_count));
        ASSERT_EQ(actual_distances.size(), expected_distances.size());
        std::sort(actual_distances.begin(), actual_distances.end());
        std::sort(expected_distances.begin(), expected_distances.end());
        for (std::size_t i = 0; i < actual_distances.size(); ++i) {
          EXPECT_NEAR(actual_distances[i],
                      expected_distances[i],
                      0.002f * std::max(1.0f, std::abs(expected_distances[i])));
        }
      }
      // External benchmark batches must use the same resident bitmap rows.
      for (int64_t q = 0; q < 7; q += 2) {
        params.filter_query_offset = q + 2;
        auto part                  = this->search(
          index, params, queries.data() + q * dim, std::min<int64_t>(2, 7 - q), k, filter);
        for (int64_t local = 0; local < std::min<int64_t>(2, 7 - q); ++local) {
          std::vector<float> batch_distances, full_distances;
          std::unordered_set<int64_t> seen;
          for (int64_t rank = 0; rank < k; ++rank) {
            const auto pos      = local * k + rank;
            const auto full_pos = (q + local) * k + rank;
            if (actual.ids[full_pos] != std::numeric_limits<int64_t>::max()) {
              full_distances.push_back(actual.distances[full_pos]);
            }
            if (part.ids[pos] == std::numeric_limits<int64_t>::max()) {
              EXPECT_EQ(part.distances[pos], std::numeric_limits<float>::max());
            } else {
              ASSERT_GE(part.ids[pos], 0);
              ASSERT_LT(part.ids[pos], rows);
              EXPECT_TRUE(passes(q + 2 + local, part.ids[pos]));
              EXPECT_TRUE(seen.insert(part.ids[pos]).second);
              batch_distances.push_back(part.distances[pos]);
            }
          }
          // IVF's materialized select-k path does not promise rank order.
          std::sort(batch_distances.begin(), batch_distances.end());
          std::sort(full_distances.begin(), full_distances.end());
          ASSERT_EQ(batch_distances.size(), full_distances.size());
          for (std::size_t i = 0; i < batch_distances.size(); ++i) {
            EXPECT_NEAR(batch_distances[i],
                        full_distances[i],
                        0.002f * std::max(1.0f, std::abs(full_distances[i])));
          }
        }
      }
    }
  }
}

TYPED_TEST(IvfBitmapFilterTest, QueryRowsSurviveKernelGridAndInternalBatchBoundaries)
{
  constexpr int64_t n_queries = 65539;  // Cross both IVF-Flat and IVF-SQ grid-y limits.
  auto stream                 = raft::resource::get_cuda_stream(this->res);
  auto host_data              = this->data(rows);
  auto host_queries           = this->data(n_queries);
  std::vector<uint32_t> words((n_queries * rows + 31) / 32, 0);
  for (int64_t q = 0; q < n_queries; ++q) {
    const auto bit = q * rows + (q * 7) % rows;
    words[bit / 32] |= uint32_t{1} << (bit % 32);
  }
  rmm::device_uvector<float> dataset(host_data.size(), stream);
  rmm::device_uvector<float> queries(host_queries.size(), stream);
  rmm::device_uvector<uint32_t> bitmap(words.size(), stream);
  raft::copy(dataset.data(), host_data.data(), host_data.size(), stream);
  raft::copy(queries.data(), host_queries.data(), host_queries.size(), stream);
  raft::copy(bitmap.data(), words.data(), words.size(), stream);
  auto index =
    TypeParam::build(this->res,
                     raft::make_device_matrix_view<const float, int64_t>(dataset.data(), rows, dim),
                     distance::DistanceType::L2Expanded);
  typename TypeParam::params_t params;
  params.n_probes = 4;
  bitmap_filter_t filter(core::bitmap_view<uint32_t, int64_t>(bitmap.data(), n_queries, rows));
  auto result = this->search(index, params, queries.data(), n_queries, 1, filter);
  for (int64_t q = 0; q < n_queries; ++q) {
    ASSERT_EQ(result.ids[q], (q * 7) % rows) << "query=" << q;
  }
}

TYPED_TEST(IvfBitmapFilterTest, DynamicBatchingRejectsBitmapFilters)
{
  if constexpr (std::is_same_v<TypeParam, SqBackend>) {
    GTEST_SKIP() << "IVF-SQ has no dynamic batching wrapper";
  } else {
    auto stream    = raft::resource::get_cuda_stream(this->res);
    auto host_data = this->data(rows);
    rmm::device_uvector<float> dataset(host_data.size(), stream);
    raft::copy(dataset.data(), host_data.data(), host_data.size(), stream);
    auto index = TypeParam::build(
      this->res,
      raft::make_device_matrix_view<const float, int64_t>(dataset.data(), rows, dim),
      distance::DistanceType::L2Expanded);
    bitmap_filter_t filter(core::bitmap_view<uint32_t, int64_t>(nullptr, 7, rows));
    typename TypeParam::params_t search_params;
    dynamic_batching::index_params batch_params;
    batch_params.k              = 8;
    batch_params.max_batch_size = 2;
    batch_params.n_queues       = 1;
    using batch_index_t         = dynamic_batching::index<float, int64_t>;
    EXPECT_THROW((batch_index_t(this->res, batch_params, index, search_params, &filter)),
                 raft::logic_error);
  }
}

TEST(IvfBitmapFilterValidation, RejectsWrongDimensionsAndOutOfRangeOffsets)
{
  bitmap_filter_t filter(core::bitmap_view<uint32_t, int64_t>(nullptr, 7, rows));
  EXPECT_NO_THROW(filtering::validate_ivf_bitmap_filter(filter, rows, 2, 5));
  EXPECT_THROW(filtering::validate_ivf_bitmap_filter(filter, rows + 1, 2, 0), raft::logic_error);
  EXPECT_THROW(filtering::validate_ivf_bitmap_filter(filter, rows, 3, 5), raft::logic_error);
  EXPECT_THROW(filtering::validate_ivf_bitmap_filter(filter, rows, 1, UINT32_MAX),
               raft::logic_error);
  bitmap_filter_t large(core::bitmap_view<uint32_t, int64_t>(nullptr, int64_t(UINT32_MAX) + 2, 1));
  EXPECT_THROW(filtering::validate_ivf_bitmap_filter(large, 1, 2, UINT32_MAX), raft::logic_error);
}

}  // namespace
}  // namespace cuvs::neighbors
