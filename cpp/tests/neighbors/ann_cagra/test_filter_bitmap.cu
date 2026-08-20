/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/core/bitmap.hpp>
#include <cuvs/neighbors/cagra.hpp>

#include "../../../bench/ann/src/common/output_set_metrics.hpp"
#include "../../../src/neighbors/cagra_benchmark.hpp"
#include "../../../src/neighbors/detail/cagra/navix_bitmap_seeds.cuh"
#include "../../../src/neighbors/detail/cagra/sample_filter_utils.cuh"

#include <raft/core/bitset.cuh>
#include <raft/core/copy.cuh>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/random/rng.cuh>
#include <raft/util/integer_utils.hpp>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/device_uvector.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace cuvs::neighbors::cagra {
namespace {

constexpr std::int64_t kRows    = 769;  // Deliberately not word-aligned.
constexpr std::int64_t kDim     = 16;
constexpr std::int64_t kQueries = 7;  // Deliberately not divisible by max_queries=2.
constexpr std::int64_t k        = 8;

using bitmap_filter_t      = cuvs::neighbors::filtering::bitmap_filter<std::uint32_t, std::int64_t>;
using accumulator_filter_t = detail::CagraSampleFilterWithRuntimeState<bitmap_filter_t, true>;
using seeded_filter_t = detail::CagraSampleFilterWithBitmapSeedRuntimeState<accumulator_filter_t>;
using offset_seeded_filter_t = detail::CagraSampleFilterWithBaseQueryIdOffset<seeded_filter_t>;
static_assert(!detail::cagra_filter_uses_bitmap_seeds<bitmap_filter_t>::value);
static_assert(detail::cagra_filter_uses_bitmap_seeds<seeded_filter_t>::value);
static_assert(detail::cagra_filter_uses_bitmap_seeds<offset_seeded_filter_t>::value);
static_assert(detail::cagra_filter_uses_passing_accumulator<offset_seeded_filter_t>::value);
static_assert(!detail::cagra_filter_uses_navix<offset_seeded_filter_t>::value);

template <typename OutputIndexT>
struct bitmap_search_result {
  std::vector<OutputIndexT> neighbors;
  std::vector<float> distances;
};

class device_bitmap {
 public:
  template <typename Predicate>
  device_bitmap(raft::resources const& res,
                std::int64_t rows,
                std::int64_t cols,
                Predicate predicate,
                bool set_padding_bits = false)
    : rows_(rows),
      cols_(cols),
      words_(raft::div_rounding_up_safe<std::int64_t>(rows * cols, 32),
             raft::resource::get_cuda_stream(res))
  {
    assign(res, std::move(predicate), set_padding_bits);
  }

  template <typename Predicate>
  void assign(raft::resources const& res, Predicate predicate, bool set_padding_bits = false)
  {
    std::vector<std::uint32_t> host(words_.size(), 0);
    for (std::int64_t row = 0; row < rows_; ++row) {
      for (std::int64_t col = 0; col < cols_; ++col) {
        if (predicate(row, col)) {
          const auto bit = row * cols_ + col;
          host[static_cast<std::size_t>(bit / 32)] |= std::uint32_t{1} << (bit % 32);
        }
      }
    }
    if (set_padding_bits && !host.empty()) {
      const auto valid_in_last_word = static_cast<unsigned>((rows_ * cols_) % 32);
      if (valid_in_last_word != 0) {
        host.back() |= ~((std::uint32_t{1} << valid_in_last_word) - 1);
      }
    }
    raft::copy(words_.data(), host.data(), host.size(), raft::resource::get_cuda_stream(res));
  }

  auto view()
  {
    return cuvs::core::bitmap_view<std::uint32_t, std::int64_t>(words_.data(), rows_, cols_);
  }

  auto filter()
  {
    return cuvs::neighbors::filtering::bitmap_filter<std::uint32_t, std::int64_t>(view());
  }

 private:
  std::int64_t rows_;
  std::int64_t cols_;
  rmm::device_uvector<std::uint32_t> words_;
};

std::string query_specific_udf_source()
{
  return R"cpp(
    __device__ bool cuvs_filter_udf(uint32_t query_id, source_index_t source_id, void*)
    {
      return ((source_id * 7u + query_id * 11u) % 5u) < 2u;
    }
  )cpp";
}

class CagraBitmapFilterTest : public ::testing::TestWithParam<cagra::search_algo> {
 protected:
  void SetUp() override
  {
    dataset.emplace(raft::make_device_matrix<float, std::int64_t>(res, kRows, kDim));
    queries.emplace(raft::make_device_matrix<float, std::int64_t>(res, kQueries, kDim));

    raft::random::RngState rng(1234ULL);
    raft::random::uniform(res, rng, dataset->data_handle(), dataset->size(), -1.0f, 1.0f);
    raft::random::uniform(res, rng, queries->data_handle(), queries->size(), -1.0f, 1.0f);

    cagra::index_params index_params;
    index_params.metric                    = cuvs::distance::DistanceType::L2Expanded;
    index_params.graph_degree              = 32;
    index_params.intermediate_graph_degree = 64;
    index_params.graph_build_params =
      cagra::graph_build_params::nn_descent_params(index_params.intermediate_graph_degree);
    index.emplace(cagra::build(res, index_params, raft::make_const_mdspan(dataset->view())));
    raft::resource::sync_stream(res);
  }

  void TearDown() override
  {
    index.reset();
    queries.reset();
    dataset.reset();
    raft::resource::sync_stream(res);
  }

  cagra::search_params params(std::int64_t result_k = k, float filtering_rate = -1.0f) const
  {
    cagra::search_params result;
    result.algo              = GetParam();
    result.itopk_size        = std::max<std::int64_t>(64, result_k);
    result.max_queries       = 2;
    result.thread_block_size = 256;
    result.filtering_rate    = filtering_rate;
    return result;
  }

  template <typename OutputIndexT = std::uint32_t>
  bitmap_search_result<OutputIndexT> search_dynamic(
    raft::resources const& search_res,
    cuvs::neighbors::filtering::base_filter const& filter,
    cagra::search_params const& search_params,
    std::int64_t result_k = k)
  {
    auto neighbors =
      raft::make_device_matrix<OutputIndexT, std::int64_t>(search_res, kQueries, result_k);
    auto distances = raft::make_device_matrix<float, std::int64_t>(search_res, kQueries, result_k);
    cagra::search(search_res,
                  search_params,
                  *index,
                  raft::make_const_mdspan(queries->view()),
                  neighbors.view(),
                  distances.view(),
                  filter);

    bitmap_search_result<OutputIndexT> result{
      std::vector<OutputIndexT>(static_cast<std::size_t>(kQueries * result_k)),
      std::vector<float>(static_cast<std::size_t>(kQueries * result_k))};
    auto stream = raft::resource::get_cuda_stream(search_res);
    raft::copy(result.neighbors.data(), neighbors.data_handle(), result.neighbors.size(), stream);
    raft::copy(result.distances.data(), distances.data_handle(), result.distances.size(), stream);
    raft::resource::sync_stream(search_res);
    return result;
  }

  template <typename IndexT>
  static void expect_same(bitmap_search_result<IndexT> const& expected,
                          bitmap_search_result<IndexT> const& actual)
  {
    ASSERT_EQ(expected.neighbors.size(), actual.neighbors.size());
    ASSERT_EQ(expected.distances.size(), actual.distances.size());
    for (std::size_t i = 0; i < expected.neighbors.size(); ++i) {
      SCOPED_TRACE(::testing::Message() << "result position " << i);
      EXPECT_EQ(expected.neighbors[i], actual.neighbors[i]);
      EXPECT_EQ(expected.distances[i], actual.distances[i]);
    }
  }

  raft::device_resources res;
  std::optional<raft::device_matrix<float, std::int64_t>> dataset;
  std::optional<raft::device_matrix<float, std::int64_t>> queries;
  std::optional<cagra::index<float, std::uint32_t>> index;
};

TEST_P(CagraBitmapFilterTest, AllPassingMatchesBitsetAcrossTopKBoundaries)
{
  device_bitmap bitmap(res, kQueries, kRows, [](auto, auto) { return true; });
  auto bitmap_filter = bitmap.filter();
  cuvs::core::bitset<std::uint32_t, std::int64_t> bitset(res, kRows, true);
  auto bitset_filter = cuvs::neighbors::filtering::bitset_filter(bitset.view());

  for (auto result_k : {std::int64_t{1},
                        std::int64_t{8},
                        std::int64_t{31},
                        std::int64_t{32},
                        std::int64_t{33},
                        std::int64_t{64}}) {
    SCOPED_TRACE(::testing::Message() << "top-k " << result_k);
    auto search_params = params(result_k, 0.0f);
    auto expected      = search_dynamic(res, bitset_filter, search_params, result_k);
    auto actual        = search_dynamic(res, bitmap_filter, search_params, result_k);
    for (std::int64_t query_id = 0; query_id < kQueries; ++query_id) {
      const auto row_begin = static_cast<std::size_t>(query_id * result_k);
      const auto row_end   = row_begin + static_cast<std::size_t>(result_k);
      std::int64_t overlap = 0;
      float previous       = -std::numeric_limits<float>::infinity();
      for (auto pos = row_begin; pos < row_end; ++pos) {
        EXPECT_LT(actual.neighbors[pos], static_cast<std::uint32_t>(kRows));
        EXPECT_TRUE(std::isfinite(actual.distances[pos]));
        EXPECT_GE(actual.distances[pos], previous);
        previous = actual.distances[pos];
        overlap += std::find(expected.neighbors.begin() + row_begin,
                             expected.neighbors.begin() + row_end,
                             actual.neighbors[pos]) != expected.neighbors.begin() + row_end;
      }
      // MULTI_CTA combines concurrently produced partial queues. Near the approximate-search
      // frontier, otherwise equivalent filter kernels can differ by a few tail candidates.
      const auto minimum_overlap = (95 * result_k + 99) / 100;
      EXPECT_GE(overlap, minimum_overlap);
    }
  }
}

TEST_P(CagraBitmapFilterTest, DistinctRowsMatchEquivalentUdfAcrossInternalTiles)
{
  auto predicate = [](std::int64_t query_id, std::int64_t source_id) {
    return ((source_id * 7 + query_id * 11) % 5) < 2;
  };
  device_bitmap bitmap(res, kQueries, kRows, predicate);
  auto bitmap_filter = bitmap.filter();
  cuvs::neighbors::filtering::udf_filter udf_filter(query_specific_udf_source(), nullptr, 0.6f);

  auto search_params = params(k, 0.6f);
  auto expected      = search_dynamic(res, udf_filter, search_params);
  auto actual        = search_dynamic(res, bitmap_filter, search_params);
  expect_same(expected, actual);

  for (std::int64_t query_id = 0; query_id < kQueries; ++query_id) {
    for (std::int64_t rank = 0; rank < k; ++rank) {
      const auto source_id = actual.neighbors[static_cast<std::size_t>(query_id * k + rank)];
      ASSERT_LT(source_id, static_cast<std::uint32_t>(kRows));
      EXPECT_TRUE(predicate(query_id, source_id));
    }
  }
}

TEST_P(CagraBitmapFilterTest, LegacyNavixBitmapMatchesEquivalentUdfExactly)
{
  auto predicate = [](std::int64_t query_id, std::int64_t source_id) {
    return ((source_id * 7 + query_id * 11) % 5) < 2;
  };
  device_bitmap bitmap(res, kQueries, kRows, predicate);
  auto bitmap_filter = bitmap.filter();
  cuvs::neighbors::filtering::udf_filter udf_filter(query_specific_udf_source(), nullptr, 0.6f);
  auto search_params           = params(k, 0.6f);
  search_params.algo           = cagra::search_algo::SINGLE_CTA;
  search_params.filtering_rate = 0.0f;

  auto run = [&](auto const& filter, bool bitmap_path) {
    auto neighbors = raft::make_device_matrix<std::int64_t, std::int64_t>(res, kQueries, k);
    auto distances = raft::make_device_matrix<float, std::int64_t>(res, kQueries, k);
    if (bitmap_path) {
      detail::benchmark_search_navix_bitmap<float>(res,
                                                   search_params,
                                                   *index,
                                                   raft::make_const_mdspan(queries->view()),
                                                   neighbors.view(),
                                                   distances.view(),
                                                   filter,
                                                   0,
                                                   0);
    } else {
      detail::benchmark_search_navix_udf<float>(res,
                                                search_params,
                                                *index,
                                                raft::make_const_mdspan(queries->view()),
                                                neighbors.view(),
                                                distances.view(),
                                                filter,
                                                0);
    }
    bitmap_search_result<std::int64_t> result{std::vector<std::int64_t>(neighbors.size()),
                                              std::vector<float>(distances.size())};
    auto stream = raft::resource::get_cuda_stream(res);
    raft::copy(result.neighbors.data(), neighbors.data_handle(), neighbors.size(), stream);
    raft::copy(result.distances.data(), distances.data_handle(), distances.size(), stream);
    raft::resource::sync_stream(res);
    return result;
  };

  expect_same(run(udf_filter, false), run(bitmap_filter, true));
}

TEST_P(CagraBitmapFilterTest, DefaultAccumulatorBitmapMatchesEquivalentUdfExactly)
{
  auto predicate = [](std::int64_t query_id, std::int64_t source_id) {
    return ((source_id * 7 + query_id * 11) % 5) < 2;
  };
  device_bitmap bitmap(res, kQueries, kRows, predicate);
  auto bitmap_filter = bitmap.filter();
  cuvs::neighbors::filtering::udf_filter udf_filter(query_specific_udf_source(), nullptr, 0.6f);
  auto search_params           = params(k, 0.0f);
  search_params.algo           = cagra::search_algo::SINGLE_CTA;
  search_params.filtering_rate = 0.0f;

  auto run = [&](auto const& filter, bool bitmap_path) {
    auto neighbors = raft::make_device_matrix<std::int64_t, std::int64_t>(res, kQueries, k);
    auto distances = raft::make_device_matrix<float, std::int64_t>(res, kQueries, k);
    if (bitmap_path) {
      detail::benchmark_search_bitmap_with_query_offset<float>(
        res,
        search_params,
        *index,
        raft::make_const_mdspan(queries->view()),
        neighbors.view(),
        distances.view(),
        filter,
        0,
        true);
    } else {
      detail::benchmark_search_favor_udf_with_sampled_rates<float>(
        res,
        search_params,
        *index,
        raft::make_const_mdspan(queries->view()),
        neighbors.view(),
        distances.view(),
        filter,
        nullptr,
        true);
    }
    bitmap_search_result<std::int64_t> result{std::vector<std::int64_t>(neighbors.size()),
                                              std::vector<float>(distances.size())};
    auto stream = raft::resource::get_cuda_stream(res);
    raft::copy(result.neighbors.data(), neighbors.data_handle(), neighbors.size(), stream);
    raft::copy(result.distances.data(), distances.data_handle(), distances.size(), stream);
    raft::resource::sync_stream(res);
    return result;
  };

  auto actual = run(bitmap_filter, true);
  expect_same(run(udf_filter, false), actual);
  for (std::int64_t query_id = 0; query_id < kQueries; ++query_id) {
    for (std::int64_t rank = 0; rank < k; ++rank) {
      const auto source = actual.neighbors[static_cast<std::size_t>(query_id * k + rank)];
      ASSERT_LT(source, kRows);
      EXPECT_TRUE(predicate(query_id, source));
    }
  }
}

TEST_P(CagraBitmapFilterTest, PreservesOriginalWordSizeMetadata)
{
  auto predicate = [](std::int64_t query_id, std::int64_t source_id) {
    return ((source_id * 7 + query_id * 11) % 5) < 2;
  };
  const auto total_bits = kQueries * kRows;
  std::vector<std::uint64_t> host_words(
    static_cast<std::size_t>(raft::div_rounding_up_safe<std::int64_t>(total_bits, 64)), 0);
  for (std::int64_t query_id = 0; query_id < kQueries; ++query_id) {
    for (std::int64_t source_id = 0; source_id < kRows; ++source_id) {
      if (predicate(query_id, source_id)) {
        const auto bit = query_id * kRows + source_id;
        host_words[static_cast<std::size_t>(bit / 64)] |= std::uint64_t{1} << (bit % 64);
      }
    }
  }

  rmm::device_uvector<std::uint64_t> device_words(host_words.size(),
                                                  raft::resource::get_cuda_stream(res));
  raft::copy(device_words.data(),
             host_words.data(),
             host_words.size(),
             raft::resource::get_cuda_stream(res));
  auto bitmap_view = cuvs::core::bitmap_view<std::uint32_t, std::int64_t>(
    reinterpret_cast<std::uint32_t*>(device_words.data()), kQueries, kRows, 64);
  auto bitmap_filter =
    cuvs::neighbors::filtering::bitmap_filter<std::uint32_t, std::int64_t>(bitmap_view);
  cuvs::neighbors::filtering::udf_filter udf_filter(query_specific_udf_source(), nullptr, 0.6f);

  auto search_params = params(k, 0.6f);
  expect_same(search_dynamic(res, udf_filter, search_params),
              search_dynamic(res, bitmap_filter, search_params));
}

TEST_P(CagraBitmapFilterTest, RuntimeApiSupportsBothOutputIndexTypes)
{
  device_bitmap bitmap(res, kQueries, kRows, [](auto query_id, auto source_id) {
    return ((query_id + source_id) % 3) != 0;
  });
  auto filter        = bitmap.filter();
  auto search_params = params(k, 1.0f / 3.0f);

  auto uint32_result = search_dynamic<std::uint32_t>(res, filter, search_params);
  auto int64_result  = search_dynamic<std::int64_t>(res, filter, search_params);
  ASSERT_EQ(uint32_result.neighbors.size(), int64_result.neighbors.size());
  ASSERT_EQ(uint32_result.distances, int64_result.distances);
  for (std::size_t i = 0; i < uint32_result.neighbors.size(); ++i) {
    EXPECT_EQ(static_cast<std::int64_t>(uint32_result.neighbors[i]), int64_result.neighbors[i]);
  }
}

TEST_P(CagraBitmapFilterTest, RejectAllAndUnderfilledRowsUseCorrectSentinels)
{
  device_bitmap reject_all(res, kQueries, kRows, [](auto, auto) { return false; });
  auto reject_filter = reject_all.filter();
  auto rejected      = search_dynamic(res, reject_filter, params(k, -1.0f));
  for (std::size_t pos = 0; pos < rejected.neighbors.size(); ++pos) {
    EXPECT_EQ(rejected.neighbors[pos], std::numeric_limits<std::uint32_t>::max());
    EXPECT_EQ(rejected.distances[pos], std::numeric_limits<float>::max());
  }

  auto sparse_predicate = [](std::int64_t query_id, std::int64_t source_id) {
    return source_id == query_id || source_id == query_id + 101 || source_id == query_id + 503;
  };
  device_bitmap sparse(res, kQueries, kRows, sparse_predicate);
  auto sparse_filter = sparse.filter();
  auto underfilled   = search_dynamic(res, sparse_filter, params(k, 0.996f));
  for (std::int64_t query_id = 0; query_id < kQueries; ++query_id) {
    bool saw_sentinel        = false;
    std::int64_t valid_count = 0;
    float previous_distance  = -std::numeric_limits<float>::infinity();
    for (std::int64_t rank = 0; rank < k; ++rank) {
      const auto pos       = static_cast<std::size_t>(query_id * k + rank);
      const auto source_id = underfilled.neighbors[pos];
      if (source_id == std::numeric_limits<std::uint32_t>::max()) {
        saw_sentinel = true;
        EXPECT_EQ(underfilled.distances[pos], std::numeric_limits<float>::max());
      } else {
        EXPECT_FALSE(saw_sentinel);
        EXPECT_TRUE(sparse_predicate(query_id, source_id));
        EXPECT_GE(underfilled.distances[pos], previous_distance);
        ++valid_count;
        previous_distance = underfilled.distances[pos];
      }
    }
    EXPECT_LE(valid_count, 3);
    EXPECT_GE(k - valid_count, 5);
  }
}

TEST_P(CagraBitmapFilterTest, AutomaticRateMatchesExactAggregateAndExplicitSearch)
{
  std::int64_t passing_count = 0;
  auto predicate             = [&passing_count](std::int64_t query_id, std::int64_t source_id) {
    const bool pass = source_id % (query_id + 2) == 0;
    passing_count += pass ? 1 : 0;
    return pass;
  };
  device_bitmap bitmap(res, kQueries, kRows, predicate, true);
  auto filter = bitmap.filter();

  const auto expected_rate =
    static_cast<float>(kQueries * kRows - passing_count) / static_cast<float>(kQueries * kRows);
  auto automatic = search_dynamic(res, filter, params(k, -1.0f));
  auto explicit_ = search_dynamic(res, filter, params(k, expected_rate));
  expect_same(explicit_, automatic);
}

TEST_P(CagraBitmapFilterTest, PayloadTracksContentPointerAndNonDefaultStream)
{
  auto even = [](auto, auto source_id) { return source_id % 2 == 0; };
  auto odd  = [](auto, auto source_id) { return source_id % 2 != 0; };
  device_bitmap bitmap(res, kQueries, kRows, even);
  auto filter = bitmap.filter();

  auto even_result = search_dynamic(res, filter, params(k, 0.5f));
  bitmap.assign(res, odd);
  auto odd_result = search_dynamic(res, filter, params(k, 0.5f));
  EXPECT_NE(even_result.neighbors, odd_result.neighbors);
  for (auto source_id : odd_result.neighbors) {
    if (source_id < static_cast<std::uint32_t>(kRows)) { EXPECT_EQ(source_id % 2, 1u); }
  }

  device_bitmap second_bitmap(res, kQueries, kRows, even);
  auto second_filter = second_bitmap.filter();
  // The bitmap storage was populated on res's stream; make it visible before a search on a
  // different stream. The search path itself is responsible for ordering its cached payload.
  raft::resource::sync_stream(res);
  raft::device_resources alternate_res;
  raft::resource::set_cuda_stream(alternate_res, rmm::cuda_stream_per_thread);
  auto alternate_result = search_dynamic(alternate_res, second_filter, params(k, 0.5f));
  expect_same(even_result, alternate_result);
}

TEST_P(CagraBitmapFilterTest, SourceIndexRemappingUsesExternalIds)
{
  auto source_indices = raft::make_device_vector<std::uint32_t, std::int64_t>(res, kRows);
  std::vector<std::uint32_t> host_source_indices(static_cast<std::size_t>(kRows));
  for (std::int64_t row = 0; row < kRows; ++row) {
    host_source_indices[static_cast<std::size_t>(row)] =
      static_cast<std::uint32_t>(kRows - 1 - row);
  }
  raft::copy(source_indices.data_handle(),
             host_source_indices.data(),
             host_source_indices.size(),
             raft::resource::get_cuda_stream(res));
  index->update_source_indices(std::move(source_indices));

  auto predicate = [](std::int64_t query_id, std::int64_t source_id) {
    return ((source_id * 7 + query_id * 11) % 5) < 2;
  };
  device_bitmap bitmap(res, kQueries, kRows, predicate);
  auto bitmap_filter = bitmap.filter();
  cuvs::neighbors::filtering::udf_filter udf_filter(query_specific_udf_source(), nullptr, 0.6f);
  auto search_params = params(k, 0.6f);

  auto expected = search_dynamic(res, udf_filter, search_params);
  auto actual   = search_dynamic(res, bitmap_filter, search_params);
  expect_same(expected, actual);
  for (std::int64_t query_id = 0; query_id < kQueries; ++query_id) {
    for (std::int64_t rank = 0; rank < k; ++rank) {
      auto source_id = actual.neighbors[static_cast<std::size_t>(query_id * k + rank)];
      ASSERT_LT(source_id, static_cast<std::uint32_t>(kRows));
      EXPECT_TRUE(predicate(query_id, source_id));
    }
  }

  // The bitmap-seed prepass scans internal IDs when source remapping is present, while both the
  // predicate and final output remain in source-ID space.
  auto navix_neighbors = raft::make_device_matrix<std::int64_t, std::int64_t>(res, kQueries, k);
  auto navix_distances = raft::make_device_matrix<float, std::int64_t>(res, kQueries, k);
  auto stream          = raft::resource::get_cuda_stream(res);
  rmm::device_uvector<std::uint32_t> navix_seeds(kQueries * k, stream);
  rmm::device_uvector<std::uint32_t> navix_counts(kQueries, stream);
  rmm::device_uvector<std::uint32_t> navix_inspected(kQueries, stream);
  search_params.algo = cagra::search_algo::SINGLE_CTA;
  detail::benchmark_search_navix_bitmap_seeded<float>(res,
                                                      search_params,
                                                      *index,
                                                      raft::make_const_mdspan(queries->view()),
                                                      navix_neighbors.view(),
                                                      navix_distances.view(),
                                                      bitmap_filter,
                                                      0,
                                                      navix_seeds.data(),
                                                      navix_counts.data(),
                                                      navix_inspected.data(),
                                                      0);
  std::vector<std::int64_t> host_navix(navix_neighbors.size());
  std::vector<std::uint32_t> host_seeds(navix_seeds.size());
  raft::copy(host_navix.data(), navix_neighbors.data_handle(), navix_neighbors.size(), stream);
  raft::copy(host_seeds.data(), navix_seeds.data(), navix_seeds.size(), stream);
  raft::resource::sync_stream(res);
  std::vector<std::uint32_t> expected_internal_seeds;
  for (std::uint32_t internal = 0; internal < kRows && expected_internal_seeds.size() < k;
       ++internal) {
    const auto source = static_cast<std::uint32_t>(kRows - 1 - internal);
    if (predicate(0, source)) { expected_internal_seeds.push_back(internal); }
  }
  EXPECT_EQ(std::vector<std::uint32_t>(host_seeds.begin(), host_seeds.begin() + k),
            expected_internal_seeds);
  for (std::int64_t query_id = 0; query_id < kQueries; ++query_id) {
    for (std::int64_t rank = 0; rank < k; ++rank) {
      const auto source = host_navix[static_cast<std::size_t>(query_id * k + rank)];
      if (source != std::numeric_limits<std::int64_t>::max()) {
        EXPECT_TRUE(predicate(query_id, source));
      }
    }
  }

  // Default and default+accumulator consume the identical internal-ID seed rows without enabling
  // NaviX traversal.  Their public-facing results must remain in remapped source-ID space.
  for (const bool passing_accumulator : {false, true}) {
    SCOPED_TRACE(::testing::Message() << "passing accumulator " << passing_accumulator);
    auto seeded_neighbors = raft::make_device_matrix<std::int64_t, std::int64_t>(res, kQueries, k);
    auto seeded_distances = raft::make_device_matrix<float, std::int64_t>(res, kQueries, k);
    detail::benchmark_search_bitmap_seeded<float>(res,
                                                  search_params,
                                                  *index,
                                                  raft::make_const_mdspan(queries->view()),
                                                  seeded_neighbors.view(),
                                                  seeded_distances.view(),
                                                  bitmap_filter,
                                                  0,
                                                  navix_seeds.data(),
                                                  navix_counts.data(),
                                                  navix_inspected.data(),
                                                  passing_accumulator);
    std::vector<std::int64_t> host_seeded(seeded_neighbors.size());
    std::vector<float> host_seeded_distances(seeded_distances.size());
    std::vector<std::uint32_t> host_default_seeds(navix_seeds.size());
    raft::copy(host_seeded.data(), seeded_neighbors.data_handle(), seeded_neighbors.size(), stream);
    raft::copy(host_seeded_distances.data(),
               seeded_distances.data_handle(),
               seeded_distances.size(),
               stream);
    raft::copy(host_default_seeds.data(), navix_seeds.data(), navix_seeds.size(), stream);
    raft::resource::sync_stream(res);
    EXPECT_EQ(
      std::vector<std::uint32_t>(host_default_seeds.begin(), host_default_seeds.begin() + k),
      expected_internal_seeds);
    for (std::int64_t query_id = 0; query_id < kQueries; ++query_id) {
      std::vector<std::int64_t> unique;
      float previous_distance = -std::numeric_limits<float>::infinity();
      bool saw_sentinel       = false;
      for (std::int64_t rank = 0; rank < k; ++rank) {
        const auto pos    = static_cast<std::size_t>(query_id * k + rank);
        const auto source = host_seeded[pos];
        if (source == std::numeric_limits<std::int64_t>::max()) {
          saw_sentinel = true;
          EXPECT_EQ(host_seeded_distances[pos], std::numeric_limits<float>::max());
          continue;
        }
        EXPECT_FALSE(saw_sentinel);
        EXPECT_TRUE(predicate(query_id, source));
        EXPECT_TRUE(std::isfinite(host_seeded_distances[pos]));
        EXPECT_GE(host_seeded_distances[pos], previous_distance);
        EXPECT_EQ(std::find(unique.begin(), unique.end(), source), unique.end());
        unique.push_back(source);
        previous_distance = host_seeded_distances[pos];
      }
    }
  }
}

TEST_P(CagraBitmapFilterTest, NavixSeedPrepassIsStableAndHonorsQueryOffset)
{
  constexpr std::uint32_t seed_k = 10;
  auto predicate                 = [](std::int64_t query_id, std::int64_t source_id) {
    if (query_id == 0) { return false; }
    if (query_id == 1) {
      return source_id == 0 || source_id == 2 || source_id == 31 || source_id == 32 ||
             source_id == 33 || source_id == 768;
    }
    if (query_id == 2) { return source_id % 2 == 0; }
    return source_id == query_id;
  };
  device_bitmap bitmap(res, kQueries, kRows, predicate);
  auto filter = bitmap.filter();
  auto stream = raft::resource::get_cuda_stream(res);
  rmm::device_uvector<std::uint32_t> seeds(3 * seed_k, stream);
  rmm::device_uvector<std::uint32_t> counts(3, stream);
  rmm::device_uvector<std::uint32_t> inspected(3, stream);
  detail::select_navix_bitmap_seeds(res,
                                    filter,
                                    static_cast<const std::uint32_t*>(nullptr),
                                    1,
                                    3,
                                    static_cast<std::uint32_t>(kRows),
                                    seed_k,
                                    seeds.data(),
                                    counts.data(),
                                    inspected.data());

  std::vector<std::uint32_t> host_seeds(seeds.size());
  std::vector<std::uint32_t> host_counts(counts.size());
  raft::copy(host_seeds.data(), seeds.data(), seeds.size(), stream);
  raft::copy(host_counts.data(), counts.data(), counts.size(), stream);
  raft::resource::sync_stream(res);

  EXPECT_EQ(host_counts, (std::vector<std::uint32_t>{6, 10, 1}));
  EXPECT_EQ(std::vector<std::uint32_t>(host_seeds.begin(), host_seeds.begin() + 6),
            (std::vector<std::uint32_t>{0, 2, 31, 32, 33, 768}));
  EXPECT_EQ(
    std::vector<std::uint32_t>(host_seeds.begin() + seed_k, host_seeds.begin() + 2 * seed_k),
    (std::vector<std::uint32_t>{0, 2, 4, 6, 8, 10, 12, 14, 16, 18}));
  EXPECT_EQ(host_seeds[2 * seed_k], 3u);
}

TEST_P(CagraBitmapFilterTest, BitmapSeededNavixReturnsOnlyPassingNodesAndEmptySentinels)
{
  auto predicate = [](std::int64_t query_id, std::int64_t source_id) {
    if (query_id == 0) { return false; }
    if (query_id == 1) {
      return source_id == 5 || source_id == 105 || source_id == 205 || source_id == 305;
    }
    return (source_id + 3 * query_id) % 4 == 0;
  };
  device_bitmap bitmap(res, kQueries, kRows, predicate);
  auto filter    = bitmap.filter();
  auto stream    = raft::resource::get_cuda_stream(res);
  auto neighbors = raft::make_device_matrix<std::int64_t, std::int64_t>(res, kQueries, k);
  auto distances = raft::make_device_matrix<float, std::int64_t>(res, kQueries, k);
  rmm::device_uvector<std::uint32_t> seeds(kQueries * k, stream);
  rmm::device_uvector<std::uint32_t> counts(kQueries, stream);
  rmm::device_uvector<std::uint32_t> inspected(kQueries, stream);
  auto search_params           = params();
  search_params.algo           = cagra::search_algo::SINGLE_CTA;
  search_params.filtering_rate = 0.0f;

  detail::benchmark_search_navix_bitmap_seeded<float>(res,
                                                      search_params,
                                                      *index,
                                                      raft::make_const_mdspan(queries->view()),
                                                      neighbors.view(),
                                                      distances.view(),
                                                      filter,
                                                      0,
                                                      seeds.data(),
                                                      counts.data(),
                                                      inspected.data(),
                                                      0);

  std::vector<std::int64_t> host_neighbors(neighbors.size());
  std::vector<float> host_distances(distances.size());
  raft::copy(host_neighbors.data(), neighbors.data_handle(), neighbors.size(), stream);
  raft::copy(host_distances.data(), distances.data_handle(), distances.size(), stream);
  raft::resource::sync_stream(res);
  for (std::int64_t query_id = 0; query_id < kQueries; ++query_id) {
    bool saw_sentinel = false;
    for (std::int64_t rank = 0; rank < k; ++rank) {
      const auto pos = static_cast<std::size_t>(query_id * k + rank);
      if (host_neighbors[pos] == std::numeric_limits<std::int64_t>::max()) {
        saw_sentinel = true;
        EXPECT_EQ(host_distances[pos], std::numeric_limits<float>::max());
      } else {
        EXPECT_FALSE(saw_sentinel);
        ASSERT_GE(host_neighbors[pos], 0);
        ASSERT_LT(host_neighbors[pos], kRows);
        EXPECT_TRUE(predicate(query_id, host_neighbors[pos]));
        EXPECT_TRUE(std::isfinite(host_distances[pos]));
      }
    }
  }
  for (std::int64_t rank = 0; rank < k; ++rank) {
    EXPECT_EQ(host_neighbors[rank], std::numeric_limits<std::int64_t>::max());
    EXPECT_EQ(host_distances[rank], std::numeric_limits<float>::max());
  }
}

TEST_P(CagraBitmapFilterTest,
       BitmapSeededDefaultIsDeterministicForOffsetEmptyUnderfilledAndDuplicatePaths)
{
  constexpr std::uint32_t query_offset = 2;
  auto predicate                       = [](std::int64_t query_id, std::int64_t source_id) {
    if (query_id == query_offset) { return false; }
    if (query_id == query_offset + 1) {
      return source_id == 0 || source_id == 2 || source_id == 31 || source_id == 768;
    }
    return ((source_id * 13 + query_id * 7) % 11) < 3;
  };
  device_bitmap bitmap(res, kQueries + query_offset, kRows, predicate, true);
  auto filter        = bitmap.filter();
  auto search_params = params(k, 0.0f);
  search_params.algo = cagra::search_algo::SINGLE_CTA;
  auto stream        = raft::resource::get_cuda_stream(res);
  std::vector<float> host_dataset(dataset->size());
  std::vector<float> host_queries(queries->size());
  raft::copy(host_dataset.data(), dataset->data_handle(), dataset->size(), stream);
  raft::copy(host_queries.data(), queries->data_handle(), queries->size(), stream);
  raft::resource::sync_stream(res);

  {
    auto invalid_params = search_params;
    invalid_params.algo = cagra::search_algo::MULTI_CTA;
    auto neighbors      = raft::make_device_matrix<std::int64_t, std::int64_t>(res, kQueries, k);
    auto distances      = raft::make_device_matrix<float, std::int64_t>(res, kQueries, k);
    rmm::device_uvector<std::uint32_t> seeds(kQueries * k, stream);
    rmm::device_uvector<std::uint32_t> counts(kQueries, stream);
    EXPECT_THROW(
      detail::benchmark_search_bitmap_seeded<float>(res,
                                                    invalid_params,
                                                    *index,
                                                    raft::make_const_mdspan(queries->view()),
                                                    neighbors.view(),
                                                    distances.view(),
                                                    filter,
                                                    query_offset,
                                                    seeds.data(),
                                                    counts.data(),
                                                    nullptr,
                                                    false),
      std::exception);

    invalid_params            = search_params;
    invalid_params.persistent = true;
    EXPECT_THROW(
      detail::benchmark_search_bitmap_seeded<float>(res,
                                                    invalid_params,
                                                    *index,
                                                    raft::make_const_mdspan(queries->view()),
                                                    neighbors.view(),
                                                    distances.view(),
                                                    filter,
                                                    query_offset,
                                                    seeds.data(),
                                                    counts.data(),
                                                    nullptr,
                                                    false),
      std::exception);

    invalid_params             = search_params;
    invalid_params.filter_mode = cagra::filtering_mode::FAVOR;
    EXPECT_THROW(
      detail::benchmark_search_bitmap_seeded<float>(res,
                                                    invalid_params,
                                                    *index,
                                                    raft::make_const_mdspan(queries->view()),
                                                    neighbors.view(),
                                                    distances.view(),
                                                    filter,
                                                    query_offset,
                                                    seeds.data(),
                                                    counts.data(),
                                                    nullptr,
                                                    false),
      std::exception);

    invalid_params = search_params;
    EXPECT_THROW(
      detail::benchmark_search_bitmap_seeded<float>(res,
                                                    invalid_params,
                                                    *index,
                                                    raft::make_const_mdspan(queries->view()),
                                                    neighbors.view(),
                                                    distances.view(),
                                                    filter,
                                                    query_offset,
                                                    nullptr,
                                                    counts.data(),
                                                    nullptr,
                                                    false),
      std::exception);
  }

  struct seeded_run {
    bitmap_search_result<std::int64_t> output;
    std::vector<std::uint32_t> seeds;
    std::vector<std::uint32_t> counts;
  };
  auto run = [&](bool passing_accumulator) {
    auto neighbors = raft::make_device_matrix<std::int64_t, std::int64_t>(res, kQueries, k);
    auto distances = raft::make_device_matrix<float, std::int64_t>(res, kQueries, k);
    rmm::device_uvector<std::uint32_t> seeds(kQueries * k, stream);
    rmm::device_uvector<std::uint32_t> counts(kQueries, stream);
    rmm::device_uvector<std::uint32_t> inspected(kQueries, stream);
    detail::benchmark_search_bitmap_seeded<float>(res,
                                                  search_params,
                                                  *index,
                                                  raft::make_const_mdspan(queries->view()),
                                                  neighbors.view(),
                                                  distances.view(),
                                                  filter,
                                                  query_offset,
                                                  seeds.data(),
                                                  counts.data(),
                                                  inspected.data(),
                                                  passing_accumulator);
    seeded_run result{
      {std::vector<std::int64_t>(neighbors.size()), std::vector<float>(distances.size())},
      std::vector<std::uint32_t>(seeds.size()),
      std::vector<std::uint32_t>(counts.size())};
    raft::copy(result.output.neighbors.data(), neighbors.data_handle(), neighbors.size(), stream);
    raft::copy(result.output.distances.data(), distances.data_handle(), distances.size(), stream);
    raft::copy(result.seeds.data(), seeds.data(), seeds.size(), stream);
    raft::copy(result.counts.data(), counts.data(), counts.size(), stream);
    raft::resource::sync_stream(res);
    return result;
  };

  for (const bool passing_accumulator : {false, true}) {
    SCOPED_TRACE(::testing::Message() << "passing accumulator " << passing_accumulator);
    const auto first  = run(passing_accumulator);
    const auto second = run(passing_accumulator);
    expect_same(first.output, second.output);
    EXPECT_EQ(first.counts, second.counts);
    for (std::int64_t query_id = 0; query_id < kQueries; ++query_id) {
      const auto row_begin = static_cast<std::size_t>(query_id * k);
      const auto count     = static_cast<std::size_t>(first.counts[query_id]);
      EXPECT_EQ(std::vector<std::uint32_t>(first.seeds.begin() + row_begin,
                                           first.seeds.begin() + row_begin + count),
                std::vector<std::uint32_t>(second.seeds.begin() + row_begin,
                                           second.seeds.begin() + row_begin + count));
    }
    ASSERT_EQ(first.counts.front(), 0u);
    ASSERT_EQ(first.counts[1], 4u);
    EXPECT_EQ(std::vector<std::uint32_t>(first.seeds.begin() + k, first.seeds.begin() + k + 4),
              (std::vector<std::uint32_t>{0, 2, 31, 768}));

    for (std::int64_t query_id = 0; query_id < kQueries; ++query_id) {
      std::vector<std::int64_t> unique;
      float previous_distance = -std::numeric_limits<float>::infinity();
      bool saw_sentinel       = false;
      for (std::int64_t rank = 0; rank < k; ++rank) {
        const auto pos       = static_cast<std::size_t>(query_id * k + rank);
        const auto source_id = first.output.neighbors[pos];
        if (source_id == std::numeric_limits<std::int64_t>::max()) {
          saw_sentinel = true;
          EXPECT_EQ(first.output.distances[pos], std::numeric_limits<float>::max());
          continue;
        }
        EXPECT_FALSE(saw_sentinel);
        ASSERT_GE(source_id, 0);
        ASSERT_LT(source_id, kRows);
        EXPECT_TRUE(predicate(query_id + query_offset, source_id));
        EXPECT_TRUE(std::isfinite(first.output.distances[pos]));
        EXPECT_GE(first.output.distances[pos], previous_distance);
        // Graph paths can rediscover a seed or child many times; visited-state suppression must
        // prevent duplicates in both the fused output and the passing accumulator.
        EXPECT_EQ(std::find(unique.begin(), unique.end(), source_id), unique.end());
        float exact_distance = 0.0f;
        for (std::int64_t dim = 0; dim < kDim; ++dim) {
          const auto delta = host_queries[static_cast<std::size_t>(query_id * kDim + dim)] -
                             host_dataset[static_cast<std::size_t>(source_id * kDim + dim)];
          exact_distance += delta * delta;
        }
        EXPECT_NEAR(
          first.output.distances[pos], exact_distance, 1.0e-2f * std::max(1.0f, exact_distance));
        unique.push_back(source_id);
        previous_distance = first.output.distances[pos];
      }
    }

    for (std::int64_t rank = 0; rank < k; ++rank) {
      EXPECT_EQ(first.output.neighbors[rank], std::numeric_limits<std::int64_t>::max());
      EXPECT_EQ(first.output.distances[rank], std::numeric_limits<float>::max());
    }
    if (passing_accumulator) {
      std::vector<std::int64_t> underfilled;
      for (std::int64_t rank = 0; rank < k; ++rank) {
        const auto source = first.output.neighbors[static_cast<std::size_t>(k + rank)];
        if (source != std::numeric_limits<std::int64_t>::max()) { underfilled.push_back(source); }
      }
      std::sort(underfilled.begin(), underfilled.end());
      EXPECT_EQ(underfilled, (std::vector<std::int64_t>{0, 2, 31, 768}));
    }
  }
}

TEST_P(CagraBitmapFilterTest, BitmapSeededControlDoesNotLeakIntoUnseededLauncher)
{
  if (GetParam() != cagra::search_algo::SINGLE_CTA) { GTEST_SKIP(); }
  device_bitmap bitmap(res, kQueries, kRows, [](auto query_id, auto source_id) {
    return ((source_id * 5 + query_id * 3) % 7) < 3;
  });
  auto filter        = bitmap.filter();
  auto search_params = params(k, 0.0f);
  search_params.algo = cagra::search_algo::SINGLE_CTA;
  auto stream        = raft::resource::get_cuda_stream(res);

  auto run_unseeded = [&](bool passing_accumulator) {
    auto neighbors = raft::make_device_matrix<std::int64_t, std::int64_t>(res, kQueries, k);
    auto distances = raft::make_device_matrix<float, std::int64_t>(res, kQueries, k);
    detail::benchmark_search_bitmap_with_query_offset<float>(
      res,
      search_params,
      *index,
      raft::make_const_mdspan(queries->view()),
      neighbors.view(),
      distances.view(),
      filter,
      0,
      passing_accumulator);
    bitmap_search_result<std::int64_t> result{std::vector<std::int64_t>(neighbors.size()),
                                              std::vector<float>(distances.size())};
    raft::copy(result.neighbors.data(), neighbors.data_handle(), neighbors.size(), stream);
    raft::copy(result.distances.data(), distances.data_handle(), distances.size(), stream);
    raft::resource::sync_stream(res);
    return result;
  };

  for (const bool passing_accumulator : {false, true}) {
    const auto before = run_unseeded(passing_accumulator);
    auto neighbors    = raft::make_device_matrix<std::int64_t, std::int64_t>(res, kQueries, k);
    auto distances    = raft::make_device_matrix<float, std::int64_t>(res, kQueries, k);
    rmm::device_uvector<std::uint32_t> seeds(kQueries * k, stream);
    rmm::device_uvector<std::uint32_t> counts(kQueries, stream);
    detail::benchmark_search_bitmap_seeded<float>(res,
                                                  search_params,
                                                  *index,
                                                  raft::make_const_mdspan(queries->view()),
                                                  neighbors.view(),
                                                  distances.view(),
                                                  filter,
                                                  0,
                                                  seeds.data(),
                                                  counts.data(),
                                                  nullptr,
                                                  passing_accumulator);
    raft::resource::sync_stream(res);
    const auto after = run_unseeded(passing_accumulator);
    expect_same(before, after);
  }
}

TEST_P(CagraBitmapFilterTest, RejectsShapeAndUnsupportedModes)
{
  device_bitmap valid_bitmap(res, kQueries, kRows, [](auto, auto) { return true; });
  auto valid_filter = valid_bitmap.filter();

  device_bitmap wrong_rows(res, kQueries - 1, kRows, [](auto, auto) { return true; });
  auto wrong_rows_filter = wrong_rows.filter();
  EXPECT_THROW(search_dynamic(res, wrong_rows_filter, params(k, 0.0f)), std::exception);

  device_bitmap wrong_cols(res, kQueries, kRows - 1, [](auto, auto) { return true; });
  auto wrong_cols_filter = wrong_cols.filter();
  EXPECT_THROW(search_dynamic(res, wrong_cols_filter, params(k, 0.0f)), std::exception);

  auto multi_kernel_params = params(k, 0.0f);
  multi_kernel_params.algo = cagra::search_algo::MULTI_KERNEL;
  EXPECT_THROW(search_dynamic(res, valid_filter, multi_kernel_params), std::exception);

  auto persistent_params       = params(k, 0.0f);
  persistent_params.algo       = cagra::search_algo::SINGLE_CTA;
  persistent_params.persistent = true;
  EXPECT_THROW(search_dynamic(res, valid_filter, persistent_params), std::exception);

  auto favor_params        = params(k, 0.0f);
  favor_params.filter_mode = cagra::filtering_mode::FAVOR;
  EXPECT_THROW(search_dynamic(res, valid_filter, favor_params), std::exception);

  for (auto invalid_rate : {1.0f,
                            std::numeric_limits<float>::infinity(),
                            -std::numeric_limits<float>::infinity(),
                            std::numeric_limits<float>::quiet_NaN()}) {
    auto invalid_rate_params = params(k, invalid_rate);
    EXPECT_THROW(search_dynamic(res, valid_filter, invalid_rate_params), std::exception);
  }
}

TEST(CagraBitmapRecallAccounting, TreatsOutputsAsASetAndRejectsSentinels)
{
  const std::int64_t signed_outputs[] = {7, 7, -1, 768, 769};
  EXPECT_TRUE(cuvs::bench::detail::is_first_output_occurrence(signed_outputs, 0));
  EXPECT_FALSE(cuvs::bench::detail::is_first_output_occurrence(signed_outputs, 1));
  EXPECT_TRUE(cuvs::bench::detail::is_first_output_occurrence(signed_outputs, 2));
  EXPECT_TRUE(cuvs::bench::detail::is_valid_source_id(signed_outputs[0], 769));
  EXPECT_FALSE(cuvs::bench::detail::is_valid_source_id(signed_outputs[2], 769));
  EXPECT_TRUE(cuvs::bench::detail::is_valid_source_id(signed_outputs[3], 769));
  EXPECT_FALSE(cuvs::bench::detail::is_valid_source_id(signed_outputs[4], 769));

  const std::uint32_t unsigned_outputs[] = {
    3, 3, std::numeric_limits<std::uint32_t>::max()};
  EXPECT_FALSE(cuvs::bench::detail::is_first_output_occurrence(unsigned_outputs, 1));
  EXPECT_FALSE(cuvs::bench::detail::is_valid_source_id(unsigned_outputs[2], 769));
}

INSTANTIATE_TEST_CASE_P(CagraBitmapFilters,
                        CagraBitmapFilterTest,
                        ::testing::Values(cagra::search_algo::SINGLE_CTA,
                                          cagra::search_algo::MULTI_CTA,
                                          cagra::search_algo::AUTO));

}  // namespace
}  // namespace cuvs::neighbors::cagra
