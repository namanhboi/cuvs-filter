/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/core/bitset.hpp>
#include <cuvs/neighbors/cagra.hpp>

#include <raft/core/copy.cuh>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/random/rng.cuh>

#include <gtest/gtest.h>

#include <thrust/device_ptr.h>
#include <thrust/sequence.h>

#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

namespace cuvs::neighbors::cagra {
namespace {

constexpr int64_t kRows    = 768;
constexpr int64_t kDim     = 16;
constexpr int64_t kQueries = 8;
constexpr int64_t k        = 8;

struct search_result {
  std::vector<uint32_t> neighbors;
  std::vector<float> distances;
};

class CagraFavorSearchTest : public ::testing::Test {
 protected:
  void SetUp() override
  {
    dataset.emplace(raft::make_device_matrix<float, int64_t>(res, kRows, kDim));
    queries.emplace(raft::make_device_matrix<float, int64_t>(res, kQueries, kDim));

    raft::random::RngState rng(1234ULL);
    raft::random::uniform(res, rng, dataset->data_handle(), dataset->size(), -1.0f, 1.0f);
    raft::random::uniform(res, rng, queries->data_handle(), queries->size(), -1.0f, 1.0f);

    cagra::index_params params;
    params.metric                    = cuvs::distance::DistanceType::L2Expanded;
    params.graph_degree              = 32;
    params.intermediate_graph_degree = 64;
    params.graph_build_params =
      cagra::graph_build_params::nn_descent_params(params.intermediate_graph_degree);
    index.emplace(cagra::build(res, params, raft::make_const_mdspan(dataset->view())));
    raft::resource::sync_stream(res);
  }

  auto make_filter(int64_t removed_count)
    -> cuvs::neighbors::filtering::bitset_filter<uint32_t, int64_t>
  {
    if (removed_count == 0) {
      bitsets.emplace_back(res, kRows, true);
      return cuvs::neighbors::filtering::bitset_filter<uint32_t, int64_t>(bitsets.back().view());
    }
    auto removed = raft::make_device_vector<int64_t, int64_t>(res, removed_count);
    thrust::sequence(raft::resource::get_thrust_policy(res),
                     thrust::device_pointer_cast(removed.data_handle()),
                     thrust::device_pointer_cast(removed.data_handle() + removed_count));
    raft::resource::sync_stream(res);
    bitsets.emplace_back(res, removed.view(), kRows);
    return cuvs::neighbors::filtering::bitset_filter<uint32_t, int64_t>(bitsets.back().view());
  }

  auto run(cagra::search_params params,
           cuvs::neighbors::filtering::base_filter const& filter,
           int64_t num_queries = kQueries)
    -> search_result
  {
    auto neighbors = raft::make_device_matrix<uint32_t, int64_t>(res, num_queries, k);
    auto distances = raft::make_device_matrix<float, int64_t>(res, num_queries, k);
    auto query_view =
      raft::make_device_matrix_view<const float, int64_t>(queries->data_handle(), num_queries, kDim);
    cagra::search(res,
                  params,
                  *index,
                  query_view,
                  neighbors.view(),
                  distances.view(),
                  filter);

    search_result result{std::vector<uint32_t>(neighbors.size()),
                         std::vector<float>(distances.size())};
    auto stream = raft::resource::get_cuda_stream(res);
    raft::copy(result.neighbors.data(), neighbors.data_handle(), neighbors.size(), stream);
    raft::copy(result.distances.data(), distances.data_handle(), distances.size(), stream);
    raft::resource::sync_stream(res);
    return result;
  }

  static auto params() -> cagra::search_params
  {
    cagra::search_params result;
    result.algo              = cagra::search_algo::SINGLE_CTA;
    result.itopk_size        = 64;
    result.max_queries       = 3;
    result.thread_block_size = 256;
    return result;
  }

  static auto multi_cta_params() -> cagra::search_params
  {
    auto result         = params();
    result.algo         = cagra::search_algo::MULTI_CTA;
    result.search_width = 1;
    result.max_queries  = 1;
    return result;
  }

  raft::resources res;
  std::optional<raft::device_matrix<float, int64_t>> dataset;
  std::optional<raft::device_matrix<float, int64_t>> queries;
  std::optional<cagra::index<float, uint32_t>> index;
  std::vector<cuvs::core::bitset<uint32_t, int64_t>> bitsets;
};

class CagraFavorUint8SearchTest : public ::testing::Test {
 protected:
  void SetUp() override
  {
    dataset.emplace(raft::make_device_matrix<uint8_t, int64_t>(res, kRows, kDim));
    queries.emplace(raft::make_device_matrix<uint8_t, int64_t>(res, kQueries, kDim));

    std::vector<uint8_t> host_dataset(dataset->size());
    std::vector<uint8_t> host_queries(queries->size());
    for (int64_t row = 0; row < kRows; ++row) {
      for (int64_t dim = 0; dim < kDim; ++dim) {
        host_dataset[row * kDim + dim] =
          static_cast<uint8_t>((row * 17 + dim * 13 + (row >> 8) * 31) & 0xff);
      }
    }
    for (int64_t row = 0; row < kQueries; ++row) {
      for (int64_t dim = 0; dim < kDim; ++dim) {
        host_queries[row * kDim + dim] =
          static_cast<uint8_t>((row * 29 + dim * 7 + 3) & 0xff);
      }
    }
    auto stream = raft::resource::get_cuda_stream(res);
    raft::copy(dataset->data_handle(), host_dataset.data(), host_dataset.size(), stream);
    raft::copy(queries->data_handle(), host_queries.data(), host_queries.size(), stream);

    cagra::index_params params;
    params.metric                    = cuvs::distance::DistanceType::L2Expanded;
    params.graph_degree              = 32;
    params.intermediate_graph_degree = 64;
    params.graph_build_params =
      cagra::graph_build_params::nn_descent_params(params.intermediate_graph_degree);
    index.emplace(cagra::build(res, params, raft::make_const_mdspan(dataset->view())));
    raft::resource::sync_stream(res);
  }

  auto make_filter(int64_t removed_count)
    -> cuvs::neighbors::filtering::bitset_filter<uint32_t, int64_t>
  {
    if (removed_count == 0) {
      bitsets.emplace_back(res, kRows, true);
      return cuvs::neighbors::filtering::bitset_filter<uint32_t, int64_t>(bitsets.back().view());
    }
    auto removed = raft::make_device_vector<int64_t, int64_t>(res, removed_count);
    thrust::sequence(raft::resource::get_thrust_policy(res),
                     thrust::device_pointer_cast(removed.data_handle()),
                     thrust::device_pointer_cast(removed.data_handle() + removed_count));
    raft::resource::sync_stream(res);
    bitsets.emplace_back(res, removed.view(), kRows);
    return cuvs::neighbors::filtering::bitset_filter<uint32_t, int64_t>(bitsets.back().view());
  }

  auto run(cagra::search_params params,
           cuvs::neighbors::filtering::base_filter const& filter,
           int64_t num_queries = kQueries)
    -> search_result
  {
    auto neighbors = raft::make_device_matrix<uint32_t, int64_t>(res, num_queries, k);
    auto distances = raft::make_device_matrix<float, int64_t>(res, num_queries, k);
    auto query_view = raft::make_device_matrix_view<const uint8_t, int64_t>(
      queries->data_handle(), num_queries, kDim);
    cagra::search(res,
                  params,
                  *index,
                  query_view,
                  neighbors.view(),
                  distances.view(),
                  filter);

    search_result result{std::vector<uint32_t>(neighbors.size()),
                         std::vector<float>(distances.size())};
    auto stream = raft::resource::get_cuda_stream(res);
    raft::copy(result.neighbors.data(), neighbors.data_handle(), neighbors.size(), stream);
    raft::copy(result.distances.data(), distances.data_handle(), distances.size(), stream);
    raft::resource::sync_stream(res);
    return result;
  }

  static auto params() -> cagra::search_params
  {
    cagra::search_params result;
    result.algo              = cagra::search_algo::SINGLE_CTA;
    result.itopk_size        = 64;
    result.max_queries       = 3;
    result.thread_block_size = 256;
    return result;
  }

  static auto multi_cta_params() -> cagra::search_params
  {
    auto result         = params();
    result.algo         = cagra::search_algo::MULTI_CTA;
    result.search_width = 1;
    result.max_queries  = 1;
    return result;
  }

  raft::resources res;
  std::optional<raft::device_matrix<uint8_t, int64_t>> dataset;
  std::optional<raft::device_matrix<uint8_t, int64_t>> queries;
  std::optional<cagra::index<uint8_t, uint32_t>> index;
  std::vector<cuvs::core::bitset<uint32_t, int64_t>> bitsets;
};

TEST_F(CagraFavorSearchTest, ExplicitDefaultPreservesExistingResults)
{
  auto filter                 = make_filter(kRows / 2);
  auto implicit_params        = params();
  auto explicit_params        = implicit_params;
  explicit_params.filter_mode = cagra::filtering_mode::DEFAULT;

  auto implicit_result = run(implicit_params, filter);
  auto explicit_result = run(explicit_params, filter);
  EXPECT_EQ(implicit_result.neighbors, explicit_result.neighbors);
  EXPECT_EQ(implicit_result.distances, explicit_result.distances);
}

TEST_F(CagraFavorSearchTest, AcceptAllMatchesDefaultSearch)
{
  auto filter                = make_filter(0);
  auto default_params        = params();
  auto favor_params          = default_params;
  favor_params.filter_mode   = cagra::filtering_mode::FAVOR;
  favor_params.favor_delta_d = 100.0f;

  auto expected = run(default_params, filter);
  auto actual   = run(favor_params, filter);
  EXPECT_EQ(expected.neighbors, actual.neighbors);
  EXPECT_EQ(expected.distances, actual.distances);
}

TEST_F(CagraFavorSearchTest, ReturnsOnlyPassingRows)
{
  auto filter                = make_filter(kRows / 2);
  auto favor_params          = params();
  favor_params.filter_mode   = cagra::filtering_mode::FAVOR;
  favor_params.favor_delta_d = 100.0f;

  auto result = run(favor_params, filter);
  for (auto neighbor : result.neighbors) {
    ASSERT_NE(neighbor, std::numeric_limits<uint32_t>::max());
    EXPECT_GE(neighbor, static_cast<uint32_t>(kRows / 2));
  }
}

TEST_F(CagraFavorSearchTest, CompactsRejectedRowsFromFinalTopK)
{
  auto filter                = make_filter(kRows / 2);
  auto favor_params          = params();
  favor_params.filter_mode   = cagra::filtering_mode::FAVOR;
  favor_params.favor_delta_d = 0.0f;

  auto result = run(favor_params, filter);
  for (auto neighbor : result.neighbors) {
    ASSERT_NE(neighbor, std::numeric_limits<uint32_t>::max());
    EXPECT_GE(neighbor, static_cast<uint32_t>(kRows / 2));
  }
}

TEST_F(CagraFavorSearchTest, EmptyFilterSkipsSearchAndReturnsSentinels)
{
  auto filter                = make_filter(kRows);
  auto favor_params          = params();
  favor_params.filter_mode   = cagra::filtering_mode::FAVOR;
  favor_params.favor_delta_d = 100.0f;

  auto result = run(favor_params, filter);
  for (auto neighbor : result.neighbors) {
    EXPECT_EQ(neighbor, std::numeric_limits<uint32_t>::max());
  }
  for (auto distance : result.distances) {
    EXPECT_EQ(distance, std::numeric_limits<float>::max());
  }
}

TEST_F(CagraFavorSearchTest, RejectsMissingDeltaAndNonBitsetFilter)
{
  auto filter               = make_filter(kRows / 2);
  auto missing_delta        = params();
  missing_delta.filter_mode = cagra::filtering_mode::FAVOR;
  EXPECT_ANY_THROW(run(missing_delta, filter));

  auto favor_params          = params();
  favor_params.filter_mode   = cagra::filtering_mode::FAVOR;
  favor_params.favor_delta_d = 100.0f;
  cuvs::neighbors::filtering::none_sample_filter no_filter;
  EXPECT_ANY_THROW(run(favor_params, no_filter));
}

TEST_F(CagraFavorUint8SearchTest, ExplicitDefaultPreservesExistingResults)
{
  auto filter                 = make_filter(kRows / 2);
  auto implicit_params        = params();
  auto explicit_params        = implicit_params;
  explicit_params.filter_mode = cagra::filtering_mode::DEFAULT;

  auto implicit_result = run(implicit_params, filter);
  auto explicit_result = run(explicit_params, filter);
  EXPECT_EQ(implicit_result.neighbors, explicit_result.neighbors);
  EXPECT_EQ(implicit_result.distances, explicit_result.distances);
}

TEST_F(CagraFavorUint8SearchTest, AcceptAllMatchesDefaultSearch)
{
  auto filter                = make_filter(0);
  auto default_params        = params();
  auto favor_params          = default_params;
  favor_params.filter_mode   = cagra::filtering_mode::FAVOR;
  favor_params.favor_delta_d = 100.0f;

  auto expected = run(default_params, filter);
  auto actual   = run(favor_params, filter);
  EXPECT_EQ(expected.neighbors, actual.neighbors);
  EXPECT_EQ(expected.distances, actual.distances);
}

TEST_F(CagraFavorUint8SearchTest, ReturnsOnlyPassingRows)
{
  auto filter                = make_filter(kRows / 2);
  auto favor_params          = params();
  favor_params.filter_mode   = cagra::filtering_mode::FAVOR;
  favor_params.favor_delta_d = 100.0f;

  auto result = run(favor_params, filter);
  for (auto neighbor : result.neighbors) {
    ASSERT_NE(neighbor, std::numeric_limits<uint32_t>::max());
    EXPECT_GE(neighbor, static_cast<uint32_t>(kRows / 2));
  }
}

TEST_F(CagraFavorUint8SearchTest, ZeroPenaltyCompactsRejectedRows)
{
  auto filter                = make_filter(kRows / 2);
  auto favor_params          = params();
  favor_params.filter_mode   = cagra::filtering_mode::FAVOR;
  favor_params.favor_delta_d = 0.0f;

  auto result = run(favor_params, filter);
  for (auto neighbor : result.neighbors) {
    ASSERT_NE(neighbor, std::numeric_limits<uint32_t>::max());
    EXPECT_GE(neighbor, static_cast<uint32_t>(kRows / 2));
  }
}

TEST_F(CagraFavorSearchTest, MultiCtaBatchOneAcceptAllMatchesDefault)
{
  auto filter                = make_filter(0);
  auto default_params        = multi_cta_params();
  auto favor_params          = default_params;
  favor_params.filter_mode   = cagra::filtering_mode::FAVOR;
  favor_params.favor_delta_d = 100.0f;

  auto expected = run(default_params, filter, 1);
  auto actual   = run(favor_params, filter, 1);
  EXPECT_EQ(expected.neighbors, actual.neighbors);
  EXPECT_EQ(expected.distances, actual.distances);
}

TEST_F(CagraFavorSearchTest, MultiCtaBatchOneReturnsOnlyPassingRows)
{
  auto filter                = make_filter(kRows / 2);
  auto favor_params          = multi_cta_params();
  favor_params.filter_mode   = cagra::filtering_mode::FAVOR;
  favor_params.favor_delta_d = 100.0f;

  auto result = run(favor_params, filter, 1);
  for (auto neighbor : result.neighbors) {
    ASSERT_NE(neighbor, std::numeric_limits<uint32_t>::max());
    EXPECT_GE(neighbor, static_cast<uint32_t>(kRows / 2));
  }
}

TEST_F(CagraFavorSearchTest, MultiCtaAdjustedTraversalReturnsOnlyPassingRows)
{
  auto filter                 = make_filter(kRows / 2);
  auto favor_params           = multi_cta_params();
  favor_params.itopk_size     = 32;
  favor_params.filtering_rate = 0.5;
  favor_params.filter_mode    = cagra::filtering_mode::FAVOR;
  favor_params.favor_delta_d  = 100.0f;

  auto result = run(favor_params, filter, 1);
  for (auto neighbor : result.neighbors) {
    ASSERT_NE(neighbor, std::numeric_limits<uint32_t>::max());
    EXPECT_GE(neighbor, static_cast<uint32_t>(kRows / 2));
  }
}

TEST_F(CagraFavorSearchTest, MultiCtaBatchOneZeroPenaltyCompactsRejectedRows)
{
  auto filter                = make_filter(kRows / 2);
  auto favor_params          = multi_cta_params();
  favor_params.filter_mode   = cagra::filtering_mode::FAVOR;
  favor_params.favor_delta_d = 0.0f;

  auto result = run(favor_params, filter, 1);
  for (auto neighbor : result.neighbors) {
    ASSERT_NE(neighbor, std::numeric_limits<uint32_t>::max());
    EXPECT_GE(neighbor, static_cast<uint32_t>(kRows / 2));
  }
}

TEST_F(CagraFavorUint8SearchTest, MultiCtaBatchOneAcceptAllMatchesDefault)
{
  auto filter                = make_filter(0);
  auto default_params        = multi_cta_params();
  auto favor_params          = default_params;
  favor_params.filter_mode   = cagra::filtering_mode::FAVOR;
  favor_params.favor_delta_d = 100.0f;

  auto expected = run(default_params, filter, 1);
  auto actual   = run(favor_params, filter, 1);
  EXPECT_EQ(expected.neighbors, actual.neighbors);
  EXPECT_EQ(expected.distances, actual.distances);
}

TEST_F(CagraFavorUint8SearchTest, MultiCtaBatchOneReturnsOnlyPassingRows)
{
  auto filter                = make_filter(kRows / 2);
  auto favor_params          = multi_cta_params();
  favor_params.filter_mode   = cagra::filtering_mode::FAVOR;
  favor_params.favor_delta_d = 100.0f;

  auto result = run(favor_params, filter, 1);
  for (auto neighbor : result.neighbors) {
    ASSERT_NE(neighbor, std::numeric_limits<uint32_t>::max());
    EXPECT_GE(neighbor, static_cast<uint32_t>(kRows / 2));
  }
}

}  // namespace
}  // namespace cuvs::neighbors::cagra
