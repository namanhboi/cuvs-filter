/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <gtest/gtest.h>

#include "../../../src/neighbors/cagra_benchmark.hpp"
#include "../ann_cagra.cuh"

#include <cuvs/core/bitset.hpp>
#include <cuvs/neighbors/cagra.hpp>

#include <raft/core/copy.cuh>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/random/rng.cuh>

#include <thrust/device_ptr.h>
#include <thrust/sequence.h>

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace cuvs::neighbors::cagra {
namespace {

constexpr int64_t n_rows                   = 768;
constexpr int64_t n_dim                    = 16;
constexpr int64_t n_queries                = 6;
constexpr int64_t k                        = 8;
constexpr int64_t threshold                = 192;
constexpr int64_t high_filtering_threshold = 704;
constexpr float high_filtering_rate =
  static_cast<float>(high_filtering_threshold) / static_cast<float>(n_rows);
constexpr std::uint32_t navix_adaptive_kuzu_policy   = 0;
constexpr std::uint32_t navix_one_hop_policy         = 1;
constexpr std::uint32_t navix_directed_capped_policy = 2;
constexpr std::uint32_t navix_blind_capped_policy    = 3;
constexpr std::uint32_t navix_adaptive_paper_policy  = 4;
constexpr std::uint32_t navix_serial_scheduler_mask  = std::uint32_t{1} << 8;

struct tenant_filter_context {
  const uint32_t* row_tenants;
  const uint32_t* query_tenants;
};

struct cagra_search_result {
  std::vector<uint32_t> neighbors;
  std::vector<float> distances;
};

std::string accept_all_udf_source()
{
  return R"cpp(
    __device__ bool cuvs_filter_udf(uint32_t, source_index_t, void*) { return true; }
  )cpp";
}

std::string threshold_udf_source()
{
  return R"cpp(
    __device__ bool cuvs_filter_udf(uint32_t, source_index_t source_id, void*)
    {
      return source_id >= 192;
    }
  )cpp";
}

std::string reject_all_udf_source()
{
  return R"cpp(
    __device__ bool cuvs_filter_udf(uint32_t, source_index_t, void*) { return false; }
  )cpp";
}

std::string high_filtering_rate_udf_source()
{
  return R"cpp(
    __device__ bool cuvs_filter_udf(uint32_t, source_index_t source_id, void*)
    {
      return source_id >= 704;
    }
  )cpp";
}

std::string tenant_udf_source()
{
  return R"cpp(
    struct tenant_filter_context {
      const uint32_t* row_tenants;
      const uint32_t* query_tenants;
    };

    __device__ bool cuvs_filter_udf(uint32_t query_id, source_index_t source_id, void* filter_data)
    {
      auto* ctx = static_cast<const tenant_filter_context*>(filter_data);
      return ctx->row_tenants[source_id] == ctx->query_tenants[query_id];
    }
  )cpp";
}

void expect_same_results(cagra_search_result const& expected, cagra_search_result const& actual)
{
  ASSERT_EQ(expected.neighbors, actual.neighbors);
  ASSERT_EQ(expected.distances.size(), actual.distances.size());
  for (size_t i = 0; i < expected.distances.size(); ++i) {
    EXPECT_FLOAT_EQ(expected.distances[i], actual.distances[i]);
  }
}

class CagraUdfFilterTest : public ::testing::TestWithParam<cagra::search_algo> {
 protected:
  virtual std::uint32_t graph_degree() const { return 32; }

  void SetUp() override
  {
    dataset.emplace(raft::make_device_matrix<float, int64_t>(res, n_rows, n_dim));
    queries.emplace(raft::make_device_matrix<float, int64_t>(res, n_queries, n_dim));

    raft::random::RngState rng(1234ULL);
    raft::random::uniform(res, rng, dataset->data_handle(), dataset->size(), -1.0f, 1.0f);
    raft::random::uniform(res, rng, queries->data_handle(), queries->size(), -1.0f, 1.0f);

    cagra::index_params index_params;
    index_params.metric                    = cuvs::distance::DistanceType::L2Expanded;
    index_params.graph_degree              = graph_degree();
    index_params.intermediate_graph_degree = 2 * graph_degree();
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

  cagra_search_result search(cuvs::neighbors::filtering::base_filter const& filter,
                             float filtering_rate = -1.0f,
                             bool favor           = false,
                             int64_t result_k     = k)
  {
    auto neighbors = raft::make_device_matrix<uint32_t, int64_t>(res, n_queries, result_k);
    auto distances = raft::make_device_matrix<float, int64_t>(res, n_queries, result_k);

    cagra::search_params search_params;
    search_params.algo              = GetParam();
    search_params.itopk_size        = std::max<int64_t>(64, result_k);
    search_params.max_queries       = 2;
    search_params.thread_block_size = 256;
    search_params.filtering_rate    = filtering_rate;
    if (favor) {
      search_params.filter_mode              = cagra::filtering_mode::FAVOR;
      search_params.favor_penalty            = cagra::favor_penalty_mode::CAGRA_RETENTION_SAFE;
      search_params.favor_retention_fraction = 0.0f;
      search_params.favor_delta_d            = 1.0f;
    }

    cagra::search(res,
                  search_params,
                  *index,
                  raft::make_const_mdspan(queries->view()),
                  neighbors.view(),
                  distances.view(),
                  filter);

    auto stream = raft::resource::get_cuda_stream(res);
    cagra_search_result result{std::vector<uint32_t>(n_queries * result_k),
                               std::vector<float>(n_queries * result_k)};
    raft::copy(result.neighbors.data(), neighbors.data_handle(), result.neighbors.size(), stream);
    raft::copy(result.distances.data(), distances.data_handle(), result.distances.size(), stream);
    raft::resource::sync_stream(res);
    return result;
  }

  cagra_search_result search_default_with_accumulator(
    cuvs::neighbors::filtering::base_filter const& filter,
    bool passing_accumulator,
    const float* sampled_rates = nullptr,
    cagra::search_algo algo    = cagra::search_algo::SINGLE_CTA,
    int64_t result_k           = k)
  {
    auto neighbors = raft::make_device_matrix<std::int64_t, int64_t>(res, n_queries, result_k);
    auto distances = raft::make_device_matrix<float, int64_t>(res, n_queries, result_k);

    cagra::search_params search_params;
    search_params.algo              = algo;
    search_params.filter_mode       = cagra::filtering_mode::DEFAULT;
    search_params.itopk_size        = std::max<int64_t>(64, result_k);
    search_params.max_queries       = 2;
    search_params.thread_block_size = 256;

    cagra::detail::benchmark_search_favor_udf_with_sampled_rates<float>(
      res,
      search_params,
      *index,
      raft::make_const_mdspan(queries->view()),
      neighbors.view(),
      distances.view(),
      filter,
      sampled_rates,
      passing_accumulator);

    auto stream = raft::resource::get_cuda_stream(res);
    std::vector<std::int64_t> host_neighbors(n_queries * result_k);
    cagra_search_result result{std::vector<uint32_t>(n_queries * result_k),
                               std::vector<float>(n_queries * result_k)};
    raft::copy(host_neighbors.data(), neighbors.data_handle(), host_neighbors.size(), stream);
    raft::copy(result.distances.data(), distances.data_handle(), result.distances.size(), stream);
    raft::resource::sync_stream(res);
    std::transform(host_neighbors.begin(),
                   host_neighbors.end(),
                   result.neighbors.begin(),
                   [](std::int64_t source_id) {
                     return source_id == std::numeric_limits<std::int64_t>::max()
                              ? std::numeric_limits<std::uint32_t>::max()
                              : static_cast<std::uint32_t>(source_id);
                   });
    return result;
  }

  cagra_search_result search_navix(cuvs::neighbors::filtering::base_filter const& filter,
                                   std::uint32_t navix_policy,
                                   std::uint32_t max_iterations = 0,
                                   int64_t result_k             = k)
  {
    auto neighbors = raft::make_device_matrix<std::int64_t, int64_t>(res, n_queries, result_k);
    auto distances = raft::make_device_matrix<float, int64_t>(res, n_queries, result_k);

    cagra::search_params search_params;
    search_params.algo              = cagra::search_algo::SINGLE_CTA;
    search_params.filter_mode       = cagra::filtering_mode::DEFAULT;
    search_params.itopk_size        = std::max<int64_t>(64, result_k);
    search_params.search_width      = 1;
    search_params.max_iterations    = max_iterations;
    search_params.max_queries       = 2;
    search_params.thread_block_size = 128;

    cagra::detail::benchmark_search_navix_udf<float>(res,
                                                     search_params,
                                                     *index,
                                                     raft::make_const_mdspan(queries->view()),
                                                     neighbors.view(),
                                                     distances.view(),
                                                     filter,
                                                     navix_policy);

    auto stream = raft::resource::get_cuda_stream(res);
    std::vector<std::int64_t> host_neighbors(n_queries * result_k);
    cagra_search_result result{std::vector<uint32_t>(n_queries * result_k),
                               std::vector<float>(n_queries * result_k)};
    raft::copy(host_neighbors.data(), neighbors.data_handle(), host_neighbors.size(), stream);
    raft::copy(result.distances.data(), distances.data_handle(), result.distances.size(), stream);
    raft::resource::sync_stream(res);
    std::transform(host_neighbors.begin(),
                   host_neighbors.end(),
                   result.neighbors.begin(),
                   [](std::int64_t source_id) {
                     return source_id == std::numeric_limits<std::int64_t>::max()
                              ? std::numeric_limits<std::uint32_t>::max()
                              : static_cast<std::uint32_t>(source_id);
                   });
    return result;
  }

  raft::resources res;
  std::optional<raft::device_matrix<float, int64_t>> dataset = std::nullopt;
  std::optional<raft::device_matrix<float, int64_t>> queries = std::nullopt;
  std::optional<cagra::index<float, uint32_t>> index         = std::nullopt;
};

class CagraUdfFilterDegree64Test : public CagraUdfFilterTest {
 protected:
  std::uint32_t graph_degree() const override { return 64; }
};

class CagraUdfFilterHalfTest : public ::testing::TestWithParam<cagra::search_algo> {
 protected:
  void SetUp() override
  {
    dataset.emplace(raft::make_device_matrix<half, int64_t>(res, n_rows, n_dim));
    queries.emplace(raft::make_device_matrix<half, int64_t>(res, n_queries, n_dim));

    raft::random::RngState rng(1234ULL);
    InitDataset(res,
                dataset->data_handle(),
                static_cast<std::uint32_t>(n_rows),
                static_cast<std::uint32_t>(n_dim),
                cuvs::distance::DistanceType::L2Expanded,
                rng);
    InitDataset(res,
                queries->data_handle(),
                static_cast<std::uint32_t>(n_queries),
                static_cast<std::uint32_t>(n_dim),
                cuvs::distance::DistanceType::L2Expanded,
                rng);

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

  cagra_search_result search(cuvs::neighbors::filtering::base_filter const& filter,
                             float filtering_rate = -1.0f)
  {
    auto neighbors = raft::make_device_matrix<uint32_t, int64_t>(res, n_queries, k);
    auto distances = raft::make_device_matrix<float, int64_t>(res, n_queries, k);

    cagra::search_params search_params;
    search_params.algo              = GetParam();
    search_params.itopk_size        = 64;
    search_params.max_queries       = 2;
    search_params.thread_block_size = 256;
    search_params.filtering_rate    = filtering_rate;

    cagra::search(res,
                  search_params,
                  *index,
                  raft::make_const_mdspan(queries->view()),
                  neighbors.view(),
                  distances.view(),
                  filter);

    auto stream = raft::resource::get_cuda_stream(res);
    cagra_search_result result{std::vector<uint32_t>(n_queries * k),
                               std::vector<float>(n_queries * k)};
    raft::copy(result.neighbors.data(), neighbors.data_handle(), result.neighbors.size(), stream);
    raft::copy(result.distances.data(), distances.data_handle(), result.distances.size(), stream);
    raft::resource::sync_stream(res);
    return result;
  }

  raft::resources res;
  std::optional<raft::device_matrix<half, int64_t>> dataset = std::nullopt;
  std::optional<raft::device_matrix<half, int64_t>> queries = std::nullopt;
  std::optional<cagra::index<half, uint32_t>> index         = std::nullopt;
};

TEST_P(CagraUdfFilterTest, AcceptAllMatchesNoFilter)
{
  cuvs::neighbors::filtering::none_sample_filter no_filter;
  auto expected = search(no_filter, 0.0f);

  cuvs::neighbors::filtering::udf_filter udf_filter(accept_all_udf_source(), nullptr, 0.0f);
  auto actual = search(udf_filter);

  expect_same_results(expected, actual);
}

TEST_P(CagraUdfFilterHalfTest, ThresholdReturnsOnlyValidNeighbors)
{
  float const filtering_rate = static_cast<float>(threshold) / static_cast<float>(n_rows);
  cuvs::neighbors::filtering::udf_filter udf_filter(
    threshold_udf_source(), nullptr, filtering_rate);
  auto result = search(udf_filter, filtering_rate);

  for (auto source_id : result.neighbors) {
    if (source_id < static_cast<uint32_t>(n_rows)) {
      EXPECT_GE(source_id, static_cast<uint32_t>(threshold));
    }
  }
}

TEST_P(CagraUdfFilterTest, RejectAllReturnsNoValidNeighbors)
{
  cuvs::neighbors::filtering::udf_filter udf_filter(reject_all_udf_source(), nullptr, 0.999f);
  auto result = search(udf_filter);

  for (auto source_id : result.neighbors) {
    EXPECT_EQ(source_id, std::numeric_limits<std::uint32_t>::max());
  }
}

TEST_P(CagraUdfFilterTest, HighFilteringRateReturnsOnlyValidNeighbors)
{
  cuvs::neighbors::filtering::udf_filter udf_filter(
    high_filtering_rate_udf_source(), nullptr, high_filtering_rate);
  auto result = search(udf_filter);

  for (auto source_id : result.neighbors) {
    if (source_id < static_cast<uint32_t>(n_rows)) {
      EXPECT_GE(source_id, static_cast<uint32_t>(high_filtering_threshold));
    }
  }
}

TEST_P(CagraUdfFilterTest, DefaultPassingAccumulatorIsPassiveAndRetainsPassingCandidates)
{
  if (GetParam() != cagra::search_algo::SINGLE_CTA) { GTEST_SKIP(); }

  cuvs::neighbors::filtering::udf_filter udf_filter(
    high_filtering_rate_udf_source(), nullptr, high_filtering_rate);
  const auto public_legacy  = search(udf_filter);
  const auto private_legacy = search_default_with_accumulator(udf_filter, false);
  const auto accumulated    = search_default_with_accumulator(udf_filter, true);

  expect_same_results(public_legacy, private_legacy);
  std::size_t legacy_valid{};
  std::size_t accumulated_valid{};
  for (std::size_t pos = 0; pos < accumulated.neighbors.size(); ++pos) {
    const auto legacy_id = private_legacy.neighbors[pos];
    legacy_valid += legacy_id < static_cast<std::uint32_t>(n_rows);

    const auto accumulated_id = accumulated.neighbors[pos];
    if (accumulated_id < static_cast<std::uint32_t>(n_rows)) {
      ++accumulated_valid;
      EXPECT_GE(accumulated_id, static_cast<std::uint32_t>(high_filtering_threshold));
    } else {
      EXPECT_EQ(accumulated_id, std::numeric_limits<std::uint32_t>::max());
      EXPECT_EQ(accumulated.distances[pos], std::numeric_limits<float>::max());
    }
  }
  EXPECT_GE(accumulated_valid, legacy_valid);
  EXPECT_NE(accumulated.neighbors, private_legacy.neighbors);
}

TEST_P(CagraUdfFilterTest, DefaultPassingAccumulatorRejectsRatesAndMultiCta)
{
  if (GetParam() != cagra::search_algo::SINGLE_CTA) { GTEST_SKIP(); }

  cuvs::neighbors::filtering::udf_filter udf_filter(
    high_filtering_rate_udf_source(), nullptr, high_filtering_rate);
  auto sampled_rates = raft::make_device_vector<float, int64_t>(res, n_queries);
  EXPECT_THROW(search_default_with_accumulator(udf_filter, true, sampled_rates.data_handle()),
               std::exception);
  EXPECT_THROW(
    search_default_with_accumulator(udf_filter, true, nullptr, cagra::search_algo::MULTI_CTA),
    std::exception);
}

TEST_P(CagraUdfFilterTest, PassingAccumulatorMatchesAcceptAllAcrossTopKBoundaries)
{
  if (GetParam() != cagra::search_algo::SINGLE_CTA) { GTEST_SKIP(); }

  cuvs::neighbors::filtering::none_sample_filter no_filter;
  cuvs::neighbors::filtering::udf_filter udf_filter(accept_all_udf_source(), nullptr, 0.0f);
  for (const auto result_k :
       {int64_t{1}, int64_t{10}, int64_t{31}, int64_t{32}, int64_t{33}, int64_t{64}}) {
    const auto expected = search(no_filter, 0.0f, false, result_k);
    const auto actual   = search_default_with_accumulator(
      udf_filter, true, nullptr, cagra::search_algo::SINGLE_CTA, result_k);
    expect_same_results(expected, actual);
  }
}

TEST_P(CagraUdfFilterTest, RepeatedUdfSearchWithSameSourceMatches)
{
  cuvs::neighbors::filtering::udf_filter udf_filter(accept_all_udf_source(), nullptr, 0.0f);

  auto first  = search(udf_filter);
  auto second = search(udf_filter);

  expect_same_results(first, second);
}

TEST_P(CagraUdfFilterTest, InvalidSourceThrows)
{
  cuvs::neighbors::filtering::udf_filter udf_filter("this is not valid cuda source", nullptr, 0.0f);

  EXPECT_THROW(search(udf_filter), std::exception);
}

TEST_P(CagraUdfFilterTest, ThresholdMatchesEquivalentBitset)
{
  auto removed_indices = raft::make_device_vector<int64_t, int64_t>(res, threshold);
  thrust::sequence(raft::resource::get_thrust_policy(res),
                   thrust::device_pointer_cast(removed_indices.data_handle()),
                   thrust::device_pointer_cast(removed_indices.data_handle() + threshold));
  raft::resource::sync_stream(res);

  cuvs::core::bitset<std::uint32_t, int64_t> removed_indices_bitset(
    res, removed_indices.view(), n_rows);
  cuvs::neighbors::filtering::bitset_filter bitset_filter(removed_indices_bitset.view());

  float const filtering_rate = static_cast<float>(threshold) / static_cast<float>(n_rows);
  auto expected              = search(bitset_filter, filtering_rate);

  cuvs::neighbors::filtering::udf_filter udf_filter(
    threshold_udf_source(), nullptr, filtering_rate);
  auto actual = search(udf_filter, filtering_rate);

  expect_same_results(expected, actual);
}

TEST_P(CagraUdfFilterTest, TenantContextHonorsQuerySpecificMetadata)
{
  std::vector<uint32_t> host_row_tenants(n_rows);
  std::vector<uint32_t> host_query_tenants(n_queries);
  for (int64_t i = 0; i < n_rows; ++i) {
    host_row_tenants[static_cast<size_t>(i)] = static_cast<uint32_t>((i / 5) % 3);
  }
  for (int64_t q = 0; q < n_queries; ++q) {
    host_query_tenants[static_cast<size_t>(q)] = static_cast<uint32_t>(q % 3);
  }

  auto row_tenants   = raft::make_device_vector<uint32_t, int64_t>(res, n_rows);
  auto query_tenants = raft::make_device_vector<uint32_t, int64_t>(res, n_queries);
  auto context       = raft::make_device_vector<tenant_filter_context, int64_t>(res, 1);

  auto stream = raft::resource::get_cuda_stream(res);
  raft::copy(row_tenants.data_handle(), host_row_tenants.data(), host_row_tenants.size(), stream);
  raft::copy(
    query_tenants.data_handle(), host_query_tenants.data(), host_query_tenants.size(), stream);

  tenant_filter_context host_context{row_tenants.data_handle(), query_tenants.data_handle()};
  raft::copy(context.data_handle(), &host_context, 1, stream);
  raft::resource::sync_stream(res);

  cuvs::neighbors::filtering::udf_filter udf_filter(
    tenant_udf_source(), context.data_handle(), 2.0f / 3.0f);
  auto result = search(udf_filter);

  for (int64_t q = 0; q < n_queries; ++q) {
    auto query_tenant = host_query_tenants[static_cast<size_t>(q)];
    for (int64_t i = 0; i < k; ++i) {
      auto source_id = result.neighbors[static_cast<size_t>(q * k + i)];
      ASSERT_LT(source_id, static_cast<uint32_t>(n_rows));
      EXPECT_EQ(host_row_tenants[source_id], query_tenant);
    }
  }
}

TEST_P(CagraUdfFilterTest, FavorSamplesRateAndReturnsOnlyPassingNeighbors)
{
  if (GetParam() != cagra::search_algo::SINGLE_CTA) { GTEST_SKIP(); }

  // Both scalar hints are intentionally wrong: phase-one FAVOR UDF must derive policy only from
  // its systematic predicate sample.
  cuvs::neighbors::filtering::udf_filter udf_filter(threshold_udf_source(), nullptr, 0.0f);
  auto result = search(udf_filter, 0.0f, true);

  for (int64_t q = 0; q < n_queries; ++q) {
    float previous_distance = -std::numeric_limits<float>::infinity();
    std::uint32_t previous_id{};
    std::vector<std::uint32_t> seen;
    for (int64_t rank = 0; rank < k; ++rank) {
      const auto pos       = static_cast<std::size_t>(q * k + rank);
      const auto source_id = result.neighbors[pos];
      ASSERT_LT(source_id, static_cast<std::uint32_t>(n_rows));
      EXPECT_GE(source_id, static_cast<std::uint32_t>(threshold));
      EXPECT_GE(result.distances[pos], previous_distance);
      if (result.distances[pos] == previous_distance) { EXPECT_GT(source_id, previous_id); }
      EXPECT_EQ(std::find(seen.begin(), seen.end(), source_id), seen.end());
      seen.push_back(source_id);
      previous_distance = result.distances[pos];
      previous_id       = source_id;
    }
  }
}

TEST_P(CagraUdfFilterTest, FavorRejectAllUsesInvalidSentinels)
{
  if (GetParam() != cagra::search_algo::SINGLE_CTA) { GTEST_SKIP(); }

  cuvs::neighbors::filtering::udf_filter udf_filter(reject_all_udf_source());
  auto result = search(udf_filter, -1.0f, true);
  for (std::size_t i = 0; i < result.neighbors.size(); ++i) {
    EXPECT_EQ(result.neighbors[i], std::numeric_limits<std::uint32_t>::max());
    EXPECT_EQ(result.distances[i], std::numeric_limits<float>::max());
  }
}

TEST_P(CagraUdfFilterTest, FavorTenantContextHonorsTiledQueryIds)
{
  if (GetParam() != cagra::search_algo::SINGLE_CTA) { GTEST_SKIP(); }

  std::vector<uint32_t> host_row_tenants(n_rows);
  std::vector<uint32_t> host_query_tenants(n_queries);
  for (int64_t row = 0; row < n_rows; ++row) {
    host_row_tenants[static_cast<std::size_t>(row)] = static_cast<std::uint32_t>((row / 5) % 3);
  }
  for (int64_t query = 0; query < n_queries; ++query) {
    host_query_tenants[static_cast<std::size_t>(query)] = static_cast<std::uint32_t>(query % 3);
  }

  auto row_tenants   = raft::make_device_vector<uint32_t, int64_t>(res, n_rows);
  auto query_tenants = raft::make_device_vector<uint32_t, int64_t>(res, n_queries);
  auto context       = raft::make_device_vector<tenant_filter_context, int64_t>(res, 1);
  auto stream        = raft::resource::get_cuda_stream(res);
  raft::copy(row_tenants.data_handle(), host_row_tenants.data(), host_row_tenants.size(), stream);
  raft::copy(
    query_tenants.data_handle(), host_query_tenants.data(), host_query_tenants.size(), stream);
  tenant_filter_context host_context{row_tenants.data_handle(), query_tenants.data_handle()};
  raft::copy(context.data_handle(), &host_context, 1, stream);

  cuvs::neighbors::filtering::udf_filter udf_filter(tenant_udf_source(), context.data_handle());
  auto result = search(udf_filter, -1.0f, true);
  for (int64_t query = 0; query < n_queries; ++query) {
    for (int64_t rank = 0; rank < k; ++rank) {
      const auto source_id = result.neighbors[static_cast<std::size_t>(query * k + rank)];
      ASSERT_LT(source_id, static_cast<std::uint32_t>(n_rows));
      EXPECT_EQ(host_row_tenants[source_id], host_query_tenants[static_cast<std::size_t>(query)]);
    }
  }
}

TEST_P(CagraUdfFilterTest, NavixAcceptAllOneHopMatchesDefaultAcrossQueryBatches)
{
  if (GetParam() != cagra::search_algo::SINGLE_CTA) { GTEST_SKIP(); }

  cuvs::neighbors::filtering::udf_filter udf_filter(accept_all_udf_source(), nullptr, 0.0f);
  const auto expected = search(udf_filter, 0.0f);
  const auto actual   = search_navix(udf_filter, navix_one_hop_policy);
  expect_same_results(expected, actual);
}

TEST_P(CagraUdfFilterTest, NavixRejectAllReturnsInvalidSentinels)
{
  if (GetParam() != cagra::search_algo::SINGLE_CTA) { GTEST_SKIP(); }

  cuvs::neighbors::filtering::udf_filter udf_filter(reject_all_udf_source(), nullptr, 1.0f);
  const auto result = search_navix(udf_filter, navix_adaptive_kuzu_policy, 8);
  for (std::size_t pos = 0; pos < result.neighbors.size(); ++pos) {
    EXPECT_EQ(result.neighbors[pos], std::numeric_limits<std::uint32_t>::max());
    EXPECT_EQ(result.distances[pos], std::numeric_limits<float>::max());
  }
}

TEST_P(CagraUdfFilterTest, NavixPoliciesReturnOnlyPassingUniqueSortedRows)
{
  if (GetParam() != cagra::search_algo::SINGLE_CTA) { GTEST_SKIP(); }

  cuvs::neighbors::filtering::udf_filter udf_filter(
    threshold_udf_source(), nullptr, static_cast<float>(threshold) / static_cast<float>(n_rows));
  for (const auto policy : {navix_adaptive_kuzu_policy,
                            navix_one_hop_policy,
                            navix_directed_capped_policy,
                            navix_blind_capped_policy,
                            navix_adaptive_paper_policy}) {
    const auto result = search_navix(udf_filter, policy);
    for (int64_t query = 0; query < n_queries; ++query) {
      std::vector<std::uint32_t> seen;
      float previous_distance = -std::numeric_limits<float>::infinity();
      std::uint32_t previous_id{};
      for (int64_t rank = 0; rank < k; ++rank) {
        const auto pos       = static_cast<std::size_t>(query * k + rank);
        const auto source_id = result.neighbors[pos];
        ASSERT_LT(source_id, static_cast<std::uint32_t>(n_rows));
        EXPECT_GE(source_id, static_cast<std::uint32_t>(threshold));
        EXPECT_GE(result.distances[pos], previous_distance);
        if (result.distances[pos] == previous_distance) { EXPECT_GT(source_id, previous_id); }
        EXPECT_EQ(std::find(seen.begin(), seen.end(), source_id), seen.end());
        seen.push_back(source_id);
        previous_distance = result.distances[pos];
        previous_id       = source_id;
      }
    }
  }
}

TEST_P(CagraUdfFilterTest, NavixHonorsQueryOffsetsAndSchedulerOrdering)
{
  if (GetParam() != cagra::search_algo::SINGLE_CTA) { GTEST_SKIP(); }

  std::vector<std::uint32_t> host_row_tenants(n_rows);
  std::vector<std::uint32_t> host_query_tenants(n_queries);
  for (int64_t row = 0; row < n_rows; ++row) {
    host_row_tenants[static_cast<std::size_t>(row)] = static_cast<std::uint32_t>((row / 5) % 3);
  }
  for (int64_t query = 0; query < n_queries; ++query) {
    host_query_tenants[static_cast<std::size_t>(query)] = static_cast<std::uint32_t>(query % 3);
  }

  auto row_tenants   = raft::make_device_vector<std::uint32_t, int64_t>(res, n_rows);
  auto query_tenants = raft::make_device_vector<std::uint32_t, int64_t>(res, n_queries);
  auto context       = raft::make_device_vector<tenant_filter_context, int64_t>(res, 1);
  auto stream        = raft::resource::get_cuda_stream(res);
  raft::copy(row_tenants.data_handle(), host_row_tenants.data(), host_row_tenants.size(), stream);
  raft::copy(
    query_tenants.data_handle(), host_query_tenants.data(), host_query_tenants.size(), stream);
  tenant_filter_context host_context{row_tenants.data_handle(), query_tenants.data_handle()};
  raft::copy(context.data_handle(), &host_context, 1, stream);

  cuvs::neighbors::filtering::udf_filter udf_filter(tenant_udf_source(), context.data_handle());
  const auto tiled = search_navix(udf_filter, navix_directed_capped_policy);
  const auto serial =
    search_navix(udf_filter, navix_directed_capped_policy | navix_serial_scheduler_mask);
  expect_same_results(tiled, serial);
  for (int64_t query = 0; query < n_queries; ++query) {
    for (int64_t rank = 0; rank < k; ++rank) {
      const auto source_id = tiled.neighbors[static_cast<std::size_t>(query * k + rank)];
      ASSERT_LT(source_id, static_cast<std::uint32_t>(n_rows));
      EXPECT_EQ(host_row_tenants[source_id], host_query_tenants[static_cast<std::size_t>(query)]);
    }
  }
}

TEST_P(CagraUdfFilterDegree64Test, NavixAcceptAllOneHopMatchesDefault)
{
  cuvs::neighbors::filtering::udf_filter udf_filter(accept_all_udf_source(), nullptr, 0.0f);
  const auto expected = search(udf_filter, 0.0f);
  const auto actual   = search_navix(udf_filter, navix_one_hop_policy);
  expect_same_results(expected, actual);
}

TEST_P(CagraUdfFilterDegree64Test, NavixPoliciesReturnOnlyPassingUniqueSortedRows)
{
  cuvs::neighbors::filtering::udf_filter udf_filter(
    threshold_udf_source(), nullptr, static_cast<float>(threshold) / static_cast<float>(n_rows));
  for (const auto policy : {navix_adaptive_kuzu_policy,
                            navix_one_hop_policy,
                            navix_directed_capped_policy,
                            navix_blind_capped_policy,
                            navix_adaptive_paper_policy}) {
    const auto result = search_navix(udf_filter, policy);
    for (int64_t query = 0; query < n_queries; ++query) {
      std::vector<std::uint32_t> seen;
      float previous_distance = -std::numeric_limits<float>::infinity();
      std::uint32_t previous_id{};
      for (int64_t rank = 0; rank < k; ++rank) {
        const auto pos       = static_cast<std::size_t>(query * k + rank);
        const auto source_id = result.neighbors[pos];
        ASSERT_LT(source_id, static_cast<std::uint32_t>(n_rows));
        EXPECT_GE(source_id, static_cast<std::uint32_t>(threshold));
        EXPECT_GE(result.distances[pos], previous_distance);
        if (result.distances[pos] == previous_distance) { EXPECT_GT(source_id, previous_id); }
        EXPECT_EQ(std::find(seen.begin(), seen.end(), source_id), seen.end());
        seen.push_back(source_id);
        previous_distance = result.distances[pos];
        previous_id       = source_id;
      }
    }
  }
}

TEST_P(CagraUdfFilterDegree64Test, NavixTiledSchedulerMatchesSerialReference)
{
  cuvs::neighbors::filtering::udf_filter udf_filter(
    threshold_udf_source(), nullptr, static_cast<float>(threshold) / static_cast<float>(n_rows));
  const auto tiled = search_navix(udf_filter, navix_directed_capped_policy);
  const auto serial =
    search_navix(udf_filter, navix_directed_capped_policy | navix_serial_scheduler_mask);
  expect_same_results(tiled, serial);
}

INSTANTIATE_TEST_CASE_P(CagraUdfFilters,
                        CagraUdfFilterTest,
                        ::testing::Values(cagra::search_algo::SINGLE_CTA,
                                          cagra::search_algo::MULTI_CTA,
                                          cagra::search_algo::MULTI_KERNEL));

INSTANTIATE_TEST_CASE_P(CagraUdfFilterHalf,
                        CagraUdfFilterHalfTest,
                        ::testing::Values(cagra::search_algo::SINGLE_CTA,
                                          cagra::search_algo::MULTI_CTA,
                                          cagra::search_algo::MULTI_KERNEL));

INSTANTIATE_TEST_CASE_P(CagraUdfDegree64,
                        CagraUdfFilterDegree64Test,
                        ::testing::Values(cagra::search_algo::SINGLE_CTA));

}  // namespace
}  // namespace cuvs::neighbors::cagra
