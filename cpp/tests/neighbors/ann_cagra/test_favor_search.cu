/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/core/bitset.hpp>
#include <cuvs/neighbors/cagra.hpp>

#include "../../../src/neighbors/cagra_benchmark.hpp"
#include "../../../src/neighbors/detail/cagra/favor_multi_seed_benchmark.cuh"
#include "../../../src/neighbors/detail/cagra/favor_search_diagnostics.hpp"

#include <raft/core/copy.cuh>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/random/rng.cuh>

#include <rmm/device_uvector.hpp>

#include <gtest/gtest.h>

#include <thrust/device_ptr.h>
#include <thrust/sequence.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
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
           int64_t num_queries = kQueries) -> search_result
  {
    auto neighbors  = raft::make_device_matrix<uint32_t, int64_t>(res, num_queries, k);
    auto distances  = raft::make_device_matrix<float, int64_t>(res, num_queries, k);
    auto query_view = raft::make_device_matrix_view<const float, int64_t>(
      queries->data_handle(), num_queries, kDim);
    cagra::search(res, params, *index, query_view, neighbors.view(), distances.view(), filter);

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
        host_queries[row * kDim + dim] = static_cast<uint8_t>((row * 29 + dim * 7 + 3) & 0xff);
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
           int64_t num_queries = kQueries) -> search_result
  {
    auto neighbors  = raft::make_device_matrix<uint32_t, int64_t>(res, num_queries, k);
    auto distances  = raft::make_device_matrix<float, int64_t>(res, num_queries, k);
    auto query_view = raft::make_device_matrix_view<const uint8_t, int64_t>(
      queries->data_handle(), num_queries, kDim);
    cagra::search(res, params, *index, query_view, neighbors.view(), distances.view(), filter);

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

TEST_F(CagraFavorSearchTest, FilteredSmallHashIsRejected)
{
  auto filter                           = make_filter(kRows / 2);
  auto favor_params                     = params();
  favor_params.filter_mode              = cagra::filtering_mode::FAVOR;
  favor_params.favor_delta_d            = 100.0f;
  favor_params.hashmap_mode             = cagra::hash_mode::SMALL;
  favor_params.favor_penalty            = cagra::favor_penalty_mode::CAGRA_RETENTION_SAFE;
  favor_params.filtering_rate           = 0.5f;
  favor_params.favor_retention_fraction = 0.0f;

  EXPECT_ANY_THROW(run(favor_params, filter));
}

TEST_F(CagraFavorSearchTest, TerminationShadowCapturesOrderedPassingTrajectory)
{
  auto filter                           = make_filter(kRows / 2);
  auto favor_params                     = params();
  favor_params.filter_mode              = cagra::filtering_mode::FAVOR;
  favor_params.favor_delta_d            = 100.0f;
  favor_params.favor_penalty            = cagra::favor_penalty_mode::CAGRA_RETENTION_SAFE;
  favor_params.filtering_rate           = 0.5f;
  favor_params.favor_retention_fraction = 0.0f;
  favor_params.max_iterations           = 8;

  constexpr std::uint32_t checkpoint_stride = 8;
  auto stream                               = raft::resource::get_cuda_stream(res);
  rmm::device_uvector<cagra::detail::favor_search_diagnostics::query_summary> summaries(kQueries,
                                                                                        stream);
  rmm::device_uvector<cagra::detail::favor_search_diagnostics::termination_checkpoint_record>
    checkpoints(kQueries * checkpoint_stride, stream);
  rmm::device_uvector<std::uint32_t> checkpoint_counts(kQueries, stream);
  rmm::device_uvector<cagra::detail::favor_search_diagnostics::context> context(1, stream);
  RAFT_CUDA_TRY(cudaMemsetAsync(
    checkpoint_counts.data(), 0, checkpoint_counts.size() * sizeof(std::uint32_t), stream));
  cagra::detail::favor_search_diagnostics::context context_host{};
  context_host.summaries                     = summaries.data();
  context_host.termination_checkpoints       = checkpoints.data();
  context_host.termination_checkpoint_counts = checkpoint_counts.data();
  context_host.termination_checkpoint_stride = checkpoint_stride;
  context_host.termination_record_start_iteration = 1;
  context_host.termination_start_iteration        = 5;
  context_host.termination_parent_interval        = 3;
  context_host.num_queries                   = kQueries;
  raft::copy(context.data(), &context_host, 1, stream);

  search_result result;
  {
    cagra::detail::favor_search_diagnostics::scoped_context diagnostic_scope{context.data()};
    result = run(favor_params, filter);
  }
  std::vector<cagra::detail::favor_search_diagnostics::termination_checkpoint_record>
    checkpoints_host(checkpoints.size());
  std::vector<std::uint32_t> counts_host(kQueries);
  raft::copy(checkpoints_host.data(), checkpoints.data(), checkpoints_host.size(), stream);
  raft::copy(counts_host.data(), checkpoint_counts.data(), counts_host.size(), stream);
  raft::resource::sync_stream(res);

  constexpr std::uint32_t expected_iterations[] = {1, 4, 5, 8};
  for (std::uint32_t query = 0; query < kQueries; ++query) {
    ASSERT_EQ(counts_host[query], 4);
    for (std::uint32_t slot = 0; slot < counts_host[query]; ++slot) {
      auto const& record = checkpoints_host[query * checkpoint_stride + slot];
      EXPECT_EQ(record.query_id, query);
      EXPECT_EQ(record.checkpoint, slot);
      EXPECT_EQ(record.iteration, expected_iterations[slot]);
      EXPECT_EQ(record.prefix_valid, 32);
      EXPECT_LE(record.prefix_pass, record.prefix_valid);
      EXPECT_EQ(record.output_count, std::min(record.passing_count, std::uint32_t{10}));
      if (slot != 0) {
        auto const& previous = checkpoints_host[query * checkpoint_stride + slot - 1];
        EXPECT_LE(previous.cumulative_candidate_evaluations,
                  record.cumulative_candidate_evaluations);
        EXPECT_LE(previous.cumulative_passing_candidates,
                  record.cumulative_passing_candidates);
        EXPECT_LE(previous.cumulative_candidate_duplicates,
                  record.cumulative_candidate_duplicates);
      }
      EXPECT_LT(record.frontier_best, std::numeric_limits<float>::max());
      for (std::uint32_t rank = 0; rank < record.output_count; ++rank) {
        EXPECT_GE(record.top_ids[rank], static_cast<std::uint32_t>(kRows / 2));
        if (rank != 0) { EXPECT_LE(record.top_distances[rank - 1], record.top_distances[rank]); }
      }
    }
    EXPECT_EQ(checkpoints_host[query * checkpoint_stride + counts_host[query] - 1].iteration, 8);
  }
}

TEST_F(CagraFavorSearchTest, CagraLocalPenaltyModesReturnOnlyPassingRows)
{
  for (auto mode : {cagra::favor_penalty_mode::CAGRA_QUERY_LOCAL,
                    cagra::favor_penalty_mode::CAGRA_RETENTION_SAFE}) {
    auto filter                       = make_filter(kRows / 2);
    auto favor_params                 = params();
    favor_params.filter_mode          = cagra::filtering_mode::FAVOR;
    favor_params.favor_delta_d        = 100.0f;
    favor_params.favor_penalty        = mode;
    favor_params.favor_penalty_lambda = 1.0f;

    auto result = run(favor_params, filter);
    for (auto neighbor : result.neighbors) {
      ASSERT_NE(neighbor, std::numeric_limits<uint32_t>::max());
      EXPECT_GE(neighbor, static_cast<uint32_t>(kRows / 2));
    }
  }
}

TEST_F(CagraFavorSearchTest, RetentionSafeAcceptAllMatchesDefaultSearch)
{
  auto filter                       = make_filter(0);
  auto default_params               = params();
  auto favor_params                 = default_params;
  favor_params.filter_mode          = cagra::filtering_mode::FAVOR;
  favor_params.favor_delta_d        = 100.0f;
  favor_params.favor_penalty        = cagra::favor_penalty_mode::CAGRA_RETENTION_SAFE;
  favor_params.favor_penalty_lambda = 1.0f;

  auto expected = run(default_params, filter);
  auto actual   = run(favor_params, filter);
  EXPECT_EQ(expected.neighbors, actual.neighbors);
  EXPECT_EQ(expected.distances, actual.distances);
}

TEST_F(CagraFavorSearchTest, ExplicitDefaultRetentionFractionPreservesResults)
{
  auto filter                              = make_filter(kRows / 2);
  auto implicit_params                     = params();
  implicit_params.filter_mode              = cagra::filtering_mode::FAVOR;
  implicit_params.favor_delta_d            = 100.0f;
  implicit_params.favor_penalty            = cagra::favor_penalty_mode::CAGRA_RETENTION_SAFE;
  implicit_params.favor_penalty_lambda     = 1.0f;
  auto explicit_params                     = implicit_params;
  explicit_params.favor_retention_fraction = 0.5f;

  auto expected = run(implicit_params, filter);
  auto actual   = run(explicit_params, filter);
  EXPECT_EQ(expected.neighbors, actual.neighbors);
  EXPECT_EQ(expected.distances, actual.distances);
}

TEST_F(CagraFavorSearchTest, BenchmarkKnownRateBridgeMatchesPublicFavorSearch)
{
  auto filter                              = make_filter(kRows / 2);
  auto favor_params                        = params();
  favor_params.filter_mode                 = cagra::filtering_mode::FAVOR;
  favor_params.favor_delta_d               = 100.0f;
  favor_params.favor_penalty               = cagra::favor_penalty_mode::CAGRA_RETENTION_SAFE;
  favor_params.favor_retention_fraction    = 0.0f;
  constexpr std::uint64_t expected_matches = kRows / 2;

  auto expected = run(favor_params, filter);
  auto matches  = cagra::detail::benchmark_count_favor_bitset_matches<float>(res, *index, filter);
  ASSERT_EQ(matches, expected_matches);
  favor_params.filtering_rate = static_cast<float>(kRows - matches) / static_cast<float>(kRows);

  auto neighbors = raft::make_device_matrix<std::int64_t, std::int64_t>(res, kQueries, k);
  auto distances = raft::make_device_matrix<float, std::int64_t>(res, kQueries, k);
  cagra::detail::benchmark_search_favor_with_known_filtering_rate<float>(
    res,
    favor_params,
    *index,
    raft::make_const_mdspan(queries->view()),
    neighbors.view(),
    distances.view(),
    filter);

  std::vector<std::int64_t> actual_neighbors(neighbors.size());
  std::vector<float> actual_distances(distances.size());
  auto stream = raft::resource::get_cuda_stream(res);
  raft::copy(actual_neighbors.data(), neighbors.data_handle(), neighbors.size(), stream);
  raft::copy(actual_distances.data(), distances.data_handle(), distances.size(), stream);
  raft::resource::sync_stream(res);

  ASSERT_EQ(actual_neighbors.size(), expected.neighbors.size());
  for (std::size_t i = 0; i < actual_neighbors.size(); ++i) {
    EXPECT_EQ(actual_neighbors[i], static_cast<std::int64_t>(expected.neighbors[i]));
  }
  EXPECT_EQ(actual_distances, expected.distances);
}

TEST_F(CagraFavorSearchTest, BenchmarkExplicitSeedsMatchNativeRandomInitializationAcrossChunks)
{
  auto filter                           = make_filter(0);
  auto favor_params                     = params();
  favor_params.filter_mode              = cagra::filtering_mode::FAVOR;
  favor_params.favor_delta_d            = 100.0f;
  favor_params.favor_penalty            = cagra::favor_penalty_mode::CAGRA_RETENTION_SAFE;
  favor_params.favor_retention_fraction = 0.0f;
  constexpr std::uint32_t initial_width = 64 + 32;
  const auto xorshift64                 = [](std::uint64_t value) {
    value ^= value >> 12;
    value ^= value << 25;
    value ^= value >> 27;
    return value * 0x2545f4914f6cdd1dULL;
  };

  auto expected = run(favor_params, filter);
  std::vector<std::uint32_t> seeds_host(kQueries * initial_width);
  for (std::int64_t query = 0; query < kQueries; ++query) {
    for (std::uint32_t position = 0; position < initial_width; ++position) {
      seeds_host[query * initial_width + position] =
        static_cast<std::uint32_t>(xorshift64(position ^ favor_params.rand_xor_mask) % kRows);
    }
  }
  auto stream = raft::resource::get_cuda_stream(res);
  rmm::device_uvector<std::uint32_t> seeds(seeds_host.size(), stream);
  raft::copy(seeds.data(), seeds_host.data(), seeds_host.size(), stream);

  cagra::detail::favor_search_diagnostics::scoped_seeds seed_scope{seeds.data(), initial_width};
  auto actual = run(favor_params, filter);
  EXPECT_EQ(actual.neighbors, expected.neighbors);
  EXPECT_EQ(actual.distances, expected.distances);
}

TEST_F(CagraFavorSearchTest, TerminalSnapshotPreservesTaggedPreCompactionFrontier)
{
  auto filter                               = make_filter(kRows / 2);
  auto favor_params                         = params();
  favor_params.filter_mode                  = cagra::filtering_mode::FAVOR;
  favor_params.filtering_rate               = 0.5f;
  favor_params.favor_delta_d                = 100.0f;
  favor_params.favor_penalty                = cagra::favor_penalty_mode::CAGRA_RETENTION_SAFE;
  favor_params.favor_retention_fraction     = 0.0f;
  constexpr std::uint32_t terminal_stride   = 64;
  constexpr std::uint64_t terminal_elements = kQueries * terminal_stride;
  auto stream                               = raft::resource::get_cuda_stream(res);
  rmm::device_uvector<cagra::detail::favor_search_diagnostics::query_summary> summaries(kQueries,
                                                                                        stream);
  rmm::device_uvector<std::uint32_t> tagged_ids(terminal_elements, stream);
  rmm::device_uvector<float> terminal_distances(terminal_elements, stream);
  rmm::device_uvector<std::uint8_t> terminal_flags(terminal_elements, stream);
  rmm::device_uvector<cagra::detail::favor_search_diagnostics::context> context(1, stream);
  cagra::detail::favor_search_diagnostics::context context_host{};
  context_host.summaries           = summaries.data();
  context_host.terminal_tagged_ids = tagged_ids.data();
  context_host.terminal_distances  = terminal_distances.data();
  context_host.terminal_flags      = terminal_flags.data();
  context_host.terminal_stride     = terminal_stride;
  context_host.num_queries         = kQueries;
  raft::copy(context.data(), &context_host, 1, stream);

  auto expected = run(favor_params, filter);
  cagra::detail::favor_search_diagnostics::scoped_context diagnostic_scope{context.data()};
  auto actual = run(favor_params, filter);
  EXPECT_EQ(actual.neighbors, expected.neighbors);
  EXPECT_EQ(actual.distances, expected.distances);

  std::vector<std::uint32_t> tagged_ids_host(terminal_elements);
  std::vector<std::uint8_t> terminal_flags_host(terminal_elements);
  raft::copy(tagged_ids_host.data(), tagged_ids.data(), tagged_ids_host.size(), stream);
  raft::copy(terminal_flags_host.data(), terminal_flags.data(), terminal_flags_host.size(), stream);
  raft::resource::sync_stream(res);
  for (std::int64_t query = 0; query < kQueries; ++query) {
    std::uint32_t valid = 0;
    for (std::uint32_t rank = 0; rank < terminal_stride; ++rank) {
      const auto offset = query * terminal_stride + rank;
      const auto flags  = terminal_flags_host[offset];
      if ((flags & 1u) == 0) { continue; }
      ++valid;
      const auto node = tagged_ids_host[offset] & 0x7fffffffu;
      ASSERT_LT(node, kRows);
      EXPECT_EQ((flags & 4u) != 0, node >= static_cast<std::uint32_t>(kRows / 2));
    }
    EXPECT_GT(valid, 0);
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

  auto invalid_lambda                 = favor_params;
  invalid_lambda.favor_penalty        = cagra::favor_penalty_mode::CAGRA_QUERY_LOCAL;
  invalid_lambda.favor_penalty_lambda = 0.0f;
  EXPECT_ANY_THROW(run(invalid_lambda, filter));

  auto automatic_retention_fraction          = favor_params;
  automatic_retention_fraction.favor_penalty = cagra::favor_penalty_mode::CAGRA_RETENTION_SAFE;
  automatic_retention_fraction.favor_retention_fraction = 0.0f;
  EXPECT_NO_THROW(run(automatic_retention_fraction, filter));

  auto automatic_wrong_mode          = automatic_retention_fraction;
  automatic_wrong_mode.favor_penalty = cagra::favor_penalty_mode::CAGRA_QUERY_LOCAL;
  EXPECT_ANY_THROW(run(automatic_wrong_mode, filter));

  auto negative_retention_fraction                     = automatic_retention_fraction;
  negative_retention_fraction.favor_retention_fraction = -0.1f;
  EXPECT_ANY_THROW(run(negative_retention_fraction, filter));

  auto one_retention_fraction                     = automatic_retention_fraction;
  one_retention_fraction.favor_retention_fraction = 1.0f;
  EXPECT_ANY_THROW(run(one_retention_fraction, filter));

  auto multi_cta_retention_fraction          = favor_params;
  multi_cta_retention_fraction.algo          = cagra::search_algo::MULTI_CTA;
  multi_cta_retention_fraction.favor_penalty = cagra::favor_penalty_mode::CAGRA_RETENTION_SAFE;
  multi_cta_retention_fraction.favor_retention_fraction = 0.75f;
  EXPECT_ANY_THROW(run(multi_cta_retention_fraction, filter, 1));

  auto multi_cta_automatic_retention = automatic_retention_fraction;
  multi_cta_automatic_retention.algo = cagra::search_algo::MULTI_CTA;
  EXPECT_ANY_THROW(run(multi_cta_automatic_retention, filter, 1));
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
  auto filter         = make_filter(0);
  auto default_params = multi_cta_params();
  // Use one CTA so two independent approximate searches have deterministic traversal order.
  // Multi-CTA concurrency and filtering are covered separately below.
  default_params.itopk_size  = 32;
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

TEST_F(CagraFavorSearchTest, MultiCtaLocalPenaltyModesReturnOnlyPassingRows)
{
  for (auto mode : {cagra::favor_penalty_mode::CAGRA_QUERY_LOCAL,
                    cagra::favor_penalty_mode::CAGRA_RETENTION_SAFE}) {
    auto filter                       = make_filter(kRows / 2);
    auto favor_params                 = multi_cta_params();
    favor_params.filter_mode          = cagra::filtering_mode::FAVOR;
    favor_params.favor_delta_d        = 100.0f;
    favor_params.filtering_rate       = 0.5f;
    favor_params.favor_penalty        = mode;
    favor_params.favor_penalty_lambda = 1.0f;

    auto result = run(favor_params, filter, 1);
    for (auto neighbor : result.neighbors) {
      ASSERT_NE(neighbor, std::numeric_limits<uint32_t>::max());
      EXPECT_GE(neighbor, static_cast<uint32_t>(kRows / 2));
    }
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
  auto filter         = make_filter(0);
  auto default_params = multi_cta_params();
  // Use one CTA so two independent approximate searches have deterministic traversal order.
  default_params.itopk_size  = 32;
  auto favor_params          = default_params;
  favor_params.filter_mode   = cagra::filtering_mode::FAVOR;
  favor_params.favor_delta_d = 100.0f;

  auto expected = run(default_params, filter, 1);
  auto actual   = run(favor_params, filter, 1);
  EXPECT_EQ(expected.neighbors, actual.neighbors);
  EXPECT_EQ(expected.distances, actual.distances);
}

TEST_F(CagraFavorUint8SearchTest, MultiCtaRetentionSafeReturnsOnlyPassingRows)
{
  auto filter                       = make_filter(kRows / 2);
  auto favor_params                 = multi_cta_params();
  favor_params.filter_mode          = cagra::filtering_mode::FAVOR;
  favor_params.filtering_rate       = 0.5f;
  favor_params.favor_delta_d        = 100.0f;
  favor_params.favor_penalty        = cagra::favor_penalty_mode::CAGRA_RETENTION_SAFE;
  favor_params.favor_penalty_lambda = 1.0f;

  auto result = run(favor_params, filter, 1);
  for (auto neighbor : result.neighbors) {
    ASSERT_NE(neighbor, std::numeric_limits<uint32_t>::max());
    EXPECT_GE(neighbor, static_cast<uint32_t>(kRows / 2));
  }
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

TEST(CagraFavorMultiSeedMergeTest, DeduplicatesSortsAndFillsSentinels)
{
  raft::resources res;
  constexpr std::int64_t n_queries = 2;
  constexpr std::int64_t result_k  = 4;
  constexpr std::int64_t n_rounds  = 3;
  constexpr auto invalid_id        = std::numeric_limits<std::int64_t>::max();
  constexpr auto invalid_distance  = std::numeric_limits<float>::max();

  // Round-major [round, query, result]. Query 0 has duplicates across all rounds; query 1
  // underfills after deduplication and must retain sentinels in the tail.
  std::vector<std::int64_t> host_round_neighbors{
    5, 2, invalid_id, 8,          11, 11,         invalid_id, invalid_id,
    2, 3, 7,          invalid_id, 12, invalid_id, 11,         invalid_id,
    1, 5, 9,          4,          12, invalid_id, invalid_id, invalid_id};
  std::vector<float> host_round_distances{0.5f,
                                          0.2f,
                                          invalid_distance,
                                          0.8f,
                                          1.1f,
                                          1.1f,
                                          invalid_distance,
                                          invalid_distance,
                                          0.2f,
                                          0.3f,
                                          0.7f,
                                          invalid_distance,
                                          1.2f,
                                          invalid_distance,
                                          1.1f,
                                          invalid_distance,
                                          0.1f,
                                          0.5f,
                                          0.9f,
                                          0.4f,
                                          1.2f,
                                          invalid_distance,
                                          invalid_distance,
                                          invalid_distance};

  auto round_neighbors =
    raft::make_device_vector<std::int64_t, std::int64_t>(res, host_round_neighbors.size());
  auto round_distances =
    raft::make_device_vector<float, std::int64_t>(res, host_round_distances.size());
  auto neighbors = raft::make_device_vector<std::int64_t, std::int64_t>(res, n_queries * result_k);
  auto distances = raft::make_device_vector<float, std::int64_t>(res, n_queries * result_k);
  auto stream    = raft::resource::get_cuda_stream(res);
  raft::copy(round_neighbors.data_handle(),
             host_round_neighbors.data(),
             host_round_neighbors.size(),
             stream);
  raft::copy(round_distances.data_handle(),
             host_round_distances.data(),
             host_round_distances.size(),
             stream);

  cagra::detail::merge_favor_multi_seed_results(res,
                                                round_neighbors.data_handle(),
                                                round_distances.data_handle(),
                                                neighbors.data_handle(),
                                                distances.data_handle(),
                                                n_queries,
                                                result_k,
                                                n_rounds);

  std::vector<std::int64_t> host_neighbors(neighbors.size());
  std::vector<float> host_distances(distances.size());
  raft::copy(host_neighbors.data(), neighbors.data_handle(), neighbors.size(), stream);
  raft::copy(host_distances.data(), distances.data_handle(), distances.size(), stream);
  raft::resource::sync_stream(res);

  EXPECT_EQ(host_neighbors,
            (std::vector<std::int64_t>{1, 2, 3, 4, 11, 12, invalid_id, invalid_id}));
  EXPECT_EQ(
    host_distances,
    (std::vector<float>{0.1f, 0.2f, 0.3f, 0.4f, 1.1f, 1.2f, invalid_distance, invalid_distance}));
}

TEST(CagraFavorMultiSeedMergeTest, OneRoundIsIdentity)
{
  raft::resources res;
  std::vector<std::int64_t> host_neighbors{3, 4, 5, 6};
  std::vector<float> host_distances{0.3f, 0.4f, 0.5f, 0.6f};
  auto input_neighbors =
    raft::make_device_vector<std::int64_t, std::int64_t>(res, host_neighbors.size());
  auto input_distances = raft::make_device_vector<float, std::int64_t>(res, host_distances.size());
  auto output_neighbors =
    raft::make_device_vector<std::int64_t, std::int64_t>(res, host_neighbors.size());
  auto output_distances = raft::make_device_vector<float, std::int64_t>(res, host_distances.size());
  auto stream           = raft::resource::get_cuda_stream(res);
  raft::copy(input_neighbors.data_handle(), host_neighbors.data(), host_neighbors.size(), stream);
  raft::copy(input_distances.data_handle(), host_distances.data(), host_distances.size(), stream);

  cagra::detail::merge_favor_multi_seed_results(res,
                                                input_neighbors.data_handle(),
                                                input_distances.data_handle(),
                                                output_neighbors.data_handle(),
                                                output_distances.data_handle(),
                                                1,
                                                host_neighbors.size(),
                                                1);

  std::vector<std::int64_t> actual_neighbors(host_neighbors.size());
  std::vector<float> actual_distances(host_distances.size());
  raft::copy(
    actual_neighbors.data(), output_neighbors.data_handle(), host_neighbors.size(), stream);
  raft::copy(
    actual_distances.data(), output_distances.data_handle(), host_distances.size(), stream);
  raft::resource::sync_stream(res);
  EXPECT_EQ(actual_neighbors, host_neighbors);
  EXPECT_EQ(actual_distances, host_distances);
}

}  // namespace
}  // namespace cuvs::neighbors::cagra
