/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/core/bitset.hpp>
#include <cuvs/neighbors/cagra.hpp>

#include "../../../src/neighbors/cagra_benchmark.hpp"
#include "../../../src/neighbors/detail/cagra/favor_multi_seed_benchmark.cuh"
#include "../../../src/neighbors/detail/cagra/favor_penalty.cuh"
#include "../../../src/neighbors/detail/cagra/favor_search_diagnostics.hpp"
#include "../../../src/neighbors/detail/cagra/jit_lto_kernels/cagra_filter_payload.cuh"
#include "../../../src/neighbors/detail/cagra/jit_lto_kernels/device_common_jit.cuh"
#include "../../../src/neighbors/detail/cagra/jit_lto_kernels/search_single_cta_device_helpers.cuh"

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

template <typename IndexT>
struct typed_search_result {
  std::vector<IndexT> neighbors;
  std::vector<float> distances;
};

using search_result = typed_search_result<uint32_t>;

struct favor_policy_result {
  std::uint32_t finite_count;
  float penalty;
  float cutoff;
  float reference_penalty;
};

template <bool Swizzled>
__global__ void evaluate_favor_policy_kernel(const float* distances,
                                             std::uint32_t retained_size,
                                             float filtering_rate,
                                             float delta_d,
                                             favor_policy_result* result)
{
  if (threadIdx.x != 0) { return; }
  const auto finite_count =
    detail::device::favor_sorted_finite_count<Swizzled>(distances, retained_size);
  result->finite_count = finite_count;
  result->penalty =
    detail::device::favor_query_local_penalty<Swizzled>(distances, finite_count, 1000.0f, 1.0f);
  result->cutoff = detail::device::favor_retention_cutoff<Swizzled>(distances, finite_count);
  result->reference_penalty =
    detail::device::favor_reference_penalty_device(filtering_rate, retained_size, delta_d);
}

template <bool Swizzled>
__global__ void compact_invalid_kernel(std::uint32_t* indices,
                                       float* distances,
                                       std::uint32_t retained_size)
{
  detail::single_cta_search::compact_invalid_to_end_of_list<Swizzled>(
    indices, distances, retained_size);
}

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

  template <typename OutputIndexT = std::uint32_t>
  auto run_typed(cagra::search_params params,
                 cuvs::neighbors::filtering::base_filter const& filter,
                 int64_t num_queries = kQueries,
                 int64_t result_k    = k) -> typed_search_result<OutputIndexT>
  {
    auto neighbors  = raft::make_device_matrix<OutputIndexT, int64_t>(res, num_queries, result_k);
    auto distances  = raft::make_device_matrix<float, int64_t>(res, num_queries, result_k);
    auto query_view = raft::make_device_matrix_view<const float, int64_t>(
      queries->data_handle(), num_queries, kDim);
    cagra::search(res, params, *index, query_view, neighbors.view(), distances.view(), filter);

    typed_search_result<OutputIndexT> result{std::vector<OutputIndexT>(neighbors.size()),
                                             std::vector<float>(distances.size())};
    auto stream = raft::resource::get_cuda_stream(res);
    raft::copy(result.neighbors.data(), neighbors.data_handle(), neighbors.size(), stream);
    raft::copy(result.distances.data(), distances.data_handle(), distances.size(), stream);
    raft::resource::sync_stream(res);
    return result;
  }

  auto run(cagra::search_params params,
           cuvs::neighbors::filtering::base_filter const& filter,
           int64_t num_queries = kQueries,
           int64_t result_k    = k) -> search_result
  {
    return run_typed<std::uint32_t>(params, filter, num_queries, result_k);
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

template <bool Swizzled>
void check_favor_policy_layout(std::uint32_t retained_size, std::uint32_t finite_count)
{
  raft::resources res;
  auto stream = raft::resource::get_cuda_stream(res);
  std::vector<float> host_distances(retained_size, std::numeric_limits<float>::max());
  for (std::uint32_t rank = 0; rank < finite_count; ++rank) {
    const auto position      = Swizzled ? rank ^ (rank >> 5) : rank;
    host_distances[position] = static_cast<float>(rank + 1);
  }

  rmm::device_uvector<float> distances(retained_size, stream);
  rmm::device_uvector<favor_policy_result> result(1, stream);
  raft::copy(distances.data(), host_distances.data(), retained_size, stream);
  constexpr float filtering_rate = 0.9985f;
  constexpr float delta_d        = 695.924f;
  evaluate_favor_policy_kernel<Swizzled>
    <<<1, 1, 0, stream>>>(distances.data(), retained_size, filtering_rate, delta_d, result.data());
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  favor_policy_result host_result{};
  raft::copy(&host_result, result.data(), 1, stream);
  raft::resource::sync_stream(res);
  EXPECT_EQ(host_result.finite_count, finite_count);
  EXPECT_FLOAT_EQ(host_result.penalty, finite_count < 2 ? 0.0f : 1.0f);
  EXPECT_EQ(
    host_result.cutoff,
    finite_count == 0 ? std::numeric_limits<float>::max() : static_cast<float>(finite_count));
  EXPECT_NEAR(host_result.reference_penalty,
              detail::favor_reference_penalty(filtering_rate, retained_size, delta_d),
              1e-3f);
}

TEST(CagraFavorPolicyTest, UsesLogicalRanksAcrossSwizzleBoundaries)
{
  for (auto retained_size : {64u, 128u, 256u, 512u}) {
    for (auto finite_count : {0u, 1u, 31u, 32u, 33u, retained_size - 1, retained_size}) {
      check_favor_policy_layout<false>(retained_size, finite_count);
      check_favor_policy_layout<true>(retained_size, finite_count);
    }
  }
}

template <bool Swizzled>
void check_logical_compaction(std::uint32_t retained_size)
{
  raft::resources res;
  auto stream            = raft::resource::get_cuda_stream(res);
  constexpr auto invalid = std::numeric_limits<std::uint32_t>::max();
  constexpr auto msb     = std::uint32_t{1} << 31;
  const std::vector<std::uint32_t> invalid_ranks{
    0u, 31u, 32u, 33u, 63u, 64u, 127u, 255u, 256u, 511u};
  std::vector<std::uint32_t> host_indices(retained_size, invalid);
  std::vector<float> host_distances(retained_size, std::numeric_limits<float>::max());
  std::vector<std::uint32_t> expected_indices;
  std::vector<float> expected_distances;
  for (std::uint32_t rank = 0; rank < retained_size; ++rank) {
    const bool is_invalid =
      std::find(invalid_ranks.begin(), invalid_ranks.end(), rank) != invalid_ranks.end();
    auto index = is_invalid ? (rank % 2 == 0 ? invalid : invalid & ~msb) : rank + 100;
    if (!is_invalid && rank % 3 == 0) { index |= msb; }
    const auto distance =
      is_invalid ? std::numeric_limits<float>::max() : static_cast<float>(rank) + 0.25f;
    const auto position      = Swizzled ? rank ^ (rank >> 5) : rank;
    host_indices[position]   = index;
    host_distances[position] = distance;
    if (!is_invalid) {
      expected_indices.push_back(index);
      expected_distances.push_back(distance);
    }
  }

  rmm::device_uvector<std::uint32_t> indices(retained_size, stream);
  rmm::device_uvector<float> distances(retained_size, stream);
  raft::copy(indices.data(), host_indices.data(), retained_size, stream);
  raft::copy(distances.data(), host_distances.data(), retained_size, stream);
  compact_invalid_kernel<Swizzled>
    <<<1, 32, 0, stream>>>(indices.data(), distances.data(), retained_size);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
  raft::copy(host_indices.data(), indices.data(), retained_size, stream);
  raft::copy(host_distances.data(), distances.data(), retained_size, stream);
  raft::resource::sync_stream(res);

  for (std::uint32_t rank = 0; rank < retained_size; ++rank) {
    const auto position = Swizzled ? rank ^ (rank >> 5) : rank;
    if (rank < expected_indices.size()) {
      EXPECT_EQ(host_indices[position], expected_indices[rank]);
      EXPECT_EQ(host_distances[position], expected_distances[rank]);
    } else {
      EXPECT_EQ(host_indices[position], invalid);
      EXPECT_EQ(host_distances[position], std::numeric_limits<float>::max());
    }
  }
}

TEST(CagraFavorPolicyTest, StableCompactionPreservesLogicalOrder)
{
  for (auto retained_size : {64u, 128u, 256u, 512u}) {
    check_logical_compaction<false>(retained_size);
    check_logical_compaction<true>(retained_size);
  }
}

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
  context_host.summaries                          = summaries.data();
  context_host.termination_checkpoints            = checkpoints.data();
  context_host.termination_checkpoint_counts      = checkpoint_counts.data();
  context_host.termination_checkpoint_stride      = checkpoint_stride;
  context_host.termination_record_start_iteration = 1;
  context_host.termination_start_iteration        = 5;
  context_host.termination_parent_interval        = 3;
  context_host.num_queries                        = kQueries;
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
        EXPECT_LE(previous.cumulative_passing_candidates, record.cumulative_passing_candidates);
        EXPECT_LE(previous.cumulative_candidate_duplicates, record.cumulative_candidate_duplicates);
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

TEST_F(CagraFavorSearchTest, LargeTopKPreservesLogicalDistanceOrder)
{
  constexpr std::int64_t result_k = 64;
  auto filter                     = make_filter(kRows / 4);
  auto default_params             = params();
  default_params.itopk_size       = 128;
  auto favor_params               = default_params;
  favor_params.filter_mode        = cagra::filtering_mode::FAVOR;
  favor_params.favor_delta_d      = 100.0f;
  favor_params.favor_penalty      = cagra::favor_penalty_mode::CAGRA_RETENTION_SAFE;

  std::vector<float> host_dataset(dataset->size());
  std::vector<float> host_queries(queries->size());
  auto stream = raft::resource::get_cuda_stream(res);
  raft::copy(host_dataset.data(), dataset->data_handle(), dataset->size(), stream);
  raft::copy(host_queries.data(), queries->data_handle(), queries->size(), stream);
  raft::resource::sync_stream(res);

  const auto check_result = [&](const search_result& result) {
    for (std::int64_t query = 0; query < kQueries; ++query) {
      std::vector<std::uint32_t> seen;
      for (std::int64_t rank = 0; rank < result_k; ++rank) {
        const auto offset   = query * result_k + rank;
        const auto neighbor = result.neighbors[offset];
        ASSERT_NE(neighbor, std::numeric_limits<std::uint32_t>::max());
        ASSERT_GE(neighbor, static_cast<std::uint32_t>(kRows / 4));
        ASSERT_LT(neighbor, static_cast<std::uint32_t>(kRows));
        EXPECT_EQ(std::find(seen.begin(), seen.end(), neighbor), seen.end());
        seen.push_back(neighbor);
        if (rank != 0) { EXPECT_LE(result.distances[offset - 1], result.distances[offset]); }

        float expected_distance = 0.0f;
        for (std::int64_t dim = 0; dim < kDim; ++dim) {
          const auto difference = host_queries[query * kDim + dim] -
                                  host_dataset[static_cast<std::int64_t>(neighbor) * kDim + dim];
          expected_distance += difference * difference;
        }
        EXPECT_NEAR(result.distances[offset], expected_distance, 2.0e-3f);
      }
    }
  };

  check_result(run(default_params, filter, kQueries, result_k));
  check_result(run(favor_params, filter, kQueries, result_k));
}

template <typename IndexT>
void expect_underfilled_source_results(const typed_search_result<IndexT>& result)
{
  constexpr auto invalid = std::numeric_limits<IndexT>::max();
  bool saw_invalid       = false;
  for (std::size_t i = 0; i < result.neighbors.size(); ++i) {
    const auto neighbor = result.neighbors[i];
    if (neighbor == invalid) {
      saw_invalid = true;
      EXPECT_EQ(result.distances[i], std::numeric_limits<float>::max());
    } else {
      EXPECT_GE(neighbor, static_cast<IndexT>(kRows - 3));
      EXPECT_LT(neighbor, static_cast<IndexT>(kRows));
      EXPECT_LT(result.distances[i], std::numeric_limits<float>::max());
    }
  }
  EXPECT_TRUE(saw_invalid);
}

TEST_F(CagraFavorSearchTest, UnderfilledOutputsUseTypedSentinelsWithSourceMapping)
{
  std::vector<std::uint32_t> source_indices_host(kRows);
  for (std::int64_t row = 0; row < kRows; ++row) {
    source_indices_host[row] = static_cast<std::uint32_t>(kRows - 1 - row);
  }
  auto source_indices = raft::make_device_vector<std::uint32_t, std::int64_t>(res, kRows);
  auto stream         = raft::resource::get_cuda_stream(res);
  raft::copy(source_indices.data_handle(), source_indices_host.data(), kRows, stream);
  index->update_source_indices(res, raft::make_const_mdspan(source_indices.view()));
  raft::resource::sync_stream(res);

  auto filter = make_filter(kRows - 3);
  std::vector<cagra::search_params> search_configs;
  auto single_params = params();
  search_configs.push_back(single_params);
  auto multi_cta = multi_cta_params();
  search_configs.push_back(multi_cta);
  auto multi_kernel         = params();
  multi_kernel.algo         = cagra::search_algo::MULTI_KERNEL;
  multi_kernel.max_queries  = 1;
  multi_kernel.search_width = 1;
  search_configs.push_back(multi_kernel);

  for (const auto& search_config : search_configs) {
    expect_underfilled_source_results(run_typed<std::uint32_t>(search_config, filter, 1));
    expect_underfilled_source_results(run_typed<std::int64_t>(search_config, filter, 1));
  }

  for (auto favor_config : {single_params, multi_cta}) {
    favor_config.filter_mode              = cagra::filtering_mode::FAVOR;
    favor_config.filtering_rate           = static_cast<float>(kRows - 3) / kRows;
    favor_config.favor_delta_d            = 100.0f;
    favor_config.favor_penalty            = cagra::favor_penalty_mode::CAGRA_RETENTION_SAFE;
    favor_config.favor_retention_fraction = 0.5f;
    expect_underfilled_source_results(run_typed<std::uint32_t>(favor_config, filter, 1));
    expect_underfilled_source_results(run_typed<std::int64_t>(favor_config, filter, 1));
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

  auto automatic_retention_auto_algo = automatic_retention_fraction;
  automatic_retention_auto_algo.algo = cagra::search_algo::AUTO;
  EXPECT_NO_THROW(run(automatic_retention_auto_algo, filter));

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
  EXPECT_NO_THROW(run(multi_cta_automatic_retention, filter, 1));
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

TEST_F(CagraFavorSearchTest, MultiCtaAutomaticRetentionAtMidpointMatchesExplicitDefault)
{
  auto filter                               = make_filter(kRows / 2);
  auto automatic_params                     = multi_cta_params();
  automatic_params.filter_mode              = cagra::filtering_mode::FAVOR;
  automatic_params.filtering_rate           = 0.5f;
  automatic_params.favor_delta_d            = 100.0f;
  automatic_params.favor_penalty            = cagra::favor_penalty_mode::CAGRA_RETENTION_SAFE;
  automatic_params.favor_retention_fraction = 0.0f;
  auto explicit_default_params              = automatic_params;
  explicit_default_params.favor_retention_fraction = 0.5f;

  // With configured itopk=64, k=8, and 50% selectivity, the automatic formula resolves to 0.5.
  auto expected = run(explicit_default_params, filter, 1);
  auto actual   = run(automatic_params, filter, 1);
  EXPECT_EQ(expected.neighbors, actual.neighbors);
  EXPECT_EQ(expected.distances, actual.distances);
  for (auto neighbor : actual.neighbors) {
    ASSERT_NE(neighbor, std::numeric_limits<uint32_t>::max());
    EXPECT_GE(neighbor, static_cast<uint32_t>(kRows / 2));
  }
}

TEST_F(CagraFavorSearchTest, MultiCtaAutomaticRetentionExercisesSparsePolicy)
{
  constexpr int64_t passing_count = 12;
  auto filter                     = make_filter(kRows - passing_count);
  auto favor_params               = multi_cta_params();
  favor_params.filter_mode        = cagra::filtering_mode::FAVOR;
  favor_params.filtering_rate =
    static_cast<float>(kRows - passing_count) / static_cast<float>(kRows);
  favor_params.favor_delta_d            = 100.0f;
  favor_params.favor_penalty            = cagra::favor_penalty_mode::CAGRA_RETENTION_SAFE;
  favor_params.favor_retention_fraction = 0.0f;

  ASSERT_GT(detail::favor_automatic_retention_fraction(
              favor_params.filtering_rate, favor_params.itopk_size, k),
            0.5f);
  auto result = run(favor_params, filter, 1);
  for (auto neighbor : result.neighbors) {
    ASSERT_NE(neighbor, std::numeric_limits<uint32_t>::max());
    EXPECT_GE(neighbor, static_cast<uint32_t>(kRows - passing_count));
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

TEST_F(CagraFavorUint8SearchTest, MultiCtaAutomaticRetentionReturnsOnlyPassingRows)
{
  auto filter                           = make_filter(kRows / 2);
  auto favor_params                     = multi_cta_params();
  favor_params.filter_mode              = cagra::filtering_mode::FAVOR;
  favor_params.filtering_rate           = 0.5f;
  favor_params.favor_delta_d            = 100.0f;
  favor_params.favor_penalty            = cagra::favor_penalty_mode::CAGRA_RETENTION_SAFE;
  favor_params.favor_retention_fraction = 0.0f;

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
