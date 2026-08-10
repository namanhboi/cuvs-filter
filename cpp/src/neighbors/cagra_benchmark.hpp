/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/core/export.hpp>
#include <cuvs/neighbors/cagra.hpp>

#include <cstdint>

namespace cuvs::neighbors::cagra::detail {

/** Internal bridge used only by the staged multi-seed FAVOR benchmark experiment. */
template <typename T>
CUVS_EXPORT std::uint64_t benchmark_count_favor_bitset_matches(
  raft::resources const& res,
  cagra::index<T, std::uint32_t> const& index,
  cuvs::neighbors::filtering::base_filter const& sample_filter);

/**
 * Run FAVOR with an already-computed filtering_rate. Unlike the public search dispatcher, this
 * does not count the bitset again.
 */
template <typename T>
CUVS_EXPORT void benchmark_search_favor_with_known_filtering_rate(
  raft::resources const& res,
  cagra::search_params const& params,
  cagra::index<T, std::uint32_t> const& index,
  raft::device_matrix_view<const T, std::int64_t, raft::row_major> queries,
  raft::device_matrix_view<std::int64_t, std::int64_t, raft::row_major> neighbors,
  raft::device_matrix_view<float, std::int64_t, raft::row_major> distances,
  cuvs::neighbors::filtering::base_filter const& sample_filter);

/** Benchmark-only UDF sampler; outputs remain private device arrays. */
template <typename T>
CUVS_EXPORT void benchmark_estimate_favor_udf_filtering_rates(
  raft::resources const& res,
  cagra::index<T, std::uint32_t> const& index,
  std::uint32_t num_queries,
  cuvs::neighbors::filtering::base_filter const& sample_filter,
  float* filtering_rates,
  std::uint32_t* passing_counts,
  std::uint32_t sample_offset = 0);

/**
 * Run the private SINGLE_CTA UDF path. FAVOR requires sampled query-local rates; default CAGRA
 * requires a null rate pointer and never estimates selectivity.
 */
template <typename T>
CUVS_EXPORT void benchmark_search_favor_udf_with_sampled_rates(
  raft::resources const& res,
  cagra::search_params const& params,
  cagra::index<T, std::uint32_t> const& index,
  raft::device_matrix_view<const T, std::int64_t, raft::row_major> queries,
  raft::device_matrix_view<std::int64_t, std::int64_t, raft::row_major> neighbors,
  raft::device_matrix_view<float, std::int64_t, raft::row_major> distances,
  cuvs::neighbors::filtering::base_filter const& sample_filter,
  const float* filtering_rates,
  bool passing_accumulator);

/** Run the private degree-32 SINGLE_CTA NaviX traversal with in-kernel seed discovery. */
template <typename T>
CUVS_EXPORT void benchmark_search_navix_udf(
  raft::resources const& res,
  cagra::search_params const& params,
  cagra::index<T, std::uint32_t> const& index,
  raft::device_matrix_view<const T, std::int64_t, raft::row_major> queries,
  raft::device_matrix_view<std::int64_t, std::int64_t, raft::row_major> neighbors,
  raft::device_matrix_view<float, std::int64_t, raft::row_major> distances,
  cuvs::neighbors::filtering::base_filter const& sample_filter,
  std::uint32_t navix_policy);

/** Run DEFAULT SINGLE_CTA with a benchmark bitmap whose first row is query_offset. */
template <typename T>
CUVS_EXPORT void benchmark_search_bitmap_with_query_offset(
  raft::resources const& res,
  cagra::search_params const& params,
  cagra::index<T, std::uint32_t> const& index,
  raft::device_matrix_view<const T, std::int64_t, raft::row_major> queries,
  raft::device_matrix_view<std::int64_t, std::int64_t, raft::row_major> neighbors,
  raft::device_matrix_view<float, std::int64_t, raft::row_major> distances,
  cuvs::neighbors::filtering::base_filter const& sample_filter,
  std::uint32_t query_offset,
  bool passing_accumulator = false);

/** Run legacy in-kernel-seeded NaviX against a per-query bitmap. */
template <typename T>
CUVS_EXPORT void benchmark_search_navix_bitmap(
  raft::resources const& res,
  cagra::search_params const& params,
  cagra::index<T, std::uint32_t> const& index,
  raft::device_matrix_view<const T, std::int64_t, raft::row_major> queries,
  raft::device_matrix_view<std::int64_t, std::int64_t, raft::row_major> neighbors,
  raft::device_matrix_view<float, std::int64_t, raft::row_major> distances,
  cuvs::neighbors::filtering::base_filter const& sample_filter,
  std::uint32_t query_offset,
  std::uint32_t navix_policy);

/** Select passing bitmap seeds, then run strict passing-only NaviX on the same stream. */
template <typename T>
CUVS_EXPORT void benchmark_search_navix_bitmap_seeded(
  raft::resources const& res,
  cagra::search_params const& params,
  cagra::index<T, std::uint32_t> const& index,
  raft::device_matrix_view<const T, std::int64_t, raft::row_major> queries,
  raft::device_matrix_view<std::int64_t, std::int64_t, raft::row_major> neighbors,
  raft::device_matrix_view<float, std::int64_t, raft::row_major> distances,
  cuvs::neighbors::filtering::base_filter const& sample_filter,
  std::uint32_t query_offset,
  std::uint32_t* seed_ids,
  std::uint32_t* seed_counts,
  std::uint32_t* inspected_units,
  std::uint32_t navix_policy);

}  // namespace cuvs::neighbors::cagra::detail
