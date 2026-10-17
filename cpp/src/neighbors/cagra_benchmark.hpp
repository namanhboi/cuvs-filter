/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/core/export.hpp>
#include <cuvs/neighbors/cagra.hpp>

#include <cstdint>

namespace cuvs::neighbors::cagra::detail {

/** Count passing source IDs once before a bitset-filtered benchmark search. */
template <typename T>
CUVS_EXPORT std::uint64_t benchmark_count_cagra_bitset_matches(
  raft::resources const& res,
  cagra::index<T, std::uint32_t> const& index,
  cuvs::neighbors::filtering::base_filter const& sample_filter);

/**
 * Retired FAVOR compatibility bridge. Always reports an error.
 */
template <typename T>
[[deprecated("FAVOR is retired")]] CUVS_EXPORT void
benchmark_search_favor_with_known_filtering_rate(
  raft::resources const& res,
  cagra::search_params const& params,
  cagra::index<T, std::uint32_t> const& index,
  raft::device_matrix_view<const T, std::int64_t, raft::row_major> queries,
  raft::device_matrix_view<std::int64_t, std::int64_t, raft::row_major> neighbors,
  raft::device_matrix_view<float, std::int64_t, raft::row_major> distances,
  cuvs::neighbors::filtering::base_filter const& sample_filter);

/** Retired FAVOR sampler compatibility bridge. Always reports an error. */
template <typename T>
[[deprecated("FAVOR is retired")]] CUVS_EXPORT void benchmark_estimate_favor_udf_filtering_rates(
  raft::resources const& res,
  cagra::index<T, std::uint32_t> const& index,
  std::uint32_t num_queries,
  cuvs::neighbors::filtering::base_filter const& sample_filter,
  float* filtering_rates,
  std::uint32_t* passing_counts,
  std::uint32_t sample_offset = 0);

/**
 * Run the private SINGLE_CTA UDF path with optional passing-result accumulation.
 * The rate pointer must be null; Base and Retain never estimate per-query selectivity.
 */
template <typename T>
CUVS_EXPORT void benchmark_search_udf_with_result_accumulator(
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

/**
 * Select the first seed_stride passing bitmap nodes, then run default SINGLE_CTA from only those
 * seeds. A zero stride preserves the historical benchmark control and uses the result width.
 * This is a benchmark control: traversal remains default CAGRA and the optional accumulator only
 * changes result retention.
 */
template <typename T>
CUVS_EXPORT void benchmark_search_bitmap_seeded(
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
  bool passing_accumulator  = false,
  std::uint32_t seed_stride = 0);

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
  std::uint32_t seed_stride,
  std::uint32_t navix_policy);

}  // namespace cuvs::neighbors::cagra::detail
