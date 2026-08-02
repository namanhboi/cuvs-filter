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

}  // namespace cuvs::neighbors::cagra::detail
