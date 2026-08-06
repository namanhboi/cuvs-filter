/*
 * SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../../sample_filter.cuh"

#include <cuvs/neighbors/common.hpp>

#include <type_traits>
#include <utility>

namespace cuvs::neighbors::cagra::detail {

/**
 * Private runtime decoration used by FAVOR UDF search.
 *
 * The public filter API intentionally does not expose per-query rates or result-accumulator
 * policy.  The dispatcher estimates the rates, owns their device storage for the duration of the
 * search, and attaches the pointer through this internal wrapper.
 */
template <class CagraSampleFilterT>
struct CagraSampleFilterWithRuntimeState {
  CagraSampleFilterT filter;
  const float* filtering_rates{};
  bool passing_accumulator{};

  CagraSampleFilterWithRuntimeState(CagraSampleFilterT filter,
                                    const float* filtering_rates,
                                    bool passing_accumulator)
    : filter(std::move(filter)),
      filtering_rates(filtering_rates),
      passing_accumulator(passing_accumulator)
  {
  }

  _RAFT_DEVICE auto operator()(const uint32_t query_id, const uint32_t sample_id)
  {
    return filter(query_id, sample_id);
  }
};

template <class CagraSampleFilterT>
struct CagraSampleFilterWithQueryIdOffset {
  const uint32_t offset;
  CagraSampleFilterT filter;

  CagraSampleFilterWithQueryIdOffset(const uint32_t offset, const CagraSampleFilterT filter)
    : offset(offset), filter(filter)
  {
  }

  _RAFT_DEVICE auto operator()(const uint32_t query_id, const uint32_t sample_id)
  {
    return filter(query_id + offset, sample_id);
  }
};

template <class CagraSampleFilterT>
struct CagraSampleFilterT_Selector {
  using type = CagraSampleFilterWithQueryIdOffset<CagraSampleFilterT>;
};
template <>
struct CagraSampleFilterT_Selector<cuvs::neighbors::filtering::none_sample_filter> {
  using type = cuvs::neighbors::filtering::none_sample_filter;
};

// A helper function to set a query id offset
template <class CagraSampleFilterT>
inline typename CagraSampleFilterT_Selector<CagraSampleFilterT>::type set_offset(
  CagraSampleFilterT filter, const uint32_t offset)
{
  typename CagraSampleFilterT_Selector<CagraSampleFilterT>::type new_filter(offset, filter);
  return new_filter;
}
template <>
inline typename CagraSampleFilterT_Selector<cuvs::neighbors::filtering::none_sample_filter>::type
set_offset<cuvs::neighbors::filtering::none_sample_filter>(
  cuvs::neighbors::filtering::none_sample_filter filter, const uint32_t)
{
  return filter;
}

template <typename T>
struct cagra_filter_uses_passing_accumulator : std::false_type {};

template <typename InnerFilterT>
struct cagra_filter_uses_passing_accumulator<CagraSampleFilterWithRuntimeState<InnerFilterT>>
  : std::true_type {};

template <typename InnerFilterT>
struct cagra_filter_uses_passing_accumulator<CagraSampleFilterWithQueryIdOffset<InnerFilterT>>
  : cagra_filter_uses_passing_accumulator<InnerFilterT> {};

template <typename FilterT>
constexpr bool cagra_filter_passing_accumulator_enabled(const FilterT& filter)
{
  if constexpr (requires { filter.passing_accumulator; }) {
    return filter.passing_accumulator;
  } else if constexpr (requires { filter.filter; }) {
    return cagra_filter_passing_accumulator_enabled(filter.filter);
  } else {
    return false;
  }
}
}  // namespace cuvs::neighbors::cagra::detail
