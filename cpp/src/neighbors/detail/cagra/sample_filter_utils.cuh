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
 * Private runtime decoration used by benchmark UDF search.
 *
 * The public filter API intentionally does not expose per-query rates or result-accumulator
 * policy. FAVOR attaches its sampled query-local rates; default CAGRA always attaches a null rate
 * pointer and can independently select the passive passing-result accumulator.
 */
template <class CagraSampleFilterT, bool PassingAccumulator>
struct CagraSampleFilterWithRuntimeState {
  CagraSampleFilterT filter;
  const float* filtering_rates{};
  static constexpr bool passing_accumulator = PassingAccumulator;

  CagraSampleFilterWithRuntimeState(CagraSampleFilterT filter, const float* filtering_rates)
    : filter(std::move(filter)), filtering_rates(filtering_rates)
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

template <typename InnerFilterT, bool PassingAccumulator>
struct cagra_filter_uses_passing_accumulator<
  CagraSampleFilterWithRuntimeState<InnerFilterT, PassingAccumulator>>
  : std::bool_constant<PassingAccumulator> {};

template <typename InnerFilterT>
struct cagra_filter_uses_passing_accumulator<CagraSampleFilterWithQueryIdOffset<InnerFilterT>>
  : cagra_filter_uses_passing_accumulator<InnerFilterT> {};

}  // namespace cuvs::neighbors::cagra::detail
