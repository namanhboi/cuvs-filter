/*
 * SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../../sample_filter.cuh"

#include <cuvs/neighbors/common.hpp>

#include <cstdint>
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

/**
 * Private runtime decoration used by the benchmark-only NaviX SINGLE_CTA experiment.
 *
 * NaviX is deliberately selected from the filter type instead of a public search parameter. This
 * keeps the public CAGRA API and the default/FAVOR kernel specializations unchanged while still
 * allowing the JIT launcher to select a dedicated kernel binary.
 */
template <class CagraSampleFilterT>
struct CagraSampleFilterWithNavixRuntimeState {
  CagraSampleFilterT filter;
  std::uint32_t policy{};
  static constexpr bool navix = true;

  CagraSampleFilterWithNavixRuntimeState(CagraSampleFilterT filter, std::uint32_t policy)
    : filter(std::move(filter)), policy(policy)
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

template <typename T>
struct cagra_filter_uses_navix : std::false_type {};

template <typename InnerFilterT, bool PassingAccumulator>
struct cagra_filter_uses_passing_accumulator<
  CagraSampleFilterWithRuntimeState<InnerFilterT, PassingAccumulator>>
  : std::bool_constant<PassingAccumulator> {};

template <typename InnerFilterT>
struct cagra_filter_uses_navix<CagraSampleFilterWithNavixRuntimeState<InnerFilterT>>
  : std::true_type {};

template <typename InnerFilterT>
struct cagra_filter_uses_passing_accumulator<CagraSampleFilterWithQueryIdOffset<InnerFilterT>>
  : cagra_filter_uses_passing_accumulator<InnerFilterT> {};

template <typename InnerFilterT>
struct cagra_filter_uses_navix<CagraSampleFilterWithQueryIdOffset<InnerFilterT>>
  : cagra_filter_uses_navix<InnerFilterT> {};

template <typename T>
std::uint32_t cagra_filter_navix_policy(const T& filter)
{
  if constexpr (requires { filter.policy; }) {
    return filter.policy;
  } else if constexpr (requires { filter.filter; }) {
    return cagra_filter_navix_policy(filter.filter);
  } else {
    return 0;
  }
}

}  // namespace cuvs::neighbors::cagra::detail
