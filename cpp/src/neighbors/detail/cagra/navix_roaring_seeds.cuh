/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "../roaring_bitmap_filter_data.cuh"

#include <cuvs/neighbors/common.hpp>
#include <raft/core/error.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/util/cudart_utils.hpp>

#include <cstdint>

namespace cuvs::neighbors::cagra::detail {

/** Preserve the bitmap seeding policy: first passing nodes in internal graph-ID order. */
template <typename IndexT>
RAFT_KERNEL navix_select_roaring_seeds_kernel(
  cuvs::neighbors::detail::roaring_bitmap_filter_data_t<> const* filter,
  IndexT const* source_indices,
  std::uint32_t query_offset,
  std::uint32_t graph_size,
  std::uint32_t seed_stride,
  IndexT* seed_ids,
  std::uint32_t* seed_counts,
  std::uint32_t* inspected_units)
{
  constexpr unsigned warp_size = 32;
  auto const local_query       = static_cast<std::uint32_t>(blockIdx.x);
  auto const query             = static_cast<std::uint64_t>(query_offset) + local_query;
  auto const lane              = static_cast<unsigned>(threadIdx.x);
  std::uint32_t found          = 0;
  std::uint32_t units          = 0;
  auto* output                 = seed_ids + static_cast<std::uint64_t>(local_query) * seed_stride;

  if (query < filter->num_queries && filter->empty_rows[query] == 0) {
    for (std::uint64_t tile = 0; tile < graph_size && found < seed_stride; tile += warp_size) {
      auto const internal = tile + lane;
      bool passes         = false;
      if (internal < graph_size) {
        auto const source = source_indices == nullptr
                              ? internal
                              : static_cast<std::uint64_t>(source_indices[internal]);
        passes            = source < filter->dataset_rows &&
                 filter->refs[query]->contains(static_cast<std::uint32_t>(source));
      }
      auto const passing_lanes = __ballot_sync(0xffffffffu, passes);
      if (lane == 0) {
        ++units;
        auto mask = passing_lanes;
        while (mask != 0 && found < seed_stride) {
          auto const passing_lane = static_cast<unsigned>(__ffs(mask) - 1);
          output[found++]         = static_cast<IndexT>(tile + passing_lane);
          mask &= mask - 1;
        }
      }
      found = __shfl_sync(0xffffffffu, found, 0);
    }
  }
  if (lane == 0) {
    seed_counts[local_query] = found;
    // Units count 32-node tiles; the dense identity path instead counts bitmap words.
    if (inspected_units != nullptr) { inspected_units[local_query] = units; }
  }
}

template <typename IndexT>
void select_navix_bitmap_seeds(raft::resources const& res,
                               cuvs::neighbors::filtering::roaring_bitmap_filter const& filter,
                               IndexT const* source_indices,
                               std::uint32_t query_offset,
                               std::uint32_t num_queries,
                               std::uint32_t graph_size,
                               std::uint32_t seed_stride,
                               IndexT* seed_ids,
                               std::uint32_t* seed_counts,
                               std::uint32_t* inspected_units = nullptr)
{
  RAFT_EXPECTS(filter.valid(), "Roaring seeds require an initialized filter");
  RAFT_EXPECTS(seed_stride > 0 && seed_ids != nullptr && seed_counts != nullptr,
               "Roaring seed outputs must be preallocated with a positive stride");
  RAFT_EXPECTS(static_cast<std::uint64_t>(query_offset) + num_queries <= filter.num_queries(),
               "Roaring filter does not cover the requested seed query range");
  if (num_queries == 0) { return; }
  auto const* payload = static_cast<cuvs::neighbors::detail::roaring_bitmap_filter_data_t<> const*>(
    filter.device_payload());
  navix_select_roaring_seeds_kernel<<<num_queries,
                                      32,
                                      0,
                                      raft::resource::get_cuda_stream(res).get()>>>(
    payload,
    source_indices,
    query_offset,
    graph_size,
    seed_stride,
    seed_ids,
    seed_counts,
    inspected_units);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

}  // namespace cuvs::neighbors::cagra::detail
