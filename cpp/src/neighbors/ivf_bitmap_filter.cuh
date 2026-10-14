/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "sample_filter.cuh"

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/util/cudart_utils.hpp>

#include <algorithm>
#include <limits>

namespace cuvs::neighbors::filtering {

/** Validate the resident bitmap before any IVF search batches are launched. */
template <typename FilterT>
void validate_ivf_bitmap_filter(const FilterT& filter,
                                int64_t index_size,
                                uint64_t n_queries,
                                uint32_t query_offset)
{
  if constexpr (std::is_same_v<FilterT, bitmap_filter<uint32_t, int64_t>>) {
    RAFT_EXPECTS(n_queries <= std::numeric_limits<uint32_t>::max(),
                 "The number of bitmap queries exceeds uint32");
    const auto view = filter.view();
    const auto end  = uint64_t(query_offset) + n_queries;
    RAFT_EXPECTS(view.get_n_cols() == index_size,
                 "The query bitmap width must match the IVF index size");
    RAFT_EXPECTS(view.get_n_rows() >= 0 && end <= uint64_t(view.get_n_rows()),
                 "The query bitmap does not cover this search batch");
    RAFT_EXPECTS(end <= uint64_t(std::numeric_limits<uint32_t>::max()) + 1,
                 "The query bitmap row ID exceeds uint32");
  }
}

template <typename IdxT>
__global__ void normalize_ivf_bitmap_results_kernel(core::bitmap_view<uint32_t, int64_t> bitmap,
                                                    IdxT* neighbors,
                                                    float* distances,
                                                    uint64_t count,
                                                    uint32_t k,
                                                    uint32_t query_offset,
                                                    float empty_distance)
{
  for (uint64_t i = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
       i += uint64_t(gridDim.x) * blockDim.x) {
    const auto id       = neighbors[i];
    const auto row      = int64_t(query_offset) + static_cast<int64_t>(i / k);
    const auto distance = distances[i];
    // Empty selection lanes may resolve to a passing ID while retaining a dummy distance.
    // Such lanes are still missing results, rather than additional copies of that valid ID.
    const bool real_distance = distance < std::numeric_limits<float>::max() &&
                               distance > std::numeric_limits<float>::lowest();
    if (!real_distance || id < 0 || uint64_t(id) >= uint64_t(bitmap.get_n_cols()) ||
        !bitmap.test(row, id)) {
      neighbors[i] = std::numeric_limits<IdxT>::max();
      distances[i] = empty_distance;
    }
  }
}

/**
 * IVF's materialized-distance path may select a filtered ID carrying a dummy distance when fewer
 * than k eligible candidates were probed. Turn those entries into the usual IVF missing-ID
 * sentinel.
 */
template <typename FilterT, typename IdxT>
void normalize_ivf_bitmap_results(raft::resources const& res,
                                  const FilterT& filter,
                                  IdxT* neighbors,
                                  float* distances,
                                  uint64_t n_queries,
                                  uint32_t k,
                                  uint32_t query_offset)
{
  if constexpr (std::is_same_v<FilterT, bitmap_filter<uint32_t, int64_t>>) {
    const auto count = n_queries * k;
    if (count == 0) { return; }
    const auto blocks = static_cast<uint32_t>(std::min<uint64_t>((count + 255) / 256, 65535));
    const auto empty_distance = std::numeric_limits<float>::max();
    normalize_ivf_bitmap_results_kernel<<<blocks, 256, 0, raft::resource::get_cuda_stream(res)>>>(
      filter.view(), neighbors, distances, count, k, query_offset, empty_distance);
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  }
}

}  // namespace cuvs::neighbors::filtering
