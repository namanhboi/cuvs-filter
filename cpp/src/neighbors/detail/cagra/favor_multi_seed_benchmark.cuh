/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cudart_utils.hpp>
#include <raft/util/integer_utils.hpp>

#include <cmath>
#include <cstddef>
#include <limits>
#include <type_traits>

namespace cuvs::neighbors::cagra::detail {

/**
 * Merge independent FAVOR result matrices, deduplicate by id, and order by (raw distance, id).
 * The staged experiment uses at most three rounds and k=10, so a small exhaustive kernel is more
 * auditable than a general segmented sort.
 */
template <typename IndexT>
RAFT_KERNEL favor_multi_seed_merge_kernel(const IndexT* round_neighbors,
                                          const float* round_distances,
                                          IndexT* neighbors,
                                          float* distances,
                                          std::size_t n_queries,
                                          std::size_t k,
                                          std::size_t n_rounds)
{
  auto query_id = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (query_id >= n_queries) { return; }

  constexpr auto invalid_id       = std::numeric_limits<IndexT>::max();
  constexpr auto invalid_distance = std::numeric_limits<float>::max();
  auto const round_stride         = n_queries * k;
  auto const output_offset        = query_id * k;

  for (std::size_t output_col = 0; output_col < k; ++output_col) {
    auto best_id       = invalid_id;
    auto best_distance = invalid_distance;

    for (std::size_t round = 0; round < n_rounds; ++round) {
      auto const input_offset = round * round_stride + output_offset;
      for (std::size_t input_col = 0; input_col < k; ++input_col) {
        auto const candidate_id       = round_neighbors[input_offset + input_col];
        auto const candidate_distance = round_distances[input_offset + input_col];
        if (candidate_id == invalid_id || !isfinite(candidate_distance)) { continue; }
        if constexpr (std::is_signed_v<IndexT>) {
          if (candidate_id < IndexT{0}) { continue; }
        }

        bool already_selected = false;
        for (std::size_t previous_col = 0; previous_col < output_col; ++previous_col) {
          if (neighbors[output_offset + previous_col] == candidate_id) {
            already_selected = true;
            break;
          }
        }
        if (already_selected) { continue; }

        if (candidate_distance < best_distance ||
            (candidate_distance == best_distance && candidate_id < best_id)) {
          best_id       = candidate_id;
          best_distance = candidate_distance;
        }
      }
    }

    neighbors[output_offset + output_col] = best_id;
    distances[output_offset + output_col] = best_distance;
  }
}

template <typename IndexT>
inline void merge_favor_multi_seed_results(const raft::resources& res,
                                           const IndexT* round_neighbors,
                                           const float* round_distances,
                                           IndexT* neighbors,
                                           float* distances,
                                           std::size_t n_queries,
                                           std::size_t k,
                                           std::size_t n_rounds)
{
  RAFT_EXPECTS(n_queries > 0 && k > 0 && n_rounds > 0,
               "FAVOR multi-seed merge dimensions must be positive");
  constexpr unsigned int block_size = 128;
  auto const grid_size = static_cast<unsigned int>((n_queries + block_size - 1) / block_size);
  auto stream          = raft::resource::get_cuda_stream(res);
  favor_multi_seed_merge_kernel<<<grid_size, block_size, 0, stream>>>(
    round_neighbors, round_distances, neighbors, distances, n_queries, k, n_rounds);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

}  // namespace cuvs::neighbors::cagra::detail
