/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "../sample_filter_data.cuh"

#include <raft/core/error.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cudart_utils.hpp>

#include <cstdint>
#include <limits>

namespace cuvs::neighbors::cagra::detail {

/**
 * Select the first k passing graph nodes in ascending internal-node order.
 *
 * The identity path enumerates set bits from coalesced 32-word tiles.  An index with source-index
 * remapping must instead visit internal IDs in order and test their mapped source IDs.  Only lane
 * zero performs the tiny (k is normally 10) stable output compaction; the whole warp supplies
 * coalesced bitmap words or mapped predicate tests.
 */
template <typename IndexT>
RAFT_KERNEL navix_select_bitmap_seeds_kernel(cuvs::neighbors::detail::bitmap_filter_data_t bitmap,
                                             const IndexT* source_indices,
                                             std::uint32_t query_offset,
                                             std::uint32_t num_queries,
                                             std::uint32_t graph_size,
                                             std::uint32_t seed_stride,
                                             IndexT* seed_ids,
                                             std::uint32_t* seed_counts,
                                             std::uint32_t* inspected_units)
{
  static_assert(sizeof(IndexT) <= sizeof(std::uint32_t));
  constexpr unsigned warp_size = 32;
  const auto local_query       = static_cast<std::uint32_t>(blockIdx.x);
  if (local_query >= num_queries || threadIdx.x >= warp_size) { return; }

  const auto lane      = static_cast<unsigned>(threadIdx.x);
  const auto query     = static_cast<std::uint64_t>(query_offset) + local_query;
  const auto num_cols  = static_cast<std::uint64_t>(bitmap.num_cols);
  const auto row_start = query * num_cols;
  std::uint32_t found  = 0;
  std::uint32_t units  = 0;
  auto* output         = seed_ids + static_cast<std::uint64_t>(local_query) * seed_stride;

  if (query >= static_cast<std::uint64_t>(bitmap.num_rows) || bitmap.bitmap_ptr == nullptr) {
    if (lane == 0) {
      seed_counts[local_query] = 0;
      if (inspected_units != nullptr) { inspected_units[local_query] = 0; }
    }
    return;
  }

  if (source_indices == nullptr) {
    // A bitmap may describe a larger source domain than a physically compacted graph.  Without
    // an explicit source-index map, internal IDs are the source IDs, so never emit a set bit past
    // the graph descriptor even when the bitmap row has additional columns.
    const auto identity_cols = num_cols < static_cast<std::uint64_t>(graph_size)
                                 ? num_cols
                                 : static_cast<std::uint64_t>(graph_size);
    const auto first_word    = row_start / warp_size;
    const auto last_bit      = row_start + identity_cols;
    const auto last_word     = (last_bit + warp_size - 1) / warp_size;
    for (std::uint64_t tile = first_word; tile < last_word && found < seed_stride;
         tile += warp_size) {
      const auto word_index     = tile + lane;
      std::uint32_t word        = word_index < last_word ? bitmap.bitmap_ptr[word_index] : 0u;
      const auto word_bit_begin = word_index * warp_size;
      if (word_index < last_word) {
        if (word_bit_begin < row_start) {
          const auto skip = static_cast<unsigned>(row_start - word_bit_begin);
          word &= skip >= warp_size ? 0u : (~std::uint32_t{0} << skip);
        }
        if (word_bit_begin + warp_size > last_bit) {
          const auto keep = static_cast<unsigned>(last_bit - word_bit_begin);
          word &= keep >= warp_size ? ~std::uint32_t{0}
                                    : (keep == 0 ? 0u : (std::uint32_t{1} << keep) - 1u);
        }
      }

      // All lanes loaded the tile before lane zero serializes its at-most-k stable extraction.
      __syncwarp();
      if (lane == 0) {
        const auto remaining = last_word - tile;
        units += static_cast<std::uint32_t>(
          remaining < warp_size ? remaining : static_cast<std::uint64_t>(warp_size));
      }
      for (unsigned owner = 0; owner < warp_size; ++owner) {
        auto owner_word = __shfl_sync(0xffffffffu, word, owner);
        if (lane == 0 && found < seed_stride) {
          while (owner_word != 0 && found < seed_stride) {
            const auto bit          = static_cast<unsigned>(__ffs(owner_word) - 1);
            const auto absolute_bit = (tile + owner) * warp_size + bit;
            output[found++]         = static_cast<IndexT>(absolute_bit - row_start);
            owner_word &= owner_word - 1;
          }
        }
      }
      found = __shfl_sync(0xffffffffu, found, 0);
      units = __shfl_sync(0xffffffffu, units, 0);
    }
  } else {
    // The bitmap is in source-ID space. Preserve the specified ascending internal-ID policy.
    for (std::uint64_t tile = 0; tile < graph_size && found < seed_stride; tile += warp_size) {
      const auto internal = tile + lane;
      bool passes         = false;
      if (internal < graph_size) {
        const auto source = static_cast<std::uint64_t>(source_indices[internal]);
        if (source < num_cols) {
          const auto bit = row_start + source;
          passes =
            (bitmap.bitmap_ptr[bit / warp_size] & (std::uint32_t{1} << (bit % warp_size))) != 0;
        }
      }
      const auto passing_lanes = __ballot_sync(0xffffffffu, passes);
      if (lane == 0) {
        ++units;
        auto mask = passing_lanes;
        while (mask != 0 && found < seed_stride) {
          const auto passing_lane = static_cast<unsigned>(__ffs(mask) - 1);
          output[found++]         = static_cast<IndexT>(tile + passing_lane);
          mask &= mask - 1;
        }
      }
      found = __shfl_sync(0xffffffffu, found, 0);
      units = __shfl_sync(0xffffffffu, units, 0);
    }
  }

  if (lane == 0) {
    seed_counts[local_query] = found;
    if (inspected_units != nullptr) { inspected_units[local_query] = units; }
  }
}

template <typename IndexT, typename BitmapFilterT>
void select_navix_bitmap_seeds(raft::resources const& res,
                               BitmapFilterT const& filter,
                               const IndexT* source_indices,
                               std::uint32_t query_offset,
                               std::uint32_t num_queries,
                               std::uint32_t graph_size,
                               std::uint32_t seed_stride,
                               IndexT* seed_ids,
                               std::uint32_t* seed_counts,
                               std::uint32_t* inspected_units = nullptr)
{
  RAFT_EXPECTS(seed_stride > 0, "NaviX bitmap seed stride must be positive");
  RAFT_EXPECTS(seed_ids != nullptr && seed_counts != nullptr,
               "NaviX bitmap seed outputs must be preallocated");
  if (num_queries == 0) { return; }
  const auto view = filter.view();
  RAFT_EXPECTS(query_offset + static_cast<std::uint64_t>(num_queries) <=
                 static_cast<std::uint64_t>(view.get_n_rows()),
               "NaviX bitmap does not cover the requested query range");
  auto bitmap = cuvs::neighbors::detail::bitmap_filter_data_t{
    const_cast<std::uint32_t*>(view.data()),
    static_cast<std::int64_t>(view.get_n_rows()),
    static_cast<std::int64_t>(view.get_n_cols()),
    static_cast<std::int64_t>(view.get_original_nbits())};
  auto stream = raft::resource::get_cuda_stream(res);
  navix_select_bitmap_seeds_kernel<<<num_queries, 32, 0, stream>>>(bitmap,
                                                                   source_indices,
                                                                   query_offset,
                                                                   num_queries,
                                                                   graph_size,
                                                                   seed_stride,
                                                                   seed_ids,
                                                                   seed_counts,
                                                                   inspected_units);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

}  // namespace cuvs::neighbors::cagra::detail
