/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../../sample_filter.cuh"
#include "filter.cuh"

namespace cuvs::neighbors::detail {

template <typename BitsetT, typename IndexT>
__device__ bool sample_filter_none(const IndexT* const* const inds_ptrs,
                                   const uint32_t query_ix,
                                   const uint32_t cluster_ix,
                                   const uint32_t sample_ix,
                                   uint32_t* bitset_ptr,
                                   IndexT bitset_len,
                                   IndexT original_nbits,
                                   IndexT bitmap_cols)
{
  return true;
}

template <typename BitsetT, typename IndexT>
__device__ bool sample_filter_bitset(const IndexT* const* const inds_ptrs,
                                     const uint32_t query_ix,
                                     const uint32_t cluster_ix,
                                     const uint32_t sample_ix,
                                     uint32_t* bitset_ptr,
                                     IndexT bitset_len,
                                     IndexT original_nbits,
                                     IndexT bitmap_cols)
{
  auto bitset_view =
    raft::core::bitset_view<BitsetT, IndexT>{bitset_ptr, bitset_len, original_nbits};
  auto bitset_filter = cuvs::neighbors::filtering::bitset_filter<BitsetT, IndexT>{bitset_view};
  auto ivf_to_sample_filter = cuvs::neighbors::filtering::
    ivf_to_sample_filter<IndexT, cuvs::neighbors::filtering::bitset_filter<BitsetT, IndexT>>{
      inds_ptrs, bitset_filter};
  return ivf_to_sample_filter(query_ix, cluster_ix, sample_ix);
}

template <typename BitsetT, typename IndexT>
__device__ bool sample_filter_bitmap(const IndexT* const* const inds_ptrs,
                                     const uint32_t query_ix,
                                     const uint32_t cluster_ix,
                                     const uint32_t sample_ix,
                                     uint32_t* bitmap_ptr,
                                     IndexT bitmap_rows,
                                     IndexT original_nbits,
                                     IndexT bitmap_cols)
{
  // IVF lists store original dataset IDs; the bitmap columns refer to those IDs.
  const auto row = static_cast<IndexT>(query_ix);
  const auto col = inds_ptrs[cluster_ix][sample_ix];
  if (row >= bitmap_rows || col < 0 || col >= bitmap_cols) { return false; }
  const raft::core::bitmap_view<BitsetT, IndexT> view{
    bitmap_ptr, bitmap_rows, bitmap_cols, original_nbits};
  return view.test(row, col);
}

}  // namespace cuvs::neighbors::detail
