/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>

namespace cuvs::neighbors::detail {

template <typename BitsetT, typename IndexT>
__device__ bool sample_filter_none(const IndexT* const* const inds_ptr,
                                   const uint32_t query_ix,
                                   const uint32_t cluster_ix,
                                   const uint32_t sample_ix,
                                   uint32_t* bitset_ptr,
                                   IndexT bitset_len,
                                   IndexT original_nbits,
                                   IndexT bitmap_cols);

template <typename BitsetT, typename IndexT>
__device__ bool sample_filter_bitset(const IndexT* const* const inds_ptr,
                                     const uint32_t query_ix,
                                     const uint32_t cluster_ix,
                                     const uint32_t sample_ix,
                                     uint32_t* bitset_ptr,
                                     IndexT bitset_len,
                                     IndexT original_nbits,
                                     IndexT bitmap_cols);

template <typename BitsetT, typename IndexT>
__device__ bool sample_filter_bitmap(const IndexT* const* const inds_ptr,
                                     const uint32_t query_ix,
                                     const uint32_t cluster_ix,
                                     const uint32_t sample_ix,
                                     uint32_t* bitmap_ptr,
                                     IndexT bitmap_rows,
                                     IndexT original_nbits,
                                     IndexT bitmap_cols);

}  // namespace cuvs::neighbors::detail
