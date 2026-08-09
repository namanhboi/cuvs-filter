/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "extern_device_functions.cuh"

#include "../../sample_filter_data.cuh"

#include <raft/core/bitmap.cuh>
#include <raft/core/bitset.cuh>

#include <cstdint>

namespace cuvs::neighbors::detail {

template <typename SourceIndexT>
__device__ bool sample_filter_none_impl(uint32_t /*query_id*/,
                                        SourceIndexT /*node_id*/,
                                        void* /*filter_data*/)
{
  return true;
}

template <typename SourceIndexT>
__device__ bool sample_filter_bitset_impl(uint32_t /*query_id*/,
                                          SourceIndexT node_id,
                                          void* filter_data)
{
  if (filter_data == nullptr) { return true; }

  auto* data = static_cast<bitset_filter_data_t<SourceIndexT>*>(filter_data);
  if (data->bitset_ptr == nullptr) { return true; }

  raft::core::bitset_view<uint32_t, SourceIndexT> const view{
    data->bitset_ptr, data->bitset_len, data->original_nbits};
  return view.test(node_id);
}

template <typename SourceIndexT>
__device__ bool sample_filter_bitmap_impl(uint32_t query_id,
                                          SourceIndexT node_id,
                                          void* filter_data)
{
  if (filter_data == nullptr) { return true; }

  auto* data = static_cast<bitmap_filter_data_t*>(filter_data);
  if (data->bitmap_ptr == nullptr) { return true; }

  const auto row = static_cast<std::int64_t>(query_id);
  const auto col = static_cast<std::int64_t>(node_id);
  if (row < 0 || row >= data->num_rows || col < 0 || col >= data->num_cols) { return false; }

  raft::core::bitmap_view<uint32_t, std::int64_t> const view{
    data->bitmap_ptr, data->num_rows, data->num_cols, data->original_nbits};
  return view.test(row, col);
}

}  // namespace cuvs::neighbors::detail
