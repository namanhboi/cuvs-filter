/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstdint>

namespace cuvs::neighbors::cagra::detail {

/// Device payload for linked CAGRA sample filters plus query offset for wrapped filters.
template <typename SourceIndexT>
struct cagra_sample_filter {
  enum : std::uint32_t { kind_none = 0, kind_bitset = 1, kind_udf = 2, kind_bitmap = 3 };

  void* filter_data{nullptr};
  const float* filtering_rates{nullptr};
  std::uint32_t query_id_offset{0};
  std::uint32_t filter_kind{kind_none};
  std::uint32_t passing_accumulator{0};

  __device__ __forceinline__ void* sample_filter_data() { return filter_data; }

  __device__ __forceinline__ float filtering_rate(std::uint32_t query_id,
                                                  float scalar_fallback) const
  {
    return filtering_rates == nullptr ? scalar_fallback : filtering_rates[query_id];
  }

  __device__ __forceinline__ bool is_bitset() const { return filter_kind == kind_bitset; }

  __device__ __forceinline__ bool is_bitmap() const { return filter_kind == kind_bitmap; }

  __device__ __forceinline__ bool uses_passing_accumulator() const
  {
    return passing_accumulator != 0;
  }
};

}  // namespace cuvs::neighbors::cagra::detail
