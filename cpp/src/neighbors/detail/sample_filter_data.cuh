/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>

namespace cuvs::neighbors::detail {

/// Bitset (and length metadata) for linked @c sample_filter in JIT LTO; passed by value to
/// @c __global__ entry points. `bitset_ptr == nullptr` means no bitset (none filter, or not a
/// bitset at runtime on host). Also passed as @c void* to the unified JIT @c sample_filter (see
/// @c sample_filter.cuh / @c bitset_filter).
template <typename SourceIndexT>
struct bitset_filter_data_t {
  std::uint32_t* bitset_ptr{nullptr};
  SourceIndexT bitset_len{};
  SourceIndexT original_nbits{};
};

/// Row-major per-query bitmap metadata for linked CAGRA sample filters. Dimensions use int64_t
/// because the tightly packed linear bit index (`query_id * num_cols + node_id`) can exceed the
/// 32-bit CAGRA source-index range even when each individual dimension does not.
struct bitmap_filter_data_t {
  std::uint32_t* bitmap_ptr{nullptr};
  std::int64_t num_rows{};
  std::int64_t num_cols{};
  std::int64_t original_nbits{};
};

}  // namespace cuvs::neighbors::detail
