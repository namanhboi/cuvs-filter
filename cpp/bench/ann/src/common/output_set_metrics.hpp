/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <cstdint>

namespace cuvs::bench::detail {

template <typename IndexT>
[[nodiscard]] constexpr auto is_first_output_occurrence(const IndexT* candidates,
                                                        std::uint32_t rank) -> bool
{
  for (std::uint32_t previous = 0; previous < rank; ++previous) {
    if (candidates[previous] == candidates[rank]) { return false; }
  }
  return true;
}

template <typename IndexT>
[[nodiscard]] constexpr auto is_valid_source_id(IndexT id, std::size_t base_rows) -> bool
{
  return static_cast<std::uint64_t>(id) < static_cast<std::uint64_t>(base_rows);
}

}  // namespace cuvs::bench::detail
