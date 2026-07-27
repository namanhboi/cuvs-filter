/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/neighbors/cagra.hpp>

namespace cuvs::neighbors::cagra::detail {

enum class single_cta_kernel_variant : uint32_t { DEFAULT, FAVOR, FAVOR_ACCUMULATOR };

constexpr bool is_favor_mode(filtering_mode mode)
{
  return mode == filtering_mode::FAVOR || mode == filtering_mode::FAVOR_ACCUMULATOR;
}

constexpr bool uses_passing_accumulator(filtering_mode mode)
{
  return mode == filtering_mode::FAVOR_ACCUMULATOR;
}

constexpr single_cta_kernel_variant get_single_cta_kernel_variant(filtering_mode mode)
{
  if (uses_passing_accumulator(mode)) { return single_cta_kernel_variant::FAVOR_ACCUMULATOR; }
  if (is_favor_mode(mode)) { return single_cta_kernel_variant::FAVOR; }
  return single_cta_kernel_variant::DEFAULT;
}

}  // namespace cuvs::neighbors::cagra::detail
