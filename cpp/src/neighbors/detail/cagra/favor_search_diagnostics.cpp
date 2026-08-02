/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "favor_search_diagnostics.hpp"

namespace cuvs::neighbors::cagra::detail::favor_search_diagnostics {
namespace {
thread_local context* active_context              = nullptr;
thread_local const std::uint32_t* active_seed_ptr = nullptr;
thread_local std::uint32_t active_seed_count      = 0;
}  // namespace

auto get_active_context() noexcept -> context* { return active_context; }

void set_active_context(context* value) noexcept { active_context = value; }

auto get_active_seed_ptr() noexcept -> const std::uint32_t* { return active_seed_ptr; }

auto get_active_seed_count() noexcept -> std::uint32_t { return active_seed_count; }

void set_active_seeds(const std::uint32_t* values, std::uint32_t count) noexcept
{
  active_seed_ptr   = values;
  active_seed_count = count;
}
}  // namespace cuvs::neighbors::cagra::detail::favor_search_diagnostics
