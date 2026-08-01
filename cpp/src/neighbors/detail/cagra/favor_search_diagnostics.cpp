/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "favor_search_diagnostics.hpp"

namespace cuvs::neighbors::cagra::detail::favor_search_diagnostics {
namespace {
thread_local context* active_context = nullptr;
}

auto get_active_context() noexcept -> context* { return active_context; }

void set_active_context(context* value) noexcept { active_context = value; }
}  // namespace cuvs::neighbors::cagra::detail::favor_search_diagnostics
