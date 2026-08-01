/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "jit_lto_kernels/favor_search_diagnostics.cuh"

#include <cuvs/core/export.hpp>

namespace cuvs::neighbors::cagra::detail::favor_search_diagnostics {

/**
 * Bench-only, thread-local diagnostic attachment.
 *
 * This deliberately is not part of cuvs::neighbors::cagra::search_params. A null attachment takes
 * the existing launcher and kernel path byte-for-byte; only an explicitly scoped benchmark search
 * selects the diagnostic JIT fragment.
 */
CUVS_EXPORT auto get_active_context() noexcept -> context*;
CUVS_EXPORT void set_active_context(context* value) noexcept;

class scoped_context {
 public:
  explicit scoped_context(context* value) : previous_{get_active_context()}
  {
    set_active_context(value);
  }
  scoped_context(const scoped_context&)            = delete;
  auto operator=(const scoped_context&) -> scoped_context& = delete;
  ~scoped_context() { set_active_context(previous_); }

 private:
  context* previous_;
};

}  // namespace cuvs::neighbors::cagra::detail::favor_search_diagnostics
