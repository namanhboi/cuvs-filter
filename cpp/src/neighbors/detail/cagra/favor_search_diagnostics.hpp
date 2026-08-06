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

/** Host-side launch resource data captured for the active diagnostic kernel. */
struct launch_metrics {
  std::uint32_t block_size{};
  std::uint32_t dynamic_smem_bytes{};
  std::uint32_t active_blocks_per_sm{};
  std::uint32_t max_threads_per_sm{};
};

CUVS_EXPORT auto get_launch_metrics() noexcept -> launch_metrics;
CUVS_EXPORT void set_launch_metrics(launch_metrics value) noexcept;
CUVS_EXPORT void reset_launch_metrics() noexcept;

/** Benchmark-only explicit per-query initialization seeds for staged retry diagnostics. */
CUVS_EXPORT auto get_active_seed_ptr() noexcept -> const std::uint32_t*;
CUVS_EXPORT auto get_active_seed_count() noexcept -> std::uint32_t;
CUVS_EXPORT void set_active_seeds(const std::uint32_t* values, std::uint32_t count) noexcept;

class scoped_context {
 public:
  explicit scoped_context(context* value) : previous_{get_active_context()}
  {
    reset_launch_metrics();
    set_active_context(value);
  }
  scoped_context(const scoped_context&)                    = delete;
  auto operator=(const scoped_context&) -> scoped_context& = delete;
  ~scoped_context() { set_active_context(previous_); }

 private:
  context* previous_;
};

class scoped_seeds {
 public:
  scoped_seeds(const std::uint32_t* values, std::uint32_t count)
    : previous_values_{get_active_seed_ptr()}, previous_count_{get_active_seed_count()}
  {
    set_active_seeds(values, count);
  }
  scoped_seeds(const scoped_seeds&)                    = delete;
  auto operator=(const scoped_seeds&) -> scoped_seeds& = delete;
  ~scoped_seeds() { set_active_seeds(previous_values_, previous_count_); }

 private:
  const std::uint32_t* previous_values_;
  std::uint32_t previous_count_;
};

}  // namespace cuvs::neighbors::cagra::detail::favor_search_diagnostics
