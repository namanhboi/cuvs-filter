/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/neighbors/cagra.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace cuvs::neighbors::cagra::detail {

inline float favor_penalty_coefficient(float filtering_rate, size_t effective_itopk)
{
  if (filtering_rate <= 0.0f) { return 0.0f; }
  auto const selectivity = 1.0f - filtering_rate;
  auto const ef          = static_cast<float>(effective_itopk);
  return filtering_rate * (ef - selectivity) / (2.0f * selectivity * ef);
}

inline float favor_reference_penalty(float filtering_rate, size_t effective_itopk, float delta_d)
{
  if (filtering_rate <= 0.0f || delta_d == 0.0f) { return 0.0f; }
  // Keep the operation types and ordering identical to the original device implementation.
  auto const selectivity = 1.0f - filtering_rate;
  auto const ef          = static_cast<float>(effective_itopk);
  return filtering_rate * (ef - selectivity) * delta_d / (2.0f * selectivity * ef);
}

/**
 * Select a CAGRA-local retention fraction from the expected number of passing itopk entries.
 *
 * Let E = selectivity * itopk and x = E / k. The shortfall pressure is zero when E >= k and
 * saturates when E <= k / 2, matching FAVOR's half-k stopping-condition threshold. Smoothstep
 * avoids a discontinuous change in traversal ordering. The result stays in [0.5, 0.9]: 0.5 is
 * the established retention-safe midpoint and a value below one preserves strict retention
 * headroom for rejected bridge candidates.
 */
inline float favor_automatic_retention_fraction(float filtering_rate,
                                                size_t effective_itopk,
                                                size_t topk)
{
  if (topk == 0) { return 0.5f; }
  auto const selectivity     = 1.0f - filtering_rate;
  auto const expected_passes = selectivity * static_cast<float>(effective_itopk);
  auto pressure              = 2.0f * (1.0f - expected_passes / static_cast<float>(topk));
  pressure                   = std::fmax(0.0f, std::fmin(1.0f, pressure));
  auto const smooth_pressure = pressure * pressure * (3.0f - 2.0f * pressure);
  return 0.5f + 0.4f * smooth_pressure;
}

struct favor_adaptive_termination_budget {
  std::uint32_t prefix_size{};
  std::uint32_t start_iteration{};
  std::uint32_t hard_iteration{};
};

/** Smallest warp-aligned logical result set whose half-full threshold exceeds topk. */
inline std::uint32_t favor_adaptive_prefix_size(std::size_t topk)
{
  constexpr std::size_t warp_size = 32;
  const auto required             = 2 * topk + 1;
  const auto rounded              = ((required + warp_size - 1) / warp_size) * warp_size;
  return static_cast<std::uint32_t>(
    std::min<std::size_t>(rounded, std::numeric_limits<std::uint32_t>::max()));
}

/** Match CAGRA's existing automatically derived graph-depth allowance. */
inline std::uint32_t favor_graph_depth_allowance(std::int64_t dataset_size,
                                                 std::int64_t graph_degree)
{
  std::uint32_t depth          = 0;
  std::int64_t reachable_nodes = 1;
  const auto branching         = std::max<std::int64_t>(2, graph_degree / 2);
  while (reachable_nodes < dataset_size) {
    if (reachable_nodes > std::numeric_limits<std::int64_t>::max() / branching) {
      ++depth;
      break;
    }
    reachable_nodes *= branching;
    ++depth;
  }
  return depth;
}

/**
 * Resolve the sparse FAVOR continuation ceiling. The returned zero prefix disables adaptation.
 * The caller supplies CAGRA's already-resolved automatic budget as the traversal floor.
 */
inline favor_adaptive_termination_budget favor_adaptive_budget(float filtering_rate,
                                                               std::size_t effective_itopk,
                                                               std::size_t topk,
                                                               std::size_t search_width,
                                                               std::int64_t dataset_size,
                                                               std::int64_t graph_degree,
                                                               std::uint32_t base_iteration)
{
  favor_adaptive_termination_budget out{};
  const auto selectivity = 1.0f - filtering_rate;
  const auto prefix      = favor_adaptive_prefix_size(topk);
  if (!(selectivity > 0.0f && selectivity < 0.10f) || search_width == 0 || topk == 0 ||
      prefix >= effective_itopk ||
      selectivity * static_cast<float>(effective_itopk) >= static_cast<float>(topk)) {
    return out;
  }

  const auto width = static_cast<double>(search_width);
  const auto stable_ceil = [](double value) {
    // filtering_rate is a float; suppress only representation noise around integer boundaries.
    return std::ceil(value - 1.0e-6 * std::max(1.0, std::abs(value)));
  };
  const auto by_buffer = static_cast<std::uint64_t>(
    stable_ceil(static_cast<double>(effective_itopk) / width));
  const auto by_passing = static_cast<std::uint64_t>(
    stable_ceil(static_cast<double>(topk) / (static_cast<double>(selectivity) * width)));
  const auto expansion = std::max(by_buffer, by_passing);
  const auto depth     = static_cast<std::uint64_t>(
    favor_graph_depth_allowance(dataset_size, graph_degree));
  const auto hard64 = std::max<std::uint64_t>(base_iteration, depth + 4 * expansion);

  out.prefix_size    = prefix;
  out.start_iteration = base_iteration;
  out.hard_iteration = static_cast<std::uint32_t>(
    std::min<std::uint64_t>(hard64, std::numeric_limits<std::uint32_t>::max()));
  return out;
}

}  // namespace cuvs::neighbors::cagra::detail
