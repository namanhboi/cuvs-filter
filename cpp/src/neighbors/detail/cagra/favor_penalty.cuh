/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/neighbors/cagra.hpp>

#include <cmath>
#include <cstddef>

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
 * avoids a discontinuous change in traversal ordering. The result stays in [0.5, 0.9): 0.5 is
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

}  // namespace cuvs::neighbors::cagra::detail
