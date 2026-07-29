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

}  // namespace cuvs::neighbors::cagra::detail
