/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/neighbors/cagra.hpp>
#include <raft/core/error.hpp>

namespace cuvs::neighbors::cagra {

// Retain legacy symbols so old callers fail explicitly instead of silently changing methods.
#define CUVS_RETIRED_FAVOR(T)                                                              \
  float compute_favor_delta_d(                                                             \
    raft::resources const&, favor_delta_d_params const&, cagra::index<T, uint32_t> const&) \
  {                                                                                        \
    RAFT_FAIL("FAVOR and its delta-d estimator are retired");                              \
  }                                                                                        \
  void save_favor_delta_d(raft::resources const&,                                          \
                          std::string const&,                                              \
                          favor_delta_d_params const&,                                     \
                          cagra::index<T, uint32_t> const&,                                \
                          float)                                                           \
  {                                                                                        \
    RAFT_FAIL("FAVOR delta-d sidecars are retired");                                       \
  }                                                                                        \
  float load_favor_delta_d(raft::resources const&,                                         \
                           std::string const&,                                             \
                           favor_delta_d_params const&,                                    \
                           cagra::index<T, uint32_t> const&)                               \
  {                                                                                        \
    RAFT_FAIL("FAVOR delta-d sidecars are retired");                                       \
  }

CUVS_RETIRED_FAVOR(float)
CUVS_RETIRED_FAVOR(half)
CUVS_RETIRED_FAVOR(int8_t)
CUVS_RETIRED_FAVOR(uint8_t)
#undef CUVS_RETIRED_FAVOR

}  // namespace cuvs::neighbors::cagra
