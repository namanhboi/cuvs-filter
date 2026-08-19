/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "cagra_planner_base.hpp"
#include <cuvs/detail/jit_lto/cagra/cagra_fragments.hpp>
#include <cuvs/distance/distance.hpp>
#include <string>

namespace cuvs::neighbors::cagra::detail::single_cta_search {

template <typename DataTag,
          typename IndexTag,
          typename DistanceTag,
          typename SourceIndexTag,
          typename QueryTag,
          typename CodebookTag,
          typename SampleFilterJitTag>
struct CagraSingleCtaSearchPlanner
  : CagraPlannerBase<DataTag, IndexTag, DistanceTag, QueryTag, CodebookTag, SampleFilterJitTag> {
  static inline LauncherJitCache launcher_jit_cache{};

  CagraSingleCtaSearchPlanner(cuvs::distance::DistanceType /*metric*/,
                              bool /*topk_by_bitonic_sort*/,
                              bool /*bitonic_sort_and_merge_multi_warps*/,
                              uint32_t /*team_size*/,
                              uint32_t /*dataset_block_dim*/,
                              bool /*is_vpq*/,
                              uint32_t /*pq_bits*/,
                              uint32_t /*pq_len*/,
                              bool persistent    = false,
                              bool favor         = false,
                              bool navix         = false,
                              bool diagnostics   = false,
                              bool bitmap_seeded = false)
    : CagraPlannerBase<DataTag, IndexTag, DistanceTag, QueryTag, CodebookTag, SampleFilterJitTag>(
        bitmap_seeded ? "search_single_cta_bitmap_seeded"
        : navix ? (diagnostics ? "search_single_cta_navix_diagnostic" : "search_single_cta_navix")
        : persistent ? "search_single_cta_p"
                     : (diagnostics ? (favor ? "search_single_cta_favor_diagnostic"
                                             : "search_single_cta_default_diagnostic")
                                    : (favor ? "search_single_cta_favor" : "search_single_cta")),
        launcher_jit_cache)
  {
  }

  void add_bitmap_seeded_search_kernel_fragment(bool topk_by_bitonic_sort,
                                                bool bitonic_sort_and_merge_multi_warps)
  {
    auto add = [&]<bool TopkByBitonic, bool MultiWarpMerge>() {
      this->template add_static_fragment<
        fragment_tag_search_single_cta_bitmap_seeded<DataTag,
                                                     SourceIndexTag,
                                                     IndexTag,
                                                     DistanceTag,
                                                     TopkByBitonic,
                                                     MultiWarpMerge>>();
    };
    if (topk_by_bitonic_sort && bitonic_sort_and_merge_multi_warps) {
      add.template operator()<true, true>();
    } else if (topk_by_bitonic_sort) {
      add.template operator()<true, false>();
    } else if (bitonic_sort_and_merge_multi_warps) {
      add.template operator()<false, true>();
    } else {
      add.template operator()<false, false>();
    }
  }

  void add_navix_search_kernel_fragment(bool topk_by_bitonic_sort,
                                        bool bitonic_sort_and_merge_multi_warps,
                                        bool diagnostics = false)
  {
    auto add = [&]<bool TopkByBitonic, bool MultiWarpMerge>() {
      if (diagnostics) {
        this->template add_static_fragment<
          fragment_tag_search_single_cta_navix_diagnostic<DataTag,
                                                          SourceIndexTag,
                                                          IndexTag,
                                                          DistanceTag,
                                                          TopkByBitonic,
                                                          MultiWarpMerge>>();
      } else {
        this->template add_static_fragment<fragment_tag_search_single_cta_navix<DataTag,
                                                                                SourceIndexTag,
                                                                                IndexTag,
                                                                                DistanceTag,
                                                                                TopkByBitonic,
                                                                                MultiWarpMerge>>();
      }
    };
    if (topk_by_bitonic_sort && bitonic_sort_and_merge_multi_warps) {
      add.template operator()<true, true>();
    } else if (topk_by_bitonic_sort) {
      add.template operator()<true, false>();
    } else if (bitonic_sort_and_merge_multi_warps) {
      add.template operator()<false, true>();
    } else {
      add.template operator()<false, false>();
    }
  }

  void add_default_diagnostic_kernel_fragment(bool topk_by_bitonic_sort,
                                              bool bitonic_sort_and_merge_multi_warps)
  {
    auto add = [&]<bool TopkByBitonic, bool MultiWarpMerge>() {
      this->template add_static_fragment<
        fragment_tag_search_single_cta_default_diagnostic<DataTag,
                                                          SourceIndexTag,
                                                          IndexTag,
                                                          DistanceTag,
                                                          TopkByBitonic,
                                                          MultiWarpMerge>>();
    };
    if (topk_by_bitonic_sort && bitonic_sort_and_merge_multi_warps) {
      add.template operator()<true, true>();
    } else if (topk_by_bitonic_sort) {
      add.template operator()<true, false>();
    } else if (bitonic_sort_and_merge_multi_warps) {
      add.template operator()<false, true>();
    } else {
      add.template operator()<false, false>();
    }
  }

  void add_search_kernel_fragment(bool topk_by_bitonic_sort,
                                  bool bitonic_sort_and_merge_multi_warps,
                                  bool persistent)
  {
    if (persistent) {
      if (topk_by_bitonic_sort && bitonic_sort_and_merge_multi_warps) {
        this->template add_static_fragment<fragment_tag_search_single_cta_p<DataTag,
                                                                            SourceIndexTag,
                                                                            IndexTag,
                                                                            DistanceTag,
                                                                            true,
                                                                            true>>();
      } else if (topk_by_bitonic_sort && !bitonic_sort_and_merge_multi_warps) {
        this->template add_static_fragment<fragment_tag_search_single_cta_p<DataTag,
                                                                            SourceIndexTag,
                                                                            IndexTag,
                                                                            DistanceTag,
                                                                            true,
                                                                            false>>();
      } else if (!topk_by_bitonic_sort && bitonic_sort_and_merge_multi_warps) {
        this->template add_static_fragment<fragment_tag_search_single_cta_p<DataTag,
                                                                            SourceIndexTag,
                                                                            IndexTag,
                                                                            DistanceTag,
                                                                            false,
                                                                            true>>();
      } else {
        this->template add_static_fragment<fragment_tag_search_single_cta_p<DataTag,
                                                                            SourceIndexTag,
                                                                            IndexTag,
                                                                            DistanceTag,
                                                                            false,
                                                                            false>>();
      }
    } else {
      if (topk_by_bitonic_sort && bitonic_sort_and_merge_multi_warps) {
        this->template add_static_fragment<fragment_tag_search_single_cta<DataTag,
                                                                          SourceIndexTag,
                                                                          IndexTag,
                                                                          DistanceTag,
                                                                          true,
                                                                          true>>();
      } else if (topk_by_bitonic_sort && !bitonic_sort_and_merge_multi_warps) {
        this->template add_static_fragment<fragment_tag_search_single_cta<DataTag,
                                                                          SourceIndexTag,
                                                                          IndexTag,
                                                                          DistanceTag,
                                                                          true,
                                                                          false>>();
      } else if (!topk_by_bitonic_sort && bitonic_sort_and_merge_multi_warps) {
        this->template add_static_fragment<fragment_tag_search_single_cta<DataTag,
                                                                          SourceIndexTag,
                                                                          IndexTag,
                                                                          DistanceTag,
                                                                          false,
                                                                          true>>();
      } else {
        this->template add_static_fragment<fragment_tag_search_single_cta<DataTag,
                                                                          SourceIndexTag,
                                                                          IndexTag,
                                                                          DistanceTag,
                                                                          false,
                                                                          false>>();
      }
    }
  }

  void add_favor_search_kernel_fragment(bool topk_by_bitonic_sort,
                                        bool bitonic_sort_and_merge_multi_warps,
                                        bool diagnostics = false)
  {
    auto add = [&]<bool TopkByBitonic, bool MultiWarpMerge>() {
      if (diagnostics) {
        this->template add_static_fragment<
          fragment_tag_search_single_cta_favor_diagnostic<DataTag,
                                                          SourceIndexTag,
                                                          IndexTag,
                                                          DistanceTag,
                                                          TopkByBitonic,
                                                          MultiWarpMerge>>();
      } else {
        this->template add_static_fragment<fragment_tag_search_single_cta_favor<DataTag,
                                                                                SourceIndexTag,
                                                                                IndexTag,
                                                                                DistanceTag,
                                                                                TopkByBitonic,
                                                                                MultiWarpMerge>>();
      }
    };
    if (topk_by_bitonic_sort && bitonic_sort_and_merge_multi_warps) {
      add.template operator()<true, true>();
    } else if (topk_by_bitonic_sort) {
      add.template operator()<true, false>();
    } else if (bitonic_sort_and_merge_multi_warps) {
      add.template operator()<false, true>();
    } else {
      add.template operator()<false, false>();
    }
  }
};

}  // namespace cuvs::neighbors::cagra::detail::single_cta_search
