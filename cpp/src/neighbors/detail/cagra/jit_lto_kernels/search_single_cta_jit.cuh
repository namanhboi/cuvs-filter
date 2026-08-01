/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

// Device-only helpers - split out to avoid pulling host launcher code into JIT translation units
#include "search_single_cta_device_helpers.cuh"

// neighbors_device_intrinsics / memory_ops come via search_single_cta_device_helpers.cuh
#include "../hashmap.hpp"
#include "../topk_by_radix.cuh"
#include "../utils.hpp"

#include <raft/core/operators.hpp>      // For raft::shfl_xor
#include <raft/util/integer_utils.hpp>  // For raft::round_up_safe
#include <raft/util/pow2_utils.cuh>

#include <cub/warp/warp_scan.cuh>

#include <cuda/atomic>
#include <cuda/std/atomic>

#include <cassert>  // For assert()

#ifdef _CLK_BREAKDOWN
#include <cstdio>  // For printf() in debug mode
#endif

// Include extern function declarations before namespace so they're available to kernel definitions
#include "cagra_filter_payload.cuh"
#include "extern_device_functions.cuh"
// Include shared JIT device functions
#include "device_common_jit.cuh"
#include "favor_search_diagnostics.cuh"

namespace cuvs::neighbors::cagra::detail::single_cta_search {

// Sample filter extern function
// sample_filter is declared in extern_device_functions.cuh
using cuvs::neighbors::detail::sample_filter;

// JIT versions of compute_distance_to_random_nodes and compute_distance_to_child_nodes
// are now shared in device_common_jit.cuh - use fully qualified names
using cuvs::neighbors::cagra::detail::device::compute_distance_to_child_nodes_jit;
using cuvs::neighbors::cagra::detail::device::compute_distance_to_random_nodes_jit;
using cuvs::neighbors::cagra::detail::device::compute_favor_distance_to_child_nodes_jit;
using cuvs::neighbors::cagra::detail::device::compute_favor_distance_to_random_nodes_jit;
using cuvs::neighbors::cagra::detail::device::
  compute_favor_retention_safe_distance_to_child_nodes_jit;

// JIT search_core - setup_workspace/compute_distance via function pointers
template <bool FAVOR,
          bool TOPK_BY_BITONIC_SORT,
          bool BITONIC_SORT_AND_MERGE_MULTI_WARPS,
          typename DataT,
          typename IndexT,
          typename DistanceT,
          typename SourceIndexT,
          bool DIAGNOSTICS = false>
RAFT_DEVICE_INLINE_FUNCTION void search_core(
  uintptr_t result_indices_ptr,
  DistanceT* const result_distances_ptr,
  const std::uint32_t top_k,
  const DataT* const queries_ptr,
  const IndexT* const knn_graph,
  const std::uint32_t graph_degree,
  const SourceIndexT* source_indices_ptr,
  const unsigned num_distilation,
  const uint64_t rand_xor_mask,
  const IndexT* seed_ptr,
  const uint32_t num_seeds,
  IndexT* const visited_hashmap_ptr,
  const std::uint32_t max_candidates,
  const std::uint32_t max_itopk,
  const std::uint32_t internal_topk,
  const std::uint32_t search_width,
  const std::uint32_t min_iteration,
  const std::uint32_t max_iteration,
  std::uint32_t* const num_executed_iterations,
  const std::uint32_t hash_bitlen,
  const std::uint32_t small_hash_bitlen,
  const std::uint32_t small_hash_reset_interval,
  const std::uint32_t query_id,
  const std::uint32_t query_id_offset,  // Offset to add to query_id when calling filter
  const dataset_descriptor_base_t<DataT, IndexT, DistanceT>* dataset_desc,
  cagra_sample_filter<SourceIndexT> filter_payload,
  const float filtering_rate                   = 0.0f,
  const float favor_penalty_distance           = 0.0f,
  const std::uint32_t favor_penalty_mode_value = 0,
  const float favor_local_gap_multiplier       = 0.0f,
  const float favor_retention_fraction         = 0.5f,
  const IndexT graph_size                      = 0,
  favor_search_diagnostics::context* const diagnostic_context = nullptr,
  const std::uint32_t favor_adaptive_start_iteration = 0,
  const std::uint32_t favor_adaptive_prefix_size     = 0)
{
  using LOAD_T = device::LOAD_128BIT_T;

  // The launcher privately uses the high bit to request the same rejected-parent retirement
  // performed by default filtered CAGRA. Keep the public penalty-mode enum and the JIT kernel
  // signature unchanged, and mask the transport bit before interpreting the penalty mode.
  constexpr std::uint32_t favor_retire_rejected_parent_mask = std::uint32_t{1} << 31;
  const auto favor_penalty_mode =
    favor_penalty_mode_value & ~favor_retire_rejected_parent_mask;
  const bool retire_rejected_parents =
    !FAVOR || ((favor_penalty_mode_value & favor_retire_rejected_parent_mask) != 0);

  cuvs::neighbors::detail::bitset_filter_data_t<SourceIndexT> favor_bitset{};
  if constexpr (FAVOR) {
    if (favor_penalty_mode == 2 && filter_payload.filter_data != nullptr) {
      // The retention-safe path is bitset-only. Hoist its three metadata fields out of the
      // iteration loop so candidate teams do not reload the payload for every graph expansion.
      favor_bitset =
        *static_cast<const cuvs::neighbors::detail::bitset_filter_data_t<SourceIndexT>*>(
          filter_payload.filter_data);
    }
  }

  auto to_source_index = [source_indices_ptr](IndexT x) {
    return source_indices_ptr == nullptr ? static_cast<SourceIndexT>(x) : source_indices_ptr[x];
  };

  constexpr IndexT index_msb_1_mask = utils::gen_index_msb_1_mask<IndexT>::value;
  const IndexT invalid_index        = utils::get_max_value<IndexT>();

  favor_search_diagnostics::query_summary* diagnostic_summary = nullptr;
  std::int32_t diagnostic_trace_slot                            = -1;
  const auto diagnostic_query_id = query_id + query_id_offset;
  if constexpr (DIAGNOSTICS) {
    if (diagnostic_context != nullptr && diagnostic_query_id < diagnostic_context->num_queries) {
      diagnostic_summary = diagnostic_context->summaries + diagnostic_query_id;
      diagnostic_trace_slot = diagnostic_context->trace_slot_by_query == nullptr
                                ? -1
                                : diagnostic_context->trace_slot_by_query[diagnostic_query_id];
      if (threadIdx.x == 0) {
        *diagnostic_summary = {};
        diagnostic_summary->schema                  = favor_search_diagnostics::schema_version;
        diagnostic_summary->query_id                = diagnostic_query_id;
        diagnostic_summary->resolved_max_iterations = max_iteration;
        diagnostic_summary->hash_bitlen             = hash_bitlen;
        diagnostic_summary->small_hash_bitlen       = small_hash_bitlen;
        diagnostic_summary->small_hash_reset_interval = small_hash_reset_interval;
        diagnostic_summary->best_unexpanded_distance = raft::upper_bound<float>();
        diagnostic_summary->worst_retained_distance  = raft::upper_bound<float>();
        diagnostic_summary->kth_passing_raw_distance = raft::upper_bound<float>();
        for (std::uint32_t rank = 0; rank < favor_search_diagnostics::ground_truth_k; ++rank) {
          diagnostic_summary->gt_first_iteration[rank] =
            favor_search_diagnostics::invalid_iteration;
        }
      }
    }
  }

#ifdef _CLK_BREAKDOWN
  std::uint64_t clk_init                 = 0;
  std::uint64_t clk_compute_1st_distance = 0;
  std::uint64_t clk_topk                 = 0;
  std::uint64_t clk_reset_hash           = 0;
  std::uint64_t clk_pickup_parents       = 0;
  std::uint64_t clk_restore_hash         = 0;
  std::uint64_t clk_compute_distance     = 0;
  std::uint64_t clk_start;
#define _CLK_START() clk_start = clock64()
#define _CLK_REC(V)  V += clock64() - clk_start;
#else
#define _CLK_START()
#define _CLK_REC(V)
#endif
  _CLK_START();

  extern __shared__ uint8_t smem[];

  // Layout of result_buffer
  const auto result_buffer_size    = internal_topk + (search_width * graph_degree);
  const auto result_buffer_size_32 = raft::round_up_safe<uint32_t>(result_buffer_size, 32);
  const auto small_hash_size       = hashmap::get_size(small_hash_bitlen);

  // Get dim and smem_ws_size directly from base descriptor
  uint32_t dim                   = dataset_desc->args.dim;
  uint32_t smem_ws_size_in_bytes = dataset_desc->smem_ws_size_in_bytes();

  auto smem_desc =
    setup_workspace<DataT, IndexT, DistanceT>(dataset_desc, smem, queries_ptr, query_id);

  auto* __restrict__ result_indices_buffer =
    reinterpret_cast<IndexT*>(smem + smem_ws_size_in_bytes);
  auto* __restrict__ result_distances_buffer =
    reinterpret_cast<DistanceT*>(result_indices_buffer + result_buffer_size_32);
  auto* __restrict__ visited_hash_buffer =
    reinterpret_cast<IndexT*>(result_distances_buffer + result_buffer_size_32);
  auto* __restrict__ parent_list_buffer =
    reinterpret_cast<IndexT*>(visited_hash_buffer + small_hash_size);
  auto* __restrict__ topk_ws = reinterpret_cast<std::uint32_t*>(parent_list_buffer + search_width);
  auto* terminate_flag       = reinterpret_cast<std::uint32_t*>(topk_ws + 3);
  DistanceT* favor_penalty   = nullptr;
  DistanceT* favor_cutoff    = nullptr;
  std::uint32_t* smem_work_ptr;
  if constexpr (FAVOR) {
    favor_penalty = reinterpret_cast<DistanceT*>(terminate_flag + 1);
    favor_cutoff  = favor_penalty + 1;
    smem_work_ptr = reinterpret_cast<std::uint32_t*>(favor_cutoff + 1);
  } else {
    smem_work_ptr = reinterpret_cast<std::uint32_t*>(terminate_flag + 1);
  }

  // A flag for filtering.
  auto filter_flag = terminate_flag;

  if (threadIdx.x == 0) {
    terminate_flag[0] = 0;
    topk_ws[0]        = ~0u;
    if constexpr (FAVOR) {
      favor_penalty[0] = favor_penalty_mode == 0
                           ? static_cast<DistanceT>(favor_penalty_distance)
                           : DistanceT{0};
      favor_cutoff[0]  = raft::upper_bound<DistanceT>();
    }
  }
  // Init hashmap
  IndexT* local_visited_hashmap_ptr;
  if (small_hash_bitlen) {
    local_visited_hashmap_ptr = visited_hash_buffer;
  } else {
    local_visited_hashmap_ptr = visited_hashmap_ptr + (hashmap::get_size(hash_bitlen) * blockIdx.y);
  }
  hashmap::init(local_visited_hashmap_ptr, hash_bitlen, 0);
  __syncthreads();
  _CLK_REC(clk_init);

  // compute distance to randomly selecting nodes using JIT version
  _CLK_START();
  const IndexT* const local_seed_ptr = seed_ptr ? seed_ptr + (num_seeds * query_id) : nullptr;
  // Get dataset_size directly from base descriptor
  IndexT dataset_size = smem_desc->size;
  if constexpr (FAVOR) {
    if (favor_penalty_mode == 0) {
      compute_favor_distance_to_random_nodes_jit<IndexT, DistanceT, DataT, SourceIndexT>(
        result_indices_buffer,
        result_distances_buffer,
        smem_desc,
        result_buffer_size,
        num_distilation,
        rand_xor_mask,
        local_seed_ptr,
        num_seeds,
        local_visited_hashmap_ptr,
        hash_bitlen,
        graph_size,
        source_indices_ptr,
        query_id + query_id_offset,
        filter_payload,
        favor_penalty[0]);
    } else {
      // Query-local modes intentionally start with an unpenalized ordering. Avoid bitset checks
      // whose zero penalty cannot change that ordering.
      compute_distance_to_random_nodes_jit<IndexT, DistanceT, DataT>(result_indices_buffer,
                                                                     result_distances_buffer,
                                                                     smem_desc,
                                                                     result_buffer_size,
                                                                     num_distilation,
                                                                     rand_xor_mask,
                                                                     local_seed_ptr,
                                                                     num_seeds,
                                                                     local_visited_hashmap_ptr,
                                                                     hash_bitlen,
                                                                     (IndexT*)nullptr,
                                                                     0,
                                                                     0,
                                                                     1,
                                                                     graph_size);
    }
  } else {
    compute_distance_to_random_nodes_jit<IndexT, DistanceT, DataT>(result_indices_buffer,
                                                                   result_distances_buffer,
                                                                   smem_desc,
                                                                   result_buffer_size,
                                                                   num_distilation,
                                                                   rand_xor_mask,
                                                                   local_seed_ptr,
                                                                   num_seeds,
                                                                   local_visited_hashmap_ptr,
                                                                   hash_bitlen,
                                                                   (IndexT*)nullptr,
                                                                   0,
                                                                   0,
                                                                   1,
                                                                   graph_size);
  }
  __syncthreads();
  _CLK_REC(clk_compute_1st_distance);

  std::uint32_t iter = 0;
  while (1) {
    // sort
    if constexpr (TOPK_BY_BITONIC_SORT) {
      assert(blockDim.x >= 64);
      const bool bitonic_sort_and_full_multi_warps = (max_candidates > 128) ? true : false;

      // reset small-hash table.
      if ((iter + 1) % small_hash_reset_interval == 0) {
        _CLK_START();
        unsigned hash_start_tid;
        if (blockDim.x == 32) {
          hash_start_tid = 0;
        } else if (blockDim.x == 64) {
          if (bitonic_sort_and_full_multi_warps || BITONIC_SORT_AND_MERGE_MULTI_WARPS) {
            hash_start_tid = 0;
          } else {
            hash_start_tid = 32;
          }
        } else {
          if (bitonic_sort_and_full_multi_warps || BITONIC_SORT_AND_MERGE_MULTI_WARPS) {
            hash_start_tid = 64;
          } else {
            hash_start_tid = 32;
          }
        }
        hashmap::init(local_visited_hashmap_ptr, hash_bitlen, hash_start_tid);
        _CLK_REC(clk_reset_hash);
      }

      // topk with bitonic sort
      _CLK_START();
      // Default filtering and sparse automatic FAVOR retirement may invalidate expanded parents
      // between top-k updates. Restore the sorted internal-top-k invariant before merging.
      if (retire_rejected_parents && *filter_flag != 0) {
        for (unsigned i = 0; i < search_width; i++) {
          move_invalid_to_end_of_list(
            result_indices_buffer, result_distances_buffer, internal_topk);
        }
        if (threadIdx.x == 0) { *terminate_flag = 0; }
      }
      topk_by_bitonic_sort_and_merge<BITONIC_SORT_AND_MERGE_MULTI_WARPS>(
        result_distances_buffer,
        result_indices_buffer,
        max_itopk,
        internal_topk,
        result_distances_buffer + internal_topk,
        result_indices_buffer + internal_topk,
        max_candidates,
        search_width * graph_degree,
        topk_ws,
        (iter == 0));
      __syncthreads();
      _CLK_REC(clk_topk);
    } else {
      _CLK_START();
      // topk with radix block sort
      topk_by_radix_sort<IndexT>{}(max_itopk,
                                   internal_topk,
                                   result_buffer_size,
                                   reinterpret_cast<std::uint32_t*>(result_distances_buffer),
                                   result_indices_buffer,
                                   reinterpret_cast<std::uint32_t*>(result_distances_buffer),
                                   result_indices_buffer,
                                   nullptr,
                                   topk_ws,
                                   true,
                                   smem_work_ptr);
      _CLK_REC(clk_topk);

      // reset small-hash table
      if ((iter + 1) % small_hash_reset_interval == 0) {
        _CLK_START();
        hashmap::init(local_visited_hashmap_ptr, hash_bitlen);
        _CLK_REC(clk_reset_hash);
      }
    }
    __syncthreads();

    if constexpr (DIAGNOSTICS) {
      // Mark which candidates from the previous expansion survived this merge. This deliberately
      // runs only for the bounded selected-query trace, never for the 10k-query summary path.
      if (diagnostic_trace_slot >= 0 && iter > 0 &&
          iter <= diagnostic_context->max_trace_iterations) {
        const auto candidates_per_iteration = diagnostic_context->candidates_per_iteration;
        auto* previous = diagnostic_context->candidate_records +
                         (static_cast<std::uint64_t>(diagnostic_trace_slot) *
                            diagnostic_context->max_trace_iterations +
                          (iter - 1)) *
                           candidates_per_iteration;
        for (std::uint32_t j = threadIdx.x; j < candidates_per_iteration; j += blockDim.x) {
          if (!previous[j].valid || previous[j].child_id == 0xffffffffu) { continue; }
          const auto child = static_cast<IndexT>(previous[j].child_id);
          for (std::uint32_t i = 0; i < internal_topk; ++i) {
            if ((result_indices_buffer[i] & ~index_msb_1_mask) == child) {
              previous[j].survived_next_merge = 1;
              break;
            }
          }
        }
      }
      __syncthreads();
    }

    if constexpr (FAVOR) {
      if (threadIdx.x == 0) {
        if (iter == 0 && favor_penalty_mode != 0) {
          favor_penalty[0] =
            device::favor_query_local_penalty(result_distances_buffer,
                                              internal_topk,
                                              static_cast<DistanceT>(favor_penalty_distance),
                                              static_cast<DistanceT>(favor_local_gap_multiplier));
        }
        if (favor_penalty_mode == 2) {
          favor_cutoff[0] = device::favor_retention_cutoff(result_distances_buffer, internal_topk);
        }
      }
    }

    if constexpr (DIAGNOSTICS) {
      __syncthreads();
      if (threadIdx.x == 0 && diagnostic_summary != nullptr) {
        std::uint32_t valid = 0, passing = 0, rejected = 0;
        std::uint32_t unexpanded_passing = 0, unexpanded_rejected = 0;
        float best_unexpanded = raft::upper_bound<float>();
        float worst_retained  = -raft::upper_bound<float>();
        for (std::uint32_t i = 0; i < internal_topk; ++i) {
          const auto tagged = result_indices_buffer[i];
          const auto node   = tagged & ~index_msb_1_mask;
          if (node == (invalid_index & ~index_msb_1_mask)) { continue; }
          ++valid;
          const bool pass = sample_filter<SourceIndexT>(query_id + query_id_offset,
                                                        to_source_index(node),
                                                        filter_payload.sample_filter_data());
          passing += pass;
          rejected += !pass;
          if ((tagged & index_msb_1_mask) == 0) {
            unexpanded_passing += pass;
            unexpanded_rejected += !pass;
            best_unexpanded = min(best_unexpanded, static_cast<float>(result_distances_buffer[i]));
          }
          worst_retained = max(worst_retained, static_cast<float>(result_distances_buffer[i]));

          if (diagnostic_context->ground_truth_ids != nullptr) {
            const auto source = static_cast<std::uint32_t>(to_source_index(node));
            const auto* gt = diagnostic_context->ground_truth_ids +
                             static_cast<std::uint64_t>(diagnostic_query_id) *
                               favor_search_diagnostics::ground_truth_k;
            for (std::uint32_t rank = 0; rank < favor_search_diagnostics::ground_truth_k; ++rank) {
              if (source == gt[rank]) {
                diagnostic_summary->gt_seen_mask |= (1u << rank);
                if (diagnostic_summary->gt_first_iteration[rank] ==
                    favor_search_diagnostics::invalid_iteration) {
                  diagnostic_summary->gt_first_iteration[rank] = iter;
                }
              }
            }
          }
        }
        diagnostic_summary->terminal_valid             = valid;
        diagnostic_summary->terminal_pass              = passing;
        diagnostic_summary->terminal_reject            = rejected;
        diagnostic_summary->terminal_unexpanded_pass   = unexpanded_passing;
        diagnostic_summary->terminal_unexpanded_reject = unexpanded_rejected;
        diagnostic_summary->query_penalty              = static_cast<float>(favor_penalty[0]);
        diagnostic_summary->terminal_cutoff             = static_cast<float>(favor_cutoff[0]);
        diagnostic_summary->best_unexpanded_distance    = best_unexpanded;
        diagnostic_summary->worst_retained_distance     = valid ? worst_retained
                                                                 : raft::upper_bound<float>();

        if (diagnostic_trace_slot >= 0 && iter < diagnostic_context->max_trace_iterations) {
          auto* record = diagnostic_context->iteration_records +
                         static_cast<std::uint64_t>(diagnostic_trace_slot) *
                           diagnostic_context->max_trace_iterations +
                         iter;
          *record                        = {};
          record->query_id               = diagnostic_query_id;
          record->iteration              = iter;
          record->valid                  = valid;
          record->passing                = passing;
          record->rejected               = rejected;
          record->unexpanded_passing     = unexpanded_passing;
          record->unexpanded_rejected    = unexpanded_rejected;
          record->penalty                = static_cast<float>(favor_penalty[0]);
          record->cutoff                 = static_cast<float>(favor_cutoff[0]);
          record->best_unexpanded_distance = best_unexpanded;
          record->worst_retained_distance  = diagnostic_summary->worst_retained_distance;
        }
      }
      __syncthreads();
    }

    if (iter + 1 == max_iteration) {
      if constexpr (DIAGNOSTICS) {
        if (threadIdx.x == 0 && diagnostic_summary != nullptr) {
          const bool has_unexpanded = diagnostic_summary->terminal_unexpanded_pass != 0 ||
                                      diagnostic_summary->terminal_unexpanded_reject != 0;
          diagnostic_summary->reason = static_cast<std::uint32_t>(
            favor_adaptive_prefix_size != 0
              ? favor_search_diagnostics::stop_reason::adaptive_safety_cap
              : (has_unexpanded
                   ? favor_search_diagnostics::stop_reason::max_with_unexpanded_frontier
                   : favor_search_diagnostics::stop_reason::max_with_empty_frontier));
          if (diagnostic_trace_slot >= 0 && iter < diagnostic_context->max_trace_iterations) {
            diagnostic_context
              ->iteration_records[static_cast<std::uint64_t>(diagnostic_trace_slot) *
                                    diagnostic_context->max_trace_iterations +
                                  iter]
              .stop_reason = diagnostic_summary->reason;
          }
        }
      }
      break;
    }

    // pick up next parents
    if (threadIdx.x < 32) {
      _CLK_START();
      pickup_next_parents<TOPK_BY_BITONIC_SORT, IndexT>(
        terminate_flag, parent_list_buffer, result_indices_buffer, internal_topk, search_width);
      _CLK_REC(clk_pickup_parents);
    }

    // Sparse automatic FAVOR uses the first selected parent as the pre-pickup minimum of the
    // unexpanded frontier. Parent pickup only tags its index MSB; its distance and filter result
    // are unchanged, so the existing pickup and synchronization can be reused.
    if constexpr (FAVOR) {
      if (favor_adaptive_prefix_size != 0 &&
          iter + 1 >= favor_adaptive_start_iteration && threadIdx.x < 32 &&
          *terminate_flag == 0) {
        const auto lane = static_cast<std::uint32_t>(threadIdx.x);
        std::uint32_t valid_prefix = 0;
        std::uint32_t passing_prefix = 0;
        for (std::uint32_t base = 0; base < favor_adaptive_prefix_size; base += 32) {
          const auto rank = base + lane;
          bool valid      = false;
          bool passing    = false;
          if (rank < favor_adaptive_prefix_size && rank < internal_topk) {
            const auto pos = TOPK_BY_BITONIC_SORT ? device::swizzling(rank) : rank;
            const auto node = result_indices_buffer[pos] & ~index_msb_1_mask;
            valid = node != (invalid_index & ~index_msb_1_mask);
            if (valid) {
              passing = device::favor_bitset_test(favor_bitset, to_source_index(node));
            }
          }
          const auto valid_mask = __ballot_sync(0xffffffffu, valid);
          const auto pass_mask  = __ballot_sync(0xffffffffu, passing);
          if (lane == 0) {
            valid_prefix += __popc(valid_mask);
            passing_prefix += __popc(pass_mask);
          }
        }
        if (lane == 0) {
          const auto frontier_pos = parent_list_buffer[0];
          const auto boundary_rank = favor_adaptive_prefix_size - 1;
          const auto boundary_pos =
            TOPK_BY_BITONIC_SORT ? device::swizzling(boundary_rank) : boundary_rank;
          const auto required_passing = (favor_adaptive_prefix_size + 1) / 2;
          if (frontier_pos != invalid_index && valid_prefix == favor_adaptive_prefix_size &&
              passing_prefix >= required_passing &&
              result_distances_buffer[frontier_pos] > result_distances_buffer[boundary_pos]) {
            *terminate_flag = 2;
          }
        }
      }
    }

    if constexpr (DIAGNOSTICS) {
      __syncthreads();
      if (threadIdx.x == 0 && diagnostic_summary != nullptr) {
        std::uint32_t selected_pass = 0, selected_reject = 0;
        if (*terminate_flag != 2) {
          for (std::uint32_t p = 0; p < search_width; ++p) {
            if (parent_list_buffer[p] == invalid_index) { continue; }
            const auto node = result_indices_buffer[parent_list_buffer[p]] & ~index_msb_1_mask;
            const bool pass = sample_filter<SourceIndexT>(query_id + query_id_offset,
                                                          to_source_index(node),
                                                          filter_payload.sample_filter_data());
            selected_pass += pass;
            selected_reject += !pass;
          }
        }
        diagnostic_summary->expanded_pass_parents += selected_pass;
        diagnostic_summary->expanded_reject_parents += selected_reject;
        if (diagnostic_trace_slot >= 0 && iter < diagnostic_context->max_trace_iterations) {
          auto& record = diagnostic_context
                           ->iteration_records[static_cast<std::uint64_t>(diagnostic_trace_slot) *
                                                 diagnostic_context->max_trace_iterations +
                                               iter];
          record.selected_passing_parents  = selected_pass;
          record.selected_rejected_parents = selected_reject;
        }
      }
      __syncthreads();
    }

    // restore small-hash table by putting internal-topk indices in it
    if ((iter + 1) % small_hash_reset_interval == 0) {
      const unsigned first_tid = ((blockDim.x <= 32) ? 0 : 32);
      _CLK_START();
      hashmap_restore(
        local_visited_hashmap_ptr, hash_bitlen, result_indices_buffer, internal_topk, first_tid);
      _CLK_REC(clk_restore_hash);
    }
    __syncthreads();

    if (*terminate_flag && iter >= min_iteration) {
      if constexpr (DIAGNOSTICS) {
        if (threadIdx.x == 0 && diagnostic_summary != nullptr) {
          diagnostic_summary->reason = static_cast<std::uint32_t>(
            *terminate_flag == 2 ? favor_search_diagnostics::stop_reason::adaptive_converged
                                 : favor_search_diagnostics::stop_reason::frontier_exhausted);
          if (diagnostic_trace_slot >= 0 && iter < diagnostic_context->max_trace_iterations) {
            diagnostic_context
              ->iteration_records[static_cast<std::uint64_t>(diagnostic_trace_slot) *
                                    diagnostic_context->max_trace_iterations +
                                  iter]
              .stop_reason = diagnostic_summary->reason;
          }
        }
      }
      break;
    }

    __syncthreads();
    // compute the norms between child nodes and query node using JIT version
    _CLK_START();
    if constexpr (FAVOR) {
      if (favor_penalty_mode == 2) {
        const bool packed_bitset = favor_bitset.original_nbits == 0 ||
                                   favor_bitset.original_nbits == sizeof(std::uint32_t) * 8 ||
                                   favor_bitset.original_nbits == favor_bitset.bitset_len;
        if (packed_bitset && source_indices_ptr == nullptr) {
          compute_favor_retention_safe_distance_to_child_nodes_jit<IndexT,
                                                                   DistanceT,
                                                                   DataT,
                                                                   SourceIndexT,
                                                                   1,
                                                                   true,
                                                                   true,
                                                                   DIAGNOSTICS>(
            result_indices_buffer + internal_topk,
            result_distances_buffer + internal_topk,
            smem_desc,
            knn_graph,
            graph_degree,
            local_visited_hashmap_ptr,
            hash_bitlen,
            parent_list_buffer,
            result_indices_buffer,
            search_width,
            source_indices_ptr,
            favor_bitset,
            favor_penalty[0],
            favor_cutoff[0],
            static_cast<DistanceT>(favor_retention_fraction),
            nullptr,
            0,
            nullptr,
            0,
            reinterpret_cast<std::uint8_t*>(smem_work_ptr));
        } else if (packed_bitset) {
          compute_favor_retention_safe_distance_to_child_nodes_jit<IndexT,
                                                                   DistanceT,
                                                                   DataT,
                                                                   SourceIndexT,
                                                                   1,
                                                                   true,
                                                                   false,
                                                                   DIAGNOSTICS>(
            result_indices_buffer + internal_topk,
            result_distances_buffer + internal_topk,
            smem_desc,
            knn_graph,
            graph_degree,
            local_visited_hashmap_ptr,
            hash_bitlen,
            parent_list_buffer,
            result_indices_buffer,
            search_width,
            source_indices_ptr,
            favor_bitset,
            favor_penalty[0],
            favor_cutoff[0],
            static_cast<DistanceT>(favor_retention_fraction),
            nullptr,
            0,
            nullptr,
            0,
            reinterpret_cast<std::uint8_t*>(smem_work_ptr));
        } else {
          compute_favor_retention_safe_distance_to_child_nodes_jit<IndexT,
                                                                   DistanceT,
                                                                   DataT,
                                                                   SourceIndexT,
                                                                   1,
                                                                   false,
                                                                   false,
                                                                   DIAGNOSTICS>(
            result_indices_buffer + internal_topk,
            result_distances_buffer + internal_topk,
            smem_desc,
            knn_graph,
            graph_degree,
            local_visited_hashmap_ptr,
            hash_bitlen,
            parent_list_buffer,
            result_indices_buffer,
            search_width,
            source_indices_ptr,
            favor_bitset,
            favor_penalty[0],
            favor_cutoff[0],
            static_cast<DistanceT>(favor_retention_fraction),
            nullptr,
            0,
            nullptr,
            0,
            reinterpret_cast<std::uint8_t*>(smem_work_ptr));
        }
      } else {
        compute_favor_distance_to_child_nodes_jit<IndexT, DistanceT, DataT, SourceIndexT>(
          result_indices_buffer + internal_topk,
          result_distances_buffer + internal_topk,
          smem_desc,
          knn_graph,
          graph_degree,
          local_visited_hashmap_ptr,
          hash_bitlen,
          parent_list_buffer,
          result_indices_buffer,
          search_width,
          source_indices_ptr,
          query_id + query_id_offset,
          filter_payload,
          favor_penalty[0],
          favor_cutoff[0],
          false);
      }
    } else {
      compute_distance_to_child_nodes_jit<IndexT, DistanceT, DataT>(
        result_indices_buffer + internal_topk,
        result_distances_buffer + internal_topk,
        smem_desc,
        knn_graph,
        graph_degree,
        local_visited_hashmap_ptr,
        hash_bitlen,
        (IndexT*)nullptr,
        0u,
        parent_list_buffer,
        result_indices_buffer,
        search_width);
    }
    // Critical: __syncthreads() must be reached by ALL threads
    // If any thread is stuck in compute_distance_to_child_nodes_jit, this will hang
    __syncthreads();
    _CLK_REC(clk_compute_distance);

    if constexpr (DIAGNOSTICS) {
      if (threadIdx.x == 0 && diagnostic_summary != nullptr) {
        const auto num_children = search_width * graph_degree;
        std::uint32_t attempts = 0, evaluations = 0, duplicates = 0, hash_full = 0;
        std::uint32_t passing = 0, rejected = 0, penalized = 0;
        for (std::uint32_t p = 0; p < search_width; ++p) {
          attempts += parent_list_buffer[p] == invalid_index ? 0 : graph_degree;
        }
        for (std::uint32_t j = 0; j < num_children; ++j) {
          const auto hash_result = reinterpret_cast<std::uint8_t*>(smem_work_ptr)[j];
          duplicates += hash_result == static_cast<std::uint8_t>(hashmap::insert_outcome::duplicate);
          hash_full += hash_result == static_cast<std::uint8_t>(hashmap::insert_outcome::full);
          favor_search_diagnostics::candidate_record* candidate = nullptr;
          if (diagnostic_trace_slot >= 0 && iter < diagnostic_context->max_trace_iterations &&
              j < diagnostic_context->candidates_per_iteration) {
            candidate = diagnostic_context->candidate_records +
                        (static_cast<std::uint64_t>(diagnostic_trace_slot) *
                           diagnostic_context->max_trace_iterations +
                         iter) *
                          diagnostic_context->candidates_per_iteration +
                        j;
            *candidate               = {};
            candidate->query_id      = diagnostic_query_id;
            candidate->iteration     = iter;
            candidate->hash_result   = hash_result;
            candidate->valid         = hash_result != 0;
            candidate->ground_truth_rank = -1;
            const auto parent_number = j / graph_degree;
            if (parent_number < search_width && parent_list_buffer[parent_number] != invalid_index) {
              candidate->parent_id = static_cast<std::uint32_t>(
                result_indices_buffer[parent_list_buffer[parent_number]] & ~index_msb_1_mask);
            }
          }
          const auto child = result_indices_buffer[internal_topk + j] & ~index_msb_1_mask;
          if (child == (invalid_index & ~index_msb_1_mask)) { continue; }
          ++evaluations;
          const auto source = to_source_index(child);
          const bool pass   = sample_filter<SourceIndexT>(query_id + query_id_offset,
                                                        source,
                                                        filter_payload.sample_filter_data());
          passing += pass;
          rejected += !pass;

          const float final_distance = static_cast<float>(result_distances_buffer[internal_topk + j]);
          float raw_distance         = final_distance;
          float effective_penalty    = 0.0f;
          if (!pass && favor_penalty_mode == 2 && favor_penalty[0] > DistanceT{0} &&
              final_distance < raft::upper_bound<float>()) {
            const float penalty = static_cast<float>(favor_penalty[0]);
            const float cutoff  = static_cast<float>(favor_cutoff[0]);
            const float rho     = favor_retention_fraction;
            const float raw_if_full_penalty = final_distance - penalty;
            if (raw_if_full_penalty < cutoff &&
                penalty <= rho * (cutoff - raw_if_full_penalty)) {
              raw_distance      = raw_if_full_penalty;
              effective_penalty = penalty;
            } else if (rho < 1.0f) {
              raw_distance = (final_distance - rho * cutoff) / (1.0f - rho);
              effective_penalty = final_distance - raw_distance;
            }
            penalized += effective_penalty > 0.0f;
          }

          if (candidate != nullptr) {
            candidate->child_id          = static_cast<std::uint32_t>(child);
            candidate->raw_distance      = raw_distance;
            candidate->effective_penalty = effective_penalty;
            candidate->final_distance    = final_distance;
            candidate->passes_filter     = pass;
            if (diagnostic_context->ground_truth_ids != nullptr) {
              const auto* gt = diagnostic_context->ground_truth_ids +
                               static_cast<std::uint64_t>(diagnostic_query_id) *
                                 favor_search_diagnostics::ground_truth_k;
              for (std::uint32_t rank = 0; rank < favor_search_diagnostics::ground_truth_k; ++rank) {
                if (static_cast<std::uint32_t>(source) == gt[rank]) {
                  candidate->ground_truth_rank = static_cast<std::int16_t>(rank);
                  break;
                }
              }
            }
          }
        }
        diagnostic_summary->candidate_attempts += attempts;
        diagnostic_summary->candidate_evaluations += evaluations;
        diagnostic_summary->candidate_duplicate_or_full += attempts - evaluations;
        diagnostic_summary->candidate_duplicates += duplicates;
        diagnostic_summary->candidate_hash_full += hash_full;
        diagnostic_summary->passing_candidates += passing;
        diagnostic_summary->rejected_candidates += rejected;
        diagnostic_summary->penalized_candidates += penalized;
        if (diagnostic_trace_slot >= 0 && iter < diagnostic_context->max_trace_iterations) {
          auto& record = diagnostic_context
                           ->iteration_records[static_cast<std::uint64_t>(diagnostic_trace_slot) *
                                                 diagnostic_context->max_trace_iterations +
                                               iter];
          record.child_attempts          = attempts;
          record.child_evaluations       = evaluations;
          record.child_duplicate_or_full = attempts - evaluations;
          record.child_duplicates       = duplicates;
          record.child_hash_full        = hash_full;
          record.child_passing           = passing;
          record.child_rejected          = rejected;
        }
      }
      __syncthreads();
    }

    if (retire_rejected_parents) {
      // This is exactly the default filtered-CAGRA lifecycle: score a selected parent's children,
      // then retire the parent if it does not pass the filter. Sparse automatic FAVOR reuses it so
      // expanded rejected nodes do not permanently consume the fused frontier/result buffer.
      if (threadIdx.x == 0) { *filter_flag = 0; }
      __syncthreads();

      constexpr IndexT index_msb_1_mask = utils::gen_index_msb_1_mask<IndexT>::value;
      const IndexT invalid_index        = utils::get_max_value<IndexT>();

      for (unsigned p = threadIdx.x; p < search_width; p += blockDim.x) {
        if (parent_list_buffer[p] != invalid_index) {
          const auto parent_id = result_indices_buffer[parent_list_buffer[p]] & ~index_msb_1_mask;
          if (!sample_filter<SourceIndexT>(query_id + query_id_offset,
                                           to_source_index(parent_id),
                                           filter_payload.sample_filter_data())) {
            result_distances_buffer[parent_list_buffer[p]] = utils::get_max_value<DistanceT>();
            result_indices_buffer[parent_list_buffer[p]]   = invalid_index;
            *filter_flag                                   = 1;
          }
        }
      }
      __syncthreads();
    }

    iter++;
  }

  // Post process for filtering - use extern sample_filter function
  for (unsigned i = threadIdx.x; i < internal_topk + search_width * graph_degree; i += blockDim.x) {
    const auto node_id = result_indices_buffer[i] & ~index_msb_1_mask;
    bool passes_filter = true;
    if (node_id != (invalid_index & ~index_msb_1_mask)) {
      if constexpr (FAVOR) {
        if (favor_penalty_mode == 2) {
          const auto source_id     = to_source_index(node_id);
          const bool packed_bitset = favor_bitset.original_nbits == 0 ||
                                     favor_bitset.original_nbits == sizeof(std::uint32_t) * 8 ||
                                     favor_bitset.original_nbits == favor_bitset.bitset_len;
          passes_filter = packed_bitset
                            ? device::favor_packed_bitset_test(favor_bitset.bitset_ptr, source_id)
                            : device::favor_bitset_test(favor_bitset, source_id);
        } else {
          passes_filter = sample_filter<SourceIndexT>(query_id + query_id_offset,
                                                      to_source_index(node_id),
                                                      filter_payload.sample_filter_data());
        }
      } else {
        passes_filter = sample_filter<SourceIndexT>(query_id + query_id_offset,
                                                    to_source_index(node_id),
                                                    filter_payload.sample_filter_data());
      }
    }
    if (!passes_filter) {
      result_distances_buffer[i] = utils::get_max_value<DistanceT>();
      result_indices_buffer[i]   = invalid_index;
    }
  }

  __syncthreads();
  // Move invalid index items to the end of the buffer without sorting the entire buffer
  using scan_op_t    = cub::WarpScan<unsigned>;
  auto& temp_storage = *reinterpret_cast<typename scan_op_t::TempStorage*>(smem_work_ptr);

  constexpr std::uint32_t warp_size = 32;
  if (threadIdx.x < warp_size) {
    std::uint32_t num_found_valid = 0;
    for (std::uint32_t buffer_offset = 0; buffer_offset < internal_topk;
         buffer_offset += warp_size) {
      const auto src_position = buffer_offset + threadIdx.x;
      const std::uint32_t is_valid_index =
        (result_indices_buffer[src_position] & (~index_msb_1_mask)) ==
            (invalid_index & (~index_msb_1_mask))
          ? 0
          : 1;
      std::uint32_t new_position;
      scan_op_t(temp_storage).InclusiveSum(is_valid_index, new_position);
      if (is_valid_index) {
        const auto dst_position               = num_found_valid + (new_position - 1);
        result_indices_buffer[dst_position]   = result_indices_buffer[src_position];
        result_distances_buffer[dst_position] = result_distances_buffer[src_position];
      }

      num_found_valid += new_position;
      for (std::uint32_t offset = (warp_size >> 1); offset > 0; offset >>= 1) {
        const auto v = raft::shfl_xor(num_found_valid, offset);
        if ((threadIdx.x & offset) == 0) { num_found_valid = v; }
      }

      if (num_found_valid >= top_k) { break; }
    }

    if (num_found_valid < top_k) {
      for (std::uint32_t i = num_found_valid + threadIdx.x; i < internal_topk; i += warp_size) {
        result_indices_buffer[i]   = invalid_index;
        result_distances_buffer[i] = utils::get_max_value<DistanceT>();
      }
    }
  }

  // If the sufficient number of valid indexes are not in the internal topk, pick up from the
  // candidate list.
  if (top_k > internal_topk || result_indices_buffer[top_k - 1] == invalid_index) {
    __syncthreads();
    topk_by_bitonic_sort_and_merge<BITONIC_SORT_AND_MERGE_MULTI_WARPS>(
      result_distances_buffer,
      result_indices_buffer,
      max_itopk,
      internal_topk,
      result_distances_buffer + internal_topk,
      result_indices_buffer + internal_topk,
      max_candidates,
      search_width * graph_degree,
      topk_ws,
      (iter == 0));
  }
  __syncthreads();

  if constexpr (DIAGNOSTICS) {
    if (threadIdx.x == 0 && diagnostic_summary != nullptr) {
      std::uint32_t output_count = 0;
      for (std::uint32_t i = 0; i < top_k; ++i) {
        unsigned ii = TOPK_BY_BITONIC_SORT ? device::swizzling(i) : i;
        const auto node = result_indices_buffer[ii] & ~index_msb_1_mask;
        output_count += node != (invalid_index & ~index_msb_1_mask);
      }
      diagnostic_summary->output_count = output_count;
      if (output_count >= top_k) {
        const auto ii = TOPK_BY_BITONIC_SORT ? device::swizzling(top_k - 1) : top_k - 1;
        diagnostic_summary->kth_passing_raw_distance =
          static_cast<float>(result_distances_buffer[ii]);
      }
    }
  }

  // NB: The indices pointer is tagged with its element size.
  const uint32_t index_element_tag = result_indices_ptr & 0x3;
  result_indices_ptr ^= index_element_tag;
  auto write_indices =
    index_element_tag == 3
      ? [](uintptr_t ptr,
           uint32_t i,
           SourceIndexT x) { reinterpret_cast<uint64_t*>(ptr)[i] = static_cast<uint64_t>(x); }
    : index_element_tag == 2
      ? [](uintptr_t ptr,
           uint32_t i,
           SourceIndexT x) { reinterpret_cast<uint32_t*>(ptr)[i] = static_cast<uint32_t>(x); }
    : index_element_tag == 1
      ? [](uintptr_t ptr,
           uint32_t i,
           SourceIndexT x) { reinterpret_cast<uint16_t*>(ptr)[i] = static_cast<uint16_t>(x); }
      : [](uintptr_t ptr, uint32_t i, SourceIndexT x) {
          reinterpret_cast<uint8_t*>(ptr)[i] = static_cast<uint8_t>(x);
        };
  for (std::uint32_t i = threadIdx.x; i < top_k; i += blockDim.x) {
    unsigned j  = i + (top_k * query_id);
    unsigned ii = i;
    if constexpr (TOPK_BY_BITONIC_SORT) { ii = device::swizzling(i); }
    if (result_distances_ptr != nullptr) { result_distances_ptr[j] = result_distances_buffer[ii]; }

    auto internal_index =
      result_indices_buffer[ii] & ~index_msb_1_mask;  // clear most significant bit
    auto source_index = to_source_index(internal_index);
    write_indices(result_indices_ptr, j, source_index);
  }
  if (threadIdx.x == 0) {
    if constexpr (DIAGNOSTICS) {
      if (diagnostic_summary != nullptr) { diagnostic_summary->iterations = iter + 1; }
    } else if (num_executed_iterations != nullptr) {
      num_executed_iterations[query_id] = iter + 1;
    }
  }
#ifdef _CLK_BREAKDOWN
  if ((threadIdx.x == 0 || threadIdx.x == blockDim.x - 1) && ((query_id * 3) % gridDim.y < 3)) {
    printf(
      "%s:%d "
      "query, %d, thread, %d"
      ", init, %lu"
      ", 1st_distance, %lu"
      ", topk, %lu"
      ", reset_hash, %lu"
      ", pickup_parents, %lu"
      ", restore_hash, %lu"
      ", distance, %lu"
      "\n",
      __FILE__,
      __LINE__,
      query_id,
      threadIdx.x,
      clk_init,
      clk_compute_1st_distance,
      clk_topk,
      clk_reset_hash,
      clk_pickup_parents,
      clk_restore_hash,
      clk_compute_distance);
  }
#endif
}

// JIT device implementation - called from extern "C" __global__ entry in generated .cu
template <bool TOPK_BY_BITONIC_SORT,
          bool BITONIC_SORT_AND_MERGE_MULTI_WARPS,
          typename DataT,
          typename IndexT,
          typename DistanceT,
          typename SourceIndexT>
__device__ void search_kernel_jit(
  uintptr_t result_indices_ptr,
  DistanceT* const result_distances_ptr,
  const std::uint32_t top_k,
  const DataT* const queries_ptr,
  const IndexT* const knn_graph,
  const std::uint32_t graph_degree,
  const SourceIndexT* source_indices_ptr,
  const unsigned num_distilation,
  const uint64_t rand_xor_mask,
  const IndexT* seed_ptr,
  const uint32_t num_seeds,
  IndexT* const visited_hashmap_ptr,
  const std::uint32_t max_candidates,
  const std::uint32_t max_itopk,
  const std::uint32_t internal_topk,
  const std::uint32_t search_width,
  const std::uint32_t min_iteration,
  const std::uint32_t max_iteration,
  std::uint32_t* const num_executed_iterations,
  const std::uint32_t hash_bitlen,
  const std::uint32_t small_hash_bitlen,
  const std::uint32_t small_hash_reset_interval,
  const std::uint32_t query_id_offset,  // Offset to add to query_id when calling filter
  const dataset_descriptor_base_t<DataT, IndexT, DistanceT>* dataset_desc,
  const IndexT graph_size,
  cagra_sample_filter<SourceIndexT> filter_payload)
{
  const auto query_id = blockIdx.y;
  search_core<false,
              TOPK_BY_BITONIC_SORT,
              BITONIC_SORT_AND_MERGE_MULTI_WARPS,
              DataT,
              IndexT,
              DistanceT,
              SourceIndexT>(result_indices_ptr,
                            result_distances_ptr,
                            top_k,
                            queries_ptr,
                            knn_graph,
                            graph_degree,
                            source_indices_ptr,
                            num_distilation,
                            rand_xor_mask,
                            seed_ptr,
                            num_seeds,
                            visited_hashmap_ptr,
                            max_candidates,
                            max_itopk,
                            internal_topk,
                            search_width,
                            min_iteration,
                            max_iteration,
                            num_executed_iterations,
                            hash_bitlen,
                            small_hash_bitlen,
                            small_hash_reset_interval,
                            query_id,
                            query_id_offset,
                            dataset_desc,
                            filter_payload,
                            0.0f,
                            0.0f,
                            0u,
                            0.0f,
                            0.5f,
                            graph_size);
}

template <bool TOPK_BY_BITONIC_SORT,
          bool BITONIC_SORT_AND_MERGE_MULTI_WARPS,
          typename DataT,
          typename IndexT,
          typename DistanceT,
          typename SourceIndexT>
__device__ void search_favor_kernel_jit(
  uintptr_t result_indices_ptr,
  DistanceT* const result_distances_ptr,
  const std::uint32_t top_k,
  const DataT* const queries_ptr,
  const IndexT* const knn_graph,
  const std::uint32_t graph_degree,
  const SourceIndexT* source_indices_ptr,
  const unsigned num_distilation,
  const uint64_t rand_xor_mask,
  const IndexT* seed_ptr,
  const uint32_t num_seeds,
  IndexT* const visited_hashmap_ptr,
  const std::uint32_t max_candidates,
  const std::uint32_t max_itopk,
  const std::uint32_t internal_topk,
  const std::uint32_t search_width,
  const std::uint32_t min_iteration,
  const std::uint32_t max_iteration,
  std::uint32_t* const num_executed_iterations,
  const std::uint32_t hash_bitlen,
  const std::uint32_t small_hash_bitlen,
  const std::uint32_t small_hash_reset_interval,
  const std::uint32_t query_id_offset,
  const dataset_descriptor_base_t<DataT, IndexT, DistanceT>* dataset_desc,
  const IndexT graph_size,
  cagra_sample_filter<SourceIndexT> filter_payload,
  const float filtering_rate,
  const float favor_penalty_distance,
  const std::uint32_t favor_penalty_mode_value,
  const float favor_local_gap_multiplier,
  const float favor_retention_fraction,
  const std::uint32_t favor_adaptive_start_iteration,
  const std::uint32_t favor_adaptive_prefix_size)
{
  const auto query_id = blockIdx.y;
  search_core<true,
              TOPK_BY_BITONIC_SORT,
              BITONIC_SORT_AND_MERGE_MULTI_WARPS,
              DataT,
              IndexT,
              DistanceT,
              SourceIndexT>(result_indices_ptr,
                            result_distances_ptr,
                            top_k,
                            queries_ptr,
                            knn_graph,
                            graph_degree,
                            source_indices_ptr,
                            num_distilation,
                            rand_xor_mask,
                            seed_ptr,
                            num_seeds,
                            visited_hashmap_ptr,
                            max_candidates,
                            max_itopk,
                            internal_topk,
                            search_width,
                            min_iteration,
                            max_iteration,
                            num_executed_iterations,
                            hash_bitlen,
                            small_hash_bitlen,
                            small_hash_reset_interval,
                            query_id,
                            query_id_offset,
                            dataset_desc,
                            filter_payload,
                            filtering_rate,
                            favor_penalty_distance,
                            favor_penalty_mode_value,
                            favor_local_gap_multiplier,
                            favor_retention_fraction,
                            graph_size,
                            nullptr,
                            favor_adaptive_start_iteration,
                            favor_adaptive_prefix_size);
}

/** Diagnostic FAVOR entry used only by the bench-only scoped diagnostic attachment. */
template <bool TOPK_BY_BITONIC_SORT,
          bool BITONIC_SORT_AND_MERGE_MULTI_WARPS,
          typename DataT,
          typename IndexT,
          typename DistanceT,
          typename SourceIndexT>
__device__ void search_favor_diagnostic_kernel_jit(
  uintptr_t result_indices_ptr,
  DistanceT* const result_distances_ptr,
  const std::uint32_t top_k,
  const DataT* const queries_ptr,
  const IndexT* const knn_graph,
  const std::uint32_t graph_degree,
  const SourceIndexT* source_indices_ptr,
  const unsigned num_distilation,
  const uint64_t rand_xor_mask,
  const IndexT* seed_ptr,
  const uint32_t num_seeds,
  IndexT* const visited_hashmap_ptr,
  const std::uint32_t max_candidates,
  const std::uint32_t max_itopk,
  const std::uint32_t internal_topk,
  const std::uint32_t search_width,
  const std::uint32_t min_iteration,
  const std::uint32_t max_iteration,
  std::uint32_t* const diagnostic_context_ptr,
  const std::uint32_t hash_bitlen,
  const std::uint32_t small_hash_bitlen,
  const std::uint32_t small_hash_reset_interval,
  const std::uint32_t query_id_offset,
  const dataset_descriptor_base_t<DataT, IndexT, DistanceT>* dataset_desc,
  const IndexT graph_size,
  cagra_sample_filter<SourceIndexT> filter_payload,
  const float filtering_rate,
  const float favor_penalty_distance,
  const std::uint32_t favor_penalty_mode_value,
  const float favor_local_gap_multiplier,
  const float favor_retention_fraction,
  const std::uint32_t favor_adaptive_start_iteration,
  const std::uint32_t favor_adaptive_prefix_size)
{
  const auto query_id = blockIdx.y;
  search_core<true,
              TOPK_BY_BITONIC_SORT,
              BITONIC_SORT_AND_MERGE_MULTI_WARPS,
              DataT,
              IndexT,
              DistanceT,
              SourceIndexT,
              true>(result_indices_ptr,
                    result_distances_ptr,
                    top_k,
                    queries_ptr,
                    knn_graph,
                    graph_degree,
                    source_indices_ptr,
                    num_distilation,
                    rand_xor_mask,
                    seed_ptr,
                    num_seeds,
                    visited_hashmap_ptr,
                    max_candidates,
                    max_itopk,
                    internal_topk,
                    search_width,
                    min_iteration,
                    max_iteration,
                    nullptr,
                    hash_bitlen,
                    small_hash_bitlen,
                    small_hash_reset_interval,
                    query_id,
                    query_id_offset,
                    dataset_desc,
                    filter_payload,
                    filtering_rate,
                    favor_penalty_distance,
                    favor_penalty_mode_value,
                    favor_local_gap_multiplier,
                    favor_retention_fraction,
                    graph_size,
                    reinterpret_cast<favor_search_diagnostics::context*>(diagnostic_context_ptr),
                    favor_adaptive_start_iteration,
                    favor_adaptive_prefix_size);
}

// JIT persistent device implementation - called from extern "C" __global__ entry in generated .cu
template <bool TOPK_BY_BITONIC_SORT,
          bool BITONIC_SORT_AND_MERGE_MULTI_WARPS,
          typename DataT,
          typename IndexT,
          typename DistanceT,
          typename SourceIndexT>
__device__ void search_single_cta_p_impl(
  worker_handle_t* worker_handles,
  job_desc_t<job_desc_traits<DataT, IndexT, DistanceT>>* job_descriptors,
  uint32_t* completion_counters,
  const IndexT* const knn_graph,  // [dataset_size, graph_degree]
  const std::uint32_t graph_degree,
  const SourceIndexT* source_indices_ptr,
  const unsigned num_distilation,
  const uint64_t rand_xor_mask,
  const IndexT* seed_ptr,  // [num_queries, num_seeds]
  const uint32_t num_seeds,
  IndexT* const visited_hashmap_ptr,  // [num_queries, 1 << hash_bitlen]
  const std::uint32_t max_candidates,
  const std::uint32_t max_itopk,
  const std::uint32_t internal_topk,
  const std::uint32_t search_width,
  const std::uint32_t min_iteration,
  const std::uint32_t max_iteration,
  std::uint32_t* const num_executed_iterations,  // [num_queries]
  const std::uint32_t hash_bitlen,
  const std::uint32_t small_hash_bitlen,
  const std::uint32_t small_hash_reset_interval,
  const std::uint32_t query_id_offset,  // Offset to add to query_id when calling filter
  const dataset_descriptor_base_t<DataT, IndexT, DistanceT>* dataset_desc,
  cagra_sample_filter<SourceIndexT> filter_payload)
{
  using job_desc_type = job_desc_t<job_desc_traits<DataT, IndexT, DistanceT>>;
  __shared__ typename job_desc_type::input_t job_descriptor;
  __shared__ worker_handle_t::data_t worker_data;

  auto& worker_handle = worker_handles[blockIdx.y].data;
  uint32_t job_ix;

  while (true) {
    // wait the writing phase
    if (threadIdx.x == 0) {
      worker_handle_t::data_t worker_data_local;
      do {
        worker_data_local = worker_handle.load(cuda::memory_order_relaxed);
      } while (worker_data_local.handle == kWaitForWork);
      if (worker_data_local.handle != kNoMoreWork) {
        worker_handle.store({kWaitForWork}, cuda::memory_order_relaxed);
      }
      job_ix = worker_data_local.value.desc_id;
      cuda::atomic_thread_fence(cuda::memory_order_acquire, cuda::thread_scope_system);
      worker_data = worker_data_local;
    }
    if (threadIdx.x < raft::WarpSize) {
      // Sync one warp and copy descriptor data
      static_assert(job_desc_type::kBlobSize <= raft::WarpSize);
      constexpr uint32_t kMaxJobsNum = 8192;
      job_ix                         = raft::shfl(job_ix, 0);
      if (threadIdx.x < job_desc_type::kBlobSize && job_ix < kMaxJobsNum) {
        job_descriptor.blob[threadIdx.x] = job_descriptors[job_ix].input.blob[threadIdx.x];
      }
    }
    __syncthreads();
    if (worker_data.handle == kNoMoreWork) { break; }

    // reading phase
    auto result_indices_ptr    = job_descriptor.value.result_indices_ptr;
    auto* result_distances_ptr = job_descriptor.value.result_distances_ptr;
    auto* queries_ptr          = job_descriptor.value.queries_ptr;
    auto top_k                 = job_descriptor.value.top_k;
    auto n_queries             = job_descriptor.value.n_queries;
    auto query_id              = worker_data.value.query_id;

    // work phase - use JIT search_core
    search_core<false,
                TOPK_BY_BITONIC_SORT,
                BITONIC_SORT_AND_MERGE_MULTI_WARPS,
                DataT,
                IndexT,
                DistanceT,
                SourceIndexT>(result_indices_ptr,
                              result_distances_ptr,
                              top_k,
                              queries_ptr,
                              knn_graph,
                              graph_degree,
                              source_indices_ptr,
                              num_distilation,
                              rand_xor_mask,
                              seed_ptr,
                              num_seeds,
                              visited_hashmap_ptr,
                              max_candidates,
                              max_itopk,
                              internal_topk,
                              search_width,
                              min_iteration,
                              max_iteration,
                              num_executed_iterations,
                              hash_bitlen,
                              small_hash_bitlen,
                              small_hash_reset_interval,
                              query_id,
                              query_id_offset,
                              dataset_desc,
                              filter_payload,
                              0.0f,
                              0.0f);

    // make sure all writes are visible even for the host
    //     (e.g. when result buffers are in pinned memory)
    cuda::atomic_thread_fence(cuda::memory_order_release, cuda::thread_scope_system);

    // arrive to mark the end of the work phase
    __syncthreads();
    if (threadIdx.x == 0) {
      auto completed_count = atomicInc(completion_counters + job_ix, n_queries - 1) + 1;
      if (completed_count >= n_queries) {
        job_descriptors[job_ix].completion_flag.store(true, cuda::memory_order_relaxed);
      }
    }
  }
}

}  // namespace cuvs::neighbors::cagra::detail::single_cta_search
