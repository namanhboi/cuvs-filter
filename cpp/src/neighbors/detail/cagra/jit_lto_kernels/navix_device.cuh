/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "cagra_filter_payload.cuh"
#include "extern_device_functions.cuh"
#include "favor_search_diagnostics.cuh"

#include "../bitonic.hpp"
#include "../hashmap.hpp"
#include "../utils.hpp"

#include <raft/util/integer_utils.hpp>

#include <cassert>
#include <cstdint>

namespace cuvs::neighbors::cagra::detail::device {

enum class navix_policy : std::uint32_t {
  adaptive_kuzu   = 0,
  one_hop         = 1,
  directed_capped = 2,
  blind_capped    = 3,
  adaptive_paper  = 4,
};

constexpr std::uint32_t navix_serial_scheduler_mask = std::uint32_t{1} << 8;

RAFT_DEVICE_INLINE_FUNCTION auto resolve_navix_policy(std::uint32_t requested,
                                                      std::uint32_t passing_first_hop)
  -> navix_policy
{
  const auto mode = static_cast<navix_policy>(requested & 0xffu);
  if (mode == navix_policy::one_hop || mode == navix_policy::directed_capped ||
      mode == navix_policy::blind_capped) {
    return mode;
  }
  if (mode == navix_policy::adaptive_paper) {
    // Paper thresholds for a degree-32 graph.
    return passing_first_hop >= 16  ? navix_policy::one_hop
           : passing_first_hop <= 2 ? navix_policy::blind_capped
                                    : navix_policy::directed_capped;
  }
  // The released Kuzu implementation uses local selectivity >= 0.4 for one hop and its
  // distance-work inequality selects blind expansion at P <= 4 for degree 32.
  return passing_first_hop >= 13  ? navix_policy::one_hop
         : passing_first_hop <= 4 ? navix_policy::blind_capped
                                  : navix_policy::directed_capped;
}

/**
 * Retain only passing nodes in a candidate batch and publish their count CTA-wide.
 *
 * This is the handoff predicate for the in-kernel seed phase. The first batch with a non-zero
 * count becomes NaviX's initial persistent frontier. Clearing the expansion tag makes every
 * retained seed eligible for expansion after the next normal top-k merge.
 */
template <typename IndexT, typename DistanceT, typename SourceIndexT>
RAFT_DEVICE_INLINE_FUNCTION void retain_navix_passing_candidates(
  IndexT* __restrict__ candidate_indices,
  DistanceT* __restrict__ candidate_distances,
  const std::uint32_t candidate_count,
  const SourceIndexT* source_indices_ptr,
  const std::uint32_t query_id,
  cagra_sample_filter<SourceIndexT> filter_payload,
  std::uint32_t* __restrict__ passing_count)
{
  constexpr IndexT invalid_index    = ~static_cast<IndexT>(0);
  constexpr IndexT index_msb_1_mask = utils::gen_index_msb_1_mask<IndexT>::value;

  if (threadIdx.x == 0) { *passing_count = 0; }
  __syncthreads();
  for (std::uint32_t i = threadIdx.x; i < candidate_count; i += blockDim.x) {
    const auto node  = candidate_indices[i] & ~index_msb_1_mask;
    const bool valid = node != (invalid_index & ~index_msb_1_mask);
    bool passes      = false;
    if (valid) {
      const auto source =
        source_indices_ptr == nullptr ? static_cast<SourceIndexT>(node) : source_indices_ptr[node];
      passes = cuvs::neighbors::detail::sample_filter<SourceIndexT>(
        query_id, source, filter_payload.sample_filter_data());
    }
    if (passes) { atomicAdd(passing_count, 1u); }
  }
  __syncthreads();

  // A zero-yield batch must remain untouched: its rejected nodes are precisely the raw-distance
  // frontier that lets ordinary CAGRA continue looking for a seed. Only the first successful batch
  // is converted in place to a predicate-clean NaviX frontier.
  if (*passing_count != 0) {
    for (std::uint32_t i = threadIdx.x; i < candidate_count; i += blockDim.x) {
      const auto node  = candidate_indices[i] & ~index_msb_1_mask;
      const bool valid = node != (invalid_index & ~index_msb_1_mask);
      bool passes      = false;
      if (valid) {
        const auto source = source_indices_ptr == nullptr ? static_cast<SourceIndexT>(node)
                                                          : source_indices_ptr[node];
        passes            = cuvs::neighbors::detail::sample_filter<SourceIndexT>(
          query_id, source, filter_payload.sample_filter_data());
      }
      if (passes) {
        // The seed phase can hand off expanded or unexpanded storage. NaviX must see every
        // retained node as a fresh parent, so remove CAGRA's expansion tag unconditionally.
        candidate_indices[i] = node;
      } else {
        candidate_indices[i]   = invalid_index;
        candidate_distances[i] = raft::upper_bound<DistanceT>();
      }
    }
  }
  __syncthreads();
}

/** Reset the visited set to exactly the retained passing seed batch. */
template <typename IndexT>
RAFT_DEVICE_INLINE_FUNCTION void reset_navix_visited_to_seed_batch(
  IndexT* __restrict__ visited_hashmap_ptr,
  const std::uint32_t visited_hash_bitlen,
  const IndexT* __restrict__ seed_indices,
  const std::uint32_t seed_slots)
{
  constexpr IndexT invalid_index = ~static_cast<IndexT>(0);
  hashmap::init(visited_hashmap_ptr, visited_hash_bitlen, 0);
  __syncthreads();
  for (std::uint32_t i = threadIdx.x; i < seed_slots; i += blockDim.x) {
    const auto seed = seed_indices[i];
    if (seed != invalid_index) {
      (void)hashmap::insert(visited_hashmap_ptr, visited_hash_bitlen, seed);
    }
  }
  __syncthreads();
}

template <typename SourceIndexT>
RAFT_DEVICE_INLINE_FUNCTION auto navix_ground_truth_bit(
  const favor_search_diagnostics::context* diagnostic_context,
  const std::uint32_t query_id,
  const SourceIndexT source) -> std::uint32_t
{
  if (diagnostic_context == nullptr || diagnostic_context->ground_truth_ids == nullptr ||
      query_id >= diagnostic_context->num_queries) {
    return 0;
  }
  const auto* gt = diagnostic_context->ground_truth_ids +
                   static_cast<std::uint64_t>(query_id) * favor_search_diagnostics::ground_truth_k;
  for (std::uint32_t rank = 0; rank < favor_search_diagnostics::ground_truth_k; ++rank) {
    if (static_cast<std::uint32_t>(source) == gt[rank]) { return std::uint32_t{1} << rank; }
  }
  return 0;
}

/** Save the exact successful handoff batch for benchmark-only CPU replay. */
template <typename IndexT, typename DistanceT, typename SourceIndexT>
RAFT_DEVICE_INLINE_FUNCTION void capture_navix_seed_batch(
  const IndexT* seed_indices,
  const DistanceT* seed_distances,
  const std::uint32_t seed_slots,
  const std::uint32_t seed_count,
  const std::uint32_t seed_iteration,
  const std::uint32_t query_id,
  const SourceIndexT* source_indices_ptr,
  favor_search_diagnostics::query_summary* diagnostic_summary,
  favor_search_diagnostics::context* diagnostic_context)
{
  if (diagnostic_summary == nullptr || diagnostic_context == nullptr) { return; }
  if (threadIdx.x == 0) {
    diagnostic_summary->navix_seed_found     = 1;
    diagnostic_summary->navix_seed_iteration = seed_iteration;
    diagnostic_summary->navix_seed_count     = seed_count;
  }
  if (diagnostic_context->navix_seed_ids == nullptr ||
      diagnostic_context->navix_seed_distances == nullptr ||
      query_id >= diagnostic_context->num_queries) {
    return;
  }
  const auto copy_count = min(seed_slots, diagnostic_context->navix_seed_stride);
  auto* ids             = diagnostic_context->navix_seed_ids +
              static_cast<std::uint64_t>(query_id) * diagnostic_context->navix_seed_stride;
  auto* distances = diagnostic_context->navix_seed_distances +
                    static_cast<std::uint64_t>(query_id) * diagnostic_context->navix_seed_stride;
  for (std::uint32_t i = threadIdx.x; i < copy_count; i += blockDim.x) {
    ids[i]          = static_cast<std::uint32_t>(seed_indices[i]);
    distances[i]    = static_cast<float>(seed_distances[i]);
    const auto seed = seed_indices[i];
    if (seed != ~static_cast<IndexT>(0)) {
      const auto source =
        source_indices_ptr == nullptr ? static_cast<SourceIndexT>(seed) : source_indices_ptr[seed];
      const auto gt_bit = navix_ground_truth_bit(diagnostic_context, query_id, source);
      if (gt_bit != 0) {
        atomicOr(&diagnostic_summary->gt_seen_mask, gt_bit);
        atomicOr(&diagnostic_summary->navix_gt_admitted_mask, gt_bit);
      }
    }
  }
}

/**
 * Produce at most D passing candidates per selected parent using NaviX one-/two-hop expansion.
 *
 * The output is written directly to CAGRA's existing W*D candidate tail. Rejected first-hop rows
 * live only in a W*D bridge array. During two-hop traversal, each resident warp owns one bridge
 * row and its 32 lanes hold the grandchildren in registers; rows are committed in bridge order to
 * preserve capped blind/directed semantics.
 */
template <bool DIAGNOSTICS,
          typename IndexT,
          typename DistanceT,
          typename DataT,
          typename SourceIndexT>
RAFT_DEVICE_INLINE_FUNCTION void compute_navix_candidates_jit(
  IndexT* __restrict__ output_indices,
  DistanceT* __restrict__ output_distances,
  const dataset_descriptor_base_t<DataT, IndexT, DistanceT>* smem_desc,
  const IndexT* __restrict__ knn_graph,
  const std::uint32_t graph_degree,
  IndexT* __restrict__ visited_hashmap_ptr,
  const std::uint32_t visited_hash_bitlen,
  const IndexT* __restrict__ parent_positions,
  const IndexT* __restrict__ internal_topk_list,
  const std::uint32_t search_width,
  const SourceIndexT* source_indices_ptr,
  const std::uint32_t query_id,
  cagra_sample_filter<SourceIndexT> filter_payload,
  std::uint32_t* __restrict__ work_ptr,
  const std::uint32_t requested_policy,
  favor_search_diagnostics::query_summary* diagnostic_summary   = nullptr,
  favor_search_diagnostics::iteration_record* diagnostic_record = nullptr,
  favor_search_diagnostics::context* diagnostic_context         = nullptr)
{
  constexpr IndexT invalid_index    = ~static_cast<IndexT>(0);
  constexpr IndexT index_msb_1_mask = utils::gen_index_msb_1_mask<IndexT>::value;
  const auto candidate_count        = search_width * graph_degree;
  const auto lane                   = static_cast<std::uint32_t>(threadIdx.x) & 31u;
  const auto warp                   = static_cast<std::uint32_t>(threadIdx.x) >> 5u;
  const auto num_warps              = static_cast<std::uint32_t>(blockDim.x) >> 5u;

  // Degree 32 is the deliberately narrow first implementation. It maps a neighbor row exactly
  // onto a warp and is the configuration used by the filtered datasets in this experiment.
  assert(graph_degree == 32);
  assert((blockDim.x & 31u) == 0);

  auto* bridge_ids      = reinterpret_cast<IndexT*>(work_ptr);
  auto* output_counts   = reinterpret_cast<std::uint32_t*>(bridge_ids + candidate_count);
  auto* bridge_counts   = output_counts + search_width;
  auto* parent_policies = bridge_counts + search_width;

  for (std::uint32_t i = threadIdx.x; i < candidate_count; i += blockDim.x) {
    output_indices[i]   = invalid_index;
    output_distances[i] = raft::upper_bound<DistanceT>();
    bridge_ids[i]       = invalid_index;

    const auto parent_number = i / graph_degree;
    const auto parent_pos    = parent_positions[parent_number];
    if (parent_pos == invalid_index) { continue; }
    const auto parent = internal_topk_list[parent_pos] & ~index_msb_1_mask;
    const auto child =
      knn_graph[static_cast<std::uint64_t>(parent) * graph_degree + (i % graph_degree)];
    if (child == invalid_index) { continue; }
    const auto source =
      source_indices_ptr == nullptr ? static_cast<SourceIndexT>(child) : source_indices_ptr[child];
    const bool passes = cuvs::neighbors::detail::sample_filter<SourceIndexT>(
      query_id, source, filter_payload.sample_filter_data());
    if constexpr (DIAGNOSTICS) {
      if (diagnostic_summary != nullptr) {
        atomicAdd(&diagnostic_summary->navix_first_hop_checks, 1u);
        atomicAdd(&diagnostic_summary->candidate_attempts, 1u);
        atomicAdd(&diagnostic_summary->candidate_evaluations, 1u);
        atomicAdd(passes ? &diagnostic_summary->navix_first_hop_passing
                         : &diagnostic_summary->rejected_candidates,
                  1u);
        if (passes) { atomicAdd(&diagnostic_summary->passing_candidates, 1u); }
        const auto gt_bit = navix_ground_truth_bit(diagnostic_context, query_id, source);
        if (gt_bit != 0) {
          atomicOr(&diagnostic_summary->gt_seen_mask, gt_bit);
          atomicOr(&diagnostic_summary->navix_gt_first_hop_mask, gt_bit);
        }
      }
      if (diagnostic_record != nullptr) {
        atomicAdd(&diagnostic_record->navix_first_hop_checks, 1u);
        if (passes) { atomicAdd(&diagnostic_record->navix_first_hop_passing, 1u); }
      }
    }
    // The MSB is free for valid internal graph IDs and temporarily carries first-hop status.
    output_indices[i] = passes ? (child | index_msb_1_mask) : child;
  }
  for (std::uint32_t p = threadIdx.x; p < search_width; p += blockDim.x) {
    output_counts[p]   = 0;
    bridge_counts[p]   = 0;
    parent_policies[p] = static_cast<std::uint32_t>(navix_policy::one_hop);
  }
  __syncthreads();

  // Resolve the local policy from all D predicate outcomes, before visited suppression.
  for (std::uint32_t p = warp; p < search_width; p += num_warps) {
    const auto tagged       = output_indices[p * graph_degree + lane];
    const bool pass         = tagged != invalid_index && ((tagged & index_msb_1_mask) != 0);
    const auto passing_mask = __ballot_sync(0xffffffffu, pass);
    if (lane == 0) {
      const auto passing_count = static_cast<std::uint32_t>(__popc(passing_mask));
      const bool valid_parent  = parent_positions[p] != invalid_index;
      const auto policy  = valid_parent ? resolve_navix_policy(requested_policy, passing_count)
                                        : navix_policy::one_hop;
      parent_policies[p] = static_cast<std::uint32_t>(policy);
      if constexpr (DIAGNOSTICS) {
        if (valid_parent && diagnostic_summary != nullptr) {
          atomicAdd(&diagnostic_summary->navix_local_p_histogram[passing_count], 1u);
          auto* counter = policy == navix_policy::one_hop
                            ? &diagnostic_summary->navix_one_hop_parents
                            : (policy == navix_policy::directed_capped
                                 ? &diagnostic_summary->navix_directed_parents
                                 : &diagnostic_summary->navix_blind_parents);
          atomicAdd(counter, 1u);
        }
        if (valid_parent && diagnostic_record != nullptr) {
          auto* counter = policy == navix_policy::one_hop
                            ? &diagnostic_record->navix_one_hop_parents
                            : (policy == navix_policy::directed_capped
                                 ? &diagnostic_record->navix_directed_parents
                                 : &diagnostic_record->navix_blind_parents);
          atomicAdd(counter, 1u);
        }
      }
    }
  }
  __syncthreads();

  // Directed mode ranks rejected bridge rows by query distance. Blind and one-hop avoid this
  // first-hop distance work; all admitted passing candidates are measured once at the end.
  const auto team_size_bits = smem_desc->team_size_bitshift_from_smem();
  const auto team_width     = 1u << team_size_bits;
  const auto teams_per_warp = device::warp_size >> team_size_bits;
  const auto max_i          = raft::round_up_safe(candidate_count, teams_per_warp);
  const auto args           = smem_desc->args.load();
  const bool lead_lane      = (threadIdx.x & (team_width - 1u)) == 0;
  for (std::uint32_t i = threadIdx.x >> team_size_bits; i < max_i;
       i += blockDim.x >> team_size_bits) {
    const bool in_range = i < candidate_count;
    const auto tagged   = in_range ? output_indices[i] : invalid_index;
    const auto child    = tagged & ~index_msb_1_mask;
    const bool directed = in_range && parent_policies[i / graph_degree] ==
                                        static_cast<std::uint32_t>(navix_policy::directed_capped);
    const bool valid =
      directed && tagged != invalid_index && child != (invalid_index & ~index_msb_1_mask);
    const auto partial =
      valid ? cuvs::neighbors::cagra::detail::compute_distance_per_thread<DataT, IndexT, DistanceT>(
                args, child)
            : (lead_lane ? raft::upper_bound<DistanceT>() : DistanceT{0});
    const auto distance = device::team_sum(partial, team_size_bits);
    __syncwarp();
    if (in_range && lead_lane) { output_distances[i] = distance; }
  }
  __syncthreads();

  // One warp arranges each first-hop row. Multiple parents are processed concurrently when W and
  // the resident warp count allow it.
  for (std::uint32_t p = warp; p < search_width; p += num_warps) {
    const auto offset = p * graph_degree + lane;
    DistanceT keys[1] = {output_distances[offset]};
    IndexT vals[1]    = {output_indices[offset]};
    const auto policy = static_cast<navix_policy>(parent_policies[p]);
    if (policy == navix_policy::directed_capped) {
      bitonic::warp_sort<DistanceT, IndexT, 1>(keys, vals);
    }

    const auto tagged = vals[0];
    const auto child  = tagged & ~index_msb_1_mask;
    const bool valid  = tagged != invalid_index && child != (invalid_index & ~index_msb_1_mask);
    const bool passes = valid && ((tagged & index_msb_1_mask) != 0);
    // One-hop mode ignores rejected neighbors, so leave them unvisited. The same row may be a
    // useful transient bridge if it is encountered later under a parent that selects two hops.
    const bool needs_visit = passes || policy != navix_policy::one_hop;
    bool inserted          = false;
    auto insert_result     = hashmap::insert_outcome::duplicate;
    if (valid && needs_visit) {
      if constexpr (DIAGNOSTICS) {
        insert_result =
          hashmap::insert_with_outcome(visited_hashmap_ptr, visited_hash_bitlen, child);
        inserted = insert_result == hashmap::insert_outcome::inserted;
        if (diagnostic_summary != nullptr) {
          if (insert_result == hashmap::insert_outcome::duplicate) {
            atomicAdd(&diagnostic_summary->candidate_duplicates, 1u);
          } else if (insert_result == hashmap::insert_outcome::full) {
            atomicAdd(&diagnostic_summary->candidate_hash_full, 1u);
            const auto source = source_indices_ptr == nullptr ? static_cast<SourceIndexT>(child)
                                                              : source_indices_ptr[child];
            const auto gt_bit = navix_ground_truth_bit(diagnostic_context, query_id, source);
            if (gt_bit != 0) { atomicOr(&diagnostic_summary->navix_gt_hash_full_mask, gt_bit); }
          }
        }
      } else {
        inserted = hashmap::insert(visited_hashmap_ptr, visited_hash_bitlen, child) != 0;
      }
    }

    const auto passing_mask = __ballot_sync(0xffffffffu, inserted && passes);
    const auto passing_rank =
      static_cast<std::uint32_t>(__popc(passing_mask & ((std::uint32_t{1} << lane) - 1u)));

    const bool bridge      = inserted && !passes && policy != navix_policy::one_hop;
    const auto bridge_mask = __ballot_sync(0xffffffffu, bridge);
    const auto bridge_rank =
      static_cast<std::uint32_t>(__popc(bridge_mask & ((std::uint32_t{1} << lane) - 1u)));
    // Every lane already holds its tagged value in registers. Clear the original row before
    // compacting so rejected and duplicate first-hop entries cannot leak past output_counts[p].
    output_indices[offset]   = invalid_index;
    output_distances[offset] = raft::upper_bound<DistanceT>();
    __syncwarp();
    if (inserted && passes) {
      output_indices[p * graph_degree + passing_rank]   = child;
      output_distances[p * graph_degree + passing_rank] = keys[0];
      if constexpr (DIAGNOSTICS) {
        if (diagnostic_summary != nullptr) {
          atomicAdd(&diagnostic_summary->navix_admitted_candidates, 1u);
          const auto source = source_indices_ptr == nullptr ? static_cast<SourceIndexT>(child)
                                                            : source_indices_ptr[child];
          const auto gt_bit = navix_ground_truth_bit(diagnostic_context, query_id, source);
          if (gt_bit != 0) { atomicOr(&diagnostic_summary->navix_gt_admitted_mask, gt_bit); }
        }
        if (diagnostic_record != nullptr) {
          atomicAdd(&diagnostic_record->navix_admitted_candidates, 1u);
        }
      }
    }
    if (bridge) { bridge_ids[p * graph_degree + bridge_rank] = child; }
    if (lane == 0) {
      output_counts[p] = static_cast<std::uint32_t>(__popc(passing_mask));
      bridge_counts[p] = static_cast<std::uint32_t>(__popc(bridge_mask));
      if constexpr (DIAGNOSTICS) {
        if (diagnostic_summary != nullptr) {
          atomicAdd(&diagnostic_summary->navix_bridge_rows,
                    static_cast<std::uint32_t>(__popc(bridge_mask)));
        }
        if (diagnostic_record != nullptr) {
          atomicAdd(&diagnostic_record->navix_bridge_rows,
                    static_cast<std::uint32_t>(__popc(bridge_mask)));
        }
      }
    }
  }
  __syncthreads();

  const bool serial_scheduler = (requested_policy & navix_serial_scheduler_mask) != 0;
  const auto tile_warps       = serial_scheduler ? 1u : num_warps;
  for (std::uint32_t p = 0; p < search_width; ++p) {
    const auto policy = static_cast<navix_policy>(parent_policies[p]);
    if (policy == navix_policy::one_hop) { continue; }

    // Keep the loop trip count independent of the evolving output count. Every warp in the CTA
    // must encounter the same barriers even when an earlier bridge row fills the D-slot cap.
    for (std::uint32_t base = 0; base < bridge_counts[p]; base += tile_warps) {
      const bool owns_row = warp < tile_warps && base + warp < bridge_counts[p];
      const auto bridge   = owns_row ? bridge_ids[p * graph_degree + base + warp] : invalid_index;
      const bool row_started_after_cap = owns_row && output_counts[p] >= graph_degree;
      if constexpr (DIAGNOSTICS) {
        if (lane == 0 && owns_row) {
          if (diagnostic_summary != nullptr) {
            atomicAdd(&diagnostic_summary->navix_bridge_rows_loaded, 1u);
            if (row_started_after_cap) {
              atomicAdd(&diagnostic_summary->navix_bridge_rows_after_cap, 1u);
            }
          }
          if (diagnostic_record != nullptr) {
            atomicAdd(&diagnostic_record->navix_bridge_rows_loaded, 1u);
            if (row_started_after_cap) {
              atomicAdd(&diagnostic_record->navix_bridge_rows_after_cap, 1u);
            }
          }
        }
      }
      const auto grandchild =
        owns_row ? knn_graph[static_cast<std::uint64_t>(bridge) * graph_degree + lane]
                 : invalid_index;
      bool passes = false;
      if (grandchild != invalid_index) {
        const auto source = source_indices_ptr == nullptr ? static_cast<SourceIndexT>(grandchild)
                                                          : source_indices_ptr[grandchild];
        passes            = cuvs::neighbors::detail::sample_filter<SourceIndexT>(
          query_id, source, filter_payload.sample_filter_data());
        if constexpr (DIAGNOSTICS) {
          if (diagnostic_summary != nullptr) {
            atomicAdd(&diagnostic_summary->navix_second_hop_checks, 1u);
            atomicAdd(&diagnostic_summary->candidate_attempts, 1u);
            atomicAdd(&diagnostic_summary->candidate_evaluations, 1u);
            if (passes) {
              atomicAdd(&diagnostic_summary->navix_second_hop_passing, 1u);
              atomicAdd(&diagnostic_summary->passing_candidates, 1u);
            } else {
              atomicAdd(&diagnostic_summary->rejected_candidates, 1u);
            }
            const auto gt_bit = navix_ground_truth_bit(diagnostic_context, query_id, source);
            if (gt_bit != 0) {
              atomicOr(&diagnostic_summary->gt_seen_mask, gt_bit);
              atomicOr(&diagnostic_summary->navix_gt_second_hop_mask, gt_bit);
            }
          }
          if (diagnostic_record != nullptr) {
            atomicAdd(&diagnostic_record->navix_second_hop_checks, 1u);
            if (passes) { atomicAdd(&diagnostic_record->navix_second_hop_passing, 1u); }
          }
        }
      }
      __syncthreads();

      // All resident warps load concurrently, then commit in bridge order. This makes the tiled
      // scheduler semantically equivalent to the serial reference even when the D-candidate cap
      // is reached part-way through a tile.
      for (std::uint32_t commit_warp = 0; commit_warp < tile_warps; ++commit_warp) {
        if (warp == commit_warp && owns_row) {
          auto pending = __ballot_sync(0xffffffffu, passes && grandchild != invalid_index);
          auto count   = __shfl_sync(0xffffffffu, lane == 0 ? output_counts[p] : 0u, 0);
          while (pending != 0 && count < graph_degree) {
            const auto lower     = (std::uint32_t{1} << lane) - 1u;
            const auto rank      = static_cast<std::uint32_t>(__popc(pending & lower));
            const auto remaining = graph_degree - count;
            const bool attempt   = (pending & (std::uint32_t{1} << lane)) != 0 && rank < remaining;
            bool inserted        = false;
            auto insert_result   = hashmap::insert_outcome::duplicate;
            if (attempt) {
              if constexpr (DIAGNOSTICS) {
                insert_result = hashmap::insert_with_outcome(
                  visited_hashmap_ptr, visited_hash_bitlen, grandchild);
                inserted = insert_result == hashmap::insert_outcome::inserted;
                if (diagnostic_summary != nullptr) {
                  if (insert_result == hashmap::insert_outcome::duplicate) {
                    atomicAdd(&diagnostic_summary->candidate_duplicates, 1u);
                  } else if (insert_result == hashmap::insert_outcome::full) {
                    atomicAdd(&diagnostic_summary->candidate_hash_full, 1u);
                    const auto source = source_indices_ptr == nullptr
                                          ? static_cast<SourceIndexT>(grandchild)
                                          : source_indices_ptr[grandchild];
                    const auto gt_bit =
                      navix_ground_truth_bit(diagnostic_context, query_id, source);
                    if (gt_bit != 0) {
                      atomicOr(&diagnostic_summary->navix_gt_hash_full_mask, gt_bit);
                    }
                  }
                }
              } else {
                inserted =
                  hashmap::insert(visited_hashmap_ptr, visited_hash_bitlen, grandchild) != 0;
              }
            }
            const auto attempt_mask = __ballot_sync(0xffffffffu, attempt);
            const auto success_mask = __ballot_sync(0xffffffffu, inserted);
            const auto success_rank = static_cast<std::uint32_t>(__popc(success_mask & lower));
            if (inserted) {
              output_indices[p * graph_degree + count + success_rank] = grandchild;
              if constexpr (DIAGNOSTICS) {
                if (diagnostic_summary != nullptr) {
                  atomicAdd(&diagnostic_summary->navix_admitted_candidates, 1u);
                  const auto source = source_indices_ptr == nullptr
                                        ? static_cast<SourceIndexT>(grandchild)
                                        : source_indices_ptr[grandchild];
                  const auto gt_bit = navix_ground_truth_bit(diagnostic_context, query_id, source);
                  if (gt_bit != 0) {
                    atomicOr(&diagnostic_summary->navix_gt_admitted_mask, gt_bit);
                  }
                }
                if (diagnostic_record != nullptr) {
                  atomicAdd(&diagnostic_record->navix_admitted_candidates, 1u);
                }
              }
            }
            count += static_cast<std::uint32_t>(__popc(success_mask));
            pending &= ~attempt_mask;
          }
          if constexpr (DIAGNOSTICS) {
            const bool blocked =
              (pending & (std::uint32_t{1} << lane)) != 0 &&
              hashmap::search(visited_hashmap_ptr, visited_hash_bitlen, grandchild) == 0;
            if (blocked) {
              if (diagnostic_summary != nullptr) {
                atomicAdd(&diagnostic_summary->navix_cap_blocked_unique, 1u);
                const auto source = source_indices_ptr == nullptr
                                      ? static_cast<SourceIndexT>(grandchild)
                                      : source_indices_ptr[grandchild];
                const auto gt_bit = navix_ground_truth_bit(diagnostic_context, query_id, source);
                if (gt_bit != 0) {
                  atomicOr(&diagnostic_summary->navix_gt_cap_blocked_mask, gt_bit);
                }
              }
              if (diagnostic_record != nullptr) {
                atomicAdd(&diagnostic_record->navix_cap_blocked_unique, 1u);
              }
            }
          }
          if (lane == 0) { output_counts[p] = min(count, graph_degree); }
        }
        __syncthreads();
      }
    }
  }

  // Compute raw distances only for candidates that can enter the persistent passing frontier.
  for (std::uint32_t i = threadIdx.x >> team_size_bits; i < max_i;
       i += blockDim.x >> team_size_bits) {
    const bool in_range     = i < candidate_count;
    const auto child        = in_range ? output_indices[i] : invalid_index;
    const bool valid        = in_range && child != invalid_index;
    const bool has_distance = valid && output_distances[i] != raft::upper_bound<DistanceT>();
    const auto partial =
      valid && !has_distance
        ? cuvs::neighbors::cagra::detail::compute_distance_per_thread<DataT, IndexT, DistanceT>(
            args, child)
        : (lead_lane ? raft::upper_bound<DistanceT>() : DistanceT{0});
    const auto distance = device::team_sum(partial, team_size_bits);
    __syncwarp();
    if (in_range && lead_lane && !has_distance) { output_distances[i] = distance; }
  }
}

}  // namespace cuvs::neighbors::cagra::detail::device
