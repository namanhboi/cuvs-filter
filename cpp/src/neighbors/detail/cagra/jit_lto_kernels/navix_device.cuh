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

template <std::uint32_t GRAPH_DEGREE>
_RAFT_HOST_DEVICE constexpr auto resolve_navix_policy(std::uint32_t requested,
                                                      std::uint32_t passing_first_hop)
  -> navix_policy
{
  static_assert(GRAPH_DEGREE == 32 || GRAPH_DEGREE == 64,
                "NaviX supports degree-32 and degree-64 CAGRA graphs");
  const auto mode = static_cast<navix_policy>(requested & 0xffu);
  if (mode == navix_policy::one_hop || mode == navix_policy::directed_capped ||
      mode == navix_policy::blind_capped) {
    return mode;
  }
  if (mode == navix_policy::adaptive_paper) {
    // The original global-policy thresholds are 0.5 and 0.08. Use exact integer comparisons so
    // degree specialization cannot change a boundary through floating-point rounding.
    return 2u * passing_first_hop >= GRAPH_DEGREE         ? navix_policy::one_hop
           : 25u * passing_first_hop <= 2u * GRAPH_DEGREE ? navix_policy::blind_capped
                                                          : navix_policy::directed_capped;
  }
  // The released Kuzu implementation uses local selectivity P/D >= 0.4 for one hop. Below that
  // threshold it compares 0.4*P*(D+1) full-two-hop work against 2D-P directed work.
  if (5u * passing_first_hop >= 2u * GRAPH_DEGREE) { return navix_policy::one_hop; }
  return 2u * passing_first_hop * (GRAPH_DEGREE + 1u) <=
             5u * (2u * GRAPH_DEGREE - passing_first_hop)
           ? navix_policy::blind_capped
           : navix_policy::directed_capped;
}

static_assert(resolve_navix_policy<32>(0, 4) == navix_policy::blind_capped);
static_assert(resolve_navix_policy<32>(0, 5) == navix_policy::directed_capped);
static_assert(resolve_navix_policy<32>(0, 12) == navix_policy::directed_capped);
static_assert(resolve_navix_policy<32>(0, 13) == navix_policy::one_hop);
static_assert(resolve_navix_policy<64>(0, 4) == navix_policy::blind_capped);
static_assert(resolve_navix_policy<64>(0, 5) == navix_policy::directed_capped);
static_assert(resolve_navix_policy<64>(0, 25) == navix_policy::directed_capped);
static_assert(resolve_navix_policy<64>(0, 26) == navix_policy::one_hop);

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
template <std::uint32_t GRAPH_DEGREE,
          bool DIAGNOSTICS,
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
  static_assert(GRAPH_DEGREE == 32 || GRAPH_DEGREE == 64,
                "NaviX supports degree-32 and degree-64 CAGRA graphs");
  constexpr std::uint32_t items_per_lane = GRAPH_DEGREE / device::warp_size;
  constexpr IndexT invalid_index         = ~static_cast<IndexT>(0);
  constexpr IndexT index_msb_1_mask      = utils::gen_index_msb_1_mask<IndexT>::value;
  const auto candidate_count             = search_width * GRAPH_DEGREE;
  const auto lane                        = static_cast<std::uint32_t>(threadIdx.x) & 31u;
  const auto warp                        = static_cast<std::uint32_t>(threadIdx.x) >> 5u;
  const auto num_warps                   = static_cast<std::uint32_t>(blockDim.x) >> 5u;

  assert(graph_degree == GRAPH_DEGREE);
  assert((blockDim.x & 31u) == 0);

  auto* bridge_ids      = reinterpret_cast<IndexT*>(work_ptr);
  auto* output_counts   = reinterpret_cast<std::uint32_t*>(bridge_ids + candidate_count);
  auto* bridge_counts   = output_counts + search_width;
  auto* parent_policies = bridge_counts + search_width;

  for (std::uint32_t i = threadIdx.x; i < candidate_count; i += blockDim.x) {
    output_indices[i]   = invalid_index;
    output_distances[i] = raft::upper_bound<DistanceT>();
    bridge_ids[i]       = invalid_index;

    const auto parent_number = i / GRAPH_DEGREE;
    const auto parent_pos    = parent_positions[parent_number];
    if (parent_pos == invalid_index) { continue; }
    const auto parent = internal_topk_list[parent_pos] & ~index_msb_1_mask;
    const auto child =
      knn_graph[static_cast<std::uint64_t>(parent) * GRAPH_DEGREE + (i % GRAPH_DEGREE)];
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
    std::uint32_t passing_count = 0;
#pragma unroll
    for (std::uint32_t item = 0; item < items_per_lane; ++item) {
      const auto tagged = output_indices[p * GRAPH_DEGREE + item * device::warp_size + lane];
      const bool pass   = tagged != invalid_index && ((tagged & index_msb_1_mask) != 0);
      passing_count += static_cast<std::uint32_t>(__popc(__ballot_sync(0xffffffffu, pass)));
    }
    if (lane == 0) {
      const bool valid_parent = parent_positions[p] != invalid_index;
      const auto policy       = valid_parent
                                  ? resolve_navix_policy<GRAPH_DEGREE>(requested_policy, passing_count)
                                  : navix_policy::one_hop;
      parent_policies[p]      = static_cast<std::uint32_t>(policy);
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
    const bool directed = in_range && parent_policies[i / GRAPH_DEGREE] ==
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
    const auto policy = static_cast<navix_policy>(parent_policies[p]);
    DistanceT keys[items_per_lane];
    IndexT vals[items_per_lane];
#pragma unroll
    for (std::uint32_t item = 0; item < items_per_lane; ++item) {
      // warp_sort<N> linearizes items lane-major. Load directed rows in that layout so the sorted
      // sequence remains distance ordered when compacted. Non-directed rows retain the graph's
      // natural chunk-major order and use coalesced shared-memory reads.
      const auto row_slot = policy == navix_policy::directed_capped
                              ? lane * items_per_lane + item
                              : item * device::warp_size + lane;
      keys[item]          = output_distances[p * GRAPH_DEGREE + row_slot];
      vals[item]          = output_indices[p * GRAPH_DEGREE + row_slot];
    }
    if (policy == navix_policy::directed_capped) {
      bitonic::warp_sort<DistanceT, IndexT, items_per_lane>(keys, vals);
    }

    bool inserted[items_per_lane];
    bool passes[items_per_lane];
    bool bridges[items_per_lane];
    std::uint32_t passing_masks[items_per_lane];
    std::uint32_t bridge_masks[items_per_lane];
#pragma unroll
    for (std::uint32_t item = 0; item < items_per_lane; ++item) {
      const auto tagged = vals[item];
      const auto child  = tagged & ~index_msb_1_mask;
      const bool valid  = tagged != invalid_index && child != (invalid_index & ~index_msb_1_mask);
      passes[item]      = valid && ((tagged & index_msb_1_mask) != 0);
      // One-hop mode ignores rejected neighbors, so leave them unvisited. The same row may be a
      // useful transient bridge if it is encountered later under a parent that selects two hops.
      const bool needs_visit = passes[item] || policy != navix_policy::one_hop;
      inserted[item]         = false;
      if (valid && needs_visit) {
        if constexpr (DIAGNOSTICS) {
          const auto insert_result =
            hashmap::insert_with_outcome(visited_hashmap_ptr, visited_hash_bitlen, child);
          inserted[item] = insert_result == hashmap::insert_outcome::inserted;
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
          inserted[item] = hashmap::insert(visited_hashmap_ptr, visited_hash_bitlen, child) != 0;
        }
      }
      bridges[item] = inserted[item] && !passes[item] && policy != navix_policy::one_hop;
    }

#pragma unroll
    for (std::uint32_t item = 0; item < items_per_lane; ++item) {
      passing_masks[item] = __ballot_sync(0xffffffffu, inserted[item] && passes[item]);
      bridge_masks[item]  = __ballot_sync(0xffffffffu, bridges[item]);
    }
    // Every lane already holds its tagged value in registers. Clear the original row before
    // compacting so rejected and duplicate first-hop entries cannot leak past output_counts[p].
#pragma unroll
    for (std::uint32_t item = 0; item < items_per_lane; ++item) {
      const auto offset        = p * GRAPH_DEGREE + item * device::warp_size + lane;
      output_indices[offset]   = invalid_index;
      output_distances[offset] = raft::upper_bound<DistanceT>();
    }
    __syncwarp();

    const auto lower_lanes = (std::uint32_t{1} << lane) - 1u;
#pragma unroll
    for (std::uint32_t item = 0; item < items_per_lane; ++item) {
      std::uint32_t passing_rank = 0;
      std::uint32_t bridge_rank  = 0;
      if (policy == navix_policy::directed_capped) {
        // Sorted items are lane-major: count every item owned by earlier lanes, followed by the
        // earlier items in this lane.
#pragma unroll
        for (std::uint32_t other = 0; other < items_per_lane; ++other) {
          passing_rank += static_cast<std::uint32_t>(__popc(passing_masks[other] & lower_lanes));
          bridge_rank += static_cast<std::uint32_t>(__popc(bridge_masks[other] & lower_lanes));
        }
#pragma unroll
        for (std::uint32_t previous = 0; previous < item; ++previous) {
          passing_rank += (passing_masks[previous] >> lane) & 1u;
          bridge_rank += (bridge_masks[previous] >> lane) & 1u;
        }
      } else {
        // Unsorted items are chunk-major, matching the graph row's natural order.
#pragma unroll
        for (std::uint32_t previous = 0; previous < item; ++previous) {
          passing_rank += static_cast<std::uint32_t>(__popc(passing_masks[previous]));
          bridge_rank += static_cast<std::uint32_t>(__popc(bridge_masks[previous]));
        }
        passing_rank += static_cast<std::uint32_t>(__popc(passing_masks[item] & lower_lanes));
        bridge_rank += static_cast<std::uint32_t>(__popc(bridge_masks[item] & lower_lanes));
      }

      const auto child = vals[item] & ~index_msb_1_mask;
      if (inserted[item] && passes[item]) {
        output_indices[p * GRAPH_DEGREE + passing_rank]   = child;
        output_distances[p * GRAPH_DEGREE + passing_rank] = keys[item];
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
      if (bridges[item]) { bridge_ids[p * GRAPH_DEGREE + bridge_rank] = child; }
    }
    if (lane == 0) {
      std::uint32_t passing_count = 0;
      std::uint32_t bridge_count  = 0;
#pragma unroll
      for (std::uint32_t item = 0; item < items_per_lane; ++item) {
        passing_count += static_cast<std::uint32_t>(__popc(passing_masks[item]));
        bridge_count += static_cast<std::uint32_t>(__popc(bridge_masks[item]));
      }
      output_counts[p] = passing_count;
      bridge_counts[p] = bridge_count;
      if constexpr (DIAGNOSTICS) {
        if (diagnostic_summary != nullptr) {
          atomicAdd(&diagnostic_summary->navix_bridge_rows, bridge_count);
        }
        if (diagnostic_record != nullptr) {
          atomicAdd(&diagnostic_record->navix_bridge_rows, bridge_count);
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
      const auto bridge   = owns_row ? bridge_ids[p * GRAPH_DEGREE + base + warp] : invalid_index;
      const bool row_started_after_cap = owns_row && output_counts[p] >= GRAPH_DEGREE;
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
      IndexT grandchildren[items_per_lane];
      bool grandchild_passes[items_per_lane];
#pragma unroll
      for (std::uint32_t item = 0; item < items_per_lane; ++item) {
        const auto grandchild   = owns_row
                                    ? knn_graph[static_cast<std::uint64_t>(bridge) * GRAPH_DEGREE +
                                              item * device::warp_size + lane]
                                    : invalid_index;
        grandchildren[item]     = grandchild;
        grandchild_passes[item] = false;
        if (grandchild != invalid_index) {
          const auto source = source_indices_ptr == nullptr ? static_cast<SourceIndexT>(grandchild)
                                                            : source_indices_ptr[grandchild];
          grandchild_passes[item] = cuvs::neighbors::detail::sample_filter<SourceIndexT>(
            query_id, source, filter_payload.sample_filter_data());
          if constexpr (DIAGNOSTICS) {
            if (diagnostic_summary != nullptr) {
              atomicAdd(&diagnostic_summary->navix_second_hop_checks, 1u);
              atomicAdd(&diagnostic_summary->candidate_attempts, 1u);
              atomicAdd(&diagnostic_summary->candidate_evaluations, 1u);
              if (grandchild_passes[item]) {
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
              if (grandchild_passes[item]) {
                atomicAdd(&diagnostic_record->navix_second_hop_passing, 1u);
              }
            }
          }
        }
      }
      __syncthreads();

      // All resident warps load concurrently, then commit in bridge order. This makes the tiled
      // scheduler semantically equivalent to the serial reference even when the D-candidate cap
      // is reached part-way through a tile.
      for (std::uint32_t commit_warp = 0; commit_warp < tile_warps; ++commit_warp) {
        if (warp == commit_warp && owns_row) {
          auto count = __shfl_sync(0xffffffffu, lane == 0 ? output_counts[p] : 0u, 0);
#pragma unroll
          for (std::uint32_t item = 0; item < items_per_lane; ++item) {
            const auto grandchild = grandchildren[item];
            auto pending =
              __ballot_sync(0xffffffffu, grandchild_passes[item] && grandchild != invalid_index);
            while (pending != 0 && count < GRAPH_DEGREE) {
              const auto lower     = (std::uint32_t{1} << lane) - 1u;
              const auto rank      = static_cast<std::uint32_t>(__popc(pending & lower));
              const auto remaining = GRAPH_DEGREE - count;
              const bool attempt = (pending & (std::uint32_t{1} << lane)) != 0 && rank < remaining;
              bool inserted      = false;
              if (attempt) {
                if constexpr (DIAGNOSTICS) {
                  const auto insert_result = hashmap::insert_with_outcome(
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
                output_indices[p * GRAPH_DEGREE + count + success_rank] = grandchild;
                if constexpr (DIAGNOSTICS) {
                  if (diagnostic_summary != nullptr) {
                    atomicAdd(&diagnostic_summary->navix_admitted_candidates, 1u);
                    const auto source = source_indices_ptr == nullptr
                                          ? static_cast<SourceIndexT>(grandchild)
                                          : source_indices_ptr[grandchild];
                    const auto gt_bit =
                      navix_ground_truth_bit(diagnostic_context, query_id, source);
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
          }
          if (lane == 0) { output_counts[p] = min(count, GRAPH_DEGREE); }
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
