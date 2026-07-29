/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../../neighbors_device_intrinsics.cuh"
#include "../../sample_filter_data.cuh"
#include "../hashmap.hpp"
#include "../utils.hpp"
#include "extern_device_functions.cuh"

#include <cuvs/distance/distance.hpp>
#include <raft/core/operators.hpp>
#include <raft/util/integer_utils.hpp>
#include <type_traits>

namespace cuvs::neighbors::cagra::detail::device {

// Helper to check if DescriptorT has kPqBits (VPQ descriptor)
template <typename T>
struct has_kpq_bits {
  template <typename U>
  static auto test(int) -> decltype(U::kPqBits, std::true_type{});
  template <typename>
  static std::false_type test(...);
  static constexpr bool value = decltype(test<T>(0))::value;
};

template <typename T>
inline constexpr bool has_kpq_bits_v = has_kpq_bits<T>::value;

/**
 * Derive the CTA-local scalar penalty from the initial unpenalized candidate ordering.
 *
 * The interquartile span is less sensitive to isolated seed outliers than the full range. Small
 * candidate lists fall back to the full finite span. A zero/degenerate span deliberately produces
 * zero penalty, which is the safe default-traversal behavior.
 */
template <typename DistanceT>
RAFT_DEVICE_INLINE_FUNCTION DistanceT
favor_query_local_penalty(const DistanceT* __restrict__ sorted_distances,
                          const uint32_t retained_size,
                          const DistanceT reference_penalty,
                          const DistanceT local_gap_multiplier)
{
  uint32_t finite_count = 0;
  const auto upper      = raft::upper_bound<DistanceT>();
  while (finite_count < retained_size && sorted_distances[finite_count] < upper) {
    ++finite_count;
  }
  if (finite_count < 2 || reference_penalty <= DistanceT{0} ||
      local_gap_multiplier <= DistanceT{0}) {
    return DistanceT{0};
  }

  const uint32_t q1 = (finite_count - 1) / 4;
  const uint32_t q3 = (3 * (finite_count - 1)) / 4;
  DistanceT local_gap{};
  if (q3 > q1) {
    local_gap = (sorted_distances[q3] - sorted_distances[q1]) / static_cast<DistanceT>(q3 - q1);
  } else {
    local_gap = (sorted_distances[finite_count - 1] - sorted_distances[0]) /
                static_cast<DistanceT>(finite_count - 1);
  }
  if (!(local_gap > DistanceT{0})) { return DistanceT{0}; }

  const auto local_cap = local_gap_multiplier * local_gap;
  return local_cap < reference_penalty ? local_cap : reference_penalty;
}

template <typename DistanceT>
RAFT_DEVICE_INLINE_FUNCTION DistanceT
favor_retention_cutoff(const DistanceT* __restrict__ sorted_distances, const uint32_t retained_size)
{
  const auto upper = raft::upper_bound<DistanceT>();
  for (uint32_t i = retained_size; i > 0; --i) {
    if (sorted_distances[i - 1] < upper) { return sorted_distances[i - 1]; }
  }
  return upper;
}

template <typename DistanceT>
RAFT_DEVICE_INLINE_FUNCTION DistanceT favor_effective_penalty(const DistanceT raw_distance,
                                                              const DistanceT query_penalty,
                                                              const DistanceT retention_cutoff,
                                                              const bool retention_safe)
{
  if (!retention_safe) { return query_penalty; }
  const auto slack =
    retention_cutoff > raw_distance ? retention_cutoff - raw_distance : DistanceT{0};
  // Keep rejected candidates safely inside the current retention boundary. The midpoint leaves
  // equal headroom on either side and avoids floating-point ties at the cutoff.
  const auto retention_cap = static_cast<DistanceT>(0.5) * slack;
  return retention_cap < query_penalty ? retention_cap : query_penalty;
}

/**
 * Direct bitset test used only by the bitset-only retention-safe FAVOR specialization.
 *
 * This is equivalent to raft::core::bitset_view<uint32_t, SourceIndexT>::test(), including the
 * original_nbits remapping. Keeping the loaded payload in a local value lets the compiler retain
 * the bitset metadata across all candidates handled by this invocation.
 */
template <typename SourceIndexT>
RAFT_DEVICE_INLINE_FUNCTION bool favor_bitset_test(
  const cuvs::neighbors::detail::bitset_filter_data_t<SourceIndexT>& bitset,
  const SourceIndexT sample_index)
{
  if (bitset.bitset_ptr == nullptr) { return true; }

  constexpr SourceIndexT word_bits = sizeof(std::uint32_t) * 8;
  SourceIndexT word_index{};
  SourceIndexT bit_offset{};
  // A normal one-dimensional bitset has no row stride: its logical length and original length
  // are equal. Treat it like the already-supported packed cases so the hot candidate path uses
  // constant power-of-two division/modulo rather than general integer division. Sliced/strided
  // bitsets retain the exact RAFT remapping below.
  if (bitset.original_nbits == 0 || bitset.original_nbits == word_bits ||
      bitset.original_nbits == bitset.bitset_len) {
    word_index = sample_index / word_bits;
    bit_offset = sample_index % word_bits;
  } else {
    const auto original_word_index = sample_index / bitset.original_nbits;
    const auto original_bit_offset = sample_index % bitset.original_nbits;
    word_index                     = original_word_index * bitset.original_nbits / word_bits;
    if (bitset.original_nbits > word_bits) {
      word_index += original_bit_offset / word_bits;
      bit_offset = original_bit_offset % word_bits;
    } else {
      const auto ratio = word_bits / bitset.original_nbits;
      bit_offset += (original_word_index % ratio) * bitset.original_nbits;
      bit_offset += original_bit_offset % word_bits;
    }
  }

  const auto word = __ldg(bitset.bitset_ptr + word_index);
  return (word & (std::uint32_t{1} << bit_offset)) != 0;
}

template <typename SourceIndexT>
RAFT_DEVICE_INLINE_FUNCTION bool favor_packed_bitset_test(const std::uint32_t* bitset_ptr,
                                                          const SourceIndexT sample_index)
{
  constexpr SourceIndexT word_bits = sizeof(std::uint32_t) * 8;
  const auto word                  = __ldg(bitset_ptr + sample_index / word_bits);
  return (word & (std::uint32_t{1} << (sample_index % word_bits))) != 0;
}

// JIT version of compute_distance_to_random_nodes - uses const dataset_descriptor_base_t* (smem)
// Shared between single_cta and multi_cta JIT kernels
template <bool APPLY_FAVOR,
          typename IndexT,
          typename DistanceT,
          typename DataT,
          typename SourceIndexT>
RAFT_DEVICE_INLINE_FUNCTION void compute_distance_to_random_nodes_jit_impl(
  IndexT* __restrict__ result_indices_ptr,       // [num_pickup]
  DistanceT* __restrict__ result_distances_ptr,  // [num_pickup]
  const dataset_descriptor_base_t<DataT, IndexT, DistanceT>* smem_desc,
  const uint32_t num_pickup,
  const uint32_t num_distilation,
  const uint64_t rand_xor_mask,
  const IndexT* __restrict__ seed_ptr,  // [num_seeds]
  const uint32_t num_seeds,
  IndexT* __restrict__ visited_hash_ptr,
  const uint32_t visited_hash_bitlen,
  IndexT* __restrict__ traversed_hash_ptr,
  const uint32_t traversed_hash_bitlen,
  const uint32_t block_id                          = 0,
  const uint32_t num_blocks                        = 1,
  const IndexT graph_size                          = 0,
  const SourceIndexT* source_indices_ptr           = nullptr,
  const uint32_t query_id                          = 0,
  cagra_sample_filter<SourceIndexT> filter_payload = {},
  const DistanceT favor_penalty                    = DistanceT{0})
{
  uint32_t team_size_bits = smem_desc->team_size_bitshift_from_smem();
  IndexT dataset_size     = smem_desc->size;
  const auto args_load    = smem_desc->args.load();

  const auto max_i = raft::round_up_safe<uint32_t>(num_pickup, device::warp_size >> team_size_bits);
  const IndexT seed_index_limit = graph_size > 0 ? graph_size : dataset_size;

  for (uint32_t i = threadIdx.x >> team_size_bits; i < max_i; i += (blockDim.x >> team_size_bits)) {
    const bool valid_i = (i < num_pickup);

    IndexT best_index_team_local    = raft::upper_bound<IndexT>();
    DistanceT best_norm2_team_local = raft::upper_bound<DistanceT>();
    for (uint32_t j = 0; j < num_distilation; j++) {
      IndexT seed_index = 0;
      if (valid_i) {
        uint32_t gid = block_id + (num_blocks * (i + (num_pickup * j)));
        if (seed_ptr && (gid < num_seeds)) {
          seed_index = seed_ptr[gid];
        } else {
          seed_index = device::xorshift64(gid ^ rand_xor_mask) % seed_index_limit;
        }
      }

      auto norm2 = cuvs::neighbors::cagra::detail::compute_distance<DataT, IndexT, DistanceT>(
        args_load, seed_index, valid_i, team_size_bits);

      if constexpr (APPLY_FAVOR) {
        const unsigned team_width = 1u << team_size_bits;
        const unsigned lane_id    = threadIdx.x & (team_width - 1u);
        bool passes_filter        = true;
        if (valid_i && lane_id == 0) {
          auto source_id = source_indices_ptr == nullptr ? static_cast<SourceIndexT>(seed_index)
                                                         : source_indices_ptr[seed_index];
          passes_filter  = cuvs::neighbors::detail::sample_filter<SourceIndexT>(
            query_id, source_id, filter_payload.sample_filter_data());
        }
        passes_filter = __shfl_sync(0xffffffffu, passes_filter, 0, team_width);
        if (valid_i && !passes_filter) { norm2 += favor_penalty; }
      }

      if (valid_i && (norm2 < best_norm2_team_local)) {
        best_norm2_team_local = norm2;
        best_index_team_local = seed_index;
      }
    }

    const unsigned lane_id = threadIdx.x & ((1u << team_size_bits) - 1u);
    if (valid_i && lane_id == 0) {
      if (best_index_team_local != raft::upper_bound<IndexT>()) {
        if (hashmap::insert(visited_hash_ptr, visited_hash_bitlen, best_index_team_local) == 0) {
          // Deactivate this entry as insertion into visited hash table has failed.
          best_norm2_team_local = raft::upper_bound<DistanceT>();
          best_index_team_local = raft::upper_bound<IndexT>();
        } else if ((traversed_hash_ptr != nullptr) &&
                   hashmap::search<IndexT, 1>(
                     traversed_hash_ptr, traversed_hash_bitlen, best_index_team_local)) {
          // Deactivate this entry as it has been already used by others.
          best_norm2_team_local = raft::upper_bound<DistanceT>();
          best_index_team_local = raft::upper_bound<IndexT>();
        }
      }
      result_distances_ptr[i] = best_norm2_team_local;
      result_indices_ptr[i]   = best_index_team_local;
    }
  }
}

template <typename IndexT, typename DistanceT, typename DataT>
RAFT_DEVICE_INLINE_FUNCTION void compute_distance_to_random_nodes_jit(
  IndexT* __restrict__ result_indices_ptr,
  DistanceT* __restrict__ result_distances_ptr,
  const dataset_descriptor_base_t<DataT, IndexT, DistanceT>* smem_desc,
  const uint32_t num_pickup,
  const uint32_t num_distilation,
  const uint64_t rand_xor_mask,
  const IndexT* __restrict__ seed_ptr,
  const uint32_t num_seeds,
  IndexT* __restrict__ visited_hash_ptr,
  const uint32_t visited_hash_bitlen,
  IndexT* __restrict__ traversed_hash_ptr,
  const uint32_t traversed_hash_bitlen,
  const uint32_t block_id   = 0,
  const uint32_t num_blocks = 1,
  const IndexT graph_size   = 0)
{
  compute_distance_to_random_nodes_jit_impl<false, IndexT, DistanceT, DataT, IndexT>(
    result_indices_ptr,
    result_distances_ptr,
    smem_desc,
    num_pickup,
    num_distilation,
    rand_xor_mask,
    seed_ptr,
    num_seeds,
    visited_hash_ptr,
    visited_hash_bitlen,
    traversed_hash_ptr,
    traversed_hash_bitlen,
    block_id,
    num_blocks,
    graph_size);
}

template <typename IndexT, typename DistanceT, typename DataT, typename SourceIndexT>
RAFT_DEVICE_INLINE_FUNCTION void compute_favor_distance_to_random_nodes_jit(
  IndexT* __restrict__ result_indices_ptr,
  DistanceT* __restrict__ result_distances_ptr,
  const dataset_descriptor_base_t<DataT, IndexT, DistanceT>* smem_desc,
  const uint32_t num_pickup,
  const uint32_t num_distilation,
  const uint64_t rand_xor_mask,
  const IndexT* __restrict__ seed_ptr,
  const uint32_t num_seeds,
  IndexT* __restrict__ visited_hash_ptr,
  const uint32_t visited_hash_bitlen,
  const IndexT graph_size,
  const SourceIndexT* source_indices_ptr,
  const uint32_t query_id,
  cagra_sample_filter<SourceIndexT> filter_payload,
  const DistanceT favor_penalty,
  IndexT* __restrict__ traversed_hash_ptr = nullptr,
  const uint32_t traversed_hash_bitlen    = 0,
  const uint32_t block_id                 = 0,
  const uint32_t num_blocks               = 1)
{
  compute_distance_to_random_nodes_jit_impl<true, IndexT, DistanceT, DataT, SourceIndexT>(
    result_indices_ptr,
    result_distances_ptr,
    smem_desc,
    num_pickup,
    num_distilation,
    rand_xor_mask,
    seed_ptr,
    num_seeds,
    visited_hash_ptr,
    visited_hash_bitlen,
    traversed_hash_ptr,
    traversed_hash_bitlen,
    block_id,
    num_blocks,
    graph_size,
    source_indices_ptr,
    query_id,
    filter_payload,
    favor_penalty);
}

// JIT version of compute_distance_to_child_nodes - uses const dataset_descriptor_base_t* (smem)
// Shared between single_cta and multi_cta JIT kernels
template <bool APPLY_FAVOR,
          typename IndexT,
          typename DistanceT,
          typename DataT,
          typename SourceIndexT,
          int STATIC_RESULT_POSITION      = 1,
          bool RETENTION_SAFE_BITSET_ONLY = false,
          bool PACKED_BITSET              = false,
          bool DIRECT_SOURCE_ID           = false>
RAFT_DEVICE_INLINE_FUNCTION void compute_distance_to_child_nodes_jit_impl(
  IndexT* __restrict__ result_child_indices_ptr,
  DistanceT* __restrict__ result_child_distances_ptr,
  const dataset_descriptor_base_t<DataT, IndexT, DistanceT>* smem_desc,
  const IndexT* __restrict__ knn_graph,
  const uint32_t knn_k,
  IndexT* __restrict__ visited_hashmap_ptr,
  const uint32_t visited_hash_bitlen,
  IndexT* __restrict__ traversed_hashmap_ptr,
  const uint32_t traversed_hash_bitlen,
  const IndexT* __restrict__ parent_indices,
  const IndexT* __restrict__ internal_topk_list,
  const uint32_t search_width,
  int* __restrict__ result_position                = nullptr,
  const int max_result_position                    = 0,
  const SourceIndexT* source_indices_ptr           = nullptr,
  const uint32_t query_id                          = 0,
  cagra_sample_filter<SourceIndexT> filter_payload = {},
  const DistanceT favor_penalty                    = DistanceT{0},
  const DistanceT favor_retention_cutoff           = raft::upper_bound<DistanceT>(),
  const bool favor_retention_safe                  = false,
  const cuvs::neighbors::detail::bitset_filter_data_t<SourceIndexT> favor_bitset = {})
{
  constexpr IndexT index_msb_1_mask = utils::gen_index_msb_1_mask<IndexT>::value;
  constexpr IndexT invalid_index    = ~static_cast<IndexT>(0);

  // Read child indices of parents from knn graph and check if the distance computation is
  // necessary.
  for (uint32_t i = threadIdx.x; i < knn_k * search_width; i += blockDim.x) {
    const IndexT smem_parent_id = parent_indices[i / knn_k];
    IndexT child_id             = invalid_index;
    if (smem_parent_id != invalid_index) {
      const auto parent_id = internal_topk_list[smem_parent_id] & ~index_msb_1_mask;
      child_id             = knn_graph[(i % knn_k) + (static_cast<int64_t>(knn_k) * parent_id)];
    }
    if (child_id != invalid_index) {
      if (hashmap::insert(visited_hashmap_ptr, visited_hash_bitlen, child_id) == 0) {
        child_id = invalid_index;
      } else if ((traversed_hashmap_ptr != nullptr) &&
                 hashmap::search<IndexT, 1>(
                   traversed_hashmap_ptr, traversed_hash_bitlen, child_id)) {
        child_id = invalid_index;
      }
    }
    if (STATIC_RESULT_POSITION) {
      result_child_indices_ptr[i] = child_id;
    } else if (child_id != invalid_index) {
      int j                       = atomicSub(result_position, 1) - 1;
      result_child_indices_ptr[j] = child_id;
    }
  }
  __syncthreads();

  // Same inline distance pattern as search_single_cta_jit.cuh / device helpers
  const auto team_size_bits = smem_desc->team_size_bitshift_from_smem();
  const auto num_k          = knn_k * search_width;
  const auto max_i          = raft::round_up_safe(num_k, device::warp_size >> team_size_bits);
  const auto args           = smem_desc->args.load();
  const bool lead_lane      = (threadIdx.x & ((1u << team_size_bits) - 1u)) == 0;
  const uint32_t ofst       = STATIC_RESULT_POSITION ? 0 : result_position[0];

  for (uint32_t i = threadIdx.x >> team_size_bits; i < max_i; i += blockDim.x >> team_size_bits) {
    const auto j        = i + ofst;
    const bool valid_i  = STATIC_RESULT_POSITION ? (j < num_k) : (j < max_result_position);
    const auto child_id = valid_i ? result_child_indices_ptr[j] : invalid_index;

    const auto per_thread =
      (child_id != invalid_index)
        ? cuvs::neighbors::cagra::detail::compute_distance_per_thread<DataT, IndexT, DistanceT>(
            args, child_id)
        : (lead_lane ? raft::upper_bound<DistanceT>() : 0);
    const DistanceT child_dist = device::team_sum(per_thread, team_size_bits);
    __syncwarp();

    // Store the distance once, fusing FAVOR's bitset check into the lead lane.
    if (valid_i && lead_lane) {
      auto final_dist = child_dist;
      if constexpr (APPLY_FAVOR) {
        if constexpr (RETENTION_SAFE_BITSET_ONLY) {
          // Invalid candidates have upper-bound distance and fail this comparison. Combining
          // validity and retention avoids a second predicate in the bitset-only hot path.
          if (favor_penalty > DistanceT{0} && child_dist < favor_retention_cutoff) {
            const auto source_id = DIRECT_SOURCE_ID ? static_cast<SourceIndexT>(child_id)
                                                    : (source_indices_ptr == nullptr
                                                         ? static_cast<SourceIndexT>(child_id)
                                                         : source_indices_ptr[child_id]);
            const bool passes_filter =
              PACKED_BITSET ? favor_packed_bitset_test(favor_bitset.bitset_ptr, source_id)
                            : favor_bitset_test(favor_bitset, source_id);
            if (!passes_filter) {
              final_dist +=
                favor_effective_penalty(child_dist, favor_penalty, favor_retention_cutoff, true);
            }
          }
        } else {
          if (child_id != invalid_index) {
            const auto source_id = source_indices_ptr == nullptr
                                     ? static_cast<SourceIndexT>(child_id)
                                     : source_indices_ptr[child_id];
            if (!cuvs::neighbors::detail::sample_filter<SourceIndexT>(
                  query_id, source_id, filter_payload.sample_filter_data())) {
              final_dist += favor_effective_penalty(
                child_dist, favor_penalty, favor_retention_cutoff, favor_retention_safe);
            }
          }
        }
      }
      result_child_distances_ptr[j] = final_dist;
    }
  }
}

template <typename IndexT, typename DistanceT, typename DataT, int STATIC_RESULT_POSITION = 1>
RAFT_DEVICE_INLINE_FUNCTION void compute_distance_to_child_nodes_jit(
  IndexT* __restrict__ result_child_indices_ptr,
  DistanceT* __restrict__ result_child_distances_ptr,
  const dataset_descriptor_base_t<DataT, IndexT, DistanceT>* smem_desc,
  const IndexT* __restrict__ knn_graph,
  const uint32_t knn_k,
  IndexT* __restrict__ visited_hashmap_ptr,
  const uint32_t visited_hash_bitlen,
  IndexT* __restrict__ traversed_hashmap_ptr,
  const uint32_t traversed_hash_bitlen,
  const IndexT* __restrict__ parent_indices,
  const IndexT* __restrict__ internal_topk_list,
  const uint32_t search_width,
  int* __restrict__ result_position = nullptr,
  const int max_result_position     = 0)
{
  compute_distance_to_child_nodes_jit_impl<false,
                                           IndexT,
                                           DistanceT,
                                           DataT,
                                           IndexT,
                                           STATIC_RESULT_POSITION>(result_child_indices_ptr,
                                                                   result_child_distances_ptr,
                                                                   smem_desc,
                                                                   knn_graph,
                                                                   knn_k,
                                                                   visited_hashmap_ptr,
                                                                   visited_hash_bitlen,
                                                                   traversed_hashmap_ptr,
                                                                   traversed_hash_bitlen,
                                                                   parent_indices,
                                                                   internal_topk_list,
                                                                   search_width,
                                                                   result_position,
                                                                   max_result_position);
}

template <typename IndexT,
          typename DistanceT,
          typename DataT,
          typename SourceIndexT,
          int STATIC_RESULT_POSITION = 1>
RAFT_DEVICE_INLINE_FUNCTION void compute_favor_distance_to_child_nodes_jit(
  IndexT* __restrict__ result_child_indices_ptr,
  DistanceT* __restrict__ result_child_distances_ptr,
  const dataset_descriptor_base_t<DataT, IndexT, DistanceT>* smem_desc,
  const IndexT* __restrict__ knn_graph,
  const uint32_t knn_k,
  IndexT* __restrict__ visited_hashmap_ptr,
  const uint32_t visited_hash_bitlen,
  const IndexT* __restrict__ parent_indices,
  const IndexT* __restrict__ internal_topk_list,
  const uint32_t search_width,
  const SourceIndexT* source_indices_ptr,
  const uint32_t query_id,
  cagra_sample_filter<SourceIndexT> filter_payload,
  const DistanceT favor_penalty,
  const DistanceT favor_retention_cutoff     = raft::upper_bound<DistanceT>(),
  const bool favor_retention_safe            = false,
  IndexT* __restrict__ traversed_hashmap_ptr = nullptr,
  const uint32_t traversed_hash_bitlen       = 0,
  int* __restrict__ result_position          = nullptr,
  const int max_result_position              = 0)
{
  compute_distance_to_child_nodes_jit_impl<true,
                                           IndexT,
                                           DistanceT,
                                           DataT,
                                           SourceIndexT,
                                           STATIC_RESULT_POSITION>(result_child_indices_ptr,
                                                                   result_child_distances_ptr,
                                                                   smem_desc,
                                                                   knn_graph,
                                                                   knn_k,
                                                                   visited_hashmap_ptr,
                                                                   visited_hash_bitlen,
                                                                   traversed_hashmap_ptr,
                                                                   traversed_hash_bitlen,
                                                                   parent_indices,
                                                                   internal_topk_list,
                                                                   search_width,
                                                                   result_position,
                                                                   max_result_position,
                                                                   source_indices_ptr,
                                                                   query_id,
                                                                   filter_payload,
                                                                   favor_penalty,
                                                                   favor_retention_cutoff,
                                                                   favor_retention_safe);
}

template <typename IndexT,
          typename DistanceT,
          typename DataT,
          typename SourceIndexT,
          int STATIC_RESULT_POSITION = 1,
          bool PACKED_BITSET         = false,
          bool DIRECT_SOURCE_ID      = false>
RAFT_DEVICE_INLINE_FUNCTION void compute_favor_retention_safe_distance_to_child_nodes_jit(
  IndexT* __restrict__ result_child_indices_ptr,
  DistanceT* __restrict__ result_child_distances_ptr,
  const dataset_descriptor_base_t<DataT, IndexT, DistanceT>* smem_desc,
  const IndexT* __restrict__ knn_graph,
  const uint32_t knn_k,
  IndexT* __restrict__ visited_hashmap_ptr,
  const uint32_t visited_hash_bitlen,
  const IndexT* __restrict__ parent_indices,
  const IndexT* __restrict__ internal_topk_list,
  const uint32_t search_width,
  const SourceIndexT* source_indices_ptr,
  const cuvs::neighbors::detail::bitset_filter_data_t<SourceIndexT> favor_bitset,
  const DistanceT favor_penalty,
  const DistanceT favor_retention_cutoff,
  IndexT* __restrict__ traversed_hashmap_ptr = nullptr,
  const uint32_t traversed_hash_bitlen       = 0,
  int* __restrict__ result_position          = nullptr,
  const int max_result_position              = 0)
{
  compute_distance_to_child_nodes_jit_impl<true,
                                           IndexT,
                                           DistanceT,
                                           DataT,
                                           SourceIndexT,
                                           STATIC_RESULT_POSITION,
                                           true,
                                           PACKED_BITSET,
                                           DIRECT_SOURCE_ID>(result_child_indices_ptr,
                                                             result_child_distances_ptr,
                                                             smem_desc,
                                                             knn_graph,
                                                             knn_k,
                                                             visited_hashmap_ptr,
                                                             visited_hash_bitlen,
                                                             traversed_hashmap_ptr,
                                                             traversed_hash_bitlen,
                                                             parent_indices,
                                                             internal_topk_list,
                                                             search_width,
                                                             result_position,
                                                             max_result_position,
                                                             source_indices_ptr,
                                                             0,
                                                             {},
                                                             favor_penalty,
                                                             favor_retention_cutoff,
                                                             true,
                                                             favor_bitset);
}

}  // namespace cuvs::neighbors::cagra::detail::device
