/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstdint>

namespace cuvs::neighbors::cagra::detail::favor_search_diagnostics {

inline constexpr std::uint32_t schema_version    = 5;
inline constexpr std::uint32_t ground_truth_k    = 10;
inline constexpr std::uint32_t invalid_iteration = 0xffffffffu;

enum class stop_reason : std::uint32_t {
  unknown = 0,
  max_with_unexpanded_frontier,
  max_with_empty_frontier,
  frontier_exhausted,
  filter_empty_or_skipped,
  adaptive_converged,
  adaptive_safety_cap,
};

enum class hash_outcome : std::uint8_t {
  unknown = 0,
  inserted,
  duplicate,
  full,
};

/** One fixed-size record per query. The diagnostic kernel is the only writer. */
struct query_summary {
  std::uint32_t schema = schema_version;
  std::uint32_t query_id{};
  std::uint32_t iterations{};
  std::uint32_t resolved_max_iterations{};
  std::uint32_t reason{};
  std::uint32_t terminal_valid{};
  std::uint32_t terminal_pass{};
  std::uint32_t terminal_reject{};
  std::uint32_t terminal_unexpanded_pass{};
  std::uint32_t terminal_unexpanded_reject{};
  std::uint32_t expanded_pass_parents{};
  std::uint32_t expanded_reject_parents{};
  std::uint32_t candidate_attempts{};
  std::uint32_t candidate_evaluations{};
  std::uint32_t candidate_duplicate_or_full{};
  std::uint32_t candidate_duplicates{};
  std::uint32_t candidate_hash_full{};
  std::uint32_t passing_candidates{};
  std::uint32_t rejected_candidates{};
  std::uint32_t penalized_candidates{};
  std::uint32_t accumulator_observations{};
  std::uint32_t accumulator_insertions{};
  std::uint32_t gt_seen_mask{};
  std::uint32_t gt_first_iteration[ground_truth_k]{};
  std::uint32_t output_count{};
  std::uint32_t hash_bitlen{};
  std::uint32_t small_hash_bitlen{};
  std::uint32_t small_hash_reset_interval{};
  float resolved_filtering_rate{};
  float reference_penalty{};
  float query_penalty{};
  float terminal_cutoff{};
  float best_unexpanded_distance{};
  float worst_retained_distance{};
  float kth_passing_raw_distance{};
  float recall{};  // populated by the bench host after the kernel completes
};

/** One record for every retained-frontier snapshot of a selected query. */
struct iteration_record {
  std::uint32_t query_id{};
  std::uint32_t iteration{};
  std::uint32_t valid{};
  std::uint32_t passing{};
  std::uint32_t rejected{};
  std::uint32_t unexpanded_passing{};
  std::uint32_t unexpanded_rejected{};
  std::uint32_t selected_passing_parents{};
  std::uint32_t selected_rejected_parents{};
  std::uint32_t child_attempts{};
  std::uint32_t child_evaluations{};
  std::uint32_t child_duplicate_or_full{};
  std::uint32_t child_duplicates{};
  std::uint32_t child_hash_full{};
  std::uint32_t child_passing{};
  std::uint32_t child_rejected{};
  std::uint32_t stop_reason{};
  float penalty{};
  float cutoff{};
  float best_unexpanded_distance{};
  float worst_retained_distance{};
};

/** Fixed-position child record. Empty/hash-rejected child slots remain explicitly visible. */
struct candidate_record {
  std::uint32_t query_id{};
  std::uint32_t iteration{};
  std::uint32_t parent_id{0xffffffffu};
  std::uint32_t child_id{0xffffffffu};
  float raw_distance{};
  float effective_penalty{};
  float final_distance{};
  std::int16_t ground_truth_rank{-1};
  std::uint8_t passes_filter{};
  std::uint8_t hash_result{};
  std::uint8_t survived_next_merge{};
  std::uint8_t valid{};
  std::uint16_t reserved{};
};

/**
 * A sparse retained-frontier snapshot used to evaluate termination rules offline.
 *
 * Checkpoints are ordered per query. The top-k fields contain passing source IDs in the logical
 * CAGRA distance order, so their distances are raw (unpenalized) FAVOR distances.
 */
struct termination_checkpoint_record {
  std::uint32_t query_id{};
  std::uint32_t checkpoint{};
  std::uint32_t iteration{};
  std::uint32_t expanded_parents{};
  std::uint32_t cumulative_candidate_evaluations{};
  std::uint32_t cumulative_passing_candidates{};
  std::uint32_t cumulative_candidate_duplicates{};
  std::uint32_t prefix_valid{};
  std::uint32_t prefix_pass{};
  std::uint32_t passing_count{};
  std::uint32_t output_count{};
  float frontier_best{};
  float prefix_boundary{};
  float kth_passing_raw_distance{};
  std::uint32_t top_ids[ground_truth_k]{};
  float top_distances[ground_truth_k]{};
};

/** Device-resident pointer table passed privately through the iteration-statistics argument. */
struct context {
  query_summary* summaries{};
  const std::uint32_t* ground_truth_ids{};    // [num_queries, ground_truth_k]
  const std::int32_t* trace_slot_by_query{};  // -1 means summary only
  iteration_record* iteration_records{};
  candidate_record* candidate_records{};
  // Optional benchmark-only terminal snapshot. IDs retain CAGRA's expanded MSB and are written in
  // logical sorted order before final filter compaction. Normal diagnostics leave these null.
  std::uint32_t* terminal_tagged_ids{};  // [num_queries, terminal_stride]
  float* terminal_distances{};           // [num_queries, terminal_stride]
  std::uint8_t* terminal_flags{};        // bit 0 valid, bit 1 expanded, bit 2 passes filter
  std::uint32_t terminal_stride{};
  // Optional benchmark-only sparse trajectory. Counts are both the device write cursor and the
  // number of valid records for each query.
  termination_checkpoint_record* termination_checkpoints{};
  std::uint32_t* termination_checkpoint_counts{};
  std::uint32_t termination_checkpoint_stride{};
  // Checkpoint recording can begin before termination_start_iteration (the B0 eligibility point).
  // B0 and the terminal iteration are always captured even when they fall between intervals.
  std::uint32_t termination_record_start_iteration{};
  std::uint32_t termination_start_iteration{};
  std::uint32_t termination_parent_interval{};
  std::uint32_t num_queries{};
  std::uint32_t max_trace_iterations{};
  std::uint32_t candidates_per_iteration{};
};

}  // namespace cuvs::neighbors::cagra::detail::favor_search_diagnostics
