/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "../../../../src/neighbors/detail/cagra/favor_search_diagnostics.hpp"
#include "../common/blob.hpp"

#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cudart_utils.hpp>
#include <rmm/device_uvector.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace cuvs::bench::detail {

namespace favor_retry_diag = cuvs::neighbors::cagra::detail::favor_search_diagnostics;

enum class favor_retry_strategy { independent, passing, frontier, combined, oracle };

inline auto parse_favor_retry_strategy(const std::string& value) -> favor_retry_strategy
{
  if (value == "independent") { return favor_retry_strategy::independent; }
  if (value == "passing") { return favor_retry_strategy::passing; }
  if (value == "frontier") { return favor_retry_strategy::frontier; }
  if (value == "combined") { return favor_retry_strategy::combined; }
  if (value == "oracle") { return favor_retry_strategy::oracle; }
  throw std::invalid_argument(
    "favor_retry_strategy must be independent, passing, frontier, combined, or oracle");
}

inline auto favor_retry_strategy_name(favor_retry_strategy value) -> const char*
{
  switch (value) {
    case favor_retry_strategy::independent: return "independent";
    case favor_retry_strategy::passing: return "passing";
    case favor_retry_strategy::frontier: return "frontier";
    case favor_retry_strategy::combined: return "combined";
    case favor_retry_strategy::oracle: return "oracle";
  }
  return "invalid";
}

struct favor_retry_diagnostic_config {
  std::string output_directory;
  std::string ground_truth_file;
  std::string dataset;
  std::string strategy;
  std::uint32_t rounds = 0;
  std::uint32_t b0     = 0;

  [[nodiscard]] bool enabled() const noexcept { return !output_directory.empty(); }
};

/**
 * Deliberately untimed, benchmark-only retry diagnostic.
 *
 * Restart variants launch a fresh B0 traversal per round. Passing/frontier/combined variants seed
 * that fresh traversal from prior state, while oracle launches one fresh but uninterrupted search
 * at B0, 2*B0, ... and therefore measures what preserving all in-kernel state can recover.
 */
class favor_retry_diagnostic_session {
 public:
  explicit favor_retry_diagnostic_session(favor_retry_diagnostic_config config)
    : config_{std::move(config)}, strategy_{parse_favor_retry_strategy(config_.strategy)}
  {
  }

  template <typename SearchFn>
  void capture(const raft::resources& res,
               std::uint32_t num_queries,
               std::uint32_t topk,
               std::uint32_t graph_degree,
               std::uint32_t search_width,
               std::int64_t dataset_size,
               std::uint32_t configured_itopk,
               float filtering_rate,
               std::int64_t* result_indices,
               float* result_distances,
               SearchFn&& search)
  {
    std::lock_guard<std::mutex> guard{mutex_};
    auto stream = raft::resource::get_cuda_stream(res);
    if (captured_) {
      if (cached_indices_.size() != static_cast<std::uint64_t>(num_queries) * topk) {
        throw std::invalid_argument(
          "retry diagnostic session cannot be reused with different query or top-k dimensions");
      }
      copy_cached_result(stream, result_indices, result_distances);
      return;
    }
    if (config_.ground_truth_file.empty()) {
      throw std::invalid_argument("favor_retry_diagnostics_groundtruth must be specified");
    }
    if (config_.rounds < 2 || config_.rounds > 7) {
      throw std::invalid_argument("favor_retry_rounds must be between 2 and 7");
    }
    if (config_.b0 == 0) { throw std::invalid_argument("favor_retry_b0 must be nonzero"); }
    if (topk == 0 || configured_itopk == 0) {
      throw std::invalid_argument("retry diagnostics require nonzero topk and itopk");
    }

    const auto internal_topk = round_up_32(configured_itopk);
    if (internal_topk > 512) {
      throw std::invalid_argument("SINGLE_CTA retry diagnostics require rounded itopk <= 512");
    }
    const auto initial_width = internal_topk + search_width * graph_degree;
    const auto output_size   = static_cast<std::uint64_t>(num_queries) * topk;
    const auto frontier_size = static_cast<std::uint64_t>(num_queries) * internal_topk;

    auto ground_truth = load_ground_truth(num_queries, topk);
    rmm::device_uvector<std::int64_t> round_indices(output_size, stream);
    rmm::device_uvector<float> round_distances(output_size, stream);
    rmm::device_uvector<std::uint32_t> terminal_ids(frontier_size, stream);
    rmm::device_uvector<float> terminal_distances(frontier_size, stream);
    rmm::device_uvector<std::uint8_t> terminal_flags(frontier_size, stream);
    rmm::device_uvector<favor_retry_diag::query_summary> summaries(num_queries, stream);
    rmm::device_uvector<favor_retry_diag::context> context(1, stream);
    rmm::device_uvector<std::uint32_t> seeds(
      static_cast<std::uint64_t>(num_queries) * initial_width, stream);

    std::vector<std::int64_t> round_indices_host(output_size);
    std::vector<float> round_distances_host(output_size);
    std::vector<std::uint32_t> terminal_ids_host(frontier_size);
    std::vector<float> terminal_distances_host(frontier_size);
    std::vector<std::uint8_t> terminal_flags_host(frontier_size);
    std::vector<favor_retry_diag::query_summary> summaries_host(num_queries);
    std::vector<std::int64_t> accumulator_indices(output_size,
                                                  std::numeric_limits<std::int64_t>::max());
    std::vector<float> accumulator_distances(output_size, std::numeric_limits<float>::max());
    std::vector<std::int64_t> previous_indices(output_size,
                                               std::numeric_limits<std::int64_t>::max());
    std::vector<std::uint32_t> seed_host;
    std::vector<std::uint32_t> foreground_seed_counts(num_queries, 0);
    std::vector<query_round_metric> metrics;
    metrics.reserve(static_cast<std::uint64_t>(num_queries) * config_.rounds);
    std::vector<std::int64_t> all_round_outputs;
    all_round_outputs.reserve(output_size * config_.rounds);
    std::vector<std::uint32_t> selected_queries;
    std::vector<std::uint32_t> selected_terminal_ids;
    std::vector<float> selected_terminal_distances;
    std::vector<std::uint8_t> selected_terminal_flags;

    favor_retry_diag::context host_context{};
    host_context.summaries           = summaries.data();
    host_context.terminal_tagged_ids = terminal_ids.data();
    host_context.terminal_distances  = terminal_distances.data();
    host_context.terminal_flags      = terminal_flags.data();
    host_context.terminal_stride     = internal_topk;
    host_context.num_queries         = num_queries;
    RAFT_CUDA_TRY(cudaMemcpyAsync(
      context.data(), &host_context, sizeof(host_context), cudaMemcpyHostToDevice, stream));

    for (std::uint32_t round = 0; round < config_.rounds; ++round) {
      const auto budget = strategy_ == favor_retry_strategy::oracle
                            ? checked_budget(config_.b0, round + 1)
                            : config_.b0;
      const auto mask =
        strategy_ == favor_retry_strategy::independent ? independent_mask(round) : base_mask;
      const std::uint32_t* seed_ptr = nullptr;
      std::uint32_t seed_count      = 0;
      if (round != 0 && uses_saved_state()) {
        seed_host = build_retry_seeds(num_queries,
                                      topk,
                                      internal_topk,
                                      initial_width,
                                      dataset_size,
                                      accumulator_indices,
                                      terminal_ids_host,
                                      terminal_flags_host,
                                      foreground_seed_counts);
        RAFT_CUDA_TRY(cudaMemcpyAsync(seeds.data(),
                                      seed_host.data(),
                                      seed_host.size() * sizeof(std::uint32_t),
                                      cudaMemcpyHostToDevice,
                                      stream));
        seed_ptr   = seeds.data();
        seed_count = initial_width;
      } else {
        std::fill(foreground_seed_counts.begin(), foreground_seed_counts.end(), 0);
      }

      RAFT_CUDA_TRY(
        cudaMemsetAsync(terminal_flags.data(), 0, frontier_size * sizeof(std::uint8_t), stream));
      search(budget,
             mask,
             context.data(),
             seed_ptr,
             seed_count,
             round_indices.data(),
             round_distances.data());
      RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
      copy_round_to_host(round_indices,
                         round_distances,
                         terminal_ids,
                         terminal_distances,
                         terminal_flags,
                         summaries,
                         round_indices_host,
                         round_distances_host,
                         terminal_ids_host,
                         terminal_distances_host,
                         terminal_flags_host,
                         summaries_host);

      all_round_outputs.insert(
        all_round_outputs.end(), round_indices_host.begin(), round_indices_host.end());
      record_metrics(round,
                     num_queries,
                     topk,
                     internal_topk,
                     dataset_size,
                     ground_truth,
                     round_indices_host,
                     previous_indices,
                     accumulator_indices,
                     accumulator_distances,
                     round_distances_host,
                     terminal_flags_host,
                     summaries_host,
                     foreground_seed_counts,
                     metrics);
      previous_indices = round_indices_host;

      if (round == 0) {
        selected_queries = select_queries(metrics, num_queries);
        const auto selected_size =
          static_cast<std::uint64_t>(selected_queries.size()) * config_.rounds * internal_topk;
        selected_terminal_ids.reserve(selected_size);
        selected_terminal_distances.reserve(selected_size);
        selected_terminal_flags.reserve(selected_size);
      }
      append_selected_frontier(selected_queries,
                               internal_topk,
                               terminal_ids_host,
                               terminal_distances_host,
                               terminal_flags_host,
                               selected_terminal_ids,
                               selected_terminal_distances,
                               selected_terminal_flags);
    }

    cached_indices_   = std::move(accumulator_indices);
    cached_distances_ = std::move(accumulator_distances);
    write_capture(metrics,
                  all_round_outputs,
                  selected_queries,
                  selected_terminal_ids,
                  selected_terminal_distances,
                  selected_terminal_flags,
                  num_queries,
                  topk,
                  graph_degree,
                  search_width,
                  dataset_size,
                  configured_itopk,
                  internal_topk,
                  initial_width,
                  filtering_rate);
    captured_ = true;
    copy_cached_result(stream, result_indices, result_distances);
  }

 private:
  struct query_round_metric {
    std::uint32_t query{};
    std::uint32_t round{};
    std::uint32_t budget{};
    std::uint32_t individual_matches{};
    std::uint32_t accumulated_matches{};
    std::uint32_t output_count{};
    std::uint32_t new_ids{};
    std::uint32_t new_ground_truth{};
    std::uint32_t frontier_valid{};
    std::uint32_t frontier_unexpanded{};
    std::uint32_t frontier_unexpanded_pass{};
    std::uint32_t frontier_unexpanded_reject{};
    std::uint32_t seed_count{};
    std::uint32_t candidate_evaluations{};
    std::uint32_t candidate_duplicates{};
    std::uint32_t candidate_hash_full{};
    float jaccard_previous{};
  };

  struct neighbor {
    float distance;
    std::int64_t index;
  };

  static constexpr std::uint64_t base_mask = 0x128394ULL;

  [[nodiscard]] bool uses_saved_state() const noexcept
  {
    return strategy_ == favor_retry_strategy::passing ||
           strategy_ == favor_retry_strategy::frontier ||
           strategy_ == favor_retry_strategy::combined;
  }

  static auto round_up_32(std::uint32_t value) -> std::uint32_t { return (value + 31u) & ~31u; }

  static auto checked_budget(std::uint32_t b0, std::uint32_t multiplier) -> std::uint32_t
  {
    const auto result = static_cast<std::uint64_t>(b0) * multiplier;
    if (result > std::numeric_limits<std::uint32_t>::max()) {
      throw std::overflow_error("retry diagnostic iteration budget overflows uint32");
    }
    return static_cast<std::uint32_t>(result);
  }

  static auto splitmix64(std::uint64_t value) -> std::uint64_t
  {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
  }

  static auto independent_mask(std::uint32_t round) -> std::uint64_t
  {
    constexpr std::uint64_t fixed[] = {base_mask, 0x9e3779b97f58ff81ULL, 0xd1b54a32d1806e97ULL};
    return round < 3 ? fixed[round] : splitmix64(base_mask + round);
  }

  static auto xorshift64(std::uint64_t value) -> std::uint64_t
  {
    value ^= value >> 12;
    value ^= value << 25;
    value ^= value >> 27;
    return value * 0x2545f4914f6cdd1dULL;
  }

  auto load_ground_truth(std::uint32_t num_queries, std::uint32_t topk) const
    -> std::vector<std::uint32_t>
  {
    blob<std::int32_t> gt{config_.ground_truth_file};
    if (gt.n_rows() < num_queries || gt.n_cols() < topk) {
      throw std::invalid_argument("retry diagnostic ground truth does not contain nq x topk");
    }
    std::vector<std::uint32_t> result(static_cast<std::uint64_t>(num_queries) * topk);
    for (std::uint32_t query = 0; query < num_queries; ++query) {
      for (std::uint32_t rank = 0; rank < topk; ++rank) {
        result[static_cast<std::uint64_t>(query) * topk + rank] = static_cast<std::uint32_t>(
          gt.data()[static_cast<std::uint64_t>(query) * gt.n_cols() + rank]);
      }
    }
    return result;
  }

  auto build_retry_seeds(std::uint32_t num_queries,
                         std::uint32_t topk,
                         std::uint32_t internal_topk,
                         std::uint32_t initial_width,
                         std::int64_t dataset_size,
                         const std::vector<std::int64_t>& accumulator,
                         const std::vector<std::uint32_t>& terminal_ids,
                         const std::vector<std::uint8_t>& terminal_flags,
                         std::vector<std::uint32_t>& foreground_counts) const
    -> std::vector<std::uint32_t>
  {
    std::vector<std::uint32_t> result(static_cast<std::uint64_t>(num_queries) * initial_width);
    for (std::uint32_t query = 0; query < num_queries; ++query) {
      auto* output        = result.data() + static_cast<std::uint64_t>(query) * initial_width;
      std::uint32_t count = 0;
      std::unordered_set<std::uint32_t> unique;
      auto append = [&](std::uint32_t index) {
        if (index < static_cast<std::uint64_t>(dataset_size) && unique.insert(index).second &&
            count < initial_width) {
          output[count++] = index;
        }
      };

      if (strategy_ == favor_retry_strategy::passing ||
          strategy_ == favor_retry_strategy::combined) {
        for (std::uint32_t rank = 0; rank < topk; ++rank) {
          const auto index = accumulator[static_cast<std::uint64_t>(query) * topk + rank];
          if (index >= 0 && index < dataset_size) { append(static_cast<std::uint32_t>(index)); }
        }
      }
      if (strategy_ == favor_retry_strategy::frontier ||
          strategy_ == favor_retry_strategy::combined) {
        const auto offset = static_cast<std::uint64_t>(query) * internal_topk;
        for (std::uint32_t rank = 0; rank < internal_topk; ++rank) {
          const auto flags = terminal_flags[offset + rank];
          if ((flags & 1u) != 0 && (flags & 2u) == 0) {
            append(terminal_ids[offset + rank] & 0x7fffffffu);
          }
        }
      }
      foreground_counts[query] = count;
      for (std::uint32_t position = count; position < initial_width; ++position) {
        output[position] = static_cast<std::uint32_t>(
          xorshift64(static_cast<std::uint64_t>(position) ^ base_mask) % dataset_size);
      }
    }
    return result;
  }

  template <typename T>
  static void copy_device_vector(const rmm::device_uvector<T>& source, std::vector<T>& target)
  {
    RAFT_CUDA_TRY(
      cudaMemcpy(target.data(), source.data(), target.size() * sizeof(T), cudaMemcpyDeviceToHost));
  }

  static void copy_round_to_host(
    const rmm::device_uvector<std::int64_t>& round_indices,
    const rmm::device_uvector<float>& round_distances,
    const rmm::device_uvector<std::uint32_t>& terminal_ids,
    const rmm::device_uvector<float>& terminal_distances,
    const rmm::device_uvector<std::uint8_t>& terminal_flags,
    const rmm::device_uvector<favor_retry_diag::query_summary>& summaries,
    std::vector<std::int64_t>& round_indices_host,
    std::vector<float>& round_distances_host,
    std::vector<std::uint32_t>& terminal_ids_host,
    std::vector<float>& terminal_distances_host,
    std::vector<std::uint8_t>& terminal_flags_host,
    std::vector<favor_retry_diag::query_summary>& summaries_host)
  {
    copy_device_vector(round_indices, round_indices_host);
    copy_device_vector(round_distances, round_distances_host);
    copy_device_vector(terminal_ids, terminal_ids_host);
    copy_device_vector(terminal_distances, terminal_distances_host);
    copy_device_vector(terminal_flags, terminal_flags_host);
    copy_device_vector(summaries, summaries_host);
  }

  static bool valid_index(std::int64_t index, std::int64_t dataset_size)
  {
    return index >= 0 && index < dataset_size;
  }

  static auto count_matches(const std::int64_t* indices,
                            const std::uint32_t* ground_truth,
                            std::uint32_t topk,
                            std::int64_t dataset_size) -> std::uint32_t
  {
    std::uint32_t matches = 0;
    for (std::uint32_t rank = 0; rank < topk; ++rank) {
      if (!valid_index(indices[rank], dataset_size)) { continue; }
      for (std::uint32_t gt_rank = 0; gt_rank < topk; ++gt_rank) {
        if (indices[rank] == ground_truth[gt_rank]) {
          ++matches;
          break;
        }
      }
    }
    return matches;
  }

  static auto contains(const std::int64_t* indices,
                       std::uint32_t topk,
                       std::int64_t value,
                       std::int64_t dataset_size) -> bool
  {
    if (!valid_index(value, dataset_size)) { return false; }
    for (std::uint32_t rank = 0; rank < topk; ++rank) {
      if (indices[rank] == value) { return true; }
    }
    return false;
  }

  static void merge_query(const std::int64_t* incoming_indices,
                          const float* incoming_distances,
                          std::int64_t* accumulated_indices,
                          float* accumulated_distances,
                          std::uint32_t topk,
                          std::int64_t dataset_size)
  {
    std::vector<neighbor> candidates;
    candidates.reserve(2 * topk);
    for (std::uint32_t rank = 0; rank < topk; ++rank) {
      if (valid_index(accumulated_indices[rank], dataset_size)) {
        candidates.push_back({accumulated_distances[rank], accumulated_indices[rank]});
      }
      if (valid_index(incoming_indices[rank], dataset_size)) {
        candidates.push_back({incoming_distances[rank], incoming_indices[rank]});
      }
    }
    std::sort(candidates.begin(), candidates.end(), [](const neighbor& lhs, const neighbor& rhs) {
      return lhs.distance < rhs.distance || (lhs.distance == rhs.distance && lhs.index < rhs.index);
    });
    std::unordered_set<std::int64_t> unique;
    std::uint32_t output = 0;
    for (const auto& candidate : candidates) {
      if (!unique.insert(candidate.index).second) { continue; }
      accumulated_indices[output]   = candidate.index;
      accumulated_distances[output] = candidate.distance;
      if (++output == topk) { break; }
    }
    while (output < topk) {
      accumulated_indices[output]   = std::numeric_limits<std::int64_t>::max();
      accumulated_distances[output] = std::numeric_limits<float>::max();
      ++output;
    }
  }

  static void record_metrics(std::uint32_t round,
                             std::uint32_t num_queries,
                             std::uint32_t topk,
                             std::uint32_t internal_topk,
                             std::int64_t dataset_size,
                             const std::vector<std::uint32_t>& ground_truth,
                             const std::vector<std::int64_t>& round_indices,
                             const std::vector<std::int64_t>& previous_indices,
                             std::vector<std::int64_t>& accumulator_indices,
                             std::vector<float>& accumulator_distances,
                             const std::vector<float>& round_distances,
                             const std::vector<std::uint8_t>& terminal_flags,
                             const std::vector<favor_retry_diag::query_summary>& summaries,
                             const std::vector<std::uint32_t>& foreground_seed_counts,
                             std::vector<query_round_metric>& metrics)
  {
    for (std::uint32_t query = 0; query < num_queries; ++query) {
      const auto output_offset   = static_cast<std::uint64_t>(query) * topk;
      const auto frontier_offset = static_cast<std::uint64_t>(query) * internal_topk;
      auto* accumulated          = accumulator_indices.data() + output_offset;
      auto* accumulated_distance = accumulator_distances.data() + output_offset;
      const auto* current        = round_indices.data() + output_offset;
      const auto* previous       = previous_indices.data() + output_offset;
      const auto* gt             = ground_truth.data() + output_offset;

      query_round_metric metric{};
      metric.query                 = query;
      metric.round                 = round + 1;
      metric.individual_matches    = count_matches(current, gt, topk, dataset_size);
      metric.seed_count            = foreground_seed_counts[query];
      metric.candidate_evaluations = summaries[query].candidate_evaluations;
      metric.candidate_duplicates  = summaries[query].candidate_duplicates;
      metric.candidate_hash_full   = summaries[query].candidate_hash_full;
      for (std::uint32_t rank = 0; rank < topk; ++rank) {
        if (valid_index(current[rank], dataset_size)) {
          ++metric.output_count;
          if (round == 0 || !contains(previous, topk, current[rank], dataset_size)) {
            ++metric.new_ids;
          }
          if (!contains(accumulated, topk, current[rank], dataset_size)) {
            for (std::uint32_t gt_rank = 0; gt_rank < topk; ++gt_rank) {
              if (current[rank] == gt[gt_rank]) {
                ++metric.new_ground_truth;
                break;
              }
            }
          }
        }
      }
      if (round != 0) {
        std::unordered_set<std::int64_t> previous_values;
        std::unordered_set<std::int64_t> current_values;
        for (std::uint32_t rank = 0; rank < topk; ++rank) {
          if (valid_index(previous[rank], dataset_size)) { previous_values.insert(previous[rank]); }
          if (valid_index(current[rank], dataset_size)) { current_values.insert(current[rank]); }
        }
        std::uint32_t intersection = 0;
        for (auto value : current_values) {
          intersection += previous_values.count(value) != 0;
        }
        const auto union_count = previous_values.size() + current_values.size() - intersection;
        metric.jaccard_previous =
          union_count == 0 ? 1.0f : static_cast<float>(intersection) / union_count;
      }
      for (std::uint32_t rank = 0; rank < internal_topk; ++rank) {
        const auto flags = terminal_flags[frontier_offset + rank];
        if ((flags & 1u) == 0) { continue; }
        ++metric.frontier_valid;
        if ((flags & 2u) == 0) {
          ++metric.frontier_unexpanded;
          if ((flags & 4u) != 0) {
            ++metric.frontier_unexpanded_pass;
          } else {
            ++metric.frontier_unexpanded_reject;
          }
        }
      }
      merge_query(current,
                  round_distances.data() + output_offset,
                  accumulated,
                  accumulated_distance,
                  topk,
                  dataset_size);
      metric.accumulated_matches = count_matches(accumulated, gt, topk, dataset_size);
      metrics.push_back(metric);
    }
  }

  static auto select_queries(const std::vector<query_round_metric>& first_round_metrics,
                             std::uint32_t num_queries) -> std::vector<std::uint32_t>
  {
    constexpr std::uint32_t per_group = 16;
    std::vector<std::uint32_t> order(num_queries);
    for (std::uint32_t query = 0; query < num_queries; ++query) {
      order[query] = query;
    }
    auto recall = [&](std::uint32_t query) {
      return first_round_metrics[query].individual_matches;
    };
    std::sort(order.begin(), order.end(), [&](auto lhs, auto rhs) {
      return recall(lhs) < recall(rhs) || (recall(lhs) == recall(rhs) && lhs < rhs);
    });
    std::vector<std::uint32_t> selected;
    std::unordered_set<std::uint32_t> unique;
    auto take = [&](std::uint32_t query) {
      if (unique.insert(query).second) { selected.push_back(query); }
    };
    for (std::uint32_t i = 0; i < std::min(per_group, num_queries); ++i) {
      take(order[i]);
    }

    auto middle = order;
    std::sort(middle.begin(), middle.end(), [&](auto lhs, auto rhs) {
      const auto lhs_delta = std::abs(static_cast<int>(recall(lhs)) - 9);
      const auto rhs_delta = std::abs(static_cast<int>(recall(rhs)) - 9);
      return lhs_delta < rhs_delta || (lhs_delta == rhs_delta && lhs < rhs);
    });
    for (std::uint32_t i = 0; i < std::min(per_group, num_queries); ++i) {
      take(middle[i]);
    }
    for (std::uint32_t i = 0; i < std::min(per_group, num_queries); ++i) {
      take(order[num_queries - 1 - i]);
    }
    return selected;
  }

  static void append_selected_frontier(const std::vector<std::uint32_t>& selected,
                                       std::uint32_t internal_topk,
                                       const std::vector<std::uint32_t>& terminal_ids,
                                       const std::vector<float>& terminal_distances,
                                       const std::vector<std::uint8_t>& terminal_flags,
                                       std::vector<std::uint32_t>& selected_ids,
                                       std::vector<float>& selected_distances,
                                       std::vector<std::uint8_t>& selected_flags)
  {
    for (auto query : selected) {
      const auto begin = static_cast<std::uint64_t>(query) * internal_topk;
      selected_ids.insert(selected_ids.end(),
                          terminal_ids.begin() + begin,
                          terminal_ids.begin() + begin + internal_topk);
      selected_distances.insert(selected_distances.end(),
                                terminal_distances.begin() + begin,
                                terminal_distances.begin() + begin + internal_topk);
      selected_flags.insert(selected_flags.end(),
                            terminal_flags.begin() + begin,
                            terminal_flags.begin() + begin + internal_topk);
    }
  }

  template <typename T>
  static void write_binary(const std::filesystem::path& path, const std::vector<T>& values)
  {
    std::ofstream output{path, std::ios::binary};
    if (!output) { throw std::runtime_error("cannot create retry diagnostic binary output"); }
    output.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(T));
    output.close();
    if (!output) { throw std::runtime_error("cannot complete retry diagnostic binary output"); }
  }

  static auto median(std::vector<double> values) -> double
  {
    if (values.empty()) { return 0.0; }
    const auto middle = values.begin() + values.size() / 2;
    std::nth_element(values.begin(), middle, values.end());
    if (values.size() % 2 != 0) { return *middle; }
    const auto lower = *std::max_element(values.begin(), middle);
    return (lower + *middle) / 2.0;
  }

  void write_capture(const std::vector<query_round_metric>& metrics,
                     const std::vector<std::int64_t>& all_round_outputs,
                     const std::vector<std::uint32_t>& selected_queries,
                     const std::vector<std::uint32_t>& selected_terminal_ids,
                     const std::vector<float>& selected_terminal_distances,
                     const std::vector<std::uint8_t>& selected_terminal_flags,
                     std::uint32_t num_queries,
                     std::uint32_t topk,
                     std::uint32_t graph_degree,
                     std::uint32_t search_width,
                     std::int64_t dataset_size,
                     std::uint32_t configured_itopk,
                     std::uint32_t internal_topk,
                     std::uint32_t initial_width,
                     float filtering_rate) const
  {
    namespace fs = std::filesystem;
    const fs::path output_dir{config_.output_directory};
    fs::create_directories(output_dir);

    std::ofstream query_csv{output_dir / "query_round.csv"};
    query_csv << "query_id,round,budget,individual_recall,accumulated_recall,output_count,"
                 "jaccard_previous,new_ids,new_ground_truth,frontier_valid,frontier_unexpanded,"
                 "frontier_unexpanded_pass,frontier_unexpanded_reject,retry_seed_count,"
                 "candidate_evaluations,candidate_duplicates,candidate_hash_full\n"
              << std::setprecision(9);
    for (const auto& metric : metrics) {
      query_csv << metric.query << ',' << metric.round << ','
                << (strategy_ == favor_retry_strategy::oracle
                      ? checked_budget(config_.b0, metric.round)
                      : config_.b0)
                << ',' << static_cast<double>(metric.individual_matches) / topk << ','
                << static_cast<double>(metric.accumulated_matches) / topk << ','
                << metric.output_count << ',' << metric.jaccard_previous << ',' << metric.new_ids
                << ',' << metric.new_ground_truth << ',' << metric.frontier_valid << ','
                << metric.frontier_unexpanded << ',' << metric.frontier_unexpanded_pass << ','
                << metric.frontier_unexpanded_reject << ',' << metric.seed_count << ','
                << metric.candidate_evaluations << ',' << metric.candidate_duplicates << ','
                << metric.candidate_hash_full << '\n';
    }

    std::ofstream round_csv{output_dir / "round_metrics.csv"};
    round_csv << "dataset,strategy,round,budget,individual_recall,accumulated_recall,"
                 "mean_jaccard_previous,mean_new_ids,mean_new_ground_truth,"
                 "median_frontier_unexpanded,mean_frontier_unexpanded,"
                 "mean_frontier_unexpanded_pass,mean_frontier_unexpanded_reject,"
                 "mean_retry_seed_count,mean_candidate_evaluations,mean_candidate_duplicates,"
                 "sum_candidate_hash_full\n"
              << std::setprecision(9);
    for (std::uint32_t round = 1; round <= config_.rounds; ++round) {
      double individual = 0, accumulated = 0, jaccard = 0, new_ids = 0, new_gt = 0;
      double frontier = 0, frontier_pass = 0, frontier_reject = 0, seed_count = 0;
      double evaluations = 0, duplicates = 0, hash_full = 0;
      std::vector<double> frontier_values;
      frontier_values.reserve(num_queries);
      for (const auto& metric : metrics) {
        if (metric.round != round) { continue; }
        individual += metric.individual_matches;
        accumulated += metric.accumulated_matches;
        jaccard += metric.jaccard_previous;
        new_ids += metric.new_ids;
        new_gt += metric.new_ground_truth;
        frontier += metric.frontier_unexpanded;
        frontier_pass += metric.frontier_unexpanded_pass;
        frontier_reject += metric.frontier_unexpanded_reject;
        seed_count += metric.seed_count;
        evaluations += metric.candidate_evaluations;
        duplicates += metric.candidate_duplicates;
        hash_full += metric.candidate_hash_full;
        frontier_values.push_back(metric.frontier_unexpanded);
      }
      const auto queries = static_cast<double>(num_queries);
      round_csv << config_.dataset << ',' << favor_retry_strategy_name(strategy_) << ',' << round
                << ','
                << (strategy_ == favor_retry_strategy::oracle ? checked_budget(config_.b0, round)
                                                              : config_.b0)
                << ',' << individual / (queries * topk) << ',' << accumulated / (queries * topk)
                << ',' << jaccard / queries << ',' << new_ids / queries << ',' << new_gt / queries
                << ',' << median(std::move(frontier_values)) << ',' << frontier / queries << ','
                << frontier_pass / queries << ',' << frontier_reject / queries << ','
                << seed_count / queries << ',' << evaluations / queries << ','
                << duplicates / queries << ',' << static_cast<std::uint64_t>(hash_full) << '\n';
    }

    std::ofstream selected_csv{output_dir / "selected_queries.csv"};
    selected_csv << "trace_slot,query_id\n";
    for (std::size_t slot = 0; slot < selected_queries.size(); ++slot) {
      selected_csv << slot << ',' << selected_queries[slot] << '\n';
    }
    query_csv.close();
    round_csv.close();
    selected_csv.close();
    if (!query_csv || !round_csv || !selected_csv) {
      throw std::runtime_error("cannot complete retry diagnostic CSV output");
    }
    write_binary(output_dir / "round_outputs.i64bin", all_round_outputs);
    write_binary(output_dir / "selected_terminal_ids.u32bin", selected_terminal_ids);
    write_binary(output_dir / "selected_terminal_distances.f32bin", selected_terminal_distances);
    write_binary(output_dir / "selected_terminal_flags.u8bin", selected_terminal_flags);

    // Completion marker: write this only after every CSV and binary artifact is closed.
    std::ofstream manifest{output_dir / "manifest.json"};
    manifest << "{\n"
             << "  \"schema_version\": 1,\n"
             << "  \"complete\": true,\n"
             << "  \"dataset\": \"" << config_.dataset << "\",\n"
             << "  \"strategy\": \"" << favor_retry_strategy_name(strategy_) << "\",\n"
             << "  \"num_queries\": " << num_queries << ",\n"
             << "  \"topk\": " << topk << ",\n"
             << "  \"dataset_size\": " << dataset_size << ",\n"
             << "  \"graph_degree\": " << graph_degree << ",\n"
             << "  \"search_width\": " << search_width << ",\n"
             << "  \"configured_itopk\": " << configured_itopk << ",\n"
             << "  \"internal_topk\": " << internal_topk << ",\n"
             << "  \"initial_width\": " << initial_width << ",\n"
             << "  \"filtering_rate\": " << filtering_rate << ",\n"
             << "  \"rounds\": " << config_.rounds << ",\n"
             << "  \"b0\": " << config_.b0 << ",\n"
             << "  \"base_mask\": " << base_mask << ",\n"
             << "  \"selected_queries\": " << selected_queries.size() << ",\n"
             << "  \"timing_valid\": false\n"
             << "}\n";
  }

  void copy_cached_result(cudaStream_t stream,
                          std::int64_t* result_indices,
                          float* result_distances) const
  {
    RAFT_CUDA_TRY(cudaMemcpyAsync(result_indices,
                                  cached_indices_.data(),
                                  cached_indices_.size() * sizeof(std::int64_t),
                                  cudaMemcpyDefault,
                                  stream));
    RAFT_CUDA_TRY(cudaMemcpyAsync(result_distances,
                                  cached_distances_.data(),
                                  cached_distances_.size() * sizeof(float),
                                  cudaMemcpyDefault,
                                  stream));
  }

  favor_retry_diagnostic_config config_;
  favor_retry_strategy strategy_;
  std::mutex mutex_;
  bool captured_ = false;
  std::vector<std::int64_t> cached_indices_;
  std::vector<float> cached_distances_;
};

}  // namespace cuvs::bench::detail
