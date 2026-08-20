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
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace cuvs::bench::detail {

namespace favor_diag = cuvs::neighbors::cagra::detail::favor_search_diagnostics;

struct favor_diagnostic_config {
  std::string output_directory;
  std::string ground_truth_file;
  std::string selected_queries_file;
  std::string dataset;
  std::string variant;
  std::uint32_t max_trace_iterations               = 0;
  std::uint32_t termination_record_start_iteration = 0;
  std::uint32_t termination_start_iteration        = 0;
  std::uint32_t termination_parent_interval        = 0;

  [[nodiscard]] bool enabled() const noexcept { return !output_directory.empty(); }
  [[nodiscard]] bool termination_shadow_enabled() const noexcept
  {
    return termination_start_iteration != 0 || termination_parent_interval != 0;
  }
};

/** Owns one deliberately untimed diagnostic capture for a cuVS Bench CAGRA configuration. */
class favor_diagnostic_session {
 public:
  explicit favor_diagnostic_session(favor_diagnostic_config config) : config_{std::move(config)} {}

  template <typename SearchFn>
  void capture(const raft::resources& res,
               std::uint32_t num_queries,
               std::uint32_t topk,
               std::uint32_t graph_degree,
               std::uint32_t search_width,
               std::int64_t dataset_size,
               std::uint32_t itopk,
               std::uint32_t configured_max_iterations,
               float filtering_rate,
               bool device_telemetry,
               const std::int64_t* result_indices,
               SearchFn&& search)
  {
    std::lock_guard<std::mutex> guard{mutex_};
    if (captured_) {
      search();
      return;
    }
    if (config_.ground_truth_file.empty()) {
      throw std::invalid_argument("favor_diagnostics_groundtruth must be specified");
    }

    auto stream            = raft::resource::get_cuda_stream(res);
    auto ground_truth_host = load_ground_truth(num_queries);
    auto selected =
      device_telemetry ? load_selected_queries(num_queries) : std::vector<std::uint32_t>{};
    std::vector<std::int32_t> trace_slot_by_query(num_queries, -1);
    for (std::size_t slot = 0; slot < selected.size(); ++slot) {
      trace_slot_by_query[selected[slot]] = static_cast<std::int32_t>(slot);
    }

    const std::uint32_t max_trace_iterations =
      selected.empty() ? 0
                       : (config_.max_trace_iterations == 0
                            ? std::max<std::uint32_t>(configured_max_iterations, 256u)
                            : config_.max_trace_iterations);
    const std::uint32_t candidates_per_iteration = graph_degree * search_width;
    const std::uint64_t iteration_count =
      static_cast<std::uint64_t>(selected.size()) * max_trace_iterations;
    const std::uint64_t candidate_count         = iteration_count * candidates_per_iteration;
    std::uint32_t termination_checkpoint_stride = 0;
    if (config_.termination_shadow_enabled()) {
      if (config_.termination_start_iteration == 0 || config_.termination_parent_interval == 0) {
        throw std::invalid_argument(
          "termination shadow requires both start_iteration and parent_interval");
      }
      if (configured_max_iterations < config_.termination_start_iteration) {
        throw std::invalid_argument(
          "termination shadow start_iteration exceeds configured max_iterations");
      }
      const auto record_start_iteration = config_.termination_record_start_iteration == 0
                                            ? config_.termination_start_iteration
                                            : config_.termination_record_start_iteration;
      if (record_start_iteration > config_.termination_start_iteration) {
        throw std::invalid_argument(
          "termination shadow record_start_iteration must not exceed start_iteration");
      }
      const auto parent_span = (configured_max_iterations - record_start_iteration) * search_width;
      // One periodic series plus independently forced B0 and terminal checkpoints.
      termination_checkpoint_stride =
        3 + raft::ceildiv(parent_span, config_.termination_parent_interval);
    }
    const std::uint64_t termination_checkpoint_count =
      static_cast<std::uint64_t>(num_queries) * termination_checkpoint_stride;
    const std::uint32_t navix_seed_stride = itopk + graph_degree * search_width;
    const std::uint64_t navix_seed_count =
      static_cast<std::uint64_t>(num_queries) * navix_seed_stride;

    rmm::device_uvector<favor_diag::query_summary> summaries(num_queries, stream);
    rmm::device_uvector<std::uint32_t> ground_truth(ground_truth_host.size(), stream);
    rmm::device_uvector<std::int32_t> trace_slots(trace_slot_by_query.size(), stream);
    rmm::device_uvector<favor_diag::iteration_record> iterations(iteration_count, stream);
    rmm::device_uvector<favor_diag::candidate_record> candidates(candidate_count, stream);
    rmm::device_uvector<favor_diag::termination_checkpoint_record> termination_checkpoints(
      termination_checkpoint_count, stream);
    rmm::device_uvector<std::uint32_t> termination_checkpoint_counts(
      termination_checkpoint_stride == 0 ? 0 : num_queries, stream);
    rmm::device_uvector<favor_diag::context> context(1, stream);
    rmm::device_uvector<std::uint32_t> navix_seed_ids(navix_seed_count, stream);
    rmm::device_uvector<float> navix_seed_distances(navix_seed_count, stream);

    RAFT_CUDA_TRY(cudaMemcpyAsync(ground_truth.data(),
                                  ground_truth_host.data(),
                                  ground_truth_host.size() * sizeof(std::uint32_t),
                                  cudaMemcpyHostToDevice,
                                  stream));
    RAFT_CUDA_TRY(cudaMemcpyAsync(trace_slots.data(),
                                  trace_slot_by_query.data(),
                                  trace_slot_by_query.size() * sizeof(std::int32_t),
                                  cudaMemcpyHostToDevice,
                                  stream));
    if (iteration_count != 0) {
      RAFT_CUDA_TRY(cudaMemsetAsync(
        iterations.data(), 0, iteration_count * sizeof(favor_diag::iteration_record), stream));
      RAFT_CUDA_TRY(cudaMemsetAsync(
        candidates.data(), 0, candidate_count * sizeof(favor_diag::candidate_record), stream));
    }
    if (termination_checkpoint_count != 0) {
      RAFT_CUDA_TRY(cudaMemsetAsync(
        termination_checkpoints.data(),
        0,
        termination_checkpoint_count * sizeof(favor_diag::termination_checkpoint_record),
        stream));
      RAFT_CUDA_TRY(cudaMemsetAsync(
        termination_checkpoint_counts.data(), 0, num_queries * sizeof(std::uint32_t), stream));
    }
    if (navix_seed_count != 0) {
      RAFT_CUDA_TRY(cudaMemsetAsync(
        navix_seed_ids.data(), 0xff, navix_seed_count * sizeof(std::uint32_t), stream));
      RAFT_CUDA_TRY(
        cudaMemsetAsync(navix_seed_distances.data(), 0, navix_seed_count * sizeof(float), stream));
    }

    favor_diag::context host_context{};
    host_context.summaries           = summaries.data();
    host_context.ground_truth_ids    = ground_truth.data();
    host_context.trace_slot_by_query = trace_slots.data();
    host_context.iteration_records   = iterations.data();
    host_context.candidate_records   = candidates.data();
    host_context.termination_checkpoints =
      termination_checkpoint_stride == 0 ? nullptr : termination_checkpoints.data();
    host_context.termination_checkpoint_counts =
      termination_checkpoint_stride == 0 ? nullptr : termination_checkpoint_counts.data();
    host_context.termination_checkpoint_stride = termination_checkpoint_stride;
    host_context.termination_record_start_iteration =
      config_.termination_record_start_iteration == 0 ? config_.termination_start_iteration
                                                      : config_.termination_record_start_iteration;
    host_context.termination_start_iteration = config_.termination_start_iteration;
    host_context.termination_parent_interval = config_.termination_parent_interval;
    host_context.num_queries                 = num_queries;
    host_context.max_trace_iterations        = max_trace_iterations;
    host_context.candidates_per_iteration    = candidates_per_iteration;
    host_context.navix_seed_ids              = navix_seed_ids.data();
    host_context.navix_seed_distances        = navix_seed_distances.data();
    host_context.navix_seed_stride           = navix_seed_stride;
    RAFT_CUDA_TRY(cudaMemcpyAsync(
      context.data(), &host_context, sizeof(host_context), cudaMemcpyHostToDevice, stream));

    if (device_telemetry) {
      favor_diag::scoped_context diagnostic_scope{context.data()};
      search();
    } else {
      search();
    }
    RAFT_CUDA_TRY(cudaStreamSynchronize(stream));
    const auto launch_metrics = favor_diag::get_launch_metrics();

    std::vector<favor_diag::query_summary> summaries_host(num_queries);
    std::vector<favor_diag::iteration_record> iterations_host(iteration_count);
    std::vector<favor_diag::candidate_record> candidates_host(candidate_count);
    std::vector<favor_diag::termination_checkpoint_record> termination_checkpoints_host(
      termination_checkpoint_count);
    std::vector<std::uint32_t> termination_checkpoint_counts_host(
      termination_checkpoint_stride == 0 ? 0 : num_queries);
    std::vector<std::int64_t> result_indices_host(static_cast<std::uint64_t>(num_queries) * topk);
    std::vector<std::uint32_t> navix_seed_ids_host(navix_seed_count);
    std::vector<float> navix_seed_distances_host(navix_seed_count);
    if (device_telemetry) {
      RAFT_CUDA_TRY(cudaMemcpy(summaries_host.data(),
                               summaries.data(),
                               summaries_host.size() * sizeof(favor_diag::query_summary),
                               cudaMemcpyDeviceToHost));
    } else {
      for (std::uint32_t query = 0; query < num_queries; ++query) {
        summaries_host[query].schema   = favor_diag::schema_version;
        summaries_host[query].query_id = query;
      }
    }
    if (iteration_count != 0) {
      RAFT_CUDA_TRY(cudaMemcpy(iterations_host.data(),
                               iterations.data(),
                               iterations_host.size() * sizeof(favor_diag::iteration_record),
                               cudaMemcpyDeviceToHost));
      RAFT_CUDA_TRY(cudaMemcpy(candidates_host.data(),
                               candidates.data(),
                               candidates_host.size() * sizeof(favor_diag::candidate_record),
                               cudaMemcpyDeviceToHost));
    }
    if (termination_checkpoint_count != 0) {
      RAFT_CUDA_TRY(cudaMemcpy(
        termination_checkpoints_host.data(),
        termination_checkpoints.data(),
        termination_checkpoints_host.size() * sizeof(favor_diag::termination_checkpoint_record),
        cudaMemcpyDeviceToHost));
      RAFT_CUDA_TRY(cudaMemcpy(termination_checkpoint_counts_host.data(),
                               termination_checkpoint_counts.data(),
                               termination_checkpoint_counts_host.size() * sizeof(std::uint32_t),
                               cudaMemcpyDeviceToHost));
    }
    RAFT_CUDA_TRY(cudaMemcpy(result_indices_host.data(),
                             result_indices,
                             result_indices_host.size() * sizeof(std::int64_t),
                             cudaMemcpyDeviceToHost));
    if (navix_seed_count != 0) {
      RAFT_CUDA_TRY(cudaMemcpy(navix_seed_ids_host.data(),
                               navix_seed_ids.data(),
                               navix_seed_count * sizeof(std::uint32_t),
                               cudaMemcpyDeviceToHost));
      RAFT_CUDA_TRY(cudaMemcpy(navix_seed_distances_host.data(),
                               navix_seed_distances.data(),
                               navix_seed_count * sizeof(float),
                               cudaMemcpyDeviceToHost));
    }

    for (std::uint32_t query = 0; query < num_queries; ++query) {
      if (summaries_host[query].schema != favor_diag::schema_version ||
          summaries_host[query].query_id != query) {
        throw std::runtime_error(
          "diagnostic device/host schema or query identity mismatch; rebuild all CAGRA JIT kernels");
      }
      std::uint32_t matches       = 0;
      std::uint32_t valid_outputs = 0;
      for (std::uint32_t rank = 0; rank < topk; ++rank) {
        const auto candidate = result_indices_host[static_cast<std::uint64_t>(query) * topk + rank];
        const bool valid = candidate >= 0 && candidate < dataset_size;
        bool first_occurrence = valid;
        for (std::uint32_t prior = 0; valid && prior < rank; ++prior) {
          first_occurrence &=
            result_indices_host[static_cast<std::uint64_t>(query) * topk + prior] != candidate;
        }
        if (!first_occurrence) { continue; }
        ++valid_outputs;
        for (std::uint32_t gt_rank = 0; gt_rank < favor_diag::ground_truth_k; ++gt_rank) {
          if (candidate ==
              ground_truth_host[static_cast<std::uint64_t>(query) * favor_diag::ground_truth_k +
                                gt_rank]) {
            ++matches;
            summaries_host[query].navix_gt_output_mask |= std::uint32_t{1} << gt_rank;
            break;
          }
        }
      }
      summaries_host[query].recall = static_cast<float>(matches) /
                                     static_cast<float>(std::min(topk, favor_diag::ground_truth_k));
      summaries_host[query].output_count = valid_outputs;
    }

    write_capture(summaries_host,
                  iterations_host,
                  candidates_host,
                  selected,
                  num_queries,
                  topk,
                  graph_degree,
                  search_width,
                  dataset_size,
                  itopk,
                  configured_max_iterations,
                  filtering_rate,
                  max_trace_iterations,
                  candidates_per_iteration,
                  launch_metrics,
                  termination_checkpoints_host,
                  termination_checkpoint_counts_host,
                  termination_checkpoint_stride,
                  navix_seed_ids_host,
                  navix_seed_distances_host,
                  navix_seed_stride,
                  result_indices_host);
    captured_ = true;
  }

 private:
  auto load_ground_truth(std::uint32_t num_queries) const -> std::vector<std::uint32_t>
  {
    blob<std::int32_t> gt{config_.ground_truth_file};
    if (gt.n_rows() < num_queries || gt.n_cols() < favor_diag::ground_truth_k) {
      throw std::invalid_argument("diagnostic ground truth does not contain nq x 10 entries");
    }
    std::vector<std::uint32_t> result(static_cast<std::uint64_t>(num_queries) *
                                      favor_diag::ground_truth_k);
    for (std::uint32_t query = 0; query < num_queries; ++query) {
      for (std::uint32_t rank = 0; rank < favor_diag::ground_truth_k; ++rank) {
        result[static_cast<std::uint64_t>(query) * favor_diag::ground_truth_k + rank] =
          static_cast<std::uint32_t>(
            gt.data()[static_cast<std::uint64_t>(query) * gt.n_cols() + rank]);
      }
    }
    return result;
  }

  auto load_selected_queries(std::uint32_t num_queries) const -> std::vector<std::uint32_t>
  {
    std::vector<std::uint32_t> selected;
    if (config_.selected_queries_file.empty()) { return selected; }
    std::ifstream input{config_.selected_queries_file};
    if (!input) { throw std::runtime_error("cannot open selected-query file"); }
    std::unordered_set<std::uint32_t> unique;
    std::uint32_t query;
    while (input >> query) {
      if (query >= num_queries) { throw std::out_of_range("selected query is outside the batch"); }
      if (unique.insert(query).second) { selected.push_back(query); }
    }
    if (selected.size() > 64) {
      throw std::invalid_argument("at most 64 deep-trace queries are allowed");
    }
    return selected;
  }

  template <typename T>
  static void write_binary(const std::filesystem::path& path, const std::vector<T>& values)
  {
    std::ofstream output{path, std::ios::binary};
    if (!output) { throw std::runtime_error("cannot create diagnostic binary output"); }
    output.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(T));
  }

  void write_capture(
    const std::vector<favor_diag::query_summary>& summaries,
    const std::vector<favor_diag::iteration_record>& iterations,
    const std::vector<favor_diag::candidate_record>& candidates,
    const std::vector<std::uint32_t>& selected,
    std::uint32_t num_queries,
    std::uint32_t topk,
    std::uint32_t graph_degree,
    std::uint32_t search_width,
    std::int64_t dataset_size,
    std::uint32_t itopk,
    std::uint32_t configured_max_iterations,
    float filtering_rate,
    std::uint32_t max_trace_iterations,
    std::uint32_t candidates_per_iteration,
    favor_diag::launch_metrics launch_metrics,
    const std::vector<favor_diag::termination_checkpoint_record>& termination_checkpoints,
    const std::vector<std::uint32_t>& termination_checkpoint_counts,
    std::uint32_t termination_checkpoint_stride,
    const std::vector<std::uint32_t>& navix_seed_ids,
    const std::vector<float>& navix_seed_distances,
    std::uint32_t navix_seed_stride,
    const std::vector<std::int64_t>& result_indices) const
  {
    namespace fs = std::filesystem;
    const fs::path output_dir{config_.output_directory};
    fs::create_directories(output_dir);

    std::ofstream csv{output_dir / "query_summary.csv"};
    csv << "query_id,recall,iterations,resolved_max_iterations,stop_reason,terminal_valid,"
           "terminal_pass,terminal_reject,terminal_unexpanded_pass,terminal_unexpanded_reject,"
           "expanded_pass_parents,expanded_reject_parents,candidate_attempts,"
           "candidate_evaluations,candidate_duplicate_or_full,candidate_duplicates,"
           "candidate_hash_full,passing_candidates,"
           "rejected_candidates,penalized_candidates,accumulator_observations,"
           "accumulator_insertions,gt_seen_mask,output_count,hash_bitlen,"
           "small_hash_bitlen,small_hash_reset_interval,resolved_filtering_rate,"
           "reference_penalty,query_penalty,terminal_cutoff,"
           "best_unexpanded_distance,worst_retained_distance,kth_passing_raw_distance,"
           "navix_seed_found,navix_seed_iteration,navix_seed_count,"
           "navix_post_seed_iterations,navix_terminal_phase,navix_one_hop_parents,"
           "navix_directed_parents,navix_blind_parents,navix_first_hop_checks,"
           "navix_first_hop_passing,navix_bridge_rows,navix_bridge_rows_loaded,"
           "navix_bridge_rows_after_cap,navix_second_hop_checks,navix_second_hop_passing,"
           "navix_admitted_candidates,navix_cap_blocked_unique,navix_gt_first_hop_mask,"
           "navix_gt_second_hop_mask,navix_gt_admitted_mask,navix_gt_retained_mask,"
           "navix_gt_cap_blocked_mask,navix_gt_hash_full_mask,navix_gt_output_mask";
    for (std::uint32_t p = 0; p <= 32; ++p) {
      csv << ",navix_local_p_" << p;
    }
    for (std::uint32_t rank = 0; rank < favor_diag::ground_truth_k; ++rank) {
      csv << ",gt_first_iteration_" << rank;
    }
    csv << '\n' << std::setprecision(9);
    for (const auto& s : summaries) {
      csv << s.query_id << ',' << s.recall << ',' << s.iterations << ','
          << s.resolved_max_iterations << ',' << s.reason << ',' << s.terminal_valid << ','
          << s.terminal_pass << ',' << s.terminal_reject << ',' << s.terminal_unexpanded_pass << ','
          << s.terminal_unexpanded_reject << ',' << s.expanded_pass_parents << ','
          << s.expanded_reject_parents << ',' << s.candidate_attempts << ','
          << s.candidate_evaluations << ',' << s.candidate_duplicate_or_full << ','
          << s.candidate_duplicates << ',' << s.candidate_hash_full << ',' << s.passing_candidates
          << ',' << s.rejected_candidates << ',' << s.penalized_candidates << ','
          << s.accumulator_observations << ',' << s.accumulator_insertions << ',' << s.gt_seen_mask
          << ',' << s.output_count << ',' << s.hash_bitlen << ',' << s.small_hash_bitlen << ','
          << s.small_hash_reset_interval << ',' << s.resolved_filtering_rate << ','
          << s.reference_penalty << ',' << s.query_penalty << ',' << s.terminal_cutoff << ','
          << s.best_unexpanded_distance << ',' << s.worst_retained_distance << ','
          << s.kth_passing_raw_distance << ',' << s.navix_seed_found << ','
          << s.navix_seed_iteration << ',' << s.navix_seed_count << ','
          << s.navix_post_seed_iterations << ',' << s.navix_terminal_phase << ','
          << s.navix_one_hop_parents << ',' << s.navix_directed_parents << ','
          << s.navix_blind_parents << ',' << s.navix_first_hop_checks << ','
          << s.navix_first_hop_passing << ',' << s.navix_bridge_rows << ','
          << s.navix_bridge_rows_loaded << ',' << s.navix_bridge_rows_after_cap << ','
          << s.navix_second_hop_checks << ',' << s.navix_second_hop_passing << ','
          << s.navix_admitted_candidates << ',' << s.navix_cap_blocked_unique << ','
          << s.navix_gt_first_hop_mask << ',' << s.navix_gt_second_hop_mask << ','
          << s.navix_gt_admitted_mask << ',' << s.navix_gt_retained_mask << ','
          << s.navix_gt_cap_blocked_mask << ',' << s.navix_gt_hash_full_mask << ','
          << s.navix_gt_output_mask;
      for (auto count : s.navix_local_p_histogram) {
        csv << ',' << count;
      }
      for (auto first : s.gt_first_iteration) {
        csv << ',' << first;
      }
      csv << '\n';
    }

    std::ofstream selected_csv{output_dir / "selected_queries.csv"};
    selected_csv << "trace_slot,query_id\n";
    for (std::size_t slot = 0; slot < selected.size(); ++slot) {
      selected_csv << slot << ',' << selected[slot] << '\n';
    }
    write_binary(output_dir / "iteration_trace.bin", iterations);
    write_binary(output_dir / "candidate_trace.bin", candidates);
    write_binary(output_dir / "termination_checkpoints.bin", termination_checkpoints);
    write_binary(output_dir / "termination_checkpoint_counts.bin", termination_checkpoint_counts);
    write_binary(output_dir / "navix_seed_ids.bin", navix_seed_ids);
    write_binary(output_dir / "navix_seed_distances.bin", navix_seed_distances);
    write_binary(output_dir / "result_indices.i64bin", result_indices);

    std::ofstream manifest{output_dir / "manifest.json"};
    manifest << "{\n"
             << "  \"schema_version\": " << favor_diag::schema_version << ",\n"
             << "  \"dataset\": \"" << config_.dataset << "\",\n"
             << "  \"variant\": \"" << config_.variant << "\",\n"
             << "  \"num_queries\": " << num_queries << ",\n"
             << "  \"topk\": " << topk << ",\n"
             << "  \"output_set_semantics\": \"distinct_valid_output_ids_v1\",\n"
             << "  \"dataset_size\": " << dataset_size << ",\n"
             << "  \"graph_degree\": " << graph_degree << ",\n"
             << "  \"search_width\": " << search_width << ",\n"
             << "  \"itopk\": " << itopk << ",\n"
             << "  \"configured_max_iterations\": " << configured_max_iterations << ",\n"
             << "  \"filtering_rate\": " << filtering_rate << ",\n"
             << "  \"trace_slots\": " << selected.size() << ",\n"
             << "  \"max_trace_iterations\": " << max_trace_iterations << ",\n"
             << "  \"candidates_per_iteration\": " << candidates_per_iteration << ",\n"
             << "  \"block_size\": " << launch_metrics.block_size << ",\n"
             << "  \"dynamic_smem_bytes\": " << launch_metrics.dynamic_smem_bytes << ",\n"
             << "  \"active_blocks_per_sm\": " << launch_metrics.active_blocks_per_sm << ",\n"
             << "  \"max_threads_per_sm\": " << launch_metrics.max_threads_per_sm << ",\n"
             << "  \"occupancy\": "
             << (launch_metrics.max_threads_per_sm == 0
                   ? 0.0
                   : static_cast<double>(launch_metrics.block_size) *
                       launch_metrics.active_blocks_per_sm / launch_metrics.max_threads_per_sm)
             << ",\n"
             << "  \"iteration_record_size\": " << sizeof(favor_diag::iteration_record) << ",\n"
             << "  \"candidate_record_size\": " << sizeof(favor_diag::candidate_record) << ",\n"
             << "  \"navix_seed_stride\": " << navix_seed_stride << ",\n"
             << "  \"result_index_bytes\": " << sizeof(std::int64_t) << ",\n"
             << "  \"termination_record_start_iteration\": "
             << (config_.termination_record_start_iteration == 0
                   ? config_.termination_start_iteration
                   : config_.termination_record_start_iteration)
             << ",\n"
             << "  \"termination_start_iteration\": " << config_.termination_start_iteration
             << ",\n"
             << "  \"termination_parent_interval\": " << config_.termination_parent_interval
             << ",\n"
             << "  \"termination_checkpoint_stride\": " << termination_checkpoint_stride << ",\n"
             << "  \"termination_checkpoint_record_size\": "
             << sizeof(favor_diag::termination_checkpoint_record) << ",\n"
             << "  \"timing_valid\": false\n"
             << "}\n";
  }

  favor_diagnostic_config config_;
  std::mutex mutex_;
  bool captured_ = false;
};

static_assert(sizeof(favor_diag::query_summary) == 408);
static_assert(sizeof(favor_diag::iteration_record) == 132);
static_assert(sizeof(favor_diag::candidate_record) == 36);
static_assert(sizeof(favor_diag::termination_checkpoint_record) == 136);

}  // namespace cuvs::bench::detail
