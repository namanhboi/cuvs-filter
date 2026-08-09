/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "../../../../src/neighbors/cagra_benchmark.hpp"
#include "../../../../src/neighbors/detail/cagra/favor_multi_seed_benchmark.cuh"
#include "../../../../src/neighbors/detail/cagra/utils.hpp"
#include "../common/ann_types.hpp"
#include "../common/cuda_huge_page_resource.hpp"
#include "cuvs_ann_bench_utils.h"
#include "favor_retry_diagnostic_session.h"
#include "favor_search_diagnostic_session.h"
#include "filtered_dataset_adapter.h"
#include <rmm/mr/pinned_host_memory_resource.hpp>

#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/neighbors/common.hpp>
#include <cuvs/neighbors/composite/index.hpp>
#include <cuvs/neighbors/dynamic_batching.hpp>
#include <cuvs/neighbors/ivf_pq.hpp>
#include <cuvs/neighbors/nn_descent.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/operators.hpp>
#include <raft/linalg/unary_op.cuh>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/resource_ref.hpp>

#include <atomic>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <raft/util/integer_utils.hpp>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

namespace cuvs::bench {

namespace detail {

/** If persistent CAGRA search uses few benchmark threads, log a throughput hint once per process.
 */
inline void maybe_log_cagra_persistent_concurrency_hint(bool persistent_search)
{
  if (!persistent_search) { return; }

  const unsigned hc = std::max(1u, static_cast<unsigned>(std::thread::hardware_concurrency()));
  const unsigned bn = static_cast<unsigned>(std::max(0, benchmark_n_threads));
  if (bn >= 2u * hc) { return; }

  static std::atomic<bool> logged{false};
  bool expected = false;
  if (!logged.compare_exchange_strong(expected, true)) { return; }

  const unsigned threads_rec = 16u * hc;
  RAFT_LOG_INFO(
    "CAGRA persistent search benefits from high client concurrency (try `--mode=throughput "
    "--threads=1:%u`).",
    threads_rec);
}

template <typename IndexT>
RAFT_KERNEL favor_fill_empty_results_kernel(IndexT* neighbors,
                                            float* distances,
                                            std::size_t n_elements)
{
  auto element = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (element >= n_elements) { return; }
  neighbors[element] = std::numeric_limits<IndexT>::max();
  distances[element] = std::numeric_limits<float>::max();
}

template <typename IndexT>
inline void fill_empty_filter_results(const raft::resources& res,
                                      IndexT* neighbors,
                                      float* distances,
                                      std::size_t n_elements)
{
  constexpr unsigned int block_size = 128;
  auto const grid_size =
    static_cast<unsigned int>(raft::ceildiv(n_elements, std::size_t{block_size}));
  auto stream = raft::resource::get_cuda_stream(res);
  favor_fill_empty_results_kernel<<<grid_size, block_size, 0, stream>>>(
    neighbors, distances, n_elements);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

inline bool favor_adaptive_termination_enabled() noexcept
{
  auto const* value = std::getenv("CUVS_FAVOR_EXPERIMENTAL_ADAPTIVE_TERMINATION");
  return value != nullptr && value[0] == '1' && value[1] == '\0';
}

}  // namespace detail

enum class AllocatorType { kHostPinned, kHostHugePage, kDevice };
enum class CagraBuildAlgo { kAuto, kIvfPq, kNnDescent };
enum class CagraMergeType { kPhysical, kLogical };

template <typename T, typename IdxT>
class cuvs_cagra : public algo<T>, public algo_gpu {
 public:
  using search_param_base = typename algo<T>::search_param;
  using algo<T>::dim_;
  using algo<T>::metric_;

  struct search_param : public search_param_base {
    cuvs::neighbors::cagra::search_params p;
    std::optional<std::string> favor_delta_d_file;
    cuvs::neighbors::cagra::favor_delta_d_params favor_delta_d_params;
    detail::favor_diagnostic_config favor_diagnostics;
    detail::favor_retry_diagnostic_config favor_retry_diagnostics;
    /** Benchmark-only independent B0 FAVOR starts. Empty preserves normal CAGRA behavior. */
    std::vector<std::uint64_t> favor_seed_masks;
    /** Include query-local UDF sampling in every timed search (end-to-end) or reuse setup rates. */
    bool favor_udf_include_sampling = true;
    /** Benchmark-only output-policy control; false reproduces the pre-accumulator result queue. */
    bool favor_udf_passing_accumulator        = true;
    bool favor_udf_passing_accumulator_is_set = false;
    std::uint32_t favor_udf_sample_offset{};
    /** Benchmark-only NaviX traversal policy. Empty preserves the normal cuVS path. */
    std::optional<std::uint32_t> navix_policy;
    float refine_ratio;
    AllocatorType graph_mem   = AllocatorType::kDevice;
    AllocatorType dataset_mem = AllocatorType::kDevice;
    [[nodiscard]] auto needs_dataset() const -> bool override { return true; }
    /* Dynamic batching */
    bool dynamic_batching = false;
    int64_t dynamic_batching_k;
    int64_t dynamic_batching_max_batch_size     = 4;
    double dynamic_batching_dispatch_timeout_ms = 0.01;
    size_t dynamic_batching_n_queues            = 8;
    bool dynamic_batching_conservative_dispatch = false;
  };

  struct build_param {
    // The optimal defaults depend on the dataset shape and thus only available once the build
    // function is called.
    using dataset_dependent_params = std::function<cuvs::neighbors::cagra::index_params(
      raft::matrix_extent<int64_t>, cuvs::distance::DistanceType)>;
    dataset_dependent_params cagra_params;
    size_t num_dataset_splits = 1;
    CagraMergeType merge_type = CagraMergeType::kPhysical;
  };

  cuvs_cagra(Metric metric, int dim, const build_param& param, int concurrent_searches = 1)
    : algo<T>(metric, dim),
      index_params_(param),

      dataset_(std::make_shared<raft::device_matrix<T, int64_t, raft::row_major>>(
        std::move(raft::make_device_matrix<T, int64_t>(handle_, 0, 0)))),
      graph_(std::make_shared<raft::device_matrix<IdxT, int64_t, raft::row_major>>(
        std::move(raft::make_device_matrix<IdxT, int64_t>(handle_, 0, 0)))),
      input_dataset_v_(
        std::make_shared<raft::device_matrix_view<const T, int64_t, raft::row_major>>(
          nullptr, 0, 0))

  {
  }

  void build(const T* dataset, size_t nrow) final;

  void set_search_param(const search_param_base& param, const void* filter_bitset) override;

  void set_search_dataset(const T* dataset, size_t nrow) override;

  void search(const T* queries,
              int batch_size,
              int k,
              algo_base::index_type* neighbors,
              float* distances) const override;
  void search_with_query_offset(const T* queries,
                                int batch_size,
                                int k,
                                algo_base::index_type* neighbors,
                                float* distances,
                                std::size_t query_offset) const override;
  void search_base(const T* queries,
                   int batch_size,
                   int k,
                   algo_base::index_type* neighbors,
                   float* distances) const;

  [[nodiscard]] auto get_sync_stream() const noexcept -> cudaStream_t override
  {
    return handle_.get_sync_stream();
  }

  [[nodiscard]] auto uses_stream() const noexcept -> bool override
  {
    // If the algorithm uses persistent kernel, the CPU has to synchronize by the end of computing
    // the result. Hence it guarantees the benchmark CUDA stream is empty by the end of the
    // execution. Hence we inform the benchmark to not waste the time on recording & synchronizing
    // the event.
    return !search_params_.persistent;
  }

  // to enable dataset access from GPU memory
  [[nodiscard]] auto get_preference() const -> algo_property override
  {
    algo_property property;
    property.dataset_memory_type = MemoryType::kHostMmap;
    property.query_memory_type   = MemoryType::kDevice;
    return property;
  }
  [[nodiscard]] auto supports_filter_validation() const -> bool override
  {
    return udf_filter_adapter_ != nullptr;
  }
  [[nodiscard]] auto is_filter_valid(std::size_t query_id, algo_base::index_type candidate_id) const
    -> bool override
  {
    return udf_filter_adapter_ != nullptr && candidate_id >= 0 &&
           static_cast<std::uint64_t>(candidate_id) < udf_filter_adapter_->base_rows() &&
           query_id < udf_filter_adapter_->query_rows() &&
           udf_filter_adapter_->passes(static_cast<std::uint32_t>(query_id),
                                       static_cast<std::uint32_t>(candidate_id));
  }
  void save(const std::string& file) const override;
  void load(const std::string&) override;
  void save_to_hnswlib(const std::string& file) const;
  std::unique_ptr<algo<T>> copy() override;

  auto get_index() const -> const cuvs::neighbors::cagra::index<T, IdxT>* { return index_.get(); }

 private:
  // handle_ must go first to make sure it dies last and all memory allocated in pool
  configured_raft_resources handle_{};
  rmm::mr::pinned_host_memory_resource mr_pinned_;
  raft::mr::cuda_huge_page_resource mr_huge_page_;
  AllocatorType graph_mem_{AllocatorType::kDevice};
  AllocatorType dataset_mem_{AllocatorType::kDevice};
  float refine_ratio_;
  build_param index_params_;
  bool need_dataset_update_{true};
  cuvs::neighbors::cagra::search_params search_params_;
  std::shared_ptr<cuvs::neighbors::cagra::index<T, IdxT>> index_;
  std::shared_ptr<raft::device_matrix<IdxT, int64_t, raft::row_major>> graph_;
  std::shared_ptr<raft::device_matrix<T, int64_t, raft::row_major>> dataset_;
  std::shared_ptr<raft::device_matrix_view<const T, int64_t, raft::row_major>> input_dataset_v_;

  std::shared_ptr<cuvs::neighbors::dynamic_batching::index<T, algo_base::index_type>>
    dynamic_batcher_;
  cuvs::neighbors::dynamic_batching::search_params dynamic_batcher_sp_{};
  int64_t dynamic_batching_max_batch_size_;
  size_t dynamic_batching_n_queues_;
  bool dynamic_batching_conservative_dispatch_;

  std::shared_ptr<rmm::device_uvector<uint32_t>> filter_bitset_;
  std::shared_ptr<cuvs::neighbors::filtering::base_filter> filter_;
  std::shared_ptr<detail::udf_filter_adapter> udf_filter_adapter_;
  std::shared_ptr<detail::udf_filter_runtime> udf_filter_runtime_;
  std::shared_ptr<rmm::device_uvector<float>> favor_udf_sampled_rates_;
  std::shared_ptr<rmm::device_uvector<std::uint32_t>> favor_udf_sampled_passing_counts_;
  bool favor_udf_include_sampling_{true};
  bool favor_udf_passing_accumulator_{true};
  std::uint32_t favor_udf_sample_offset_{};
  std::optional<std::uint32_t> navix_policy_;
  mutable std::uint32_t udf_query_offset_{};
  bool filter_empty_{false};
  std::shared_ptr<detail::favor_diagnostic_session> favor_diagnostic_session_;
  std::shared_ptr<detail::favor_retry_diagnostic_session> favor_retry_diagnostic_session_;
  std::vector<std::uint64_t> favor_seed_masks_;
  std::vector<std::shared_ptr<cuvs::neighbors::cagra::index<T, IdxT>>> sub_indices_;

  inline rmm::device_async_resource_ref get_mr(AllocatorType mem_type)
  {
    switch (mem_type) {
      case (AllocatorType::kHostPinned): return mr_pinned_;
      case (AllocatorType::kHostHugePage): return mr_huge_page_;
      default: return rmm::mr::get_current_device_resource_ref();
    }
  }
};

template <typename T, typename IdxT>
void cuvs_cagra<T, IdxT>::build(const T* dataset, size_t nrow)
{
  auto dataset_extents = raft::make_extents<IdxT>(nrow, dim_);
  auto params          = index_params_.cagra_params(dataset_extents, parse_metric_type(metric_));

  auto dataset_view_host =
    raft::make_mdspan<const T, IdxT, raft::row_major, true, false>(dataset, dataset_extents);
  auto dataset_view_device =
    raft::make_mdspan<const T, IdxT, raft::row_major, false, true>(dataset, dataset_extents);
  bool dataset_is_on_host = raft::get_device_for_address(dataset) == -1;
  if (index_params_.num_dataset_splits <= 1) {
    index_ = std::make_shared<cuvs::neighbors::cagra::index<T, IdxT>>(std::move(
      dataset_is_on_host ? cuvs::neighbors::cagra::build(handle_, params, dataset_view_host)
                         : cuvs::neighbors::cagra::build(handle_, params, dataset_view_device)));
  } else {
    IdxT rows_per_split =
      raft::ceildiv<IdxT>(nrow, static_cast<IdxT>(index_params_.num_dataset_splits));
    for (size_t i = 0; i < index_params_.num_dataset_splits; ++i) {
      IdxT start = static_cast<IdxT>(i * rows_per_split);
      if (start >= nrow) break;
      IdxT rows        = std::min(rows_per_split, static_cast<IdxT>(nrow) - start);
      const T* sub_ptr = dataset + static_cast<size_t>(start) * dim_;
      auto sub_host =
        raft::make_host_matrix_view<const T, int64_t, raft::row_major>(sub_ptr, rows, dim_);
      auto sub_dev =
        raft::make_device_matrix_view<const T, int64_t, raft::row_major>(sub_ptr, rows, dim_);

      auto sub_index = cuvs::neighbors::cagra::index<T, IdxT>(handle_, params.metric);
      if (index_params_.merge_type == CagraMergeType::kPhysical) {
        if (dataset_is_on_host) {
          sub_index.update_dataset(handle_, sub_host);
        } else {
          sub_index.update_dataset(handle_, sub_dev);
        }
      }
      if (index_params_.merge_type == CagraMergeType::kLogical) {
        if (dataset_is_on_host) {
          sub_index = cuvs::neighbors::cagra::build(handle_, params, sub_host);
        } else {
          sub_index = cuvs::neighbors::cagra::build(handle_, params, sub_dev);
        }
      }
      auto sub_index_shared =
        std::make_shared<cuvs::neighbors::cagra::index<T, IdxT>>(std::move(sub_index));
      sub_indices_.push_back(std::move(sub_index_shared));
    }
    if (index_params_.merge_type == CagraMergeType::kPhysical) {
      std::vector<cuvs::neighbors::cagra::index<T, IdxT>*> indices;
      indices.reserve(sub_indices_.size());
      for (auto& ptr : sub_indices_) {
        indices.push_back(ptr.get());
      }

      index_ = std::make_shared<cuvs::neighbors::cagra::index<T, IdxT>>(
        std::move(cuvs::neighbors::cagra::merge(handle_, params, indices)));
    }
  }
}

inline auto allocator_to_string(AllocatorType mem_type) -> std::string
{
  if (mem_type == AllocatorType::kDevice) {
    return "device";
  } else if (mem_type == AllocatorType::kHostPinned) {
    return "host_pinned";
  } else if (mem_type == AllocatorType::kHostHugePage) {
    return "host_huge_page";
  }
  return "<invalid allocator type>";
}

template <typename T, typename IdxT>
void cuvs_cagra<T, IdxT>::set_search_param(const search_param_base& param,
                                           const void* filter_bitset)
{
  const auto& dataset_conf = configuration::singleton().get_dataset_conf();
  if (dataset_conf.udf_filter.has_value()) {
    RAFT_EXPECTS(filter_bitset == nullptr,
                 "A query-dependent UDF filter cannot be combined with a bitset filter");
    if (!udf_filter_adapter_) {
      udf_filter_adapter_ = detail::make_udf_filter_adapter(handle_, *dataset_conf.udf_filter);
    }
    RAFT_EXPECTS(index_ != nullptr, "The CAGRA index must be loaded before UDF metadata");
    RAFT_EXPECTS(udf_filter_adapter_->base_rows() == index_->size(),
                 "Filtered-dataset base metadata row count does not match the CAGRA index");
    udf_filter_runtime_ = udf_filter_adapter_->make_runtime(handle_);
    filter_ =
      std::make_shared<cuvs::neighbors::filtering::udf_filter>(udf_filter_runtime_->filter());
    filter_bitset_.reset();
  } else if (index_ && filter_bitset != nullptr) {
    udf_filter_runtime_.reset();
    udf_filter_adapter_.reset();
    auto n_words   = raft::ceildiv<size_t>(index_->size(), sizeof(uint32_t) * 8);
    auto stream    = raft::resource::get_cuda_stream(handle_);
    filter_bitset_ = std::make_shared<rmm::device_uvector<uint32_t>>(n_words, stream);
    raft::copy(
      filter_bitset_->data(), reinterpret_cast<uint32_t const*>(filter_bitset), n_words, stream);
    filter_ = make_cuvs_filter(filter_bitset_->data(), index_->size());
  } else {
    udf_filter_runtime_.reset();
    udf_filter_adapter_.reset();
    filter_bitset_.reset();
    filter_ = make_cuvs_filter(nullptr, index_ ? index_->size() : 0);
  }
  auto sp = dynamic_cast<const search_param&>(param);
  RAFT_EXPECTS(!udf_filter_runtime_ || !sp.dynamic_batching,
               "Query-dependent UDF filters do not support dynamic batching");
  RAFT_EXPECTS(!udf_filter_runtime_ || index_params_.num_dataset_splits <= 1 ||
                 index_params_.merge_type == CagraMergeType::kPhysical,
               "Query-dependent UDF filters require a physical CAGRA index");
  bool needs_dynamic_batcher_update =
    (dynamic_batching_max_batch_size_ != sp.dynamic_batching_max_batch_size) ||
    (dynamic_batching_n_queues_ != sp.dynamic_batching_n_queues) ||
    (dynamic_batching_conservative_dispatch_ != sp.dynamic_batching_conservative_dispatch);
  dynamic_batching_max_batch_size_        = sp.dynamic_batching_max_batch_size;
  dynamic_batching_n_queues_              = sp.dynamic_batching_n_queues;
  dynamic_batching_conservative_dispatch_ = sp.dynamic_batching_conservative_dispatch;
  search_params_                          = sp.p;
  refine_ratio_                           = sp.refine_ratio;
  favor_seed_masks_                       = sp.favor_seed_masks;
  if (search_params_.filter_mode == cuvs::neighbors::cagra::filtering_mode::DEFAULT) {
    favor_udf_include_sampling_ = false;
    favor_udf_passing_accumulator_ =
      sp.favor_udf_passing_accumulator_is_set ? sp.favor_udf_passing_accumulator : false;
  } else {
    favor_udf_include_sampling_    = sp.favor_udf_include_sampling;
    favor_udf_passing_accumulator_ = sp.favor_udf_passing_accumulator;
  }
  favor_udf_sample_offset_ = sp.favor_udf_sample_offset;
  navix_policy_            = sp.navix_policy;
  filter_empty_            = false;
  favor_udf_sampled_rates_.reset();
  favor_udf_sampled_passing_counts_.reset();
  const bool is_favor_filtering_mode =
    search_params_.filter_mode == cuvs::neighbors::cagra::filtering_mode::FAVOR;

  const bool needs_udf_filter_rates = udf_filter_runtime_ && is_favor_filtering_mode;
  if (needs_udf_filter_rates) {
    RAFT_EXPECTS(search_params_.filter_mode == cuvs::neighbors::cagra::filtering_mode::FAVOR,
                 "benchmark UDF sampled-rate path supports FAVOR mode");
    auto stream              = raft::resource::get_cuda_stream(handle_);
    const auto query_rows    = udf_filter_adapter_->query_rows();
    favor_udf_sampled_rates_ = std::make_shared<rmm::device_uvector<float>>(query_rows, stream);
    favor_udf_sampled_passing_counts_ =
      std::make_shared<rmm::device_uvector<std::uint32_t>>(query_rows, stream);
    if (!favor_udf_include_sampling_) {
      udf_filter_runtime_->set_query_offset(0);
      cuvs::neighbors::cagra::detail::benchmark_estimate_favor_udf_filtering_rates<T>(
        handle_,
        *index_,
        query_rows,
        *filter_,
        favor_udf_sampled_rates_->data(),
        favor_udf_sampled_passing_counts_->data(),
        favor_udf_sample_offset_);
      raft::resource::sync_stream(handle_);
    }
  }
  if (navix_policy_) {
    RAFT_EXPECTS(udf_filter_runtime_ != nullptr,
                 "The benchmark-only NaviX path requires a query-dependent UDF filter");
    RAFT_EXPECTS(search_params_.filter_mode == cuvs::neighbors::cagra::filtering_mode::DEFAULT,
                 "The benchmark-only NaviX path requires filter_mode=default");
    RAFT_EXPECTS(search_params_.algo == cuvs::neighbors::cagra::search_algo::AUTO ||
                   search_params_.algo == cuvs::neighbors::cagra::search_algo::SINGLE_CTA,
                 "The benchmark-only NaviX path supports only AUTO or SINGLE_CTA");
    RAFT_EXPECTS(!search_params_.persistent,
                 "The benchmark-only NaviX path does not support persistent search");
    RAFT_EXPECTS(!sp.dynamic_batching,
                 "The benchmark-only NaviX path does not support dynamic batching");
    RAFT_EXPECTS(sp.refine_ratio == 1.0f,
                 "The benchmark-only NaviX path does not support refinement");
    RAFT_EXPECTS(index_->graph().extent(1) == 32,
                 "The benchmark-only NaviX path currently requires graph_degree=32");
    RAFT_EXPECTS(!favor_udf_passing_accumulator_,
                 "NaviX owns the passing frontier and cannot use the result accumulator");
    RAFT_EXPECTS(!sp.favor_retry_diagnostics.enabled(),
                 "The benchmark-only NaviX path does not support FAVOR retry diagnostics");
    RAFT_EXPECTS(favor_seed_masks_.empty(),
                 "The benchmark-only NaviX path does not support FAVOR retry seeds");
  }
  if (filter_bitset_ != nullptr && search_params_.filtering_rate < 0.0f) {
    const auto num_set_bits =
      cuvs::neighbors::cagra::detail::benchmark_count_favor_bitset_matches<T>(
        handle_, *index_, *filter_);
    filter_empty_ = num_set_bits == 0;
    if (!filter_empty_) {
      const auto filtering_rate = static_cast<float>(index_->data().n_rows() - num_set_bits) /
                                  static_cast<float>(index_->data().n_rows());
      search_params_.filtering_rate = std::min(filtering_rate, std::nextafter(1.0f, 0.0f));
    }
  }
  if (!favor_seed_masks_.empty()) {
    RAFT_EXPECTS(favor_seed_masks_.size() <= 3,
                 "favor_seed_masks supports between one and three benchmark rounds");
    RAFT_EXPECTS(filter_bitset_ != nullptr, "favor_seed_masks requires a benchmark bitset filter");
    RAFT_EXPECTS(search_params_.filter_mode == cuvs::neighbors::cagra::filtering_mode::FAVOR,
                 "favor_seed_masks requires filter_mode=favor");
    RAFT_EXPECTS(search_params_.algo == cuvs::neighbors::cagra::search_algo::SINGLE_CTA,
                 "favor_seed_masks supports only algo=single_cta");
    RAFT_EXPECTS(search_params_.max_iterations == 0,
                 "favor_seed_masks is a B0 experiment and requires max_iterations=0");
    RAFT_EXPECTS(search_params_.favor_penalty ==
                     cuvs::neighbors::cagra::favor_penalty_mode::CAGRA_RETENTION_SAFE &&
                   search_params_.favor_retention_fraction == 0.0f,
                 "favor_seed_masks requires automatic CAGRA retention-safe FAVOR");
    RAFT_EXPECTS(!search_params_.persistent, "favor_seed_masks does not support persistent search");
    RAFT_EXPECTS(!sp.dynamic_batching, "favor_seed_masks does not support dynamic batching");
    RAFT_EXPECTS(sp.refine_ratio == 1.0f, "favor_seed_masks does not support refinement");
    RAFT_EXPECTS(!sp.favor_diagnostics.enabled(),
                 "favor_seed_masks does not support FAVOR trace diagnostics");
    RAFT_EXPECTS(index_params_.num_dataset_splits <= 1 ||
                   index_params_.merge_type == CagraMergeType::kPhysical,
                 "favor_seed_masks supports only physical CAGRA indices");
    RAFT_EXPECTS(
      !detail::favor_adaptive_termination_enabled(),
      "favor_seed_masks requires CUVS_FAVOR_EXPERIMENTAL_ADAPTIVE_TERMINATION to be disabled");
    for (std::size_t i = 0; i < favor_seed_masks_.size(); ++i) {
      for (std::size_t j = i + 1; j < favor_seed_masks_.size(); ++j) {
        RAFT_EXPECTS(favor_seed_masks_[i] != favor_seed_masks_[j],
                     "favor_seed_masks entries must be unique");
      }
    }
  }
  if (sp.favor_diagnostics.enabled()) {
    RAFT_EXPECTS(search_params_.algo == cuvs::neighbors::cagra::search_algo::SINGLE_CTA,
                 "FAVOR diagnostics support only SINGLE_CTA");
    RAFT_EXPECTS(!search_params_.persistent, "FAVOR diagnostics do not support persistent search");
    RAFT_EXPECTS(!sp.dynamic_batching, "FAVOR diagnostics do not support dynamic batching");
    if (sp.favor_diagnostics.termination_shadow_enabled()) {
      RAFT_EXPECTS(filter_bitset_ != nullptr,
                   "FAVOR termination shadow requires a benchmark bitset filter");
      RAFT_EXPECTS(search_params_.filter_mode == cuvs::neighbors::cagra::filtering_mode::FAVOR,
                   "FAVOR termination shadow requires filter_mode=favor");
      RAFT_EXPECTS(search_params_.max_iterations != 0,
                   "FAVOR termination shadow requires an explicit deep max_iterations cap");
      RAFT_EXPECTS(search_params_.favor_penalty ==
                       cuvs::neighbors::cagra::favor_penalty_mode::CAGRA_RETENTION_SAFE &&
                     search_params_.favor_retention_fraction == 0.0f,
                   "FAVOR termination shadow requires automatic CAGRA retention-safe FAVOR");
      RAFT_EXPECTS(sp.refine_ratio == 1.0f, "FAVOR termination shadow does not support refinement");
      RAFT_EXPECTS(favor_seed_masks_.empty(),
                   "FAVOR termination shadow does not support favor_seed_masks");
      RAFT_EXPECTS(!detail::favor_adaptive_termination_enabled(),
                   "FAVOR termination shadow requires adaptive termination to be disabled");
    }
    favor_diagnostic_session_ =
      std::make_shared<detail::favor_diagnostic_session>(sp.favor_diagnostics);
  } else {
    favor_diagnostic_session_.reset();
  }
  if (sp.favor_retry_diagnostics.enabled()) {
    RAFT_EXPECTS(filter_bitset_ != nullptr,
                 "FAVOR retry diagnostics require a benchmark bitset filter");
    RAFT_EXPECTS(search_params_.filter_mode == cuvs::neighbors::cagra::filtering_mode::FAVOR,
                 "FAVOR retry diagnostics require filter_mode=favor");
    RAFT_EXPECTS(search_params_.algo == cuvs::neighbors::cagra::search_algo::SINGLE_CTA,
                 "FAVOR retry diagnostics support only algo=single_cta");
    RAFT_EXPECTS(search_params_.max_iterations == 0,
                 "FAVOR retry diagnostics require automatic max_iterations in the base config");
    RAFT_EXPECTS(search_params_.favor_penalty ==
                     cuvs::neighbors::cagra::favor_penalty_mode::CAGRA_RETENTION_SAFE &&
                   search_params_.favor_retention_fraction == 0.0f,
                 "FAVOR retry diagnostics require automatic CAGRA retention-safe FAVOR");
    RAFT_EXPECTS(!search_params_.persistent,
                 "FAVOR retry diagnostics do not support persistent search");
    RAFT_EXPECTS(!sp.dynamic_batching, "FAVOR retry diagnostics do not support dynamic batching");
    RAFT_EXPECTS(sp.refine_ratio == 1.0f, "FAVOR retry diagnostics do not support refinement");
    RAFT_EXPECTS(favor_seed_masks_.empty(),
                 "FAVOR retry diagnostics do not support favor_seed_masks");
    RAFT_EXPECTS(!sp.favor_diagnostics.enabled(),
                 "FAVOR retry diagnostics do not support FAVOR trace diagnostics");
    RAFT_EXPECTS(index_params_.num_dataset_splits <= 1 ||
                   index_params_.merge_type == CagraMergeType::kPhysical,
                 "FAVOR retry diagnostics support only physical CAGRA indices");
    RAFT_EXPECTS(!detail::favor_adaptive_termination_enabled(),
                 "FAVOR retry diagnostics require adaptive termination to be disabled");
    favor_retry_diagnostic_session_ =
      std::make_shared<detail::favor_retry_diagnostic_session>(sp.favor_retry_diagnostics);
  } else {
    favor_retry_diagnostic_session_.reset();
  }
  if (sp.graph_mem != graph_mem_) {
    // Move graph to correct memory space
    graph_mem_ = sp.graph_mem;
    RAFT_LOG_DEBUG("moving graph to new memory space: %s", allocator_to_string(graph_mem_).c_str());
    // We create a new graph and copy to it from existing graph
    auto mr = get_mr(graph_mem_);

    // Create a new graph, then copy, and __only then__ replace the shared pointer.
    auto old_graph =
      index_->graph();  // view of graph_ if it exists, of an internal index member otherwise
    auto new_graph = raft::make_device_mdarray<IdxT, int64_t>(handle_, mr, old_graph.extents());
    raft::copy(new_graph.data_handle(),
               old_graph.data_handle(),
               old_graph.size(),
               raft::resource::get_cuda_stream(handle_));
    raft::resource::sync_stream(handle_);
    *graph_ = std::move(new_graph);

    // NB: update_graph() only stores a view in the index. We need to keep the graph object alive.
    index_->update_graph(handle_, make_const_mdspan(graph_->view()));
    needs_dynamic_batcher_update = true;
  }

  if (sp.dataset_mem != dataset_mem_ || need_dataset_update_) {
    dataset_mem_ = sp.dataset_mem;

    // First free up existing memory
    *dataset_ = raft::make_device_matrix<T, int64_t>(handle_, 0, 0);
    index_->update_dataset(handle_, make_const_mdspan(dataset_->view()));

    // Allocate space using the correct memory resource.
    RAFT_LOG_DEBUG("moving dataset to new memory space: %s",
                   allocator_to_string(dataset_mem_).c_str());

    auto mr = get_mr(dataset_mem_);
    cuvs::neighbors::cagra::detail::copy_with_padding(handle_, *dataset_, *input_dataset_v_, mr);

    auto dataset_view = raft::make_device_strided_matrix_view<const T, int64_t>(
      dataset_->data_handle(), dataset_->extent(0), this->dim_, dataset_->extent(1));
    index_->update_dataset(handle_, dataset_view);

    need_dataset_update_         = false;
    needs_dynamic_batcher_update = true;
  }

  // Deserialized CAGRA graph files do not contain the dense dataset. Load and validate the FAVOR
  // sidecar only after the benchmark's search dataset and any relocated graph have been attached.
  if (search_params_.filter_mode == cuvs::neighbors::cagra::filtering_mode::FAVOR &&
      sp.favor_delta_d_file.has_value()) {
    RAFT_EXPECTS(index_ != nullptr, "The CAGRA index must be loaded before FAVOR delta-d");
    search_params_.favor_delta_d = cuvs::neighbors::cagra::load_favor_delta_d(
      handle_, sp.favor_delta_d_file.value(), sp.favor_delta_d_params, *index_);
  }

  // dynamic batching
  if (sp.dynamic_batching) {
    if (!dynamic_batcher_ || needs_dynamic_batcher_update) {
      dynamic_batcher_ =
        std::make_shared<cuvs::neighbors::dynamic_batching::index<T, algo_base::index_type>>(
          handle_,
          cuvs::neighbors::dynamic_batching::index_params{
            {},
            sp.dynamic_batching_k,
            sp.dynamic_batching_max_batch_size,
            sp.dynamic_batching_n_queues,
            sp.dynamic_batching_conservative_dispatch},
          *index_,
          search_params_,
          filter_.get());
    }
    dynamic_batcher_sp_.dispatch_timeout_ms = sp.dynamic_batching_dispatch_timeout_ms;
  } else {
    if (dynamic_batcher_) { dynamic_batcher_.reset(); }
  }

  detail::maybe_log_cagra_persistent_concurrency_hint(search_params_.persistent);
}

template <typename T, typename IdxT>
void cuvs_cagra<T, IdxT>::set_search_dataset(const T* dataset, size_t nrow)
{
  if (index_params_.num_dataset_splits > 1 &&
      index_params_.merge_type == CagraMergeType::kLogical) {
    bool dataset_is_on_host = raft::get_device_for_address(dataset) == -1;
    IdxT rows_per_split =
      raft::ceildiv<IdxT>(nrow, static_cast<IdxT>(index_params_.num_dataset_splits));
    for (size_t i = 0; i < sub_indices_.size(); ++i) {
      IdxT start = static_cast<IdxT>(i * rows_per_split);
      if (start >= nrow) break;
      IdxT rows        = std::min(rows_per_split, static_cast<IdxT>(nrow) - start);
      const T* sub_ptr = dataset + static_cast<size_t>(start) * dim_;
      auto sub_host =
        raft::make_host_matrix_view<const T, int64_t, raft::row_major>(sub_ptr, rows, dim_);
      auto sub_dev =
        raft::make_device_matrix_view<const T, int64_t, raft::row_major>(sub_ptr, rows, dim_);
      auto sub_index = sub_indices_[i].get();
      if (index_params_.merge_type == CagraMergeType::kLogical) {
        if (dataset_is_on_host) {
          sub_index->update_dataset(handle_, sub_host);
        } else {
          sub_index->update_dataset(handle_, sub_dev);
        }
      }
    }
    need_dataset_update_ = false;
  } else {
    using ds_idx_type = decltype(index_->data().n_rows());
    bool is_vpq =
      dynamic_cast<const cuvs::neighbors::vpq_dataset<half, ds_idx_type>*>(&index_->data()) ||
      dynamic_cast<const cuvs::neighbors::vpq_dataset<float, ds_idx_type>*>(&index_->data());
    // It can happen that we are re-using a previous algo object which already has
    // the dataset set. Check if we need update.
    if (static_cast<size_t>(input_dataset_v_->extent(0)) != nrow ||
        input_dataset_v_->data_handle() != dataset) {
      *input_dataset_v_ =
        raft::make_device_matrix_view<const T, int64_t>(dataset, nrow, this->dim_);
      need_dataset_update_ = !is_vpq;  // ignore update if this is a VPQ dataset.
    }
  }
}

template <typename T, typename IdxT>
void cuvs_cagra<T, IdxT>::save(const std::string& file) const
{
  if (index_params_.num_dataset_splits > 1 &&
      index_params_.merge_type == CagraMergeType::kLogical) {
    for (size_t i = 0; i < sub_indices_.size(); ++i) {
      std::string subfile = file + (i == 0 ? "" : ".subidx." + std::to_string(i));
      cuvs::neighbors::cagra::serialize(handle_, subfile, *sub_indices_[i], false);
    }
    std::ofstream f(file + ".submeta", std::ios::out);
    f << sub_indices_.size();
    f.close();
  } else {
    using ds_idx_type = decltype(index_->data().n_rows());
    bool is_vpq =
      dynamic_cast<const cuvs::neighbors::vpq_dataset<half, ds_idx_type>*>(&index_->data()) ||
      dynamic_cast<const cuvs::neighbors::vpq_dataset<float, ds_idx_type>*>(&index_->data());
    cuvs::neighbors::cagra::serialize(handle_, file, *index_, is_vpq);
  }
}

template <typename T, typename IdxT>
void cuvs_cagra<T, IdxT>::save_to_hnswlib(const std::string& file) const
{
  cuvs::neighbors::cagra::serialize_to_hnswlib(handle_, file, *index_);
}

template <typename T, typename IdxT>
void cuvs_cagra<T, IdxT>::load(const std::string& file)
{
  std::ifstream meta(file + ".submeta", std::ios::in);
  if (index_params_.num_dataset_splits > 1 &&
      index_params_.merge_type == CagraMergeType::kLogical && meta.good()) {
    // Load multiple sub-indices for logical merge
    size_t count;
    meta >> count;
    meta.close();
    sub_indices_.clear();
    for (size_t i = 0; i < count; ++i) {
      std::string subfile = file + (i == 0 ? "" : ".subidx." + std::to_string(i));
      auto sub_index      = std::make_shared<cuvs::neighbors::cagra::index<T, IdxT>>(handle_);
      cuvs::neighbors::cagra::deserialize(handle_, subfile, sub_index.get());
      sub_indices_.push_back(std::move(sub_index));
    }
  } else {
    index_ = std::make_shared<cuvs::neighbors::cagra::index<T, IdxT>>(handle_);
    cuvs::neighbors::cagra::deserialize(handle_, file, index_.get());
  }
}

template <typename T, typename IdxT>
std::unique_ptr<algo<T>> cuvs_cagra<T, IdxT>::copy()
{
  auto result = std::make_unique<cuvs_cagra<T, IdxT>>(std::cref(*this));
  if (udf_filter_adapter_) {
    // Device metadata is immutable and shared.  The tiny device context is per copy because each
    // benchmark thread advances through the query set independently.
    result->udf_filter_runtime_ = udf_filter_adapter_->make_runtime(result->handle_);
    result->filter_             = std::make_shared<cuvs::neighbors::filtering::udf_filter>(
      result->udf_filter_runtime_->filter());
    if (result->favor_udf_include_sampling_ && favor_udf_sampled_rates_) {
      // Timed sampling writes one rate/count per query.  Give every benchmark copy independent
      // output buffers so concurrent workers cannot overwrite each other's query slices.
      auto stream           = raft::resource::get_cuda_stream(result->handle_);
      const auto query_rows = udf_filter_adapter_->query_rows();
      result->favor_udf_sampled_rates_ =
        std::make_shared<rmm::device_uvector<float>>(query_rows, stream);
      result->favor_udf_sampled_passing_counts_ =
        std::make_shared<rmm::device_uvector<std::uint32_t>>(query_rows, stream);
    }
  }
  return result;
}

template <typename T, typename IdxT>
void cuvs_cagra<T, IdxT>::search_base(
  const T* queries, int batch_size, int k, algo_base::index_type* neighbors, float* distances) const
{
  static_assert(std::is_integral_v<algo_base::index_type>);
  static_assert(std::is_integral_v<IdxT>);

  auto queries_view = raft::make_device_matrix_view<const T, int64_t>(queries, batch_size, dim_);
  auto neighbors_view =
    raft::make_device_matrix_view<algo_base::index_type, int64_t>(neighbors, batch_size, k);
  auto distances_view = raft::make_device_matrix_view<float, int64_t>(distances, batch_size, k);

  if (filter_empty_) {
    detail::fill_empty_filter_results(
      handle_, neighbors, distances, static_cast<std::size_t>(batch_size) * k);
    return;
  }

  if (navix_policy_) {
    auto run_navix = [&]() {
      cuvs::neighbors::cagra::detail::benchmark_search_navix_udf<T>(handle_,
                                                                    search_params_,
                                                                    *index_,
                                                                    queries_view,
                                                                    neighbors_view,
                                                                    distances_view,
                                                                    *filter_,
                                                                    *navix_policy_);
    };
    if (favor_diagnostic_session_) {
      favor_diagnostic_session_->capture(handle_,
                                         static_cast<std::uint32_t>(batch_size),
                                         static_cast<std::uint32_t>(k),
                                         static_cast<std::uint32_t>(index_->graph().extent(1)),
                                         static_cast<std::uint32_t>(search_params_.search_width),
                                         static_cast<std::int64_t>(index_->size()),
                                         static_cast<std::uint32_t>(search_params_.itopk_size),
                                         search_params_.max_iterations,
                                         -1.0f,
                                         true,
                                         neighbors,
                                         run_navix);
    } else {
      run_navix();
    }
    return;
  }

  const bool run_udf_accumulator_path =
    udf_filter_runtime_ &&
    (search_params_.filter_mode == cuvs::neighbors::cagra::filtering_mode::FAVOR ||
     favor_udf_passing_accumulator_);

  if (run_udf_accumulator_path) {
    float* sampled_rates = (favor_udf_sampled_rates_ && favor_udf_sampled_rates_->size() > 0)
                             ? favor_udf_sampled_rates_->data() + udf_query_offset_
                             : nullptr;
    auto run_udf_accumulated_search = [&]() {
      if (favor_udf_include_sampling_) {
        cuvs::neighbors::cagra::detail::benchmark_estimate_favor_udf_filtering_rates<T>(
          handle_,
          *index_,
          static_cast<std::uint32_t>(batch_size),
          *filter_,
          sampled_rates,
          favor_udf_sampled_passing_counts_
            ? favor_udf_sampled_passing_counts_->data() + udf_query_offset_
            : nullptr,
          favor_udf_sample_offset_);
      }
      if (search_params_.filter_mode == cuvs::neighbors::cagra::filtering_mode::FAVOR) {
        cuvs::neighbors::cagra::detail::benchmark_search_favor_udf_with_sampled_rates<T>(
          handle_,
          search_params_,
          *index_,
          queries_view,
          neighbors_view,
          distances_view,
          *filter_,
          sampled_rates,
          favor_udf_passing_accumulator_);
        return;
      }

      cuvs::neighbors::cagra::detail::benchmark_search_favor_udf_with_sampled_rates<T>(
        handle_,
        search_params_,
        *index_,
        queries_view,
        neighbors_view,
        distances_view,
        *filter_,
        sampled_rates,
        favor_udf_passing_accumulator_);
    };
    if (favor_diagnostic_session_) {
      favor_diagnostic_session_->capture(handle_,
                                         static_cast<std::uint32_t>(batch_size),
                                         static_cast<std::uint32_t>(k),
                                         static_cast<std::uint32_t>(index_->graph().extent(1)),
                                         static_cast<std::uint32_t>(search_params_.search_width),
                                         static_cast<std::int64_t>(index_->size()),
                                         static_cast<std::uint32_t>(search_params_.itopk_size),
                                         search_params_.max_iterations,
                                         -1.0f,
                                         true,
                                         neighbors,
                                         run_udf_accumulated_search);
    } else {
      run_udf_accumulated_search();
    }
    return;
  }

  if (favor_retry_diagnostic_session_) {
    const auto filtering_rate = search_params_.filtering_rate;
    auto run_retry_round =
      [&](std::uint32_t max_iterations,
          std::uint64_t rand_xor_mask,
          cuvs::neighbors::cagra::detail::favor_search_diagnostics::context* diagnostic_context,
          const std::uint32_t* seeds,
          std::uint32_t seed_count,
          std::int64_t* round_neighbors,
          float* round_distances) {
        auto params_copy           = search_params_;
        params_copy.filtering_rate = filtering_rate;
        params_copy.max_iterations = max_iterations;
        params_copy.rand_xor_mask  = rand_xor_mask;
        auto round_neighbors_view =
          raft::make_device_matrix_view<std::int64_t, std::int64_t>(round_neighbors, batch_size, k);
        auto round_distances_view =
          raft::make_device_matrix_view<float, std::int64_t>(round_distances, batch_size, k);
        cuvs::neighbors::cagra::detail::favor_search_diagnostics::scoped_context context_scope{
          diagnostic_context};
        cuvs::neighbors::cagra::detail::favor_search_diagnostics::scoped_seeds seed_scope{
          seeds, seed_count};
        cuvs::neighbors::cagra::detail::benchmark_search_favor_with_known_filtering_rate<T>(
          handle_,
          params_copy,
          *index_,
          queries_view,
          round_neighbors_view,
          round_distances_view,
          *filter_);
      };
    favor_retry_diagnostic_session_->capture(
      handle_,
      static_cast<std::uint32_t>(batch_size),
      static_cast<std::uint32_t>(k),
      static_cast<std::uint32_t>(index_->graph().extent(1)),
      static_cast<std::uint32_t>(search_params_.search_width),
      static_cast<std::int64_t>(index_->size()),
      static_cast<std::uint32_t>(search_params_.itopk_size),
      filtering_rate,
      neighbors,
      distances,
      run_retry_round);
    return;
  }

  if (!favor_seed_masks_.empty()) {
    auto params_copy = search_params_;

    if (favor_seed_masks_.size() == 1) {
      params_copy.rand_xor_mask = favor_seed_masks_.front();
      cuvs::neighbors::cagra::detail::benchmark_search_favor_with_known_filtering_rate<T>(
        handle_, params_copy, *index_, queries_view, neighbors_view, distances_view, *filter_);
      return;
    }

    auto const n_queries       = static_cast<std::size_t>(batch_size);
    auto const result_width    = static_cast<std::size_t>(k);
    auto const n_rounds        = favor_seed_masks_.size();
    auto const round_elements  = n_queries * result_width * n_rounds;
    auto const output_elements = n_queries * result_width;
    auto const outputs_on_device =
      raft::get_device_for_address(neighbors) >= 0 && raft::get_device_for_address(distances) >= 0;
    auto const workspace_size =
      round_elements * (sizeof(algo_base::index_type) + sizeof(float)) +
      (outputs_on_device ? 0 : output_elements * (sizeof(algo_base::index_type) + sizeof(float)));
    auto& workspace        = get_tmp_buffer_from_global_pool(workspace_size);
    auto* workspace_ptr    = reinterpret_cast<std::uint8_t*>(workspace.data(MemoryType::kDevice));
    auto* round_neighbors  = reinterpret_cast<algo_base::index_type*>(workspace_ptr);
    auto* merged_neighbors = neighbors;
    auto* neighbor_storage_end = round_neighbors + round_elements;
    if (!outputs_on_device) { merged_neighbors = neighbor_storage_end; }
    auto* round_distances =
      reinterpret_cast<float*>(neighbor_storage_end + (outputs_on_device ? 0 : output_elements));

    for (std::size_t round = 0; round < n_rounds; ++round) {
      params_copy.rand_xor_mask = favor_seed_masks_[round];
      auto round_neighbors_view =
        raft::make_device_matrix_view<algo_base::index_type, std::int64_t>(
          round_neighbors + round * output_elements, batch_size, k);
      auto round_distances_view = raft::make_device_matrix_view<float, std::int64_t>(
        round_distances + round * output_elements, batch_size, k);
      cuvs::neighbors::cagra::detail::benchmark_search_favor_with_known_filtering_rate<T>(
        handle_,
        params_copy,
        *index_,
        queries_view,
        round_neighbors_view,
        round_distances_view,
        *filter_);
    }

    auto* merged_distances = distances;
    if (!outputs_on_device) { merged_distances = round_distances + round_elements; }
    cuvs::neighbors::cagra::detail::merge_favor_multi_seed_results(handle_,
                                                                   round_neighbors,
                                                                   round_distances,
                                                                   merged_neighbors,
                                                                   merged_distances,
                                                                   n_queries,
                                                                   result_width,
                                                                   n_rounds);
    if (!outputs_on_device) {
      auto stream = raft::resource::get_cuda_stream(handle_);
      raft::copy(neighbors, merged_neighbors, output_elements, stream);
      raft::copy(distances, merged_distances, output_elements, stream);
    }
    return;
  }

  if (dynamic_batcher_) {
    cuvs::neighbors::dynamic_batching::search(handle_,
                                              dynamic_batcher_sp_,
                                              *dynamic_batcher_,
                                              queries_view,
                                              neighbors_view,
                                              distances_view);
  } else {
    if (index_params_.num_dataset_splits <= 1 ||
        index_params_.merge_type == CagraMergeType::kPhysical) {
      auto run_search = [&]() {
        cuvs::neighbors::cagra::search(
          handle_, search_params_, *index_, queries_view, neighbors_view, distances_view, *filter_);
      };
      if (favor_diagnostic_session_) {
        favor_diagnostic_session_->capture(handle_,
                                           static_cast<std::uint32_t>(batch_size),
                                           static_cast<std::uint32_t>(k),
                                           static_cast<std::uint32_t>(index_->graph().extent(1)),
                                           static_cast<std::uint32_t>(search_params_.search_width),
                                           static_cast<std::int64_t>(index_->size()),
                                           static_cast<std::uint32_t>(search_params_.itopk_size),
                                           search_params_.max_iterations,
                                           search_params_.filtering_rate,
                                           true,
                                           neighbors,
                                           run_search);
      } else {
        run_search();
      }
    } else {
      if (index_params_.merge_type == CagraMergeType::kLogical) {
        // TODO: index merge must happen outside of search, otherwise what are we benchmarking?
        std::vector<cuvs::neighbors::cagra::index<T, IdxT>*> cagra_indices;
        cagra_indices.reserve(sub_indices_.size());
        for (auto& ptr : sub_indices_) {
          cagra_indices.push_back(ptr.get());
        }

        raft::resources composite_handle(handle_);
        size_t n_streams = cagra_indices.size();
        raft::resource::set_cuda_stream_pool(composite_handle,
                                             std::make_shared<rmm::cuda_stream_pool>(n_streams));

        cuvs::neighbors::composite::composite_index<T, IdxT, algo_base::index_type> composite(
          cagra_indices);
        composite.search(
          composite_handle, search_params_, queries_view, neighbors_view, distances_view);
      }
    }
  }
}

template <typename T, typename IdxT>
void cuvs_cagra<T, IdxT>::search(
  const T* queries, int batch_size, int k, algo_base::index_type* neighbors, float* distances) const
{
  static_assert(std::is_integral_v<algo_base::index_type>);
  static_assert(std::is_integral_v<IdxT>);

  auto k0                       = static_cast<size_t>(refine_ratio_ * k);
  const bool disable_refinement = k0 <= static_cast<size_t>(k);
  const raft::resources& res    = handle_;
  // NOTE: caching mem_type to reduce mutex locks
  // raft::get_device_for_address call cuda API to get the pointer properties,
  // this means it locks the context mutex for a very small amount of time.
  // In the event of thread contention (such as thousands threads), this time can actually increase.
  // Hence we try to bypass this check for repeated search calls.
  thread_local MemoryType mem_type                   = MemoryType::kDevice;
  thread_local algo_base::index_type* prev_neighbors = nullptr;
  if (prev_neighbors != neighbors) {
    prev_neighbors = neighbors;
    mem_type =
      raft::get_device_for_address(neighbors) >= 0 ? MemoryType::kDevice : MemoryType::kHostPinned;
  }

  // If dynamic batching is used and there's no sync between benchmark laps, multiple sequential
  // requests can group together. The data is copied asynchronously, and if the same intermediate
  // buffer is used for multiple requests, they can override each other's data. Hence, we need to
  // allocate as much space as required by the maximum number of sequential requests.
  auto max_dyn_grouping = dynamic_batcher_ ? raft::div_rounding_up_safe<int64_t>(
                                               dynamic_batching_max_batch_size_, batch_size) *
                                               dynamic_batching_n_queues_
                                           : 1;
  auto tmp_buf_size =
    ((disable_refinement ? 0 : (sizeof(float) + sizeof(algo_base::index_type)))) * batch_size * k0;
  auto& tmp_buf = get_tmp_buffer_from_global_pool(tmp_buf_size * max_dyn_grouping);
  thread_local static int64_t group_id = 0;
  auto* candidates_ptr                 = reinterpret_cast<algo_base::index_type*>(
    reinterpret_cast<uint8_t*>(tmp_buf.data(mem_type)) + tmp_buf_size * group_id);
  group_id = (group_id + 1) % max_dyn_grouping;
  auto* candidate_dists_ptr =
    reinterpret_cast<float*>(candidates_ptr + (disable_refinement ? 0 : batch_size * k0));

  if (disable_refinement) {
    search_base(queries, batch_size, k, neighbors, distances);
  } else {
    search_base(queries, batch_size, k0, candidates_ptr, candidate_dists_ptr);

    if (mem_type == MemoryType::kHostPinned && uses_stream()) {
      // If the algorithm uses a stream to synchronize (non-persistent kernel), but the data is in
      // the pinned host memory, we need to synchronize before the refinement operation to wait for
      // the data being available for the host.
      raft::resource::sync_stream(res);
    }

    auto candidate_ixs =
      raft::make_device_matrix_view<const algo_base::index_type, algo_base::index_type>(
        candidates_ptr, batch_size, k0);
    auto queries_v =
      raft::make_device_matrix_view<const T, algo_base::index_type>(queries, batch_size, dim_);
    refine_helper(
      res, *input_dataset_v_, queries_v, candidate_ixs, k, neighbors, distances, index_->metric());
  }
}

template <typename T, typename IdxT>
void cuvs_cagra<T, IdxT>::search_with_query_offset(const T* queries,
                                                   int batch_size,
                                                   int k,
                                                   algo_base::index_type* neighbors,
                                                   float* distances,
                                                   std::size_t query_offset) const
{
  if (udf_filter_runtime_) {
    RAFT_EXPECTS(query_offset <= std::numeric_limits<std::uint32_t>::max(),
                 "Filtered-dataset query offset exceeds uint32");
    RAFT_EXPECTS(
      query_offset + static_cast<std::size_t>(batch_size) <= udf_filter_adapter_->query_rows(),
      "Filtered-dataset query metadata does not cover this benchmark batch");
    udf_filter_runtime_->set_query_offset(static_cast<std::uint32_t>(query_offset));
    udf_query_offset_ = static_cast<std::uint32_t>(query_offset);
  }
  search(queries, batch_size, k, neighbors, distances);
}
}  // namespace cuvs::bench
