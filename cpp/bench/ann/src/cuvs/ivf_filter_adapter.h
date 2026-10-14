/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "cuvs_ann_bench_utils.h"
#include "filtered_dataset_adapter.h"

#include <raft/core/bitmap.cuh>
#include <raft/linalg/map.cuh>

#include <cstdint>
#include <limits>
#include <memory>

namespace cuvs::bench::detail {

/** Immutable resident filter storage shared by copies of an IVF benchmark wrapper. */
class ivf_benchmark_filter {
 public:
  void configure(raft::resources const& res, int64_t index_size, const void* filter_bitset)
  {
    const auto& conf = configuration::singleton().get_dataset_conf();
    RAFT_EXPECTS(!conf.udf_filter.has_value(), "The IVF benchmark does not support UDF filters");
    if (!conf.bitmap_filter.has_value()) {
      bitmap_.reset();
      filter_ = make_cuvs_filter(filter_bitset, index_size);
      return;
    }
    RAFT_EXPECTS(filter_bitset == nullptr,
                 "The IVF benchmark cannot combine a shared bitset and query bitmap");
    if (!bitmap_) { bitmap_ = make_bitmap_filter_adapter(res, *conf.bitmap_filter); }
    RAFT_EXPECTS(bitmap_->base_rows() == index_size,
                 "The query bitmap width must match the IVF index size");
    filter_ =
      std::make_shared<neighbors::filtering::bitmap_filter<uint32_t, int64_t>>(bitmap_->filter());
  }

  [[nodiscard]] auto filter() const -> neighbors::filtering::base_filter const& { return *filter_; }
  [[nodiscard]] auto get() const -> neighbors::filtering::base_filter const*
  {
    return filter_.get();
  }
  [[nodiscard]] auto has_bitmap() const -> bool { return bitmap_ != nullptr; }

  /** Refinement uses the unfiltered IVF sentinel; restore the bitmap benchmark convention. */
  void normalize_results(raft::resources const& res,
                         int64_t* neighbors,
                         float* distances,
                         std::size_t n_queries,
                         uint32_t k,
                         std::size_t query_offset) const
  {
    if (!bitmap_) { return; }
    const auto bitmap = bitmap_->filter().view();
    raft::linalg::map_offset(
      res,
      raft::make_device_vector_view<float, int64_t>(distances, n_queries * k),
      [=] __device__(int64_t i) {
        const auto id       = neighbors[i];
        const auto distance = distances[i];
        if (!(distance < std::numeric_limits<float>::max() &&
              distance > std::numeric_limits<float>::lowest()) ||
            id < 0 || id >= bitmap.get_n_cols() || !bitmap.test(query_offset + i / k, id)) {
          neighbors[i] = std::numeric_limits<int64_t>::max();
          return std::numeric_limits<float>::max();
        }
        return distances[i];
      });
  }

  template <typename SearchParams>
  [[nodiscard]] auto params_for_batch(SearchParams params, std::size_t offset, int batch_size) const
    -> SearchParams
  {
    if (bitmap_) {
      RAFT_EXPECTS(offset <= std::numeric_limits<uint32_t>::max(),
                   "The query bitmap offset exceeds uint32");
      RAFT_EXPECTS(batch_size >= 0 && uint64_t(offset) + batch_size <= bitmap_->query_rows(),
                   "The query bitmap does not cover this benchmark batch");
      params.filter_query_offset = static_cast<uint32_t>(offset);
    }
    return params;
  }

  [[nodiscard]] auto passes(std::size_t query_id, int64_t candidate_id) const -> bool
  {
    return bitmap_ != nullptr && candidate_id >= 0 &&
           uint64_t(candidate_id) < bitmap_->base_rows() && query_id < bitmap_->query_rows() &&
           bitmap_->passes(static_cast<uint32_t>(query_id), static_cast<uint32_t>(candidate_id));
  }

 private:
  std::shared_ptr<bitmap_filter_adapter> bitmap_;
  std::shared_ptr<neighbors::filtering::base_filter> filter_;
};

}  // namespace cuvs::bench::detail
