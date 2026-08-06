/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "cagra_filter_payload.hpp"
#include "jit_lto_kernels/cagra_jit_launcher_factory.hpp"
#include "jit_lto_kernels/kernel_def.hpp"
#include "jit_lto_kernels/sample_filter_udf.cuh"
#include "shared_launcher_jit.hpp"

#include <raft/core/error.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>

#include <rmm/device_uvector.hpp>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>

namespace cuvs::neighbors::cagra::detail {

struct favor_udf_rate_estimate {
  favor_udf_rate_estimate(std::size_t num_queries, rmm::cuda_stream_view stream)
    : filtering_rates(num_queries, stream), passing_counts(num_queries, stream)
  {
  }

  rmm::device_uvector<float> filtering_rates;
  rmm::device_uvector<std::uint32_t> passing_counts;
  std::uint32_t sample_count{};
  std::uint32_t sample_step{};
};

inline auto favor_systematic_sample_step(std::uint64_t dataset_size) -> std::uint32_t
{
  if (dataset_size <= 1'000) { return 1; }
  if (dataset_size <= 10'000) { return 10; }
  if (dataset_size <= 100'000) { return 100; }
  return static_cast<std::uint32_t>(std::max<std::uint64_t>(1, dataset_size / 10'000));
}

template <typename DataT, typename IndexT, typename SampleFilterT>
void estimate_favor_udf_filtering_rates_into(raft::resources const& res,
                                             cagra::index<DataT, IndexT> const& index,
                                             std::uint32_t num_queries,
                                             SampleFilterT const& sample_filter,
                                             float* filtering_rates,
                                             std::uint32_t* passing_counts,
                                             std::uint32_t sample_offset = 0)
{
  static_assert(std::is_same_v<IndexT, std::uint32_t>,
                "CAGRA UDF filtering-rate sampling currently requires uint32 indices");
  RAFT_EXPECTS(index.size() > 0, "Cannot estimate a UDF filtering rate on an empty index");
  RAFT_EXPECTS(index.size() <= std::numeric_limits<std::uint32_t>::max(),
               "CAGRA UDF filtering-rate sampling supports at most uint32 rows");

  RAFT_EXPECTS(num_queries > 0, "Cannot estimate UDF filtering rates for zero queries");
  RAFT_EXPECTS(filtering_rates != nullptr, "UDF filtering-rate output must not be null");

  auto stream             = raft::resource::get_cuda_stream(res);
  const auto sample_step  = favor_systematic_sample_step(static_cast<std::uint64_t>(index.size()));
  const auto sample_count = static_cast<std::uint32_t>(
    (static_cast<std::uint64_t>(index.size()) + sample_step - 1) / sample_step);

  auto launcher =
    make_cagra_filter_rate_estimator_jit_launcher<DataT,
                                                  IndexT,
                                                  float,
                                                  IndexT,
                                                  sample_filter_jit_tag_t<SampleFilterT>>(
      make_cagra_sample_filter_udf_fragment<IndexT>(sample_filter));
  if (!launcher) { RAFT_FAIL("Failed to get the CAGRA UDF filtering-rate sampler launcher"); }

  const auto payload = extract_cagra_sample_filter<IndexT>(sample_filter, stream);
  const IndexT* source_indices_ptr =
    index.source_indices().has_value() ? index.source_indices()->data_handle() : nullptr;
  constexpr std::uint32_t block_size = 256;
  launcher->template dispatch<multi_kernel_search::estimate_filter_rate_kernel_func_t<IndexT>>(
    stream,
    dim3(num_queries, 1, 1),
    dim3(block_size, 1, 1),
    0,
    source_indices_ptr,
    static_cast<std::uint32_t>(index.size()),
    sample_step,
    sample_count,
    sample_offset % static_cast<std::uint32_t>(index.size()),
    num_queries,
    payload,
    filtering_rates,
    passing_counts);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

template <typename DataT, typename IndexT, typename SampleFilterT>
auto estimate_favor_udf_filtering_rates(raft::resources const& res,
                                        cagra::index<DataT, IndexT> const& index,
                                        std::uint32_t num_queries,
                                        SampleFilterT const& sample_filter,
                                        std::uint32_t sample_offset = 0) -> favor_udf_rate_estimate
{
  auto stream = raft::resource::get_cuda_stream(res);
  favor_udf_rate_estimate estimate(num_queries, stream);
  estimate.sample_step  = favor_systematic_sample_step(static_cast<std::uint64_t>(index.size()));
  estimate.sample_count = static_cast<std::uint32_t>(
    (static_cast<std::uint64_t>(index.size()) + estimate.sample_step - 1) / estimate.sample_step);
  estimate_favor_udf_filtering_rates_into(res,
                                          index,
                                          num_queries,
                                          sample_filter,
                                          estimate.filtering_rates.data(),
                                          estimate.passing_counts.data(),
                                          sample_offset);
  return estimate;
}

}  // namespace cuvs::neighbors::cagra::detail
