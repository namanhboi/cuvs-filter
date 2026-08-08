/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include "detail/cagra/add_nodes.cuh"
#include "detail/cagra/cagra_build.cuh"
#include "detail/cagra/cagra_merge.cuh"
#include "detail/cagra/cagra_search.cuh"
#include "detail/cagra/filter_rate_estimator.cuh"
#include "detail/cagra/graph_core.cuh"

#include "detail/ann_utils.cuh"
#include <raft/core/device_mdspan.hpp>
#include <raft/core/host_device_accessor.hpp>
#include <raft/core/mdspan.hpp>
#include <raft/core/resources.hpp>
#include <raft/linalg/norm.cuh>
#include <raft/linalg/reduce.cuh>

#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/cagra.hpp>

#include <cuvs/neighbors/common.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform_reduce.h>

namespace cuvs::neighbors::cagra {

namespace detail {

template <typename T, typename IdxT>
std::uint64_t count_favor_bitset_matches(
  raft::resources const& res,
  cagra::index<T, IdxT> const& index,
  cuvs::neighbors::filtering::bitset_filter<uint32_t, int64_t> const& filter)
{
  if (!index.source_indices().has_value()) {
    return static_cast<std::uint64_t>(filter.bitset_view_.count(res));
  }

  auto source_indices = index.source_indices().value();
  auto const* source  = source_indices.data_handle();
  auto bitset         = filter.bitset_view_;
  auto const nbits    = bitset.get_original_nbits();
  auto stream         = raft::resource::get_cuda_stream(res);
  auto first          = thrust::make_counting_iterator<int64_t>(0);
  return thrust::transform_reduce(
    thrust::cuda::par.on(stream),
    first,
    first + static_cast<int64_t>(index.size()),
    [source, bitset, nbits] __device__(int64_t row) -> std::uint64_t {
      auto source_id = static_cast<int64_t>(source[row]);
      return source_id >= 0 && source_id < nbits && bitset.test(source_id) ? 1u : 0u;
    },
    std::uint64_t{0},
    thrust::plus<std::uint64_t>{});
}

template <typename OutputIdxT>
void fill_empty_favor_results(
  raft::resources const& res,
  raft::device_matrix_view<OutputIdxT, int64_t, raft::row_major> neighbors,
  raft::device_matrix_view<float, int64_t, raft::row_major> distances)
{
  auto stream = raft::resource::get_cuda_stream(res);
  auto policy = thrust::cuda::par.on(stream);
  thrust::fill(policy,
               neighbors.data_handle(),
               neighbors.data_handle() + neighbors.size(),
               std::numeric_limits<OutputIdxT>::max());
  thrust::fill(policy,
               distances.data_handle(),
               distances.data_handle() + distances.size(),
               std::numeric_limits<float>::max());
}

}  // namespace detail

// Member function implementations for cagra::index
template <typename T, typename IdxT>
void index<T, IdxT>::compute_dataset_norms_(raft::resources const& res)
{
  // Get the dataset view
  auto dataset_view = this->dataset();

  // Allocate norms vector if not already allocated
  if (!dataset_norms_.has_value() || dataset_norms_->extent(0) != dataset_view.extent(0)) {
    dataset_norms_.reset();
    dataset_norms_ = raft::make_device_vector<float, int64_t>(res, dataset_view.extent(0));
  }

  constexpr float kScale = cuvs::spatial::knn::detail::utils::config<T>::kDivisor /
                           cuvs::spatial::knn::detail::utils::config<float>::kDivisor;

  // first scale the dataset and then compute norms
  auto scaled_sq_op = raft::compose_op(
    raft::sq_op{}, raft::div_const_op<float>{float(kScale)}, raft::cast_op<float>());
  raft::linalg::reduce<raft::Apply::ALONG_ROWS>(
    res,
    raft::make_device_matrix_view<const T, int64_t, raft::row_major>(
      dataset_view.data_handle(), dataset_view.extent(0), dataset_view.stride(0)),
    dataset_norms_->view(),
    (float)0,
    false,
    scaled_sq_op,
    raft::add_op(),
    raft::sqrt_op{});
}

/**
 * @defgroup cagra CUDA ANN Graph-based nearest neighbor search
 * @{
 */

/**
 * @brief Build a kNN graph using IVF-PQ.
 *
 * The kNN graph is the first building block for CAGRA index.
 *
 * The output is a dense matrix that stores the neighbor indices for each point in the dataset.
 * Each point has the same number of neighbors.
 *
 * See [cagra::build](#cagra::build) for an alternative method.
 *
 * The following distance metrics are supported:
 * - L2Expanded
 * - InnerProduct
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // use default index parameters based on shape of the dataset
 *   ivf_pq::index_params build_params = ivf_pq::index_params::from_dataset(dataset);
 *   ivf_pq::search_params search_params;
 *   auto knn_graph      = raft::make_host_matrix<IdxT, IdxT>(dataset.extent(0), 128);
 *   // create knn graph
 *   cagra::build_knn_graph(res, dataset, knn_graph.view(), 2, build_params, search_params);
 *   auto optimized_gaph = raft::make_host_matrix<IdxT, IdxT>(dataset.extent(0), 64);
 *   cagra::optimize(res, dataset, knn_graph.view(), optimized_graph.view());
 *   // Construct an index from dataset and optimized knn_graph
 *   auto index = cagra::index<T, IdxT>(res, build_params.metric(), dataset,
 *                                      optimized_graph.view());
 * @endcode
 *
 * @tparam DataT data element type
 * @tparam IdxT type of the dataset vector indices
 *
 * @param[in] res raft resources
 * @param[in] dataset a matrix view (host or device) to a row-major matrix [n_rows, dim]
 * @param[out] knn_graph a host matrix view to store the output knn graph [n_rows, graph_degree]
 * @param[in] ivf_pq_params ivf-pq parameters for graph build
 */
template <typename DataT, typename IdxT, typename accessor>
void build_knn_graph(
  raft::resources const& res,
  raft::mdspan<const DataT, raft::matrix_extent<int64_t>, raft::row_major, accessor> dataset,
  raft::host_matrix_view<IdxT, int64_t, raft::row_major> knn_graph,
  cagra::graph_build_params::ivf_pq_params ivf_pq_params)
{
  using internal_IdxT = typename std::make_unsigned<IdxT>::type;

  auto knn_graph_internal = raft::make_host_matrix_view<internal_IdxT, int64_t>(
    reinterpret_cast<internal_IdxT*>(knn_graph.data_handle()),
    knn_graph.extent(0),
    knn_graph.extent(1));
  auto dataset_internal =
    raft::mdspan<const DataT, raft::matrix_extent<int64_t>, raft::row_major, accessor>(
      dataset.data_handle(), dataset.extent(0), dataset.extent(1));

  cagra::detail::build_knn_graph(res, dataset_internal, knn_graph_internal, ivf_pq_params);
}

/**
 * @brief Build a kNN graph using NN-descent.
 *
 * The kNN graph is the first building block for CAGRA index.
 *
 * The output is a dense matrix that stores the neighbor indices for each point in the dataset.
 * Each point has the same number of neighbors.
 *
 * See [cagra::build](#cagra::build) for an alternative method.
 *
 * The following distance metrics are supported:
 * - L2Expanded
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   using namespace cuvs::neighbors::experimental;
 *   // use default index parameters
 *   nn_descent::index_params build_params;
 *   build_params.graph_degree = 128;
 *   auto knn_graph      = raft::make_host_matrix<IdxT, IdxT>(dataset.extent(0), 128);
 *   // create knn graph
 *   cagra::build_knn_graph(res, dataset, knn_graph.view(), build_params);
 *   auto optimized_gaph      = raft::make_host_matrix<IdxT, int64_t>(dataset.extent(0), 64);
 *   cagra::optimize(res, dataset, nn_descent_index.graph.view(), optimized_graph.view());
 *   // Construct an index from dataset and optimized knn_graph
 *   auto index = cagra::index<T, IdxT>(res, build_params.metric(), dataset,
 * optimized_graph.view());
 * @endcode
 *
 * @tparam DataT data element type
 * @tparam IdxT type of the dataset vector indices
 * @tparam accessor host or device accessor_type for the dataset
 * @param[in] res raft::resources is an object managing resources
 * @param[in] dataset input raft::host/device_matrix_view that can be located in
 *                in host or device memory
 * @param[out] knn_graph a host matrix view to store the output knn graph [n_rows, graph_degree]
 * @param[in] build_params an instance of experimental::nn_descent::index_params that are parameters
 *                     to run the nn-descent algorithm
 */
template <typename DataT,
          typename IdxT     = uint32_t,
          typename accessor = raft::host_device_accessor<cuda::std::default_accessor<DataT>,
                                                         raft::memory_type::device>>
void build_knn_graph(
  raft::resources const& res,
  raft::mdspan<const DataT, raft::matrix_extent<int64_t>, raft::row_major, accessor> dataset,
  raft::host_matrix_view<IdxT, int64_t, raft::row_major> knn_graph,
  cuvs::neighbors::nn_descent::index_params build_params)
{
  detail::build_knn_graph<DataT, IdxT>(res, dataset, knn_graph, build_params);
}

/**
 * @brief Sort a KNN graph index.
 * Preprocessing step for `cagra::optimize`: If a KNN graph is not built using
 * `cagra::build_knn_graph`, then it is necessary to call this function before calling
 * `cagra::optimize`. If the graph is built by `cagra::build_knn_graph`, it is already sorted and
 * you do not need to call this function.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   cagra::index_params build_params;
 *   auto knn_graph = raft::make_host_matrix<IdxT, IdxT>(dataset.extent(0), 128);
 *   // build KNN graph not using `cagra::build_knn_graph`
 *   // build(knn_graph, dataset, ...);
 *   // sort graph index
 *   sort_knn_graph(res, build_params.metric, dataset.view(), knn_graph.view());
 *   // optimize graph
 *   cagra::optimize(res, dataset, knn_graph.view(), optimized_graph.view());
 *   // Construct an index from dataset and optimized knn_graph
 *   auto index = cagra::index<T, IdxT>(res, build_params.metric(), dataset,
 *                                      optimized_graph.view());
 * @endcode
 *
 * @tparam DataT type of the data in the source dataset
 * @tparam IdxT type of the dataset vector indices
 *
 * @param[in] res raft resources
 * @param[in] metric metric
 * @param[in] dataset a matrix view (host or device) to a row-major matrix [n_rows, dim]
 * @param[in,out] knn_graph a matrix view (host or device) of the input knn graph [n_rows,
 * knn_graph_degree]
 */
template <typename DataT,
          typename IdxT       = uint32_t,
          typename d_accessor = raft::host_device_accessor<cuda::std::default_accessor<DataT>,
                                                           raft::memory_type::device>,
          typename g_accessor =
            raft::host_device_accessor<cuda::std::default_accessor<IdxT>, raft::memory_type::host>>
void sort_knn_graph(
  raft::resources const& res,
  cuvs::distance::DistanceType metric,
  raft::mdspan<const DataT, raft::matrix_extent<int64_t>, raft::row_major, d_accessor> dataset,
  raft::mdspan<IdxT, raft::matrix_extent<int64_t>, raft::row_major, g_accessor> knn_graph)
{
  using internal_IdxT = typename std::make_unsigned<IdxT>::type;

  using g_accessor_internal =
    raft::host_device_accessor<cuda::std::default_accessor<internal_IdxT>, g_accessor::mem_type>;
  auto knn_graph_internal =
    raft::mdspan<internal_IdxT, raft::matrix_extent<int64_t>, raft::row_major, g_accessor_internal>(
      reinterpret_cast<internal_IdxT*>(knn_graph.data_handle()),
      knn_graph.extent(0),
      knn_graph.extent(1));

  auto dataset_internal =
    raft::mdspan<const DataT, raft::matrix_extent<int64_t>, raft::row_major, d_accessor>(
      dataset.data_handle(), dataset.extent(0), dataset.extent(1));

  cagra::detail::graph::sort_knn_graph(res, metric, dataset_internal, knn_graph_internal);
}

/**
 * @brief Prune a KNN graph.
 *
 * Decrease the number of neighbors for each node.
 *
 * See [cagra::build_knn_graph](#cagra::build_knn_graph) for usage example
 *
 * @tparam IdxT type of the indices in the source dataset
 *
 * @param[in] res raft resources
 * @param[in] knn_graph a matrix view (host or device) of the input knn graph [n_rows,
 * knn_graph_degree]
 * @param[out] new_graph a host matrix view of the optimized knn graph [n_rows, graph_degree]
 */
template <typename IdxT = uint32_t,
          typename g_accessor =
            raft::host_device_accessor<cuda::std::default_accessor<IdxT>, raft::memory_type::host>>
void optimize(
  raft::resources const& res,
  raft::mdspan<IdxT, raft::matrix_extent<int64_t>, raft::row_major, g_accessor> knn_graph,
  raft::host_matrix_view<IdxT, int64_t, raft::row_major> new_graph,
  const bool guarantee_connectivity = false)
{
  detail::optimize(res, knn_graph, new_graph, guarantee_connectivity);
}

template <typename T,
          typename IdxT = uint32_t,
          typename Accessor =
            raft::host_device_accessor<cuda::std::default_accessor<T>, raft::memory_type::host>>
index<T, IdxT> build(
  raft::resources const& res,
  const index_params& params,
  raft::mdspan<const T, raft::matrix_extent<int64_t>, raft::row_major, Accessor> dataset)
{
  // Check if ACE dispatch is requested via graph_build_params
  if (std::holds_alternative<graph_build_params::ace_params>(params.graph_build_params)) {
    // ACE expects the dataset to be on host due to the large dataset size
    RAFT_EXPECTS(raft::get_device_for_address(dataset.data_handle()) == -1,
                 "ACE: Dataset must be on host for ACE build");
    auto dataset_view = raft::make_host_matrix_view<const T, int64_t, row_major>(
      dataset.data_handle(), dataset.extent(0), dataset.extent(1));
    return cuvs::neighbors::cagra::detail::build_ace<T, IdxT>(res, params, dataset_view);
  }
  return cuvs::neighbors::cagra::detail::build<T, IdxT, Accessor>(res, params, dataset);
}

/**
 * @brief Search ANN using the constructed index with the given sample filter.
 *
 * Usage example:
 * @code{.cpp}
 *   using namespace cuvs::neighbors;
 *   // use default index parameters
 *   cagra::index_params index_params;
 *   // create and fill the index from a [N, D] dataset
 *   auto index = cagra::build(res, index_params, dataset);
 *   // use default search parameters
 *   cagra::search_params search_params;
 *   // create a bitset to filter the search
 *   auto removed_indices = raft::make_device_vector<IdxT>(res, n_removed_indices);
 *   raft::core::bitset<std::uint32_t, IdxT> removed_indices_bitset(
 *     res, removed_indices.view(), dataset.extent(0));
 *   // search K nearest neighbours according to a bitset
 *   auto neighbors = raft::make_device_matrix<uint32_t>(res, n_queries, k);
 *   auto distances = raft::make_device_matrix<float>(res, n_queries, k);
 *   cagra::search_with_filtering(res, search_params, index, queries, neighbors, distances,
 *     filtering::bitset_filter(removed_indices_bitset.view()));
 * @endcode
 *
 * @tparam T data element type
 * @tparam IdxT type of the indices in the CAGRA graph
 * @tparam CagraSampleFilterT Device filter function, with the signature
 *         `(uint32_t query ix, uint32_t sample_ix) -> bool`
 * @tparam OutputIdxT type of the returned indices
 *
 * @param[in] res raft resources
 * @param[in] params configure the search
 * @param[in] idx cagra index
 * @param[in] queries a device matrix view to a row-major matrix [n_queries, index->dim()]
 * @param[out] neighbors a device matrix view to the indices of the neighbors in the source dataset
 * [n_queries, k]
 * @param[out] distances a device matrix view to the distances to the selected neighbors [n_queries,
 * k]
 * @param[in] sample_filter a device filter function that greenlights samples for a given query
 */
template <typename T, typename IdxT, typename CagraSampleFilterT, typename OutputIdxT = IdxT>
void search_with_filtering(raft::resources const& res,
                           const search_params& params,
                           const index<T, IdxT>& idx,
                           raft::device_matrix_view<const T, int64_t, raft::row_major> queries,
                           raft::device_matrix_view<OutputIdxT, int64_t, raft::row_major> neighbors,
                           raft::device_matrix_view<float, int64_t, raft::row_major> distances,
                           CagraSampleFilterT sample_filter = CagraSampleFilterT())
{
  RAFT_EXPECTS(
    queries.extent(0) == neighbors.extent(0) && queries.extent(0) == distances.extent(0),
    "Number of rows in output neighbors and distances matrices must equal the number of queries.");

  RAFT_EXPECTS(neighbors.extent(1) == distances.extent(1),
               "Number of columns in output neighbors and distances matrices must equal k");
  RAFT_EXPECTS(queries.extent(1) == idx.dim(),
               "Number of query dimensions should equal number of dimensions in the index.");

  return cagra::detail::search_main<T, OutputIdxT, CagraSampleFilterT, IdxT>(
    res, params, idx, queries, neighbors, distances, sample_filter);
}

template <typename T, typename IdxT, typename OutputIdxT = IdxT>
void search(raft::resources const& res,
            const search_params& params,
            const index<T, IdxT>& idx,
            raft::device_matrix_view<const T, int64_t, raft::row_major> queries,
            raft::device_matrix_view<OutputIdxT, int64_t, raft::row_major> neighbors,
            raft::device_matrix_view<float, int64_t, raft::row_major> distances,
            const cuvs::neighbors::filtering::base_filter& sample_filter_ref)
{
  try {
    using none_filter_type    = cuvs::neighbors::filtering::none_sample_filter;
    auto& sample_filter       = dynamic_cast<const none_filter_type&>(sample_filter_ref);
    search_params params_copy = params;
    RAFT_EXPECTS(params.filter_mode != filtering_mode::FAVOR,
                 "FAVOR filtering requires a bitset filter");
    if (params.filtering_rate < 0.0) { params_copy.filtering_rate = 0.0; }
    auto sample_filter_copy = sample_filter;
    return search_with_filtering<T, IdxT, none_filter_type, OutputIdxT>(
      res, params_copy, idx, queries, neighbors, distances, sample_filter_copy);
  } catch (const std::bad_cast&) {
  }

  try {
    auto& sample_filter =
      dynamic_cast<const cuvs::neighbors::filtering::bitset_filter<uint32_t, int64_t>&>(
        sample_filter_ref);
    search_params params_copy = params;
    if (params.filter_mode == filtering_mode::FAVOR) {
      if (params.filtering_rate < 0.0f) {
        const auto num_set_bits = detail::count_favor_bitset_matches(res, idx, sample_filter);
        if (num_set_bits == 0) {
          detail::fill_empty_favor_results(res, neighbors, distances);
          return;
        }
        const auto filtering_rate = static_cast<float>(idx.data().n_rows() - num_set_bits) /
                                    static_cast<float>(idx.data().n_rows());
        // Extremely sparse filters on indices larger than float's exact-integer range can round
        // to 1.0 even though at least one row passes. Preserve the nearest valid rate in that case.
        params_copy.filtering_rate = std::min(filtering_rate, std::nextafter(1.0f, 0.0f));
      }
    } else if (params.filtering_rate < 0.0) {
      const auto num_set_bits = sample_filter.bitset_view_.count(res);
      auto filtering_rate     = (float)(idx.data().n_rows() - num_set_bits) / idx.data().n_rows();
      const float min_filtering_rate = 0.0;
      const float max_filtering_rate = 0.999;
      params_copy.filtering_rate =
        std::min(std::max(filtering_rate, min_filtering_rate), max_filtering_rate);
    }
    auto sample_filter_copy = sample_filter;
    return search_with_filtering<T, IdxT, decltype(sample_filter_copy), OutputIdxT>(
      res, params_copy, idx, queries, neighbors, distances, sample_filter_copy);
  } catch (const std::bad_cast&) {
  }

  try {
    auto& sample_filter =
      dynamic_cast<const cuvs::neighbors::filtering::udf_filter&>(sample_filter_ref);
    search_params params_copy = params;
    if (params.filter_mode == filtering_mode::FAVOR) {
      RAFT_EXPECTS(params.algo == search_algo::AUTO || params.algo == search_algo::SINGLE_CTA,
                   "FAVOR UDF filtering currently supports only SINGLE_CTA search");
      RAFT_EXPECTS(!params.persistent, "FAVOR UDF filtering does not support persistent search");
      RAFT_EXPECTS(params.favor_penalty == favor_penalty_mode::CAGRA_RETENTION_SAFE,
                   "FAVOR UDF filtering requires CAGRA_RETENTION_SAFE scoring");
      RAFT_EXPECTS(params.favor_retention_fraction == 0.0f,
                   "FAVOR UDF filtering currently requires automatic retention");

      // Phase-one FAVOR UDF policy is sampling-only.  In particular, YFCC never uploads or uses
      // precomputed exact selectivity, and caller-provided scalar hints cannot silently change the
      // experiment. AUTO is resolved explicitly to the sole supported traversal here.
      params_copy.algo = search_algo::SINGLE_CTA;
      auto estimate    = detail::estimate_favor_udf_filtering_rates(
        res, idx, static_cast<std::uint32_t>(queries.extent(0)), sample_filter);
      // The search plan still requires a valid scalar; each CTA replaces it with its private
      // sampled query rate before deriving the penalty coefficient and retention fraction.
      params_copy.filtering_rate = 0.0f;
      auto runtime_filter =
        detail::CagraSampleFilterWithRuntimeState<cuvs::neighbors::filtering::udf_filter, true>{
          sample_filter, estimate.filtering_rates.data()};
      return search_with_filtering<T, IdxT, decltype(runtime_filter), OutputIdxT>(
        res, params_copy, idx, queries, neighbors, distances, runtime_filter);
    }

    if (params.filtering_rate < 0.0) {
      const float min_filtering_rate = 0.0f;
      const float max_filtering_rate = 0.999f;
      params_copy.filtering_rate =
        sample_filter.filtering_rate < 0.0f
          ? 0.0f
          : std::min(std::max(sample_filter.filtering_rate, min_filtering_rate),
                     max_filtering_rate);
    }
    auto sample_filter_copy = sample_filter;
    return search_with_filtering<T, IdxT, decltype(sample_filter_copy), OutputIdxT>(
      res, params_copy, idx, queries, neighbors, distances, sample_filter_copy);
  } catch (const std::bad_cast&) {
    RAFT_FAIL("Unsupported sample filter type");
  }
}

template <class T, class IdxT, class Accessor>
void extend(
  raft::resources const& handle,
  raft::mdspan<const T, raft::matrix_extent<int64_t>, raft::row_major, Accessor> additional_dataset,
  cuvs::neighbors::cagra::index<T, IdxT>& index,
  const cagra::extend_params& params,
  std::optional<raft::device_matrix_view<T, int64_t, raft::layout_stride>> ndv,
  std::optional<raft::device_matrix_view<IdxT, int64_t>> ngv)
{
  cagra::extend_core<T, IdxT, Accessor>(handle, additional_dataset, index, params, ndv, ngv);
}

template <class T, class IdxT>
index<T, IdxT> merge(raft::resources const& handle,
                     const cagra::index_params& params,
                     std::vector<cuvs::neighbors::cagra::index<T, IdxT>*>& indices,
                     const cuvs::neighbors::filtering::base_filter& row_filter)
{
  return cagra::detail::merge<T, IdxT>(handle, params, indices, row_filter);
}

/** @} */  // end group cagra

}  // namespace cuvs::neighbors::cagra

#define CUVS_INST_CAGRA_MERGE(T, IdxT)                                                  \
  auto merge(raft::resources const& handle,                                             \
             const cuvs::neighbors::cagra::index_params& params,                        \
             std::vector<cuvs::neighbors::cagra::index<T, IdxT>*>& indices,             \
             const cuvs::neighbors::filtering::base_filter& row_filter)                 \
    -> cuvs::neighbors::cagra::index<T, IdxT>                                           \
  {                                                                                     \
    return cuvs::neighbors::cagra::merge<T, IdxT>(handle, params, indices, row_filter); \
  }
