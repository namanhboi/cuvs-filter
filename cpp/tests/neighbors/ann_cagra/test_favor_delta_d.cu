/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../../../src/neighbors/detail/cagra/favor_penalty.cuh"
#include <cuvs/neighbors/cagra.hpp>

#include <raft/core/copy.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

namespace {

TEST(CagraFavorPenalty, ReferenceCoefficient)
{
  using cuvs::neighbors::cagra::detail::favor_penalty_coefficient;
  using cuvs::neighbors::cagra::detail::favor_reference_penalty;

  auto const reference   = favor_reference_penalty(0.9f, 64, 100.0f);
  auto const coefficient = favor_penalty_coefficient(0.9f, 64);
  EXPECT_FLOAT_EQ(reference, 0.9f * (64.0f - 0.1f) * 100.0f / (2.0f * 0.1f * 64.0f));
  EXPECT_FLOAT_EQ(coefficient, 0.9f * (64.0f - 0.1f) / (2.0f * 0.1f * 64.0f));
  EXPECT_FLOAT_EQ(favor_penalty_coefficient(0.0f, 64), 0.0f);
  EXPECT_FLOAT_EQ(favor_reference_penalty(0.9f, 64, 0.0f), 0.0f);
}

TEST(CagraFavorPenalty, AutomaticRetentionTracksExpectedPassingOccupancy)
{
  using cuvs::neighbors::cagra::detail::favor_automatic_retention_fraction;

  // At least k expected passing entries preserves the established midpoint.
  EXPECT_FLOAT_EQ(favor_automatic_retention_fraction(0.9f, 128, 10), 0.5f);
  // At most k/2 expected passing entries saturates at the strict-headroom ceiling.
  EXPECT_FLOAT_EQ(favor_automatic_retention_fraction(0.99f, 500, 10), 0.9f);
  EXPECT_FLOAT_EQ(favor_automatic_retention_fraction(0.99f, 256, 10), 0.9f);

  auto const near_half = favor_automatic_retention_fraction(0.99f, 512, 10);
  EXPECT_GT(near_half, 0.89f);
  EXPECT_LT(near_half, 0.9f);
  EXPECT_FLOAT_EQ(favor_automatic_retention_fraction(0.99f, 512, 0), 0.5f);
}

TEST(CagraFavorPenalty, SparseAdaptiveTerminationBudget)
{
  using cuvs::neighbors::cagra::detail::favor_adaptive_budget;
  using cuvs::neighbors::cagra::detail::favor_adaptive_prefix_size;

  EXPECT_EQ(favor_adaptive_prefix_size(10), 32);
  EXPECT_EQ(favor_adaptive_prefix_size(16), 64);

  auto const one_million = favor_adaptive_budget(0.99f, 512, 10, 1, 1'000'000, 32, 517);
  EXPECT_EQ(one_million.prefix_size, 32);
  EXPECT_EQ(one_million.start_iteration, 517);
  EXPECT_EQ(one_million.hard_iteration, 4005);

  auto const ten_million = favor_adaptive_budget(0.99f, 512, 10, 1, 10'000'000, 32, 518);
  EXPECT_EQ(ten_million.prefix_size, 32);
  EXPECT_EQ(ten_million.start_iteration, 518);
  EXPECT_EQ(ten_million.hard_iteration, 4006);

  auto const width_four = favor_adaptive_budget(0.99f, 512, 10, 4, 1'000'000, 32, 133);
  EXPECT_EQ(width_four.hard_iteration, 1005);

  // Ten percent and a prefix with no remaining frontier reserve are deliberately ineligible.
  EXPECT_EQ(favor_adaptive_budget(0.90f, 512, 10, 1, 1'000'000, 32, 517).prefix_size, 0);
  EXPECT_EQ(favor_adaptive_budget(0.99f, 32, 10, 1, 1'000'000, 32, 37).prefix_size, 0);
}

struct temporary_file {
  std::string name;
  temporary_file()
  {
    static std::atomic<uint64_t> sequence{};
    name = (std::filesystem::temp_directory_path() /
            ("cuvs_favor_delta_d_" + std::to_string(sequence++) + ".bin"))
             .string();
  }
  ~temporary_file() { std::filesystem::remove(name); }
};

template <typename T>
auto make_index(raft::resources const& res,
                std::vector<float> const& values,
                uint32_t dimension,
                std::vector<uint32_t> const& edges,
                uint32_t degree,
                cuvs::distance::DistanceType metric = cuvs::distance::DistanceType::L2Expanded)
  -> cuvs::neighbors::cagra::index<T, uint32_t>
{
  auto rows    = values.size() / dimension;
  auto dataset = raft::make_host_matrix<T, int64_t>(rows, dimension);
  auto graph   = raft::make_host_matrix<uint32_t, int64_t>(rows, degree);
  for (size_t i = 0; i < values.size(); ++i) {
    dataset.data_handle()[i] = static_cast<T>(values[i]);
  }
  std::copy(edges.begin(), edges.end(), graph.data_handle());
  return {
    res, metric, raft::make_const_mdspan(dataset.view()), raft::make_const_mdspan(graph.view())};
}

auto read_file(std::string const& name) -> std::array<unsigned char, 80>
{
  std::array<unsigned char, 80> bytes{};
  std::ifstream input(name, std::ios::binary);
  input.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
  EXPECT_TRUE(input);
  return bytes;
}

void write_file(std::string const& name, std::array<unsigned char, 80> const& bytes)
{
  std::ofstream output(name, std::ios::binary | std::ios::trunc);
  output.write(reinterpret_cast<char const*>(bytes.data()), bytes.size());
  ASSERT_TRUE(output);
}

uint64_t sidecar_checksum(std::array<unsigned char, 80> const& bytes)
{
  uint64_t hash = 1469598103934665603ull;
  for (size_t i = 0; i < 72; ++i) {
    hash ^= bytes[i];
    hash *= 1099511628211ull;
  }
  return hash;
}

template <typename Value>
void patch_sidecar(std::array<unsigned char, 80>* bytes, size_t offset, Value value, bool checksum)
{
  std::memcpy(bytes->data() + offset, &value, sizeof(value));
  if (checksum) {
    auto sum = sidecar_checksum(*bytes);
    std::memcpy(bytes->data() + 72, &sum, sizeof(sum));
  }
}

float cpu_delta(std::vector<float> const& data,
                std::vector<uint32_t> const& graph,
                uint32_t degree,
                uint32_t alpha,
                uint32_t beta,
                uint32_t depth,
                uint32_t dimension = 1)
{
  auto n       = static_cast<uint32_t>(data.size() / dimension);
  double total = 0.0;
  for (uint32_t root = 0; root < n; ++root) {
    std::vector<uint32_t> candidates;
    std::vector<uint32_t> frontier{root};
    for (uint32_t level = 0; level < depth; ++level) {
      std::vector<uint32_t> next;
      for (auto node : frontier) {
        for (uint32_t edge = 0; edge < degree; ++edge) {
          auto candidate = graph[node * degree + edge];
          if (candidate >= n || candidate == root ||
              std::find(candidates.begin(), candidates.end(), candidate) != candidates.end()) {
            continue;
          }
          candidates.push_back(candidate);
          next.push_back(candidate);
        }
      }
      frontier = std::move(next);
    }
    if (candidates.size() < beta) { continue; }
    std::vector<float> distances;
    for (auto candidate : candidates) {
      float distance = 0.0f;
      for (uint32_t d = 0; d < dimension; ++d) {
        auto difference = data[root * dimension + d] - data[candidate * dimension + d];
        distance += difference * difference;
      }
      distances.push_back(distance);
    }
    std::sort(distances.begin(), distances.end());
    total += 5.0 * (distances[beta - 1] - distances[alpha - 1]) / (beta - alpha);
  }
  return static_cast<float>(total / n);
}

TEST(CagraFavorDeltaD, MatchesDeterministicCpuBfs)
{
  raft::resources res;
  constexpr uint32_t n = 8, degree = 3;
  std::vector<float> values{0, 1, 2, 4, 7, 11, 16, 22};
  // Includes self edges, duplicates, a cycle, an invalid edge, and a disconnected final root.
  std::vector<uint32_t> edges{0, 1, 1, 2, 3, 0, 3, 4, 1, 4, 5, 2,
                              5, 6, 3, 6, 0, 4, 0, 1, 5, 7, 7, 99};
  auto dataset = raft::make_host_matrix<float, int64_t>(n, 1);
  auto graph   = raft::make_host_matrix<uint32_t, int64_t>(n, degree);
  std::copy(values.begin(), values.end(), dataset.data_handle());
  std::copy(edges.begin(), edges.end(), graph.data_handle());
  cuvs::neighbors::cagra::index<float, uint32_t> index(res,
                                                       cuvs::distance::DistanceType::L2Expanded,
                                                       raft::make_const_mdspan(dataset.view()),
                                                       raft::make_const_mdspan(graph.view()));

  for (uint32_t depth : {1u, 2u, 3u}) {
    auto expected = cpu_delta(values, edges, degree, 1, 3, depth);
    auto actual   = cuvs::neighbors::cagra::compute_favor_delta_d(res, {1, 3, depth}, index);
    EXPECT_FLOAT_EQ(actual, expected);
  }

  for (uint32_t dimension : {65u, 129u, 513u}) {
    std::vector<float> multi_dimensional(n * dimension);
    for (uint32_t row = 0; row < n; ++row) {
      for (uint32_t d = 0; d < dimension; ++d) {
        multi_dimensional[row * dimension + d] =
          static_cast<float>((row + 1) * ((d % 17) + 1)) / static_cast<float>(d + 3);
      }
    }
    auto multi_index = make_index<float>(res, multi_dimensional, dimension, edges, degree);
    auto expected    = cpu_delta(multi_dimensional, edges, degree, 1, 3, 2, dimension);
    auto actual      = cuvs::neighbors::cagra::compute_favor_delta_d(res, {1, 3, 2}, multi_index);
    EXPECT_NEAR(actual, expected, std::max(1e-5f, std::abs(expected) * 1e-5f));
  }
}

TEST(CagraFavorDeltaD, ValidatesPublicParametersAndAttachments)
{
  raft::resources res;
  cuvs::neighbors::cagra::index<float, uint32_t> empty(res);
  EXPECT_ANY_THROW(cuvs::neighbors::cagra::compute_favor_delta_d(res, {0, 2, 1}, empty));
  EXPECT_ANY_THROW(cuvs::neighbors::cagra::compute_favor_delta_d(res, {2, 2, 1}, empty));
  EXPECT_ANY_THROW(cuvs::neighbors::cagra::compute_favor_delta_d(res, {1, 1025, 1}, empty));
  EXPECT_ANY_THROW(cuvs::neighbors::cagra::compute_favor_delta_d(res, {1, 2, 0}, empty));
  EXPECT_ANY_THROW(cuvs::neighbors::cagra::compute_favor_delta_d(res, {1, 2, 1}, empty));
}

template <typename T>
class CagraFavorDeltaDSidecarTyped : public ::testing::Test {};
using SidecarTypes = ::testing::Types<float, half, int8_t, uint8_t>;
TYPED_TEST_SUITE(CagraFavorDeltaDSidecarTyped, SidecarTypes);

TYPED_TEST(CagraFavorDeltaDSidecarTyped, RoundTripsExactScalarAndParameterCombinations)
{
  raft::resources res;
  std::vector<float> values{0, 1, 2, 3, 4, 5};
  std::vector<uint32_t> edges{1, 2, 2, 3, 3, 4, 4, 5, 5, 0, 0, 1};
  auto index = make_index<TypeParam>(res, values, 1, edges, 2);
  for (auto params : {cuvs::neighbors::cagra::favor_delta_d_params{1, 2, 2},
                      cuvs::neighbors::cagra::favor_delta_d_params{2, 3, 3}}) {
    temporary_file file;
    float value = std::nextafter(3.25f, 4.0f);
    cuvs::neighbors::cagra::save_favor_delta_d(res, file.name, params, index, value);
    auto loaded = cuvs::neighbors::cagra::load_favor_delta_d(res, file.name, params, index);
    EXPECT_EQ(std::bit_cast<uint32_t>(loaded), std::bit_cast<uint32_t>(value));
    EXPECT_EQ(index.size(), 6);
    EXPECT_EQ(index.dim(), 1);
    EXPECT_EQ(index.graph_degree(), 2);
  }
}

TEST(CagraFavorDeltaDSidecar, RejectsContentAndMetadataMismatches)
{
  raft::resources res;
  std::vector<float> values{0, 1, 2, 3, 4, 5};
  std::vector<uint32_t> edges{1, 2, 2, 3, 3, 4, 4, 5, 5, 0, 0, 1};
  auto index = make_index<float>(res, values, 1, edges, 2);
  temporary_file file;
  cuvs::neighbors::cagra::favor_delta_d_params params{1, 2, 2};
  cuvs::neighbors::cagra::save_favor_delta_d(res, file.name, params, index, 2.5f);

  auto changed_values = values;
  changed_values[3] += 1;
  auto changed_dataset = make_index<float>(res, changed_values, 1, edges, 2);
  EXPECT_THROW(cuvs::neighbors::cagra::load_favor_delta_d(res, file.name, params, changed_dataset),
               std::runtime_error);

  auto reordered_values = values;
  std::swap(reordered_values[0], reordered_values[1]);
  auto reordered_dataset = make_index<float>(res, reordered_values, 1, edges, 2);
  EXPECT_THROW(
    cuvs::neighbors::cagra::load_favor_delta_d(res, file.name, params, reordered_dataset),
    std::runtime_error);

  auto changed_edges = edges;
  changed_edges[3]   = 0;
  auto changed_graph = make_index<float>(res, values, 1, changed_edges, 2);
  EXPECT_THROW(cuvs::neighbors::cagra::load_favor_delta_d(res, file.name, params, changed_graph),
               std::runtime_error);

  auto wrong_metric =
    make_index<float>(res, values, 1, edges, 2, cuvs::distance::DistanceType::InnerProduct);
  EXPECT_THROW(cuvs::neighbors::cagra::load_favor_delta_d(res, file.name, params, wrong_metric),
               std::runtime_error);

  std::vector<float> two_dimensional{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
  auto wrong_dimension = make_index<float>(res, two_dimensional, 2, edges, 2);
  EXPECT_THROW(cuvs::neighbors::cagra::load_favor_delta_d(res, file.name, params, wrong_dimension),
               std::runtime_error);

  std::vector<uint32_t> degree_three{1, 2, 3, 2, 3, 4, 3, 4, 5, 4, 5, 0, 5, 0, 1, 0, 1, 2};
  auto wrong_degree = make_index<float>(res, values, 1, degree_three, 3);
  EXPECT_THROW(cuvs::neighbors::cagra::load_favor_delta_d(res, file.name, params, wrong_degree),
               std::runtime_error);

  std::vector<float> shorter_values{0, 1, 2, 3, 4};
  std::vector<uint32_t> shorter_edges{1, 2, 2, 3, 3, 4, 4, 0, 0, 1};
  auto wrong_size = make_index<float>(res, shorter_values, 1, shorter_edges, 2);
  EXPECT_THROW(cuvs::neighbors::cagra::load_favor_delta_d(res, file.name, params, wrong_size),
               std::runtime_error);

  auto wrong_type = make_index<uint8_t>(res, values, 1, edges, 2);
  EXPECT_THROW(cuvs::neighbors::cagra::load_favor_delta_d(res, file.name, params, wrong_type),
               std::runtime_error);
  EXPECT_THROW(cuvs::neighbors::cagra::load_favor_delta_d(res, file.name, {1, 3, 2}, index),
               std::runtime_error);
  EXPECT_THROW(cuvs::neighbors::cagra::load_favor_delta_d(res, file.name, {1, 2, 3}, index),
               std::runtime_error);
}

TEST(CagraFavorDeltaDSidecar, IgnoresDenseDatasetRowPadding)
{
  raft::resources res;
  constexpr int64_t rows = 4, dimension = 2, stride = 4;
  std::vector<uint32_t> edges{1, 2, 2, 3, 3, 0, 0, 1};
  std::vector<float> values{0, 1, 2, 3, 4, 5, 6, 7};
  auto first  = make_index<float>(res, values, dimension, edges, 2);
  auto second = make_index<float>(res, values, dimension, edges, 2);

  auto first_storage  = raft::make_device_vector<float, int64_t>(res, rows * stride);
  auto second_storage = raft::make_device_vector<float, int64_t>(res, rows * stride);
  std::vector<float> first_host(rows * stride), second_host(rows * stride);
  for (int64_t row = 0; row < rows; ++row) {
    first_host[row * stride] = second_host[row * stride] = values[row * dimension];
    first_host[row * stride + 1] = second_host[row * stride + 1] = values[row * dimension + 1];
    first_host[row * stride + 2] = first_host[row * stride + 3] = 11.0f + row;
    second_host[row * stride + 2] = second_host[row * stride + 3] = 91.0f + row;
  }
  auto stream = raft::resource::get_cuda_stream(res);
  raft::copy(first_storage.data_handle(), first_host.data(), first_host.size(), stream);
  raft::copy(second_storage.data_handle(), second_host.data(), second_host.size(), stream);
  first.update_dataset(res,
                       raft::make_device_strided_matrix_view<const float, int64_t>(
                         first_storage.data_handle(), rows, dimension, stride));
  second.update_dataset(res,
                        raft::make_device_strided_matrix_view<const float, int64_t>(
                          second_storage.data_handle(), rows, dimension, stride));

  temporary_file file;
  cuvs::neighbors::cagra::favor_delta_d_params params{1, 2, 2};
  cuvs::neighbors::cagra::save_favor_delta_d(res, file.name, params, first, 4.5f);
  EXPECT_FLOAT_EQ(cuvs::neighbors::cagra::load_favor_delta_d(res, file.name, params, second), 4.5f);
}

TEST(CagraFavorDeltaDSidecar, RejectsMalformedFilesAndNonFiniteValues)
{
  raft::resources res;
  std::vector<float> values{0, 1, 2, 3, 4, 5};
  std::vector<uint32_t> edges{1, 2, 2, 3, 3, 4, 4, 5, 5, 0, 0, 1};
  auto index = make_index<float>(res, values, 1, edges, 2);
  cuvs::neighbors::cagra::favor_delta_d_params params{1, 2, 2};
  temporary_file valid;
  cuvs::neighbors::cagra::save_favor_delta_d(res, valid.name, params, index, 2.5f);
  auto original = read_file(valid.name);

  temporary_file missing;
  EXPECT_THROW(cuvs::neighbors::cagra::load_favor_delta_d(res, missing.name, params, index),
               std::runtime_error);
  for (size_t length : {0u, 1u, 79u}) {
    temporary_file truncated;
    std::ofstream output(truncated.name, std::ios::binary);
    output.write(reinterpret_cast<char const*>(original.data()), length);
    output.close();
    EXPECT_THROW(cuvs::neighbors::cagra::load_favor_delta_d(res, truncated.name, params, index),
                 std::runtime_error);
  }
  {
    temporary_file extended;
    std::ofstream output(extended.name, std::ios::binary);
    output.write(reinterpret_cast<char const*>(original.data()), original.size());
    output.put('x');
    output.close();
    EXPECT_THROW(cuvs::neighbors::cagra::load_favor_delta_d(res, extended.name, params, index),
                 std::runtime_error);
  }
  for (auto mutation : {std::pair<size_t, uint32_t>{8, 2}, {12, 2}}) {
    temporary_file unsupported;
    auto bytes = original;
    patch_sidecar(&bytes, mutation.first, mutation.second, false);
    write_file(unsupported.name, bytes);
    EXPECT_THROW(cuvs::neighbors::cagra::load_favor_delta_d(res, unsupported.name, params, index),
                 std::runtime_error);
  }
  {
    temporary_file bad_magic;
    auto bytes = original;
    bytes[0] ^= 1;
    write_file(bad_magic.name, bytes);
    EXPECT_THROW(cuvs::neighbors::cagra::load_favor_delta_d(res, bad_magic.name, params, index),
                 std::runtime_error);
  }
  {
    temporary_file corrupt;
    auto bytes = original;
    bytes[52] ^= 1;
    write_file(corrupt.name, bytes);
    EXPECT_THROW(cuvs::neighbors::cagra::load_favor_delta_d(res, corrupt.name, params, index),
                 std::runtime_error);
  }
  for (float invalid : {std::numeric_limits<float>::infinity(),
                        -std::numeric_limits<float>::infinity(),
                        std::numeric_limits<float>::quiet_NaN()}) {
    EXPECT_ANY_THROW(
      cuvs::neighbors::cagra::save_favor_delta_d(res, valid.name, params, index, invalid));
    temporary_file non_finite;
    auto bytes = original;
    patch_sidecar(&bytes, 52, invalid, true);
    write_file(non_finite.name, bytes);
    EXPECT_THROW(cuvs::neighbors::cagra::load_favor_delta_d(res, non_finite.name, params, index),
                 std::runtime_error);
  }
}

TEST(CagraFavorDeltaDSidecar, RequiresAttachedDenseDatasetAndGraph)
{
  raft::resources res;
  cuvs::neighbors::cagra::index<float, uint32_t> empty(res);
  temporary_file file;
  EXPECT_ANY_THROW(cuvs::neighbors::cagra::save_favor_delta_d(res, file.name, {1, 2, 1}, empty, 1));

  std::vector<float> values{0, 1, 2};
  std::vector<uint32_t> edges{1, 2, 0};
  auto complete = make_index<float>(res, values, 1, edges, 1);
  cuvs::neighbors::cagra::save_favor_delta_d(res, file.name, {1, 2, 1}, complete, 1);
  EXPECT_ANY_THROW(cuvs::neighbors::cagra::load_favor_delta_d(res, file.name, {1, 2, 1}, empty));

  auto vq      = raft::make_device_matrix<float, uint32_t>(res, 1, 1);
  auto pq      = raft::make_device_matrix<float, uint32_t>(res, 2, 1);
  auto encoded = raft::make_device_matrix<uint8_t, int64_t>(res, 3, 1);
  complete.update_dataset(
    res,
    cuvs::neighbors::vpq_dataset<float, int64_t>{std::move(vq), std::move(pq), std::move(encoded)});
  EXPECT_ANY_THROW(
    cuvs::neighbors::cagra::save_favor_delta_d(res, file.name, {1, 2, 1}, complete, 1));
  EXPECT_ANY_THROW(cuvs::neighbors::cagra::load_favor_delta_d(res, file.name, {1, 2, 1}, complete));

  auto disk_index = make_index<float>(res, values, 1, edges, 1);
  temporary_file disk_dataset;
  auto [dataset_fd, header_size] =
    cuvs::util::create_numpy_file<float>(disk_dataset.name, {values.size(), 1});
  (void)header_size;
  disk_index.update_dataset(res, std::move(dataset_fd));
  EXPECT_ANY_THROW(
    cuvs::neighbors::cagra::save_favor_delta_d(res, file.name, {1, 2, 1}, disk_index, 1));
  EXPECT_ANY_THROW(
    cuvs::neighbors::cagra::load_favor_delta_d(res, file.name, {1, 2, 1}, disk_index));
}

}  // namespace
