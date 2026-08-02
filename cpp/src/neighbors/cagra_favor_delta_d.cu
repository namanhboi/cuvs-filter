/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuvs/core/cuda_fp16.hpp>
#include <cuvs/neighbors/cagra.hpp>

#include <raft/core/copy.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/util/cudart_utils.hpp>
#include <rmm/device_uvector.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace cuvs::neighbors::cagra::detail {

namespace {

constexpr std::array<char, 8> sidecar_magic{'C', 'U', 'V', 'S', 'D', 'D', '\r', '\n'};
constexpr uint32_t sidecar_file_version      = 1;
constexpr uint32_t sidecar_algorithm_version = 1;

enum class sidecar_element_type : uint32_t { f32 = 1, f16 = 2, i8 = 3, u8 = 4 };

struct favor_delta_d_sidecar {
  char magic[8];
  uint32_t file_version;
  uint32_t algorithm_version;
  uint32_t element_type;
  uint32_t metric;
  uint64_t index_size;
  uint32_t dimension;
  uint32_t graph_degree;
  uint32_t alpha;
  uint32_t beta;
  uint32_t bfs_depth;
  float delta_d;
  uint64_t graph_fingerprint;
  uint64_t dataset_fingerprint;
  uint64_t header_checksum;
};

static_assert(std::is_trivially_copyable_v<favor_delta_d_sidecar>);
static_assert(sizeof(favor_delta_d_sidecar) == 80);
static_assert(offsetof(favor_delta_d_sidecar, file_version) == 8);
static_assert(offsetof(favor_delta_d_sidecar, index_size) == 24);
static_assert(offsetof(favor_delta_d_sidecar, dimension) == 32);
static_assert(offsetof(favor_delta_d_sidecar, delta_d) == 52);
static_assert(offsetof(favor_delta_d_sidecar, graph_fingerprint) == 56);
static_assert(offsetof(favor_delta_d_sidecar, header_checksum) == 72);

template <typename T>
constexpr sidecar_element_type element_type_code();
template <>
constexpr sidecar_element_type element_type_code<float>()
{
  return sidecar_element_type::f32;
}
template <>
constexpr sidecar_element_type element_type_code<half>()
{
  return sidecar_element_type::f16;
}
template <>
constexpr sidecar_element_type element_type_code<int8_t>()
{
  return sidecar_element_type::i8;
}
template <>
constexpr sidecar_element_type element_type_code<uint8_t>()
{
  return sidecar_element_type::u8;
}

uint64_t checksum_bytes(void const* data, size_t bytes)
{
  auto p        = static_cast<unsigned char const*>(data);
  uint64_t hash = 1469598103934665603ull;
  for (size_t i = 0; i < bytes; ++i) {
    hash ^= p[i];
    hash *= 1099511628211ull;
  }
  return hash;
}

__host__ __device__ uint64_t mix_fingerprint(uint64_t value)
{
  value += 0x9e3779b97f4a7c15ull;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ull;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebull;
  return value ^ (value >> 31);
}

template <typename T>
__global__ void fingerprint_strided_kernel(
  const T* data, uint64_t rows, uint64_t columns, uint64_t stride, uint64_t* fingerprint)
{
  auto count = rows * columns;
  for (uint64_t position = blockIdx.x * blockDim.x + threadIdx.x; position < count;
       position += static_cast<uint64_t>(gridDim.x) * blockDim.x) {
    auto row       = position / columns;
    auto col       = position - row * columns;
    auto bytes     = reinterpret_cast<unsigned char const*>(data + row * stride + col);
    uint64_t value = mix_fingerprint(position ^ (sizeof(T) * 0x517cc1b727220a95ull));
    for (uint32_t byte = 0; byte < sizeof(T); ++byte) {
      value = mix_fingerprint(value ^ (static_cast<uint64_t>(bytes[byte]) << ((byte & 7) * 8)));
    }
    atomicXor(reinterpret_cast<unsigned long long*>(fingerprint),
              static_cast<unsigned long long>(value));
  }
}

template <typename T>
uint64_t device_fingerprint(raft::resources const& res,
                            T const* data,
                            uint64_t rows,
                            uint64_t columns,
                            uint64_t stride,
                            uint64_t seed)
{
  auto stream = raft::resource::get_cuda_stream(res);
  rmm::device_uvector<uint64_t> result(1, stream);
  RAFT_CUDA_TRY(cudaMemsetAsync(result.data(), 0, sizeof(uint64_t), stream));
  auto count  = rows * columns;
  auto blocks = static_cast<unsigned>(std::min<uint64_t>((count + 255) / 256, 1024));
  if (blocks != 0) {
    fingerprint_strided_kernel<<<blocks, 256, 0, stream>>>(
      data, rows, columns, stride, result.data());
    RAFT_CUDA_TRY(cudaPeekAtLastError());
  }
  uint64_t host_result{};
  raft::copy(&host_result, result.data(), 1, stream);
  raft::resource::sync_stream(res);
  return host_result ^ seed ^ mix_fingerprint(rows) ^ mix_fingerprint(columns);
}

template <typename T>
void validate_persistence_index(cagra::index<T, uint32_t> const& index)
{
  RAFT_EXPECTS(!index.dataset_fd().has_value() && !index.graph_fd().has_value(),
               "FAVOR delta-d sidecar requires an in-memory index, not a disk-backed index");
  auto dataset = index.dataset();
  auto graph   = index.graph();
  RAFT_EXPECTS(dataset.data_handle() != nullptr && dataset.extent(0) > 0 && dataset.extent(1) > 0,
               "FAVOR delta-d sidecar requires an attached dense device dataset");
  RAFT_EXPECTS(graph.data_handle() != nullptr && graph.extent(0) > 0 && graph.extent(1) > 0,
               "FAVOR delta-d sidecar requires an attached dense device graph");
  RAFT_EXPECTS(dataset.extent(0) == graph.extent(0),
               "FAVOR delta-d sidecar dataset and graph row counts must match");
}

void validate_persistence_params(favor_delta_d_params const& params, uint64_t index_size)
{
  RAFT_EXPECTS(params.alpha >= 1 && params.alpha < params.beta && params.beta <= 1024,
               "FAVOR delta-d sidecar requires 1 <= alpha < beta <= 1024");
  RAFT_EXPECTS(params.bfs_depth >= 1, "FAVOR delta-d sidecar requires bfs_depth >= 1");
  RAFT_EXPECTS(params.beta < index_size, "FAVOR delta-d sidecar requires beta < index size");
}

template <typename T>
auto make_sidecar(raft::resources const& res,
                  favor_delta_d_params const& params,
                  cagra::index<T, uint32_t> const& index,
                  float delta_d) -> favor_delta_d_sidecar
{
  validate_persistence_index(index);
  validate_persistence_params(params, index.size());
  RAFT_EXPECTS(std::isfinite(delta_d), "FAVOR delta-d sidecar value must be finite");
  auto dataset = index.dataset();
  auto graph   = index.graph();
  favor_delta_d_sidecar sidecar{};
  std::memcpy(sidecar.magic, sidecar_magic.data(), sidecar_magic.size());
  sidecar.file_version        = sidecar_file_version;
  sidecar.algorithm_version   = sidecar_algorithm_version;
  sidecar.element_type        = static_cast<uint32_t>(element_type_code<T>());
  sidecar.metric              = static_cast<uint32_t>(index.metric());
  sidecar.index_size          = index.size();
  sidecar.dimension           = index.dim();
  sidecar.graph_degree        = index.graph_degree();
  sidecar.alpha               = params.alpha;
  sidecar.beta                = params.beta;
  sidecar.bfs_depth           = params.bfs_depth;
  sidecar.delta_d             = delta_d;
  sidecar.graph_fingerprint   = device_fingerprint(res,
                                                 graph.data_handle(),
                                                 graph.extent(0),
                                                 graph.extent(1),
                                                 graph.extent(1),
                                                 0x67726170682d6464ull);
  sidecar.dataset_fingerprint = device_fingerprint(res,
                                                   dataset.data_handle(),
                                                   dataset.extent(0),
                                                   dataset.extent(1),
                                                   dataset.stride(0),
                                                   0x646174617365742dull);
  sidecar.header_checksum =
    checksum_bytes(&sidecar, offsetof(favor_delta_d_sidecar, header_checksum));
  return sidecar;
}

template <typename T>
void save_favor_delta_d_impl(raft::resources const& res,
                             std::string const& filename,
                             favor_delta_d_params const& params,
                             cagra::index<T, uint32_t> const& index,
                             float delta_d)
{
  auto sidecar = make_sidecar(res, params, index, delta_d);
  std::ofstream output(filename, std::ios::binary | std::ios::trunc);
  if (!output) {
    throw std::runtime_error("cannot open FAVOR delta-d sidecar for writing: " + filename);
  }
  output.write(reinterpret_cast<char const*>(&sidecar), sizeof(sidecar));
  output.flush();
  if (!output) { throw std::runtime_error("failed to write FAVOR delta-d sidecar: " + filename); }
}

template <typename T>
float load_favor_delta_d_impl(raft::resources const& res,
                              std::string const& filename,
                              favor_delta_d_params const& expected_params,
                              cagra::index<T, uint32_t> const& index)
{
  std::ifstream input(filename, std::ios::binary | std::ios::ate);
  if (!input) { throw std::runtime_error("cannot open FAVOR delta-d sidecar: " + filename); }
  auto length = input.tellg();
  if (length < static_cast<std::streamoff>(sizeof(favor_delta_d_sidecar))) {
    throw std::runtime_error("truncated FAVOR delta-d sidecar: " + filename);
  }
  if (length > static_cast<std::streamoff>(sizeof(favor_delta_d_sidecar))) {
    throw std::runtime_error("FAVOR delta-d sidecar has trailing data: " + filename);
  }
  input.seekg(0);
  favor_delta_d_sidecar stored{};
  input.read(reinterpret_cast<char*>(&stored), sizeof(stored));
  if (!input) { throw std::runtime_error("truncated FAVOR delta-d sidecar: " + filename); }
  if (std::memcmp(stored.magic, sidecar_magic.data(), sidecar_magic.size()) != 0) {
    throw std::runtime_error("invalid FAVOR delta-d sidecar magic: " + filename);
  }
  if (stored.file_version != sidecar_file_version) {
    throw std::runtime_error("unsupported FAVOR delta-d sidecar file version: " + filename);
  }
  if (stored.algorithm_version != sidecar_algorithm_version) {
    throw std::runtime_error("unsupported FAVOR delta-d calculation algorithm version: " +
                             filename);
  }
  auto checksum = checksum_bytes(&stored, offsetof(favor_delta_d_sidecar, header_checksum));
  if (checksum != stored.header_checksum) {
    throw std::runtime_error("corrupt FAVOR delta-d sidecar header: " + filename);
  }
  if (!std::isfinite(stored.delta_d)) {
    throw std::runtime_error("non-finite FAVOR delta-d sidecar value: " + filename);
  }
  if (stored.element_type != static_cast<uint32_t>(element_type_code<T>())) {
    throw std::runtime_error("FAVOR delta-d sidecar element type mismatch: " + filename);
  }
  validate_persistence_index(index);
  validate_persistence_params(expected_params, index.size());
  if (stored.alpha != expected_params.alpha || stored.beta != expected_params.beta ||
      stored.bfs_depth != expected_params.bfs_depth) {
    throw std::runtime_error("FAVOR delta-d sidecar parameter mismatch: " + filename);
  }
  if (stored.metric != static_cast<uint32_t>(index.metric())) {
    throw std::runtime_error("FAVOR delta-d sidecar metric mismatch: " + filename);
  }
  if (stored.index_size != index.size()) {
    throw std::runtime_error("FAVOR delta-d sidecar index size mismatch: " + filename);
  }
  if (stored.dimension != index.dim()) {
    throw std::runtime_error("FAVOR delta-d sidecar dimension mismatch: " + filename);
  }
  if (stored.graph_degree != index.graph_degree()) {
    throw std::runtime_error("FAVOR delta-d sidecar graph degree mismatch: " + filename);
  }
  auto actual = make_sidecar(res, expected_params, index, stored.delta_d);
  if (stored.graph_fingerprint != actual.graph_fingerprint) {
    throw std::runtime_error("FAVOR delta-d sidecar graph fingerprint mismatch: " + filename);
  }
  if (stored.dataset_fingerprint != actual.dataset_fingerprint) {
    throw std::runtime_error("FAVOR delta-d sidecar dataset fingerprint mismatch: " + filename);
  }
  return stored.delta_d;
}

}  // namespace

template <typename T>
__device__ float as_float(T x)
{
  return static_cast<float>(x);
}

template <typename T>
__global__ void favor_delta_d_kernel(const T* dataset,
                                     int64_t stride,
                                     uint32_t n_rows,
                                     uint32_t dim,
                                     const uint32_t* graph,
                                     uint32_t degree,
                                     uint32_t alpha,
                                     uint32_t beta,
                                     uint32_t bfs_depth,
                                     uint32_t capacity,
                                     uint32_t seen_capacity,
                                     uint32_t team_size_bits,
                                     cuvs::distance::DistanceType metric,
                                     float* roots)
{
  auto root = blockIdx.x;
  if (root >= n_rows) { return; }

  extern __shared__ unsigned char storage[];
  auto nodes     = reinterpret_cast<uint32_t*>(storage);
  auto distances = reinterpret_cast<float*>(storage);
  auto seen      = nodes + capacity;
  __shared__ uint32_t count;

  constexpr auto empty = std::numeric_limits<uint32_t>::max();
  if (threadIdx.x == 0) { count = 0; }
  for (uint32_t i = threadIdx.x; i < seen_capacity; i += blockDim.x) {
    seen[i] = empty;
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    uint32_t frontier_begin = 0;
    uint32_t frontier_end   = 1;

    // The root is an implicit queue entry. Subsequent frontier entries live in nodes[].
    for (uint32_t depth = 0; depth < bfs_depth; ++depth) {
      auto level_end = frontier_end;
      for (uint32_t q = frontier_begin; q < level_end; ++q) {
        auto src = q == 0 ? root : nodes[q - 1];
        for (uint32_t edge = 0; edge < degree; ++edge) {
          auto candidate = graph[static_cast<uint64_t>(src) * degree + edge];
          if (candidate >= n_rows || candidate == root) { continue; }
          if (count == capacity) { continue; }

          auto slot = (candidate * 0x9e3779b9u) & (seen_capacity - 1);
          while (seen[slot] != empty && seen[slot] != candidate) {
            slot = (slot + 1) & (seen_capacity - 1);
          }
          if (seen[slot] == candidate) { continue; }
          seen[slot]     = candidate;
          nodes[count++] = candidate;
        }
      }
      frontier_begin = level_end;
      frontier_end   = count + 1;
    }
  }
  __syncthreads();

  if (count < beta) {
    if (threadIdx.x == 0) { roots[root] = 0.0f; }
    return;
  }

  // CAGRA stages the query in shared memory once and reuses it across all candidate teams. The
  // current root is the query for this delta-d block; the BFS hash storage is dead after this point
  // and can be reused for the float query vector.
  auto root_values = reinterpret_cast<float*>(seen);
  for (uint32_t d = threadIdx.x; d < dim; d += blockDim.x) {
    root_values[d] = as_float(dataset[static_cast<uint64_t>(root) * stride + d]);
  }
  __syncthreads();

  // Match CAGRA's standard distance-comparison layout: one power-of-two thread team per
  // candidate, contiguous dimensions across lanes, lane-local accumulation, then an XOR team
  // reduction. CAGRA uses teams 8/16/32 for dimensions <=128/<=256/>256 respectively.
  auto const team_size       = 1u << team_size_bits;
  auto const lane_id         = threadIdx.x & (team_size - 1);
  auto const team_id         = threadIdx.x >> team_size_bits;
  auto const teams_per_block = blockDim.x >> team_size_bits;
  auto const max_i           = ((count + teams_per_block - 1) / teams_per_block) * teams_per_block;
  for (uint32_t i = team_id; i < max_i; i += teams_per_block) {
    auto const valid     = i < count;
    auto const candidate = valid ? nodes[i] : uint32_t{0};
    float value          = 0.0f;
    float root_norm      = 0.0f;
    float candidate_norm = 0.0f;
    if (valid) {
      for (uint32_t d = lane_id; d < dim; d += team_size) {
        auto const x = root_values[d];
        auto const y = as_float(dataset[static_cast<uint64_t>(candidate) * stride + d]);
        if (metric == cuvs::distance::DistanceType::L2Expanded) {
          auto const diff = x - y;
          value += diff * diff;
        } else if (metric == cuvs::distance::DistanceType::L1) {
          value += fabsf(x - y);
        } else {
          value -= x * y;
          if (metric == cuvs::distance::DistanceType::CosineExpanded) {
            root_norm += x * x;
            candidate_norm += y * y;
          }
        }
      }
    }
    for (uint32_t reduction_stride = team_size >> 1; reduction_stride > 0; reduction_stride >>= 1) {
      value += __shfl_xor_sync(0xffffffffu, value, reduction_stride, team_size);
      root_norm += __shfl_xor_sync(0xffffffffu, root_norm, reduction_stride, team_size);
      candidate_norm += __shfl_xor_sync(0xffffffffu, candidate_norm, reduction_stride, team_size);
    }
    if (valid && lane_id == 0) {
      if (metric == cuvs::distance::DistanceType::CosineExpanded) {
        auto const denom = sqrtf(root_norm * candidate_norm);
        value            = denom > 0.0f ? value / denom : 0.0f;
      }
      distances[i] = value;
    }
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    // Only the first beta order statistics are needed. Select them from the complete BFS
    // neighborhood without sorting the (potentially much larger) remainder.
    for (uint32_t i = 0; i < beta; ++i) {
      uint32_t minimum = i;
      for (uint32_t j = i + 1; j < count; ++j) {
        if (distances[j] < distances[minimum]) { minimum = j; }
      }
      auto value         = distances[i];
      distances[i]       = distances[minimum];
      distances[minimum] = value;
    }
    roots[root] =
      5.0f * (distances[beta - 1] - distances[alpha - 1]) / static_cast<float>(beta - alpha);
  }
}

template <typename T>
float compute_favor_delta_d_impl(raft::resources const& res,
                                 favor_delta_d_params const& params,
                                 cagra::index<T, uint32_t> const& index)
{
  RAFT_EXPECTS(params.alpha >= 1 && params.alpha < params.beta && params.beta <= 1024,
               "FAVOR delta-d requires 1 <= alpha < beta <= 1024");
  RAFT_EXPECTS(params.bfs_depth >= 1, "FAVOR delta-d requires bfs_depth >= 1");
  RAFT_EXPECTS(params.beta < index.size(), "FAVOR delta-d requires beta < index size");
  RAFT_EXPECTS(!index.dataset_fd().has_value() && !index.graph_fd().has_value(),
               "FAVOR delta-d does not support disk-backed indices");

  auto dataset = index.dataset();
  auto graph   = index.graph();
  RAFT_EXPECTS(dataset.data_handle() != nullptr && dataset.extent(0) > 0 && dataset.extent(1) > 0,
               "FAVOR delta-d requires an attached dense device dataset");
  RAFT_EXPECTS(graph.data_handle() != nullptr && graph.extent(0) > 0 && graph.extent(1) > 0,
               "FAVOR delta-d requires an attached dense device graph");
  RAFT_EXPECTS(dataset.extent(0) == graph.extent(0),
               "FAVOR delta-d dataset and graph row counts must match");
  RAFT_EXPECTS(index.metric() == cuvs::distance::DistanceType::L2Expanded ||
                 index.metric() == cuvs::distance::DistanceType::L1 ||
                 index.metric() == cuvs::distance::DistanceType::InnerProduct ||
                 index.metric() == cuvs::distance::DistanceType::CosineExpanded,
               "FAVOR delta-d does not support this CAGRA metric");

  auto stream = raft::resource::get_cuda_stream(res);
  rmm::device_uvector<float> roots(dataset.extent(0), stream);
  uint64_t capacity = 0;
  uint64_t frontier = 1;
  for (uint32_t depth = 0; depth < params.bfs_depth && capacity < index.size() - 1; ++depth) {
    frontier = std::min<uint64_t>(frontier * graph.extent(1), index.size() - 1);
    capacity = std::min<uint64_t>(capacity + frontier, index.size() - 1);
  }
  RAFT_EXPECTS(capacity >= params.beta, "BFS depth and graph degree cannot yield beta candidates");
  RAFT_EXPECTS(capacity <= std::numeric_limits<uint32_t>::max() / 2,
               "Complete BFS neighborhood is too large for FAVOR delta-d");
  uint64_t seen_capacity = 1;
  while (seen_capacity < 2 * capacity) {
    seen_capacity *= 2;
  }
  constexpr uint64_t block_size = 256;
  auto const reusable_bytes =
    std::max<uint64_t>(seen_capacity * sizeof(uint32_t), dataset.extent(1) * sizeof(float));
  auto const shared_bytes   = capacity * sizeof(uint32_t) + reusable_bytes;
  auto const team_size_bits = dataset.extent(1) <= 128 ? 3u : dataset.extent(1) <= 256 ? 4u : 5u;
  int device{};
  int max_shared_bytes{};
  RAFT_CUDA_TRY(cudaGetDevice(&device));
  RAFT_CUDA_TRY(
    cudaDeviceGetAttribute(&max_shared_bytes, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
  RAFT_EXPECTS(shared_bytes <= static_cast<size_t>(max_shared_bytes),
               "Complete BFS neighborhood requires %zu shared-memory bytes, device supports %d",
               shared_bytes,
               max_shared_bytes);
  if (shared_bytes > 48 * 1024) {
    RAFT_CUDA_TRY(cudaFuncSetAttribute(favor_delta_d_kernel<T>,
                                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                                       static_cast<int>(shared_bytes)));
  }
  favor_delta_d_kernel<<<dataset.extent(0), block_size, shared_bytes, stream>>>(
    dataset.data_handle(),
    dataset.stride(0),
    dataset.extent(0),
    dataset.extent(1),
    graph.data_handle(),
    graph.extent(1),
    params.alpha,
    params.beta,
    params.bfs_depth,
    static_cast<uint32_t>(capacity),
    static_cast<uint32_t>(seen_capacity),
    team_size_bits,
    index.metric(),
    roots.data());
  RAFT_CUDA_TRY(cudaPeekAtLastError());

  // A host reduction avoids introducing another dependency into this diagnostic API.
  std::vector<float> host_roots(roots.size());
  raft::copy(host_roots.data(), roots.data(), roots.size(), stream);
  raft::resource::sync_stream(res);
  double sum = 0.0;
  for (auto value : host_roots) {
    sum += value;
  }
  return static_cast<float>(sum / static_cast<double>(host_roots.size()));
}

}  // namespace cuvs::neighbors::cagra::detail

namespace cuvs::neighbors::cagra {

#define CUVS_INST_FAVOR_DELTA_D(T)                                                 \
  float compute_favor_delta_d(raft::resources const& res,                          \
                              favor_delta_d_params const& params,                  \
                              cagra::index<T, uint32_t> const& index)              \
  {                                                                                \
    return detail::compute_favor_delta_d_impl(res, params, index);                 \
  }                                                                                \
  void save_favor_delta_d(raft::resources const& res,                              \
                          std::string const& filename,                             \
                          favor_delta_d_params const& params,                      \
                          cagra::index<T, uint32_t> const& index,                  \
                          float delta_d)                                           \
  {                                                                                \
    detail::save_favor_delta_d_impl(res, filename, params, index, delta_d);        \
  }                                                                                \
  float load_favor_delta_d(raft::resources const& res,                             \
                           std::string const& filename,                            \
                           favor_delta_d_params const& expected_params,            \
                           cagra::index<T, uint32_t> const& index)                 \
  {                                                                                \
    return detail::load_favor_delta_d_impl(res, filename, expected_params, index); \
  }

CUVS_INST_FAVOR_DELTA_D(float)
CUVS_INST_FAVOR_DELTA_D(half)
CUVS_INST_FAVOR_DELTA_D(int8_t)
CUVS_INST_FAVOR_DELTA_D(uint8_t)

#undef CUVS_INST_FAVOR_DELTA_D

}  // namespace cuvs::neighbors::cagra
