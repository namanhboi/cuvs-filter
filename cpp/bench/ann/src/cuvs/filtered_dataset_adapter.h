/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "../common/conf.hpp"

#include <cuvs/neighbors/common.hpp>

#include <raft/core/copy.cuh>
#include <raft/core/error.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_uvector.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace cuvs::bench::detail {

/** Runtime state is deliberately per benchmark algo copy so concurrent query offsets cannot race.
 */
class udf_filter_runtime {
 public:
  virtual ~udf_filter_runtime()                                                       = default;
  virtual void set_query_offset(std::uint32_t query_offset)                           = 0;
  [[nodiscard]] virtual auto filter() const -> cuvs::neighbors::filtering::udf_filter = 0;
};

/** Generic benchmark-side contract for query-dependent filtered datasets. */
class udf_filter_adapter {
 public:
  virtual ~udf_filter_adapter()                                                   = default;
  [[nodiscard]] virtual auto name() const -> const char*                          = 0;
  [[nodiscard]] virtual auto base_rows() const -> std::uint32_t                   = 0;
  [[nodiscard]] virtual auto query_rows() const -> std::uint32_t                  = 0;
  [[nodiscard]] virtual auto arity(std::uint32_t query_id) const -> std::uint32_t = 0;
  [[nodiscard]] virtual auto passes(std::uint32_t query_id, std::uint32_t candidate_id) const
    -> bool = 0;
  [[nodiscard]] virtual auto make_runtime(raft::resources const& res) const
    -> std::shared_ptr<udf_filter_runtime> = 0;
};

/** Host representation of the BigANN sparse-matrix file, without its all-one value array. */
struct spmat_csr {
  std::uint32_t rows{};
  std::uint32_t cols{};
  std::uint64_t duplicate_entries{};
  std::vector<std::uint32_t> offsets;
  std::vector<std::uint32_t> columns;

  explicit spmat_csr(std::string const& path)
  {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) { throw std::runtime_error("Cannot open sparse metadata: " + path); }
    const auto file_size = input.tellg();
    input.seekg(0);

    std::int64_t header[3]{};
    input.read(reinterpret_cast<char*>(header), sizeof(header));
    if (!input || header[0] <= 0 || header[1] <= 0 || header[2] < 0 ||
        static_cast<std::uint64_t>(header[0]) > std::numeric_limits<std::uint32_t>::max() ||
        static_cast<std::uint64_t>(header[1]) > std::numeric_limits<std::uint32_t>::max() ||
        static_cast<std::uint64_t>(header[2]) > std::numeric_limits<std::uint32_t>::max()) {
      throw std::runtime_error("Invalid sparse metadata header: " + path);
    }
    rows           = static_cast<std::uint32_t>(header[0]);
    cols           = static_cast<std::uint32_t>(header[1]);
    const auto nnz = static_cast<std::uint64_t>(header[2]);

    constexpr auto header_bytes = std::uint64_t{3 * sizeof(std::int64_t)};
    const auto pointer_bytes    = (static_cast<std::uint64_t>(rows) + 1) * sizeof(std::int64_t);
    const auto index_bytes      = nnz * sizeof(std::int32_t);
    const auto value_bytes      = nnz * sizeof(float);
    const auto expected_size    = header_bytes + pointer_bytes + index_bytes + value_bytes;
    if (file_size < 0 || static_cast<std::uint64_t>(file_size) != expected_size) {
      throw std::runtime_error("Sparse metadata size does not match its header: " + path);
    }

    offsets.resize(static_cast<std::size_t>(rows) + 1);
    constexpr std::size_t chunk_size = 1u << 20;
    std::vector<std::int64_t> signed_chunk(chunk_size);
    for (std::size_t first = 0; first < offsets.size(); first += chunk_size) {
      const auto count = std::min(chunk_size, offsets.size() - first);
      input.read(reinterpret_cast<char*>(signed_chunk.data()),
                 static_cast<std::streamsize>(count * sizeof(std::int64_t)));
      if (!input) { throw std::runtime_error("Truncated sparse row pointers: " + path); }
      for (std::size_t i = 0; i < count; ++i) {
        const auto value = signed_chunk[i];
        if (value < 0 || static_cast<std::uint64_t>(value) > nnz) {
          throw std::runtime_error("Sparse row pointer is out of range: " + path);
        }
        offsets[first + i] = static_cast<std::uint32_t>(value);
      }
    }
    if (offsets.front() != 0 || offsets.back() != nnz ||
        !std::is_sorted(offsets.begin(), offsets.end())) {
      throw std::runtime_error("Sparse row pointers are not monotonic: " + path);
    }

    columns.resize(static_cast<std::size_t>(nnz));
    std::vector<std::int32_t> column_chunk(chunk_size);
    for (std::size_t first = 0; first < columns.size(); first += chunk_size) {
      const auto count = std::min(chunk_size, columns.size() - first);
      input.read(reinterpret_cast<char*>(column_chunk.data()),
                 static_cast<std::streamsize>(count * sizeof(std::int32_t)));
      if (!input) { throw std::runtime_error("Truncated sparse column indices: " + path); }
      for (std::size_t i = 0; i < count; ++i) {
        const auto value = column_chunk[i];
        if (value < 0 || static_cast<std::uint32_t>(value) >= cols) {
          throw std::runtime_error("Sparse column index is out of range: " + path);
        }
        columns[first + i] = static_cast<std::uint32_t>(value);
      }
    }

    // BigANN stores an explicit float value for every nonzero.  The contains-all adapter accepts
    // only the documented unit-valued representation but does not retain those redundant values.
    std::vector<float> value_chunk(chunk_size);
    for (std::size_t first = 0; first < columns.size(); first += chunk_size) {
      const auto count = std::min(chunk_size, columns.size() - first);
      input.read(reinterpret_cast<char*>(value_chunk.data()),
                 static_cast<std::streamsize>(count * sizeof(float)));
      if (!input) { throw std::runtime_error("Truncated sparse values: " + path); }
      if (!std::all_of(
            value_chunk.begin(), value_chunk.begin() + count, [](float x) { return x == 1.0f; })) {
        throw std::runtime_error("Sparse contains-all metadata must contain unit values: " + path);
      }
    }

    std::vector<std::uint32_t> last_seen(cols, std::numeric_limits<std::uint32_t>::max());
    for (std::uint32_t row = 0; row < rows; ++row) {
      for (auto pos = offsets[row]; pos < offsets[row + 1]; ++pos) {
        const auto column = columns[pos];
        duplicate_entries += last_seen[column] == row;
        last_seen[column] = row;
      }
    }
  }
};

struct spmat_contains_all_device_context {
  const std::uint32_t* base_offsets{};
  const std::uint32_t* base_columns{};
  const std::uint32_t* query_offsets{};
  const std::uint32_t* query_columns{};
  std::uint32_t base_rows{};
  std::uint32_t query_rows{};
  std::uint32_t query_offset{};
};

struct spmat_contains_all_storage {
  spmat_contains_all_storage(raft::resources const& res,
                             std::string const& base_path,
                             std::string const& query_path)
    : base(base_path),
      queries(query_path),
      base_offsets(base.offsets.size(), raft::resource::get_cuda_stream(res)),
      base_columns(base.columns.size(), raft::resource::get_cuda_stream(res)),
      query_offsets(queries.offsets.size(), raft::resource::get_cuda_stream(res)),
      query_columns(queries.columns.size(), raft::resource::get_cuda_stream(res))
  {
    if (base.cols != queries.cols) {
      throw std::runtime_error("Base/query sparse metadata column counts differ");
    }
    auto stream = raft::resource::get_cuda_stream(res);
    raft::copy(base_offsets.data(), base.offsets.data(), base.offsets.size(), stream);
    raft::copy(base_columns.data(), base.columns.data(), base.columns.size(), stream);
    raft::copy(query_offsets.data(), queries.offsets.data(), queries.offsets.size(), stream);
    raft::copy(query_columns.data(), queries.columns.data(), queries.columns.size(), stream);
    raft::resource::sync_stream(res);
  }

  spmat_csr base;
  spmat_csr queries;
  rmm::device_uvector<std::uint32_t> base_offsets;
  rmm::device_uvector<std::uint32_t> base_columns;
  rmm::device_uvector<std::uint32_t> query_offsets;
  rmm::device_uvector<std::uint32_t> query_columns;
};

inline auto spmat_contains_all_udf_source() -> std::string
{
  return R"cpp(
    struct spmat_contains_all_device_context {
      const uint32_t* base_offsets;
      const uint32_t* base_columns;
      const uint32_t* query_offsets;
      const uint32_t* query_columns;
      uint32_t base_rows;
      uint32_t query_rows;
      uint32_t query_offset;
    };

    __device__ bool cuvs_spmat_contains_all(uint32_t local_query_id,
                                            source_index_t source_id,
                                            void* filter_data)
    {
      const auto* ctx         = static_cast<const spmat_contains_all_device_context*>(filter_data);
      const uint32_t query_id = ctx->query_offset + local_query_id;
      if (source_id >= ctx->base_rows || query_id >= ctx->query_rows) { return false; }
      const uint32_t query_begin = ctx->query_offsets[query_id];
      const uint32_t query_end   = ctx->query_offsets[query_id + 1];
      const uint32_t base_begin  = ctx->base_offsets[source_id];
      const uint32_t base_end    = ctx->base_offsets[source_id + 1];
      for (uint32_t query_pos = query_begin; query_pos < query_end; ++query_pos) {
        bool found               = false;
        const uint32_t query_tag = ctx->query_columns[query_pos];
        for (uint32_t base_pos = base_begin; base_pos < base_end; ++base_pos) {
          if (ctx->base_columns[base_pos] == query_tag) {
            found = true;
            break;
          }
        }
        if (!found) { return false; }
      }
      return true;
    }
  )cpp";
}

class spmat_contains_all_runtime final : public udf_filter_runtime {
 public:
  spmat_contains_all_runtime(raft::resources const& res,
                             std::shared_ptr<const spmat_contains_all_storage> storage)
    : storage_(std::move(storage)),
      stream_(raft::resource::get_cuda_stream(res)),
      context_(1, stream_)
  {
    set_query_offset(0);
  }

  void set_query_offset(std::uint32_t query_offset) override
  {
    RAFT_EXPECTS(query_offset < storage_->queries.rows,
                 "Filtered-dataset query offset is out of range");
    const spmat_contains_all_device_context context{storage_->base_offsets.data(),
                                                    storage_->base_columns.data(),
                                                    storage_->query_offsets.data(),
                                                    storage_->query_columns.data(),
                                                    storage_->base.rows,
                                                    storage_->queries.rows,
                                                    query_offset};
    RAFT_CUDA_TRY(
      cudaMemcpyAsync(context_.data(), &context, sizeof(context), cudaMemcpyHostToDevice, stream_));
  }

  [[nodiscard]] auto filter() const -> cuvs::neighbors::filtering::udf_filter override
  {
    return cuvs::neighbors::filtering::udf_filter{
      spmat_contains_all_udf_source(),
      const_cast<spmat_contains_all_device_context*>(context_.data()),
      -1.0f,
      "cuvs_spmat_contains_all"};
  }

 private:
  std::shared_ptr<const spmat_contains_all_storage> storage_;
  rmm::cuda_stream_view stream_;
  rmm::device_uvector<spmat_contains_all_device_context> context_;
};

class spmat_contains_all_adapter final : public udf_filter_adapter {
 public:
  spmat_contains_all_adapter(raft::resources const& res,
                             configuration::dataset_conf::udf_filter_conf const& conf)
    : storage_(std::make_shared<spmat_contains_all_storage>(
        res, conf.base_metadata_file, conf.query_metadata_file))
  {
  }

  [[nodiscard]] auto name() const -> const char* override { return "spmat_contains_all"; }
  [[nodiscard]] auto base_rows() const -> std::uint32_t override { return storage_->base.rows; }
  [[nodiscard]] auto query_rows() const -> std::uint32_t override { return storage_->queries.rows; }
  [[nodiscard]] auto arity(std::uint32_t query_id) const -> std::uint32_t override
  {
    if (query_id >= query_rows()) { throw std::out_of_range("Filter query id is out of range"); }
    return storage_->queries.offsets[query_id + 1] - storage_->queries.offsets[query_id];
  }
  [[nodiscard]] auto passes(std::uint32_t query_id, std::uint32_t candidate_id) const
    -> bool override
  {
    if (query_id >= query_rows() || candidate_id >= base_rows()) { return false; }
    const auto& queries = storage_->queries;
    const auto& base    = storage_->base;
    for (auto q = queries.offsets[query_id]; q < queries.offsets[query_id + 1]; ++q) {
      bool found = false;
      for (auto b = base.offsets[candidate_id]; b < base.offsets[candidate_id + 1]; ++b) {
        if (base.columns[b] == queries.columns[q]) {
          found = true;
          break;
        }
      }
      if (!found) { return false; }
    }
    return true;
  }
  [[nodiscard]] auto make_runtime(raft::resources const& res) const
    -> std::shared_ptr<udf_filter_runtime> override
  {
    return std::make_shared<spmat_contains_all_runtime>(res, storage_);
  }

 private:
  std::shared_ptr<spmat_contains_all_storage> storage_;
};

inline auto make_udf_filter_adapter(raft::resources const& res,
                                    configuration::dataset_conf::udf_filter_conf const& conf)
  -> std::shared_ptr<udf_filter_adapter>
{
  if (conf.adapter == "spmat_contains_all") {
    return std::make_shared<spmat_contains_all_adapter>(res, conf);
  }
  throw std::runtime_error("Unsupported filtered-dataset adapter: " + conf.adapter);
}

}  // namespace cuvs::bench::detail
