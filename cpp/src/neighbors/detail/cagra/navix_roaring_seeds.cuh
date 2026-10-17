/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "../roaring_bitmap_filter_data.cuh"

#include <cuvs/neighbors/common.hpp>
#include <raft/core/error.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/util/cudart_utils.hpp>

#include <cstdint>

namespace cuvs::neighbors::cagra::detail {

/** Preserve the bitmap seeding policy: first passing nodes in internal graph-ID order. */
template <typename IndexT>
RAFT_KERNEL navix_select_roaring_seeds_kernel(
  cuvs::neighbors::detail::roaring_bitmap_filter_data_t<> const* filter,
  IndexT const* source_indices,
  std::uint32_t query_offset,
  std::uint32_t graph_size,
  std::uint32_t seed_stride,
  IndexT* seed_ids,
  std::uint32_t* seed_counts,
  std::uint32_t* inspected_units)
{
  constexpr unsigned warp_size = 32;
  auto const local_query       = static_cast<std::uint32_t>(blockIdx.x);
  auto const query             = static_cast<std::uint64_t>(query_offset) + local_query;
  auto const lane              = static_cast<unsigned>(threadIdx.x);
  std::uint32_t found          = 0;
  std::uint32_t units          = 0;
  auto* output                 = seed_ids + static_cast<std::uint64_t>(local_query) * seed_stride;

  if (query < filter->num_queries && filter->empty_rows[query] == 0) {
    bool enumerated = false;
    if (source_indices == nullptr) {
      // cuVS from_ids emits portable no-run containers. Enumerate their sorted prefix instead
      // of testing every absent ID. Keep the membership fallback for other encodings and remaps.
      auto const* bytes  = reinterpret_cast<std::uint8_t const*>(filter->refs[query]->data());
      auto const* header = reinterpret_cast<std::uint32_t const*>(bytes);
      if (header[0] == 12346u) {
        enumerated            = true;
        auto const containers = header[1];
        auto const* key_cards = reinterpret_cast<std::uint16_t const*>(bytes + 8);
        auto const* offsets = reinterpret_cast<std::uint32_t const*>(bytes + 8 + 4ull * containers);
        for (std::uint32_t c = 0; c < containers && found < seed_stride; ++c) {
          auto const base = std::uint32_t{key_cards[2 * c]} << 16;
          if (base >= graph_size) { break; }
          auto const cardinality = std::uint32_t{key_cards[2 * c + 1]} + 1;
          auto const* payload    = reinterpret_cast<std::uint16_t const*>(bytes + offsets[c]);
          if (cardinality <= 4096) {
            for (std::uint32_t start = 0; start < cardinality && found < seed_stride;
                 start += warp_size) {
              auto const position = start + lane;
              auto const id       = position < cardinality ? base + payload[position] : graph_size;
              auto const mask     = __ballot_sync(0xffffffffu, id < graph_size);
              auto const rank     = __popc(mask & ((std::uint32_t{1} << lane) - 1));
              if (id < graph_size && found + rank < seed_stride) {
                output[found + rank] = static_cast<IndexT>(id);
              }
              found += min(static_cast<std::uint32_t>(__popc(mask)), seed_stride - found);
              ++units;
            }
          } else {
            // Container payloads are only 16-bit aligned after odd-length array containers.
            // Use two aligned halfword loads rather than an unaligned uint32/uint64 load.
            for (std::uint32_t start = 0; start < 2048 && found < seed_stride; start += warp_size) {
              auto const word_id = start + lane;
              auto const first   = base + word_id * 32;
              auto word          = std::uint32_t{payload[2 * word_id]} |
                          (std::uint32_t{payload[2 * word_id + 1]} << 16);
              if (first >= graph_size) {
                word = 0;
              } else if (graph_size - first < 32) {
                word &= (std::uint32_t{1} << (graph_size - first)) - 1;
              }
              auto inclusive = static_cast<std::uint32_t>(__popc(word));
              for (unsigned delta = 1; delta < warp_size; delta *= 2) {
                auto const prior = __shfl_up_sync(0xffffffffu, inclusive, delta);
                if (lane >= delta) { inclusive += prior; }
              }
              auto const before = inclusive - static_cast<std::uint32_t>(__popc(word));
              auto remaining    = word;
              auto position     = found + before;
              while (remaining != 0 && position < seed_stride) {
                output[position++] = static_cast<IndexT>(first + __ffs(remaining) - 1);
                remaining &= remaining - 1;
              }
              auto const total = __shfl_sync(0xffffffffu, inclusive, warp_size - 1);
              found += min(total, seed_stride - found);
              ++units;
            }
          }
        }
      }
    }
    for (std::uint64_t tile = 0; !enumerated && tile < graph_size && found < seed_stride;
         tile += warp_size) {
      auto const internal = tile + lane;
      bool passes         = false;
      if (internal < graph_size) {
        auto const source = source_indices == nullptr
                              ? internal
                              : static_cast<std::uint64_t>(source_indices[internal]);
        passes            = source < filter->dataset_rows &&
                 filter->refs[query]->contains(static_cast<std::uint32_t>(source));
      }
      auto const passing_lanes = __ballot_sync(0xffffffffu, passes);
      if (lane == 0) {
        ++units;
        auto mask = passing_lanes;
        while (mask != 0 && found < seed_stride) {
          auto const passing_lane = static_cast<unsigned>(__ffs(mask) - 1);
          output[found++]         = static_cast<IndexT>(tile + passing_lane);
          mask &= mask - 1;
        }
      }
      found = __shfl_sync(0xffffffffu, found, 0);
    }
  }
  if (lane == 0) {
    seed_counts[local_query] = found;
    // Units are representation-specific: array-entry/bitmap-word tiles, or remapped ID tiles.
    if (inspected_units != nullptr) { inspected_units[local_query] = units; }
  }
}

template <typename IndexT>
void select_navix_bitmap_seeds(raft::resources const& res,
                               cuvs::neighbors::filtering::roaring_bitmap_filter const& filter,
                               IndexT const* source_indices,
                               std::uint32_t query_offset,
                               std::uint32_t num_queries,
                               std::uint32_t graph_size,
                               std::uint32_t seed_stride,
                               IndexT* seed_ids,
                               std::uint32_t* seed_counts,
                               std::uint32_t* inspected_units = nullptr)
{
  RAFT_EXPECTS(filter.valid(), "Roaring seeds require an initialized filter");
  RAFT_EXPECTS(seed_stride > 0 && seed_ids != nullptr && seed_counts != nullptr,
               "Roaring seed outputs must be preallocated with a positive stride");
  RAFT_EXPECTS(static_cast<std::uint64_t>(query_offset) + num_queries <= filter.num_queries(),
               "Roaring filter does not cover the requested seed query range");
  if (num_queries == 0) { return; }
  auto const* payload = static_cast<cuvs::neighbors::detail::roaring_bitmap_filter_data_t<> const*>(
    filter.device_payload());
  navix_select_roaring_seeds_kernel<<<num_queries,
                                      32,
                                      0,
                                      raft::resource::get_cuda_stream(res).get()>>>(
    payload,
    source_indices,
    query_offset,
    graph_size,
    seed_stride,
    seed_ids,
    seed_counts,
    inspected_units);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

}  // namespace cuvs::neighbors::cagra::detail
