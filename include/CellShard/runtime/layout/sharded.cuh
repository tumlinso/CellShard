#pragma once

#ifndef CELLSHARD_ENABLE_CELLERATOR_QUANTIZED
#define CELLSHARD_ENABLE_CELLERATOR_QUANTIZED 1
#endif

#include "../../core/offset_span.cuh"
#include "../../core/real.cuh"
#include "../../access/fallback_adapters.cuh"
#include "../../formats/compressed.cuh"
#include "../../formats/blocked_ell.cuh"
#if CELLSHARD_ENABLE_CELLERATOR_QUANTIZED
#include "../../formats/quantized_blocked_ell.cuh"
#endif
#include "../../formats/sliced_ell.cuh"
#include "../../formats/dense.cuh"
#include "../../formats/diagonal.cuh"
#include "../../formats/triplet.cuh"
#if defined(__has_include)
#if __has_include(<Cellerator/interop/cellshard_access.cuh>)
#include <Cellerator/interop/cellshard_access.cuh>
#endif
#endif

#include <cstddef>
#include <CellShard/domain.hh>

namespace cellshard {

// Metadata-only stitched matrix view.
//
// The split is intentional:
// - parts[] may or may not be materialized on host
// - partition_rows/partition_nnz/partition_aux stay valid as cheap metadata
// - shard_offsets provide coarser scheduling groups for runtime code
template<typename MatrixT>
struct alignas(16) sharded {
    unsigned long rows;
    unsigned long cols;
    unsigned long nnz;

    unsigned long num_partitions;
    unsigned long partition_capacity;
    MatrixT **parts;
    unsigned long *partition_offsets;
    unsigned long *partition_rows;
    unsigned long *partition_nnz;
    unsigned long *partition_aux;

    unsigned long num_shards;
    unsigned long shard_capacity;
    unsigned long *shard_offsets;
    // O(1) shard -> [first_part, last_part) lookup table. This is derived from
    // shard_offsets/partition_offsets and keeps shard-boundary queries off the
    // binary-search path in the hot runtime code.
    unsigned long *shard_parts;
};

// CS-FOUND-LEGACY: this describes a compatibility row partition. The caller
// supplies semantic IDs explicitly; physical shard grouping remains only
// legacy locality metadata and is not biological ownership.
struct legacy_row_partition_binding {
    domain_binding binding{};
    std::uint64_t canonical_generation = 0;
    unsigned long global_row_begin = 0;
    unsigned long row_count = 0;
    unsigned long physical_shard_group = 0;
};

template<typename MatrixT>
inline bool adapt_legacy_row_partition(
    const sharded<MatrixT> &matrix, unsigned long partition_index,
    domain_binding explicit_binding, std::uint64_t canonical_generation,
    unsigned long physical_shard_group,
    legacy_row_partition_binding *out) noexcept {
    if (out == nullptr || partition_index >= matrix.num_partitions
        || matrix.partition_offsets == nullptr || canonical_generation == 0
        || !valid_domain_binding_role(explicit_binding.role)
        || !explicit_binding.domain.valid() || !explicit_binding.map.valid()
        || !explicit_binding.partition.valid() || !explicit_binding.order.valid()) {
        return false;
    }
    const unsigned long begin = matrix.partition_offsets[partition_index];
    const unsigned long end = matrix.partition_offsets[partition_index + 1];
    if (end <= begin) return false;
    *out = {explicit_binding, canonical_generation, begin, end - begin,
            physical_shard_group};
    return true;
}

// Zero metadata and pointers. No deallocation happens here.
template<typename MatrixT>
__host__ __device__ __forceinline__ void init(sharded<MatrixT> * __restrict__ m) {
    m->rows = 0;
    m->cols = 0;
    m->nnz = 0;
    m->num_partitions = 0;
    m->partition_capacity = 0;
    m->parts = 0;
    m->partition_offsets = 0;
    m->partition_rows = 0;
    m->partition_nnz = 0;
    m->partition_aux = 0;
    m->num_shards = 0;
    m->shard_capacity = 0;
    m->shard_offsets = 0;
    m->shard_parts = 0;
}

// Default auxiliary metadata for formats that do not need it.
template<typename MatrixT>
__host__ __device__ __forceinline__ unsigned long partition_aux(const MatrixT *m) {
    return access::payload_traits<MatrixT>::aux(m);
}

template<typename MatrixT>
__host__ __device__ __forceinline__ unsigned long partition_nnz(const MatrixT *m) {
    return access::payload_traits<MatrixT>::nnz(m);
}

// Row -> part lookup over partition_offsets[].
template<typename MatrixT>
__host__ __device__ __forceinline__ unsigned long find_partition(const sharded<MatrixT> * __restrict__ m, unsigned long row) {
    if (m->partition_offsets == 0 || m->num_partitions == 0) return m->num_partitions;
    return find_offset_span(row, m->partition_offsets, m->num_partitions);
}

// Row -> shard lookup over shard_offsets[].
template<typename MatrixT>
__host__ __device__ __forceinline__ unsigned long find_shard(const sharded<MatrixT> * __restrict__ m, unsigned long row) {
    if (m->shard_offsets == 0 || m->num_shards == 0) return m->num_shards;
    return find_offset_span(row, m->shard_offsets, m->num_shards);
}

// Boundary helpers keep row ownership explicit and cheap.
template<typename MatrixT>
__host__ __device__ __forceinline__ unsigned long first_row_in_partition(const sharded<MatrixT> * __restrict__ m, unsigned long partId) {
    if (partId >= m->num_partitions || m->partition_offsets == 0) return m->rows;
    return m->partition_offsets[partId];
}

template<typename MatrixT>
__host__ __device__ __forceinline__ unsigned long last_row_in_partition(const sharded<MatrixT> * __restrict__ m, unsigned long partId) {
    if (partId >= m->num_partitions || m->partition_offsets == 0) return m->rows;
    return m->partition_offsets[partId + 1];
}

// Legal resharding cut points are exactly the part boundaries.
template<typename MatrixT>
__host__ __device__ __forceinline__ int row_is_partition_boundary(const sharded<MatrixT> * __restrict__ m, unsigned long row) {
    if (row == 0 || row == m->rows) return 1;
    if (m->partition_offsets == 0 || m->num_partitions == 0) return 0;
    const unsigned long hit = find_partition(m, row);
    if (hit >= m->num_partitions) return 0;
    return m->partition_offsets[hit] == row;
}

// at() only works if the target part is already materialized in parts[].
template<typename MatrixT>
__host__ __device__ __forceinline__ const real::storage_t *at(const sharded<MatrixT> * __restrict__ m, unsigned long r, types::idx_t c) {
    const unsigned long partId = find_partition(m, r);
    MatrixT *part = 0;
    if (partId >= m->num_partitions) return 0;
    part = m->parts[partId];
    if (part == 0) return 0;
    return access::payload_traits<MatrixT>::debug_at(part, r - m->partition_offsets[partId], c);
}

template<typename MatrixT>
__host__ __device__ __forceinline__ real::storage_t *at(sharded<MatrixT> * __restrict__ m, unsigned long r, types::idx_t c) {
    const unsigned long partId = find_partition(m, r);
    MatrixT *part = 0;
    if (partId >= m->num_partitions) return 0;
    part = m->parts[partId];
    if (part == 0) return 0;
    return const_cast<real::storage_t *>(access::payload_traits<MatrixT>::debug_at(part, r - m->partition_offsets[partId], c));
}

// Shard membership is derived from row boundaries and therefore stays aligned
// to whole parts.
template<typename MatrixT>
__host__ __device__ __forceinline__ unsigned long first_partition_in_shard(const sharded<MatrixT> * __restrict__ m, unsigned long shardId) {
    if (m->shard_parts != 0 && shardId < m->num_shards) return m->shard_parts[shardId];
    if (shardId >= m->num_shards || m->num_partitions == 0) return m->num_partitions;
    return find_partition(m, m->shard_offsets[shardId]);
}

template<typename MatrixT>
__host__ __device__ __forceinline__ unsigned long last_partition_in_shard(const sharded<MatrixT> * __restrict__ m, unsigned long shardId) {
    unsigned long rowEnd = 0;
    if (m->shard_parts != 0 && shardId < m->num_shards) return m->shard_parts[shardId + 1];
    if (shardId >= m->num_shards) return m->num_partitions;
    rowEnd = m->shard_offsets[shardId + 1];
    if (rowEnd == 0) return 0;
    return find_partition(m, rowEnd - 1) + 1;
}

template<typename MatrixT>
__host__ __device__ __forceinline__ unsigned long first_row_in_shard(const sharded<MatrixT> * __restrict__ m, unsigned long shardId) {
    if (shardId >= m->num_shards || m->shard_offsets == 0) return m->rows;
    return m->shard_offsets[shardId];
}

template<typename MatrixT>
__host__ __device__ __forceinline__ unsigned long last_row_in_shard(const sharded<MatrixT> * __restrict__ m, unsigned long shardId) {
    if (shardId >= m->num_shards || m->shard_offsets == 0) return m->rows;
    return m->shard_offsets[shardId + 1];
}

template<typename MatrixT>
__host__ __device__ __forceinline__ unsigned long partition_count_in_shard(const sharded<MatrixT> * __restrict__ m, unsigned long shardId) {
    const unsigned long begin = first_partition_in_shard(m, shardId);
    const unsigned long end = last_partition_in_shard(m, shardId);
    if (begin >= end) return 0;
    return end - begin;
}

template<typename MatrixT>
__host__ __device__ __forceinline__ unsigned long rows_in_shard(const sharded<MatrixT> * __restrict__ m, unsigned long shardId) {
    const unsigned long rowBegin = first_row_in_shard(m, shardId);
    const unsigned long rowEnd = last_row_in_shard(m, shardId);
    if (rowBegin >= rowEnd) return 0;
    return rowEnd - rowBegin;
}

// Loaded-state checks look only at host materialization state.
template<typename MatrixT>
__host__ __device__ __forceinline__ int partition_loaded(const sharded<MatrixT> * __restrict__ m, unsigned long partId) {
    if (partId >= m->num_partitions) return 0;
    return m->parts[partId] != 0;
}

template<typename MatrixT>
__host__ __device__ __forceinline__ int shard_loaded(const sharded<MatrixT> * __restrict__ m, unsigned long shardId) {
    unsigned long begin = 0;
    unsigned long end = 0;
    unsigned long i = 0;

    if (shardId >= m->num_shards) return 0;
    begin = first_partition_in_shard(m, shardId);
    end = last_partition_in_shard(m, shardId);
    for (i = begin; i < end; ++i) {
        if (!partition_loaded(m, i)) return 0;
    }
    return 1;
}

// Metadata reductions over the parts that make up one shard.
template<typename MatrixT>
__host__ __device__ __forceinline__ unsigned long nnz_in_shard(const sharded<MatrixT> * __restrict__ m, unsigned long shardId) {
    unsigned long begin = 0;
    unsigned long end = 0;
    unsigned long i = 0;
    unsigned long total = 0;

    if (shardId >= m->num_shards) return 0;
    begin = first_partition_in_shard(m, shardId);
    end = last_partition_in_shard(m, shardId);
    for (i = begin; i < end; ++i) total += m->partition_nnz[i];
    return total;
}

// Host-side footprint estimates. If a part is not materialized, the active
// access payload trait reconstructs bytes from partition metadata.
template<typename MatrixT>
__host__ __device__ __forceinline__ std::size_t partition_bytes(const sharded<MatrixT> *m, unsigned long partId) {
    if (partId >= m->num_partitions) return 0;
    return access::payload_traits<MatrixT>::host_bytes(
        m->parts[partId],
        m->partition_rows[partId],
        m->cols,
        m->partition_nnz[partId],
        m->partition_aux[partId]);
}

template<typename MatrixT>
__host__ __device__ __forceinline__ std::size_t bytes(const sharded<MatrixT> * __restrict__ m) {
    unsigned long i = 0;
    std::size_t total = sizeof(*m);
    total += (std::size_t) m->partition_capacity * sizeof(MatrixT *);
    total += (std::size_t) (m->partition_capacity + 1) * sizeof(unsigned long);
    total += (std::size_t) m->partition_capacity * sizeof(unsigned long);
    total += (std::size_t) m->partition_capacity * sizeof(unsigned long);
    total += (std::size_t) m->partition_capacity * sizeof(unsigned long);
    total += (std::size_t) (m->shard_capacity + 1) * sizeof(unsigned long);
    total += (std::size_t) (m->shard_capacity + 1) * sizeof(unsigned long);
    for (i = 0; i < m->num_partitions; ++i) total += partition_bytes(m, i);
    return total;
}

template<typename MatrixT>
__host__ __device__ __forceinline__ std::size_t shard_bytes(const sharded<MatrixT> * __restrict__ m, unsigned long shardId) {
    unsigned long begin = 0;
    unsigned long end = 0;
    unsigned long i = 0;
    std::size_t total = 0;

    if (shardId >= m->num_shards) return 0;
    begin = first_partition_in_shard(m, shardId);
    end = last_partition_in_shard(m, shardId);
    for (i = begin; i < end; ++i) total += partition_bytes(m, i);
    return total;
}

} // namespace cellshard
