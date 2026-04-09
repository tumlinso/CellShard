#pragma once

#include "../offset_span.cuh"
#include "../real.cuh"
#include "../formats/compressed.cuh"
#include "../formats/dense.cuh"
#include "../formats/diagonal.cuh"
#include "../formats/triplet.cuh"

#include <cstddef>

namespace cellshard {

// Metadata-only stitched matrix view.
//
// The split is intentional:
// - parts[] may or may not be materialized on host
// - part_rows/part_nnz/part_aux stay valid as cheap metadata
// - shard_offsets provide coarser scheduling groups for runtime code
template<typename MatrixT>
struct alignas(16) sharded {
    unsigned long rows;
    unsigned long cols;
    unsigned long nnz;

    unsigned long num_parts;
    unsigned long part_capacity;
    MatrixT **parts;
    unsigned long *part_offsets;
    unsigned long *part_rows;
    unsigned long *part_nnz;
    unsigned long *part_aux;

    unsigned long num_shards;
    unsigned long shard_capacity;
    unsigned long *shard_offsets;
    // O(1) shard -> [first_part, last_part) lookup table. This is derived from
    // shard_offsets/part_offsets and keeps shard-boundary queries off the
    // binary-search path in the hot runtime code.
    unsigned long *shard_parts;
};

// Zero metadata and pointers. No deallocation happens here.
template<typename MatrixT>
__host__ __device__ __forceinline__ void init(sharded<MatrixT> * __restrict__ m) {
    m->rows = 0;
    m->cols = 0;
    m->nnz = 0;
    m->num_parts = 0;
    m->part_capacity = 0;
    m->parts = 0;
    m->part_offsets = 0;
    m->part_rows = 0;
    m->part_nnz = 0;
    m->part_aux = 0;
    m->num_shards = 0;
    m->shard_capacity = 0;
    m->shard_offsets = 0;
    m->shard_parts = 0;
}

// Default auxiliary metadata for formats that do not need it.
template<typename MatrixT>
__host__ __device__ __forceinline__ unsigned int part_aux(const MatrixT *) {
    return 0;
}

// Dense parts derive nnz from rows*cols.
__host__ __device__ __forceinline__ unsigned long part_nnz(const dense *m) {
    return (unsigned long) m->rows * (unsigned long) m->cols;
}

// Generic sparse path reads nnz directly from the materialized part.
template<typename MatrixT>
__host__ __device__ __forceinline__ unsigned long part_nnz(const MatrixT *m) {
    return m->nnz;
}

// DIA needs num_diagonals when the part payload is absent.
__host__ __device__ __forceinline__ unsigned int part_aux(const sparse::dia *m) {
    return m->num_diagonals;
}

// Compressed storage needs axis metadata when only the sharded header is live.
__host__ __device__ __forceinline__ unsigned int part_aux(const sparse::compressed *m) {
    return m->axis;
}

// Row -> part lookup over part_offsets[].
template<typename MatrixT>
__host__ __device__ __forceinline__ unsigned long find_part(const sharded<MatrixT> * __restrict__ m, unsigned long row) {
    if (m->part_offsets == 0 || m->num_parts == 0) return m->num_parts;
    return find_offset_span(row, m->part_offsets, m->num_parts);
}

// Row -> shard lookup over shard_offsets[].
template<typename MatrixT>
__host__ __device__ __forceinline__ unsigned long find_shard(const sharded<MatrixT> * __restrict__ m, unsigned long row) {
    if (m->shard_offsets == 0 || m->num_shards == 0) return m->num_shards;
    return find_offset_span(row, m->shard_offsets, m->num_shards);
}

// Boundary helpers keep row ownership explicit and cheap.
template<typename MatrixT>
__host__ __device__ __forceinline__ unsigned long first_row_in_part(const sharded<MatrixT> * __restrict__ m, unsigned long partId) {
    if (partId >= m->num_parts || m->part_offsets == 0) return m->rows;
    return m->part_offsets[partId];
}

template<typename MatrixT>
__host__ __device__ __forceinline__ unsigned long last_row_in_part(const sharded<MatrixT> * __restrict__ m, unsigned long partId) {
    if (partId >= m->num_parts || m->part_offsets == 0) return m->rows;
    return m->part_offsets[partId + 1];
}

// Legal resharding cut points are exactly the part boundaries.
template<typename MatrixT>
__host__ __device__ __forceinline__ int row_is_part_boundary(const sharded<MatrixT> * __restrict__ m, unsigned long row) {
    if (row == 0 || row == m->rows) return 1;
    if (m->part_offsets == 0 || m->num_parts == 0) return 0;
    const unsigned long hit = find_part(m, row);
    if (hit >= m->num_parts) return 0;
    return m->part_offsets[hit] == row;
}

// at() only works if the target part is already materialized in parts[].
template<typename MatrixT>
__host__ __device__ __forceinline__ const real::storage_t *at(const sharded<MatrixT> * __restrict__ m, unsigned long r, types::idx_t c) {
    const unsigned long partId = find_part(m, r);
    MatrixT *part = 0;
    if (partId >= m->num_parts) return 0;
    part = m->parts[partId];
    if (part == 0) return 0;
    return at(part, (types::dim_t) (r - m->part_offsets[partId]), c);
}

template<typename MatrixT>
__host__ __device__ __forceinline__ real::storage_t *at(sharded<MatrixT> * __restrict__ m, unsigned long r, types::idx_t c) {
    const unsigned long partId = find_part(m, r);
    MatrixT *part = 0;
    if (partId >= m->num_parts) return 0;
    part = m->parts[partId];
    if (part == 0) return 0;
    return at(part, (types::dim_t) (r - m->part_offsets[partId]), c);
}

// Shard membership is derived from row boundaries and therefore stays aligned
// to whole parts.
template<typename MatrixT>
__host__ __device__ __forceinline__ unsigned long first_part_in_shard(const sharded<MatrixT> * __restrict__ m, unsigned long shardId) {
    if (m->shard_parts != 0 && shardId < m->num_shards) return m->shard_parts[shardId];
    if (shardId >= m->num_shards || m->num_parts == 0) return m->num_parts;
    return find_part(m, m->shard_offsets[shardId]);
}

template<typename MatrixT>
__host__ __device__ __forceinline__ unsigned long last_part_in_shard(const sharded<MatrixT> * __restrict__ m, unsigned long shardId) {
    unsigned long rowEnd = 0;
    if (m->shard_parts != 0 && shardId < m->num_shards) return m->shard_parts[shardId + 1];
    if (shardId >= m->num_shards) return m->num_parts;
    rowEnd = m->shard_offsets[shardId + 1];
    if (rowEnd == 0) return 0;
    return find_part(m, rowEnd - 1) + 1;
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
__host__ __device__ __forceinline__ unsigned long part_count_in_shard(const sharded<MatrixT> * __restrict__ m, unsigned long shardId) {
    const unsigned long begin = first_part_in_shard(m, shardId);
    const unsigned long end = last_part_in_shard(m, shardId);
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
__host__ __device__ __forceinline__ int part_loaded(const sharded<MatrixT> * __restrict__ m, unsigned long partId) {
    if (partId >= m->num_parts) return 0;
    return m->parts[partId] != 0;
}

template<typename MatrixT>
__host__ __device__ __forceinline__ int shard_loaded(const sharded<MatrixT> * __restrict__ m, unsigned long shardId) {
    unsigned long begin = 0;
    unsigned long end = 0;
    unsigned long i = 0;

    if (shardId >= m->num_shards) return 0;
    begin = first_part_in_shard(m, shardId);
    end = last_part_in_shard(m, shardId);
    for (i = begin; i < end; ++i) {
        if (!part_loaded(m, i)) return 0;
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
    begin = first_part_in_shard(m, shardId);
    end = last_part_in_shard(m, shardId);
    for (i = begin; i < end; ++i) total += m->part_nnz[i];
    return total;
}

// Host-side footprint estimates. If a part is not materialized, bytes are
// reconstructed from metadata only.
__host__ __device__ __forceinline__ std::size_t part_bytes(const sharded<dense> *m, unsigned long partId) {
    if (partId >= m->num_parts) return 0;
    if (m->parts[partId] != 0) return bytes(m->parts[partId]);
    return sizeof(dense) + (std::size_t) m->part_nnz[partId] * sizeof(real::storage_t);
}

__host__ __device__ __forceinline__ std::size_t part_bytes(const sharded<sparse::compressed> *m, unsigned long partId) {
    const unsigned long ptr_dim = m->part_aux[partId] == sparse::compressed_by_col ? m->cols : m->part_rows[partId];
    if (partId >= m->num_parts) return 0;
    if (m->parts[partId] != 0) return bytes(m->parts[partId]);
    return sizeof(sparse::compressed)
        + (std::size_t) (ptr_dim + 1) * sizeof(types::ptr_t)
        + (std::size_t) m->part_nnz[partId] * sizeof(types::idx_t)
        + (std::size_t) m->part_nnz[partId] * sizeof(real::storage_t);
}

__host__ __device__ __forceinline__ std::size_t part_bytes(const sharded<sparse::coo> *m, unsigned long partId) {
    if (partId >= m->num_parts) return 0;
    if (m->parts[partId] != 0) return bytes(m->parts[partId]);
    return sizeof(sparse::coo)
        + (std::size_t) m->part_nnz[partId] * sizeof(types::idx_t)
        + (std::size_t) m->part_nnz[partId] * sizeof(types::idx_t)
        + (std::size_t) m->part_nnz[partId] * sizeof(real::storage_t);
}

__host__ __device__ __forceinline__ std::size_t part_bytes(const sharded<sparse::dia> *m, unsigned long partId) {
    if (partId >= m->num_parts) return 0;
    if (m->parts[partId] != 0) return bytes(m->parts[partId]);
    return sizeof(sparse::dia)
        + (std::size_t) m->part_aux[partId] * sizeof(int)
        + (std::size_t) m->part_nnz[partId] * sizeof(real::storage_t);
}

template<typename MatrixT>
__host__ __device__ __forceinline__ std::size_t bytes(const sharded<MatrixT> * __restrict__ m) {
    unsigned long i = 0;
    std::size_t total = sizeof(*m);
    total += (std::size_t) m->part_capacity * sizeof(MatrixT *);
    total += (std::size_t) (m->part_capacity + 1) * sizeof(unsigned long);
    total += (std::size_t) m->part_capacity * sizeof(unsigned long);
    total += (std::size_t) m->part_capacity * sizeof(unsigned long);
    total += (std::size_t) m->part_capacity * sizeof(unsigned long);
    total += (std::size_t) (m->shard_capacity + 1) * sizeof(unsigned long);
    total += (std::size_t) (m->shard_capacity + 1) * sizeof(unsigned long);
    for (i = 0; i < m->num_parts; ++i) total += part_bytes(m, i);
    return total;
}

template<typename MatrixT>
__host__ __device__ __forceinline__ std::size_t shard_bytes(const sharded<MatrixT> * __restrict__ m, unsigned long shardId) {
    unsigned long begin = 0;
    unsigned long end = 0;
    unsigned long i = 0;
    std::size_t total = 0;

    if (shardId >= m->num_shards) return 0;
    begin = first_part_in_shard(m, shardId);
    end = last_part_in_shard(m, shardId);
    for (i = begin; i < end; ++i) total += part_bytes(m, i);
    return total;
}

} // namespace cellshard
