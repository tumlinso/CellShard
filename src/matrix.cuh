#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <climits>

#include "types.cuh"

namespace cellshard {

template<typename OffsetT>
__host__ __device__ __forceinline__ OffsetT find_offset_span(OffsetT row, const OffsetT *offsets, OffsetT count) {
    OffsetT lo = 0;
    OffsetT hi = count;
    while (lo < hi) {
        const OffsetT mid = lo + ((hi - lo) >> 1);
        const OffsetT mid_offset = offsets[mid];
        const OffsetT next_offset = offsets[mid + 1];
        if (row < mid_offset) hi = mid;
        else if (row >= next_offset) lo = mid + 1;
        else return mid;
    }
    return count;
}

namespace sparse {

template<typename IndexT, typename RowPtrT, typename ColIdxT>
struct csr_base {
    IndexT rows;
    IndexT cols;
    IndexT nnz;
    RowPtrT rowPtr;
    ColIdxT colIdx;
};

} // namespace sparse

} // namespace cellshard

#include "dense.cuh"
#include "sparse/csr.cuh"
#include "sparse/conversion/csX_to_coo.cuh"
#include "sparse/coo.cuh"
#include "sparse/dia.cuh"
#include "sharded.cuh"

namespace cellshard {

template<typename MatrixT>
__host__ __device__ __forceinline__ unsigned int part_aux(const MatrixT *) {
    return 0;
}

__host__ __device__ __forceinline__ unsigned int part_aux(const sparse::dia *m) {
    return m->num_diagonals;
}

__host__ __device__ __forceinline__ std::size_t part_bytes(const sharded<dense> *m, unsigned long partId) {
    if (partId >= m->num_parts) return 0;
    if (m->parts[partId] != 0) return bytes(m->parts[partId]);
    return sizeof(dense) + (std::size_t) m->part_nnz[partId] * sizeof(__half);
}

__host__ __device__ __forceinline__ std::size_t part_bytes(const sharded<sparse::csr> *m, unsigned long partId) {
    if (partId >= m->num_parts) return 0;
    if (m->parts[partId] != 0) return bytes(m->parts[partId]);
    return sizeof(sparse::csr)
        + (std::size_t) (m->part_rows[partId] + 1) * sizeof(unsigned int)
        + (std::size_t) m->part_nnz[partId] * sizeof(unsigned int)
        + (std::size_t) m->part_nnz[partId] * sizeof(__half);
}

__host__ __device__ __forceinline__ std::size_t part_bytes(const sharded<sparse::coo> *m, unsigned long partId) {
    if (partId >= m->num_parts) return 0;
    if (m->parts[partId] != 0) return bytes(m->parts[partId]);
    return sizeof(sparse::coo)
        + (std::size_t) m->part_nnz[partId] * sizeof(unsigned int)
        + (std::size_t) m->part_nnz[partId] * sizeof(unsigned int)
        + (std::size_t) m->part_nnz[partId] * sizeof(__half);
}

__host__ __device__ __forceinline__ std::size_t part_bytes(const sharded<sparse::dia> *m, unsigned long partId) {
    if (partId >= m->num_parts) return 0;
    if (m->parts[partId] != 0) return bytes(m->parts[partId]);
    return sizeof(sparse::dia)
        + (std::size_t) m->part_aux[partId] * sizeof(int)
        + (std::size_t) m->part_nnz[partId] * sizeof(__half);
}

__host__ __forceinline__ void destroy(dense *m) {
    if (m == 0) return;
    clear(m);
    delete m;
}

__host__ __forceinline__ void destroy(sparse::csr *m) {
    if (m == 0) return;
    sparse::clear(m);
    delete m;
}

__host__ __forceinline__ void destroy(sparse::coo *m) {
    if (m == 0) return;
    sparse::clear(m);
    delete m;
}

__host__ __forceinline__ void destroy(sparse::dia *m) {
    if (m == 0) return;
    sparse::clear(m);
    delete m;
}

} // namespace cellshard
