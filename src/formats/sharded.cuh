#pragma once

#include "../matrix.cuh"

namespace cellshard {

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
};

template<typename MatrixT> __host__ inline void destroy(MatrixT *m) {
    if (m == 0) return;
    clear(m);
    delete m;
}
template<typename MatrixT> __host__ __device__ __forceinline__ unsigned int part_aux(const MatrixT *m);
template<typename MatrixT> __host__ __device__ __forceinline__ std::size_t part_bytes(const sharded<MatrixT> *m, unsigned long partId);

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
}

template<typename MatrixT>
__host__ __forceinline__ void clear(sharded<MatrixT> * __restrict__ m) {
    unsigned long i = 0;
    if (m->parts != 0) {
        for (i = 0; i < m->num_parts; ++i) destroy(m->parts[i]);
    }
    std::free(m->parts);
    std::free(m->part_offsets);
    std::free(m->part_rows);
    std::free(m->part_nnz);
    std::free(m->part_aux);
    std::free(m->shard_offsets);
    init(m);
}

template<typename MatrixT>
__host__ __forceinline__ int reserve_parts(sharded<MatrixT> * __restrict__ m, unsigned long capacity) {
    MatrixT **newParts = 0;
    unsigned long *newOffsets = 0;
    unsigned long *newRows = 0;
    unsigned long *newNnz = 0;
    unsigned long *newAux = 0;

    if (capacity <= m->part_capacity) return 1;
    newParts = (MatrixT **) std::calloc((std::size_t) capacity, sizeof(MatrixT *));
    newOffsets = (unsigned long *) std::calloc((std::size_t) (capacity + 1), sizeof(unsigned long));
    newRows = (unsigned long *) std::calloc((std::size_t) capacity, sizeof(unsigned long));
    newNnz = (unsigned long *) std::calloc((std::size_t) capacity, sizeof(unsigned long));
    newAux = (unsigned long *) std::calloc((std::size_t) capacity, sizeof(unsigned long));
    if (newParts == 0 || newOffsets == 0 || newRows == 0 || newNnz == 0 || newAux == 0) {
        std::free(newParts);
        std::free(newOffsets);
        std::free(newRows);
        std::free(newNnz);
        std::free(newAux);
        return 0;
    }
    if (m->num_parts != 0) {
        std::memcpy(newParts, m->parts, (std::size_t) m->num_parts * sizeof(MatrixT *));
        std::memcpy(newOffsets, m->part_offsets, (std::size_t) (m->num_parts + 1) * sizeof(unsigned long));
        std::memcpy(newRows, m->part_rows, (std::size_t) m->num_parts * sizeof(unsigned long));
        std::memcpy(newNnz, m->part_nnz, (std::size_t) m->num_parts * sizeof(unsigned long));
        std::memcpy(newAux, m->part_aux, (std::size_t) m->num_parts * sizeof(unsigned long));
    }

    std::free(m->parts);
    std::free(m->part_offsets);
    std::free(m->part_rows);
    std::free(m->part_nnz);
    std::free(m->part_aux);
    m->parts = newParts;
    m->part_offsets = newOffsets;
    m->part_rows = newRows;
    m->part_nnz = newNnz;
    m->part_aux = newAux;
    m->part_capacity = capacity;
    return 1;
}

template<typename MatrixT>
__host__ __forceinline__ int reserve_shards(sharded<MatrixT> * __restrict__ m, unsigned long capacity) {
    unsigned long *newOffsets = 0;

    if (capacity <= m->shard_capacity) return 1;
    newOffsets = (unsigned long *) std::calloc((std::size_t) (capacity + 1), sizeof(unsigned long));
    if (newOffsets == 0) return 0;
    if (m->shard_offsets != 0 && m->num_shards != 0) {
        std::memcpy(newOffsets, m->shard_offsets, (std::size_t) (m->num_shards + 1) * sizeof(unsigned long));
    }
    std::free(m->shard_offsets);
    m->shard_offsets = newOffsets;
    m->shard_capacity = capacity;
    return 1;
}

template<typename MatrixT>
__host__ __forceinline__ void rebuild_part_offsets(sharded<MatrixT> * __restrict__ m) {
    unsigned long i = 0;

    m->rows = 0;
    m->nnz = 0;
    if (m->part_offsets == 0) return;
    m->part_offsets[0] = 0;
    for (i = 0; i < m->num_parts; ++i) {
        m->part_offsets[i + 1] = m->part_offsets[i] + m->part_rows[i];
        m->nnz += m->part_nnz[i];
    }
    m->rows = m->part_offsets[m->num_parts];
}

template<typename MatrixT>
__host__ __forceinline__ int set_shards_to_parts(sharded<MatrixT> * __restrict__ m) {
    unsigned long i = 0;
    if (!reserve_shards(m, m->num_parts)) return 0;
    m->num_shards = m->num_parts;
    for (i = 0; i <= m->num_parts; ++i) m->shard_offsets[i] = m->part_offsets[i];
    return 1;
}

template<typename MatrixT>
__host__ __device__ __forceinline__ unsigned long find_part(const sharded<MatrixT> * __restrict__ m, unsigned long row) {
    if (m->part_offsets == 0 || m->num_parts == 0) return m->num_parts;
    return find_offset_span(row, m->part_offsets, m->num_parts);
}

template<typename MatrixT>
__host__ __device__ __forceinline__ unsigned long find_shard(const sharded<MatrixT> * __restrict__ m, unsigned long row) {
    if (m->shard_offsets == 0 || m->num_shards == 0) return m->num_shards;
    return find_offset_span(row, m->shard_offsets, m->num_shards);
}

template<typename MatrixT>
__host__ __device__ __forceinline__ const __half *at(const sharded<MatrixT> * __restrict__ m, unsigned long r, unsigned int c) {
    unsigned long partId = find_part(m, r);
    MatrixT *part = 0;
    if (partId >= m->num_parts) return 0;
    part = m->parts[partId];
    if (part == 0) return 0;
    return at(part, (unsigned int) (r - m->part_offsets[partId]), c);
}

template<typename MatrixT>
__host__ __device__ __forceinline__ __half *at(sharded<MatrixT> * __restrict__ m, unsigned long r, unsigned int c) {
    unsigned long partId = find_part(m, r);
    MatrixT *part = 0;
    if (partId >= m->num_parts) return 0;
    part = m->parts[partId];
    if (part == 0) return 0;
    return at(part, (unsigned int) (r - m->part_offsets[partId]), c);
}

template<typename MatrixT>
__host__ __forceinline__ int append_part(sharded<MatrixT> * __restrict__ m, MatrixT *part) {
    unsigned long next = 0;

    if (m->num_parts == m->part_capacity) {
        next = m->part_capacity == 0 ? 4 : m->part_capacity << 1;
        if (!reserve_parts(m, next)) return 0;
    }
    m->parts[m->num_parts] = part;
    m->part_rows[m->num_parts] = part != 0 ? part->rows : 0;
    m->part_nnz[m->num_parts] = part != 0 ? part->nnz : 0;
    m->part_aux[m->num_parts] = part != 0 ? ::cellshard::part_aux(part) : 0;
    ++m->num_parts;
    if (part != 0) {
        if (m->cols == 0) m->cols = part->cols;
    }
    rebuild_part_offsets(m);
    return set_shards_to_parts(m);
}

template<typename MatrixT>
__host__ __forceinline__ int concatenate(sharded<MatrixT> * __restrict__ dst, sharded<MatrixT> * __restrict__ src) {
    unsigned long i = 0;

    if (src->num_parts == 0) return 1;
    if (!reserve_parts(dst, dst->num_parts + src->num_parts)) return 0;
    for (i = 0; i < src->num_parts; ++i) {
        dst->parts[dst->num_parts + i] = src->parts[i];
        dst->part_rows[dst->num_parts + i] = src->part_rows[i];
        dst->part_nnz[dst->num_parts + i] = src->part_nnz[i];
        dst->part_aux[dst->num_parts + i] = src->part_aux[i];
        src->parts[i] = 0;
        src->part_rows[i] = 0;
        src->part_nnz[i] = 0;
        src->part_aux[i] = 0;
    }
    dst->num_parts += src->num_parts;
    src->num_parts = 0;
    if (dst->cols == 0) dst->cols = src->cols;
    rebuild_part_offsets(dst);
    set_shards_to_parts(dst);
    rebuild_part_offsets(src);
    src->rows = 0;
    src->nnz = 0;
    src->num_shards = 0;
    return 1;
}

template<typename MatrixT>
__host__ __forceinline__ int set_equal_shards(sharded<MatrixT> * __restrict__ m, unsigned long count) {
    unsigned long base = 0;
    unsigned long rem = 0;
    unsigned long row = 0;
    unsigned long i = 0;

    if (count == 0) {
        m->num_shards = 0;
        return 1;
    }
    if (!reserve_shards(m, count)) return 0;
    m->num_shards = count;
    base = m->rows / count;
    rem = m->rows % count;
    for (i = 0; i < count; ++i) {
        m->shard_offsets[i] = row;
        row += base + (i < rem ? 1 : 0);
    }
    m->shard_offsets[count] = m->rows;
    return 1;
}

template<typename MatrixT>
__host__ __forceinline__ int reshard(sharded<MatrixT> * __restrict__ m, unsigned long count, const unsigned long * __restrict__ offsets) {
    unsigned long i = 0;

    if (count == 0 || offsets == 0) {
        m->num_shards = 0;
        return 1;
    }
    if (offsets[0] != 0 || offsets[count] != m->rows) return 0;
    for (i = 0; i < count; ++i) {
        if (offsets[i] > offsets[i + 1]) return 0;
    }
    if (!reserve_shards(m, count)) return 0;
    m->num_shards = count;
    for (i = 0; i <= count; ++i) m->shard_offsets[i] = offsets[i];
    return 1;
}

template<typename MatrixT>
__host__ __device__ __forceinline__ unsigned long first_part_in_shard(const sharded<MatrixT> * __restrict__ m, unsigned long shardId) {
    if (shardId >= m->num_shards || m->num_parts == 0) return m->num_parts;
    return find_part(m, m->shard_offsets[shardId]);
}

template<typename MatrixT>
__host__ __device__ __forceinline__ unsigned long last_part_in_shard(const sharded<MatrixT> * __restrict__ m, unsigned long shardId) {
    unsigned long rowEnd = 0;
    if (shardId >= m->num_shards) return m->num_parts;
    rowEnd = m->shard_offsets[shardId + 1];
    if (rowEnd == 0) return 0;
    return find_part(m, rowEnd - 1) + 1;
}

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
    for (i = 0; i < m->num_parts; ++i) total += part_bytes(m, i);
    return total;
}

template<typename MatrixT>
__host__ __forceinline__ int set_shards_by_part_bytes(sharded<MatrixT> * __restrict__ m, std::size_t max_bytes) {
    std::size_t used = 0;
    std::size_t partBytes = 0;
    unsigned long shardCount = 0;
    unsigned long i = 0;

    if (max_bytes == 0) return set_shards_to_parts(m);
    if (!reserve_shards(m, m->num_parts)) return 0;

    m->shard_offsets[0] = 0;
    shardCount = 0;
    used = 0;
    for (i = 0; i < m->num_parts; ++i) {
        partBytes = part_bytes(m, i);
        if (partBytes == 0) continue;
        if (used != 0 && used + partBytes > max_bytes) {
            ++shardCount;
            m->shard_offsets[shardCount] = m->part_offsets[i];
            used = 0;
        }
        used += partBytes;
    }

    if (m->num_parts != 0) {
        ++shardCount;
        m->shard_offsets[shardCount] = m->rows;
    }
    m->num_shards = shardCount;
    return 1;
}

} // namespace cellshard
