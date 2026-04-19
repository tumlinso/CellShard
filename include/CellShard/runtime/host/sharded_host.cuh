#pragma once

#include "../layout/sharded.cuh"
#include "../storage/shard_storage.cuh"
#include "../../io/csh5/api.cuh"
#include "../../io/pack/packfile.cuh"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/types.h>

namespace cellshard {

template<typename MatrixT>
__host__ __forceinline__ void destroy(MatrixT *m) {
    if (m == 0) return;
    clear(m);
    delete m;
}

template<typename MatrixT>
__host__ __forceinline__ void clear(sharded<MatrixT> * __restrict__ m) {
    unsigned long i = 0;
    if (m->parts != 0) {
        for (i = 0; i < m->num_partitions; ++i) destroy(m->parts[i]);
    }
    std::free(m->parts);
    std::free(m->partition_offsets);
    std::free(m->partition_rows);
    std::free(m->partition_nnz);
    std::free(m->partition_aux);
    std::free(m->shard_offsets);
    std::free(m->shard_parts);
    init(m);
}

template<typename MatrixT>
__host__ __forceinline__ int reserve_partitions(sharded<MatrixT> * __restrict__ m, unsigned long capacity) {
    MatrixT **newParts = 0;
    unsigned long *newOffsets = 0;
    unsigned long *newRows = 0;
    unsigned long *newNnz = 0;
    unsigned long *newAux = 0;

    if (capacity <= m->partition_capacity) return 1;
    // Growth here is host-side metadata churn only. The existing MatrixT*
    // payloads are not cloned, but the metadata arrays are reallocated and
    // memcpy'd into fresh storage.
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
    if (m->num_partitions != 0) {
        std::memcpy(newParts, m->parts, (std::size_t) m->num_partitions * sizeof(MatrixT *));
        std::memcpy(newOffsets, m->partition_offsets, (std::size_t) (m->num_partitions + 1) * sizeof(unsigned long));
        std::memcpy(newRows, m->partition_rows, (std::size_t) m->num_partitions * sizeof(unsigned long));
        std::memcpy(newNnz, m->partition_nnz, (std::size_t) m->num_partitions * sizeof(unsigned long));
        std::memcpy(newAux, m->partition_aux, (std::size_t) m->num_partitions * sizeof(unsigned long));
    }

    std::free(m->parts);
    std::free(m->partition_offsets);
    std::free(m->partition_rows);
    std::free(m->partition_nnz);
    std::free(m->partition_aux);
    m->parts = newParts;
    m->partition_offsets = newOffsets;
    m->partition_rows = newRows;
    m->partition_nnz = newNnz;
    m->partition_aux = newAux;
    m->partition_capacity = capacity;
    return 1;
}

template<typename MatrixT>
__host__ __forceinline__ int reserve_shards(sharded<MatrixT> * __restrict__ m, unsigned long capacity) {
    unsigned long *newOffsets = 0;
    unsigned long *newParts = 0;

    if (capacity <= m->shard_capacity) return 1;
    // Shard growth reallocates only the row-offset table.
    newOffsets = (unsigned long *) std::calloc((std::size_t) (capacity + 1), sizeof(unsigned long));
    newParts = (unsigned long *) std::calloc((std::size_t) (capacity + 1), sizeof(unsigned long));
    if (newOffsets == 0 || newParts == 0) {
        std::free(newOffsets);
        std::free(newParts);
        return 0;
    }
    if (m->shard_offsets != 0 && m->num_shards != 0) {
        std::memcpy(newOffsets, m->shard_offsets, (std::size_t) (m->num_shards + 1) * sizeof(unsigned long));
        if (m->shard_parts != 0) {
            std::memcpy(newParts, m->shard_parts, (std::size_t) (m->num_shards + 1) * sizeof(unsigned long));
        }
    }
    std::free(m->shard_offsets);
    std::free(m->shard_parts);
    m->shard_offsets = newOffsets;
    m->shard_parts = newParts;
    m->shard_capacity = capacity;
    return 1;
}

template<typename MatrixT>
__host__ __forceinline__ void rebuild_shard_parts(sharded<MatrixT> * __restrict__ m) {
    unsigned long shard = 0;
    unsigned long part = 0;

    // Precompute the exact part-span for every shard once, so repeated shard
    // staging, release, bucket, and byte-estimation calls avoid row->part
    // binary searches entirely.
    if (m->shard_parts == 0 || m->shard_offsets == 0 || m->partition_offsets == 0) return;
    for (shard = 0; shard < m->num_shards; ++shard) {
        const unsigned long row_begin = m->shard_offsets[shard];
        while (part < m->num_partitions && m->partition_offsets[part] < row_begin) ++part;
        m->shard_parts[shard] = part;
    }
    m->shard_parts[m->num_shards] = m->num_partitions;
}

template<typename MatrixT>
__host__ __forceinline__ void rebuild_partition_offsets(sharded<MatrixT> * __restrict__ m) {
    unsigned long i = 0;

    m->rows = 0;
    m->nnz = 0;
    if (m->partition_offsets == 0) return;
    m->partition_offsets[0] = 0;
    for (i = 0; i < m->num_partitions; ++i) {
        m->partition_offsets[i + 1] = m->partition_offsets[i] + m->partition_rows[i];
        m->nnz += m->partition_nnz[i];
    }
    m->rows = m->partition_offsets[m->num_partitions];
}

template<typename MatrixT>
__host__ __forceinline__ int define_partitions(sharded<MatrixT> * __restrict__ m,
                                          unsigned long cols,
                                          unsigned long num_partitions,
                                          const unsigned long * __restrict__ partition_rows,
                                          const unsigned long * __restrict__ partition_nnz,
                                          const unsigned long * __restrict__ partition_aux_in) {
    unsigned long i = 0;

    clear(m);
    init(m);
    if (!reserve_partitions(m, num_partitions)) return 0;

    m->cols = cols;
    m->num_partitions = num_partitions;
    for (i = 0; i < num_partitions; ++i) {
        m->parts[i] = 0;
        m->partition_rows[i] = partition_rows != 0 ? partition_rows[i] : 0;
        m->partition_nnz[i] = partition_nnz != 0 ? partition_nnz[i] : 0;
        m->partition_aux[i] = partition_aux_in != 0 ? partition_aux_in[i] : 0;
    }
    rebuild_partition_offsets(m);
    return set_shards_to_partitions(m);
}

template<typename MatrixT>
__host__ __forceinline__ int set_shards_to_partitions(sharded<MatrixT> * __restrict__ m) {
    unsigned long i = 0;
    if (!reserve_shards(m, m->num_partitions)) return 0;
    m->num_shards = m->num_partitions;
    for (i = 0; i <= m->num_partitions; ++i) m->shard_offsets[i] = m->partition_offsets[i];
    for (i = 0; i <= m->num_partitions; ++i) m->shard_parts[i] = i;
    return 1;
}

template<typename MatrixT>
__host__ __forceinline__ int append_partition(sharded<MatrixT> * __restrict__ m, MatrixT *part) {
    unsigned long next = 0;

    if (m->num_partitions == m->partition_capacity) {
        // Capacity growth copies metadata arrays. The part payload pointer is
        // inserted directly; append_partition does not clone MatrixT payload.
        next = m->partition_capacity == 0 ? 4 : m->partition_capacity << 1;
        if (!reserve_partitions(m, next)) return 0;
    }
    m->parts[m->num_partitions] = part;
    m->partition_rows[m->num_partitions] = part != 0 ? part->rows : 0;
    m->partition_nnz[m->num_partitions] = part != 0 ? ::cellshard::partition_nnz(part) : 0;
    m->partition_aux[m->num_partitions] = part != 0 ? ::cellshard::partition_aux(part) : 0;
    ++m->num_partitions;
    if (part != 0 && m->cols == 0) m->cols = part->cols;
    rebuild_partition_offsets(m);
    return set_shards_to_partitions(m);
}

template<typename MatrixT>
__host__ __forceinline__ int concatenate(sharded<MatrixT> * __restrict__ dst, sharded<MatrixT> * __restrict__ src) {
    unsigned long i = 0;

    if (src->num_partitions == 0) return 1;
    // This is metadata/pointer movement, not deep copy. Ownership of src->parts
    // transfers into dst one pointer at a time.
    if (!reserve_partitions(dst, dst->num_partitions + src->num_partitions)) return 0;
    for (i = 0; i < src->num_partitions; ++i) {
        dst->parts[dst->num_partitions + i] = src->parts[i];
        dst->partition_rows[dst->num_partitions + i] = src->partition_rows[i];
        dst->partition_nnz[dst->num_partitions + i] = src->partition_nnz[i];
        dst->partition_aux[dst->num_partitions + i] = src->partition_aux[i];
        src->parts[i] = 0;
        src->partition_rows[i] = 0;
        src->partition_nnz[i] = 0;
        src->partition_aux[i] = 0;
    }
    dst->num_partitions += src->num_partitions;
    src->num_partitions = 0;
    if (dst->cols == 0) dst->cols = src->cols;
    rebuild_partition_offsets(dst);
    set_shards_to_partitions(dst);
    rebuild_partition_offsets(src);
    src->rows = 0;
    src->nnz = 0;
    src->num_shards = 0;
    if (src->shard_parts != 0) src->shard_parts[0] = 0;
    return 1;
}

template<typename MatrixT>
__host__ __forceinline__ int set_equal_shards(sharded<MatrixT> * __restrict__ m, unsigned long count) {
    unsigned long shardCount = 0;
    unsigned long rows = 0;
    unsigned long target = 0;
    unsigned long i = 0;

    if (count == 0) {
        m->num_shards = 0;
        return 1;
    }
    if (m->num_partitions == 0) {
        m->num_shards = 0;
        return 1;
    }
    if (count >= m->num_partitions) return set_shards_to_partitions(m);
    if (!reserve_shards(m, count)) return 0;
    target = (m->rows + count - 1) / count;
    m->shard_offsets[0] = 0;
    shardCount = 0;
    rows = 0;
    for (i = 0; i < m->num_partitions; ++i) {
        const unsigned long parts_left = m->num_partitions - (i + 1);
        const unsigned long shards_left = count - (shardCount + 1);
        rows += m->partition_rows[i];
        if (shards_left == 0) continue;
        if (rows >= target && parts_left >= shards_left) {
            ++shardCount;
            m->shard_offsets[shardCount] = m->partition_offsets[i + 1];
            rows = 0;
        }
    }
    ++shardCount;
    m->shard_offsets[shardCount] = m->rows;
    m->num_shards = shardCount;
    rebuild_shard_parts(m);
    return 1;
}

template<typename MatrixT>
__host__ __forceinline__ int build_shard_offsets_by_rows(const sharded<MatrixT> * __restrict__ m,
                                                         unsigned long target_rows_per_shard,
                                                         unsigned long **out_offsets,
                                                         unsigned long *out_count) {
    unsigned long *offsets = 0;
    unsigned long shard_count = 0;
    unsigned long used = 0;
    unsigned long rows = 0;
    unsigned long i = 0;

    if (out_offsets == 0 || out_count == 0) return 0;
    *out_offsets = 0;
    *out_count = 0;
    if (m->num_partitions == 0) return 1;

    offsets = (unsigned long *) std::calloc((std::size_t) (m->num_partitions + 1), sizeof(unsigned long));
    if (offsets == 0) return 0;
    offsets[0] = 0;

    if (target_rows_per_shard == 0) {
        for (i = 0; i <= m->num_partitions; ++i) offsets[i] = m->partition_offsets[i];
        *out_offsets = offsets;
        *out_count = m->num_partitions;
        return 1;
    }

    for (i = 0; i < m->num_partitions; ++i) {
        rows = m->partition_rows[i];
        if (used != 0 && used + rows > target_rows_per_shard) {
            ++shard_count;
            offsets[shard_count] = m->partition_offsets[i];
            used = 0;
        }
        used += rows;
    }
    ++shard_count;
    offsets[shard_count] = m->rows;
    *out_offsets = offsets;
    *out_count = shard_count;
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
        if (!row_is_partition_boundary(m, offsets[i])) return 0;
    }
    if (!row_is_partition_boundary(m, offsets[count])) return 0;
    if (!reserve_shards(m, count)) return 0;
    m->num_shards = count;
    for (i = 0; i <= count; ++i) m->shard_offsets[i] = offsets[i];
    rebuild_shard_parts(m);
    return 1;
}

template<typename MatrixT>
__host__ __forceinline__ int set_shards_by_nnz(sharded<MatrixT> * __restrict__ m, unsigned long max_nnz) {
    unsigned long used = 0;
    unsigned long shardCount = 0;
    unsigned long i = 0;

    if (max_nnz == 0) return set_shards_to_partitions(m);
    if (!reserve_shards(m, m->num_partitions)) return 0;

    m->shard_offsets[0] = 0;
    shardCount = 0;
    used = 0;
    for (i = 0; i < m->num_partitions; ++i) {
        if (m->partition_nnz[i] == 0) continue;
        if (used != 0 && used + m->partition_nnz[i] > max_nnz) {
            ++shardCount;
            m->shard_offsets[shardCount] = m->partition_offsets[i];
            used = 0;
        }
        used += m->partition_nnz[i];
    }

    if (m->num_partitions != 0) {
        ++shardCount;
        m->shard_offsets[shardCount] = m->rows;
    }
    m->num_shards = shardCount;
    rebuild_shard_parts(m);
    return 1;
}

template<typename MatrixT>
__host__ __forceinline__ int set_shards_by_part_bytes(sharded<MatrixT> * __restrict__ m, std::size_t max_bytes) {
    std::size_t used = 0;
    std::size_t partBytes = 0;
    unsigned long shardCount = 0;
    unsigned long i = 0;

    if (max_bytes == 0) return set_shards_to_partitions(m);
    if (!reserve_shards(m, m->num_partitions)) return 0;

    m->shard_offsets[0] = 0;
    shardCount = 0;
    used = 0;
    for (i = 0; i < m->num_partitions; ++i) {
        partBytes = partition_bytes(m, i);
        if (partBytes == 0) continue;
        if (used != 0 && used + partBytes > max_bytes) {
            ++shardCount;
            m->shard_offsets[shardCount] = m->partition_offsets[i];
            used = 0;
        }
        used += partBytes;
    }

    if (m->num_partitions != 0) {
        ++shardCount;
        m->shard_offsets[shardCount] = m->rows;
    }
    m->num_shards = shardCount;
    rebuild_shard_parts(m);
    return 1;
}

template<typename MatrixT>
__host__ __forceinline__ int fetch_partition(sharded<MatrixT> *m, const shard_storage *s, unsigned long partId) {
    (void) m;
    (void) s;
    (void) partId;
    return 0;
}

__host__ __forceinline__ int fetch_partition(sharded<sparse::compressed> *m, const shard_storage *s, unsigned long partId) {
    (void) m;
    (void) s;
    (void) partId;
    std::fprintf(stderr, "Error: legacy compressed .csh5 partition fetch is no longer supported\n");
    return 0;
}

__host__ __forceinline__ int fetch_partition(sharded<sparse::blocked_ell> *m, const shard_storage *s, unsigned long partId) {
    shard_storage *storage = const_cast<shard_storage *>(s);

    if (partId >= m->num_partitions || storage == 0 || storage->backend != shard_storage_backend_dataset_h5) return 0;
    return fetch_dataset_blocked_ell_h5_partition(m, s, partId);
}

__host__ __forceinline__ int fetch_partition(sharded<sparse::quantized_blocked_ell> *m, const shard_storage *s, unsigned long partId) {
    shard_storage *storage = const_cast<shard_storage *>(s);

    if (partId >= m->num_partitions || storage == 0 || storage->backend != shard_storage_backend_dataset_h5) return 0;
    return fetch_dataset_quantized_blocked_ell_h5_partition(m, s, partId);
}

__host__ __forceinline__ int fetch_partition(sharded<sparse::sliced_ell> *m, const shard_storage *s, unsigned long partId) {
    shard_storage *storage = const_cast<shard_storage *>(s);

    if (partId >= m->num_partitions || storage == 0 || storage->backend != shard_storage_backend_dataset_h5) return 0;
    return fetch_dataset_sliced_ell_h5_partition(m, s, partId);
}

template<typename MatrixT>
__host__ __forceinline__ int fetch_all_partitions(sharded<MatrixT> *m, const shard_storage *s) {
    (void) m;
    (void) s;
    return 0;
}

__host__ __forceinline__ int fetch_all_partitions(sharded<sparse::compressed> *m, const shard_storage *s) {
    (void) m;
    (void) s;
    std::fprintf(stderr, "Error: legacy compressed .csh5 partition fetch is no longer supported\n");
    return 0;
}

__host__ __forceinline__ int fetch_all_partitions(sharded<sparse::blocked_ell> *m, const shard_storage *s) {
    unsigned long i = 0;
    shard_storage *storage = const_cast<shard_storage *>(s);

    if (storage == 0 || storage->backend != shard_storage_backend_dataset_h5) return 0;
    for (i = 0; i < m->num_partitions; ++i) {
        if (!fetch_dataset_blocked_ell_h5_partition(m, s, i)) return 0;
    }
    if (m->num_shards == 0) return set_shards_to_partitions(m);
    return 1;
}

__host__ __forceinline__ int fetch_all_partitions(sharded<sparse::quantized_blocked_ell> *m, const shard_storage *s) {
    unsigned long i = 0;
    shard_storage *storage = const_cast<shard_storage *>(s);

    if (storage == 0 || storage->backend != shard_storage_backend_dataset_h5) return 0;
    for (i = 0; i < m->num_partitions; ++i) {
        if (!fetch_dataset_quantized_blocked_ell_h5_partition(m, s, i)) return 0;
    }
    if (m->num_shards == 0) return set_shards_to_partitions(m);
    return 1;
}

__host__ __forceinline__ int fetch_all_partitions(sharded<sparse::sliced_ell> *m, const shard_storage *s) {
    unsigned long i = 0;
    shard_storage *storage = const_cast<shard_storage *>(s);

    if (storage == 0 || storage->backend != shard_storage_backend_dataset_h5) return 0;
    for (i = 0; i < m->num_partitions; ++i) {
        if (!fetch_dataset_sliced_ell_h5_partition(m, s, i)) return 0;
    }
    if (m->num_shards == 0) return set_shards_to_partitions(m);
    return 1;
}

template<typename MatrixT>
__host__ __forceinline__ int fetch_shard(sharded<MatrixT> *m, const shard_storage *s, unsigned long shardId) {
    (void) m;
    (void) s;
    (void) shardId;
    return 0;
}

__host__ __forceinline__ int fetch_shard(sharded<sparse::compressed> *m, const shard_storage *s, unsigned long shardId) {
    (void) m;
    (void) s;
    (void) shardId;
    std::fprintf(stderr, "Error: legacy compressed .csh5 shard fetch is no longer supported\n");
    return 0;
}

__host__ __forceinline__ int fetch_shard(sharded<sparse::blocked_ell> *m, const shard_storage *s, unsigned long shardId) {
    shard_storage *storage = const_cast<shard_storage *>(s);

    if (shardId >= m->num_shards || storage == 0 || storage->backend != shard_storage_backend_dataset_h5) return 0;
    return fetch_dataset_blocked_ell_h5_shard(m, s, shardId);
}

__host__ __forceinline__ int fetch_shard(sharded<sparse::quantized_blocked_ell> *m, const shard_storage *s, unsigned long shardId) {
    shard_storage *storage = const_cast<shard_storage *>(s);

    if (shardId >= m->num_shards || storage == 0 || storage->backend != shard_storage_backend_dataset_h5) return 0;
    return fetch_dataset_quantized_blocked_ell_h5_shard(m, s, shardId);
}

__host__ __forceinline__ int fetch_shard(sharded<sparse::sliced_ell> *m, const shard_storage *s, unsigned long shardId) {
    shard_storage *storage = const_cast<shard_storage *>(s);

    if (shardId >= m->num_shards || storage == 0 || storage->backend != shard_storage_backend_dataset_h5) return 0;
    return fetch_dataset_sliced_ell_h5_shard(m, s, shardId);
}

template<typename MatrixT>
__host__ __forceinline__ int drop_partition(sharded<MatrixT> *m, unsigned long partId) {
    if (partId >= m->num_partitions) return 0;
    // Drop only releases the host materialization for this part. Packfile bytes
    // remain on disk, and any device residency is managed separately.
    destroy(m->parts[partId]);
    m->parts[partId] = 0;
    return 1;
}

template<typename MatrixT>
__host__ __forceinline__ int drop_all_partitions(sharded<MatrixT> *m) {
    unsigned long i = 0;
    for (i = 0; i < m->num_partitions; ++i) {
        if (!drop_partition(m, i)) return 0;
    }
    return 1;
}

template<typename MatrixT>
__host__ __forceinline__ int drop_shard(sharded<MatrixT> *m, unsigned long shardId) {
    unsigned long begin = 0;
    unsigned long end = 0;
    unsigned long i = 0;

    if (shardId >= m->num_shards) return 0;
    begin = first_partition_in_shard(m, shardId);
    end = last_partition_in_shard(m, shardId);
    for (i = begin; i < end; ++i) {
        if (!drop_partition(m, i)) return 0;
    }
    return 1;
}

} // namespace cellshard
