#pragma once

#include "sharded.cuh"
#include "shard_paths.cuh"
#include "series_h5.cuh"
#include "../disk/matrix.cuh"

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
        for (i = 0; i < m->num_parts; ++i) destroy(m->parts[i]);
    }
    std::free(m->parts);
    std::free(m->part_offsets);
    std::free(m->part_rows);
    std::free(m->part_nnz);
    std::free(m->part_aux);
    std::free(m->shard_offsets);
    std::free(m->shard_parts);
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
    if (m->shard_parts == 0 || m->shard_offsets == 0 || m->part_offsets == 0) return;
    for (shard = 0; shard < m->num_shards; ++shard) {
        const unsigned long row_begin = m->shard_offsets[shard];
        while (part < m->num_parts && m->part_offsets[part] < row_begin) ++part;
        m->shard_parts[shard] = part;
    }
    m->shard_parts[m->num_shards] = m->num_parts;
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
__host__ __forceinline__ int define_parts(sharded<MatrixT> * __restrict__ m,
                                          unsigned long cols,
                                          unsigned long num_parts,
                                          const unsigned long * __restrict__ part_rows,
                                          const unsigned long * __restrict__ part_nnz,
                                          const unsigned long * __restrict__ part_aux_in) {
    unsigned long i = 0;

    clear(m);
    init(m);
    if (!reserve_parts(m, num_parts)) return 0;

    m->cols = cols;
    m->num_parts = num_parts;
    for (i = 0; i < num_parts; ++i) {
        m->parts[i] = 0;
        m->part_rows[i] = part_rows != 0 ? part_rows[i] : 0;
        m->part_nnz[i] = part_nnz != 0 ? part_nnz[i] : 0;
        m->part_aux[i] = part_aux_in != 0 ? part_aux_in[i] : 0;
    }
    rebuild_part_offsets(m);
    return set_shards_to_parts(m);
}

template<typename MatrixT>
__host__ __forceinline__ int set_shards_to_parts(sharded<MatrixT> * __restrict__ m) {
    unsigned long i = 0;
    if (!reserve_shards(m, m->num_parts)) return 0;
    m->num_shards = m->num_parts;
    for (i = 0; i <= m->num_parts; ++i) m->shard_offsets[i] = m->part_offsets[i];
    for (i = 0; i <= m->num_parts; ++i) m->shard_parts[i] = i;
    return 1;
}

template<typename MatrixT>
__host__ __forceinline__ int append_part(sharded<MatrixT> * __restrict__ m, MatrixT *part) {
    unsigned long next = 0;

    if (m->num_parts == m->part_capacity) {
        // Capacity growth copies metadata arrays. The part payload pointer is
        // inserted directly; append_part does not clone MatrixT payload.
        next = m->part_capacity == 0 ? 4 : m->part_capacity << 1;
        if (!reserve_parts(m, next)) return 0;
    }
    m->parts[m->num_parts] = part;
    m->part_rows[m->num_parts] = part != 0 ? part->rows : 0;
    m->part_nnz[m->num_parts] = part != 0 ? ::cellshard::part_nnz(part) : 0;
    m->part_aux[m->num_parts] = part != 0 ? ::cellshard::part_aux(part) : 0;
    ++m->num_parts;
    if (part != 0 && m->cols == 0) m->cols = part->cols;
    rebuild_part_offsets(m);
    return set_shards_to_parts(m);
}

template<typename MatrixT>
__host__ __forceinline__ int concatenate(sharded<MatrixT> * __restrict__ dst, sharded<MatrixT> * __restrict__ src) {
    unsigned long i = 0;

    if (src->num_parts == 0) return 1;
    // This is metadata/pointer movement, not deep copy. Ownership of src->parts
    // transfers into dst one pointer at a time.
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
    if (m->num_parts == 0) {
        m->num_shards = 0;
        return 1;
    }
    if (count >= m->num_parts) return set_shards_to_parts(m);
    if (!reserve_shards(m, count)) return 0;
    target = (m->rows + count - 1) / count;
    m->shard_offsets[0] = 0;
    shardCount = 0;
    rows = 0;
    for (i = 0; i < m->num_parts; ++i) {
        const unsigned long parts_left = m->num_parts - (i + 1);
        const unsigned long shards_left = count - (shardCount + 1);
        rows += m->part_rows[i];
        if (shards_left == 0) continue;
        if (rows >= target && parts_left >= shards_left) {
            ++shardCount;
            m->shard_offsets[shardCount] = m->part_offsets[i + 1];
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
    if (m->num_parts == 0) return 1;

    offsets = (unsigned long *) std::calloc((std::size_t) (m->num_parts + 1), sizeof(unsigned long));
    if (offsets == 0) return 0;
    offsets[0] = 0;

    if (target_rows_per_shard == 0) {
        for (i = 0; i <= m->num_parts; ++i) offsets[i] = m->part_offsets[i];
        *out_offsets = offsets;
        *out_count = m->num_parts;
        return 1;
    }

    for (i = 0; i < m->num_parts; ++i) {
        rows = m->part_rows[i];
        if (used != 0 && used + rows > target_rows_per_shard) {
            ++shard_count;
            offsets[shard_count] = m->part_offsets[i];
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
        if (!row_is_part_boundary(m, offsets[i])) return 0;
    }
    if (!row_is_part_boundary(m, offsets[count])) return 0;
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

    if (max_nnz == 0) return set_shards_to_parts(m);
    if (!reserve_shards(m, m->num_parts)) return 0;

    m->shard_offsets[0] = 0;
    shardCount = 0;
    used = 0;
    for (i = 0; i < m->num_parts; ++i) {
        if (m->part_nnz[i] == 0) continue;
        if (used != 0 && used + m->part_nnz[i] > max_nnz) {
            ++shardCount;
            m->shard_offsets[shardCount] = m->part_offsets[i];
            used = 0;
        }
        used += m->part_nnz[i];
    }

    if (m->num_parts != 0) {
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
    rebuild_shard_parts(m);
    return 1;
}

template<typename MatrixT>
__host__ __forceinline__ int load_part_from_open_packfile(sharded<MatrixT> *m,
                                                          shard_storage *s,
                                                          unsigned long partId,
                                                          std::uint64_t *cursor) {
    MatrixT *part = 0;
    int ok = 0;
    const std::uint64_t offset = s->locators[partId].offset;

    if (partId >= m->num_parts || s == 0 || partId >= s->capacity || s->packfile_fp == 0 || s->locators == 0) return 0;
    if (m->parts[partId] != 0) destroy(m->parts[partId]);
    m->parts[partId] = 0;

    // This is the host-side fast path for staged reads:
    // - reuse one open packfile handle
    // - avoid reopening the packfile per part
    // - avoid reseeking when shard parts were stored contiguously
    // - keep the decode path identical once the file cursor is in place
    part = new MatrixT;
    init(part);
    if (cursor == 0 || *cursor != offset) {
        if (fseeko(s->packfile_fp, (off_t) offset, SEEK_SET) != 0) {
            destroy(part);
            return 0;
        }
        if (cursor != 0) *cursor = offset;
    }
    if (!load(s->packfile_fp, part)) {
        destroy(part);
        return 0;
    }
    if (cursor != 0) *cursor = offset + s->locators[partId].bytes;
    if (part->rows != m->part_rows[partId]) goto fail;
    if (::cellshard::part_nnz(part) != m->part_nnz[partId]) goto fail;
    if (m->cols != 0 && part->cols != m->cols) goto fail;
    if (::cellshard::part_aux(part) != m->part_aux[partId]) goto fail;
    m->parts[partId] = part;
    ok = 1;

fail:
    if (!ok) destroy(part);
    return ok;
}

template<typename MatrixT>
__host__ __forceinline__ int fetch_part(sharded<MatrixT> *m, const shard_storage *s, unsigned long partId) {
    std::uint64_t cursor = 0;
    shard_storage *storage = const_cast<shard_storage *>(s);
    // Synchronous host materialization from packfile. This is explicit I/O and
    // allocation work, not a cheap metadata operation.
    if (partId >= m->num_parts || storage == 0 || partId >= storage->capacity || storage->packfile_path == 0 || storage->locators == 0) return 0;
    if (!ensure_packfile_open(storage)) return 0;
    return load_part_from_open_packfile(m, storage, partId, &cursor);
}

__host__ __forceinline__ int fetch_part(sharded<sparse::compressed> *m, const shard_storage *s, unsigned long partId) {
    std::uint64_t cursor = 0;
    shard_storage *storage = const_cast<shard_storage *>(s);

    if (partId >= m->num_parts || storage == 0 || storage->packfile_path == 0) return 0;
    if (storage->backend == shard_storage_backend_series_h5) {
        return fetch_series_compressed_h5_part(m, s, partId);
    }
    if (partId >= storage->capacity || storage->locators == 0) return 0;
    if (!ensure_packfile_open(storage)) return 0;
    return load_part_from_open_packfile(m, storage, partId, &cursor);
}

__host__ __forceinline__ int fetch_part(sharded<sparse::blocked_ell> *m, const shard_storage *s, unsigned long partId) {
    std::uint64_t cursor = 0;
    shard_storage *storage = const_cast<shard_storage *>(s);

    if (partId >= m->num_parts || storage == 0 || storage->packfile_path == 0) return 0;
    if (storage->backend == shard_storage_backend_series_h5) {
        return fetch_series_blocked_ell_h5_part(m, s, partId);
    }
    if (partId >= storage->capacity || storage->locators == 0) return 0;
    if (!ensure_packfile_open(storage)) return 0;
    return load_part_from_open_packfile(m, storage, partId, &cursor);
}

template<typename MatrixT>
__host__ __forceinline__ int fetch_all_parts(sharded<MatrixT> *m, const shard_storage *s) {
    unsigned long i = 0;
    std::uint64_t cursor = 0;
    shard_storage *storage = const_cast<shard_storage *>(s);
    if (storage == 0 || storage->packfile_path == 0 || storage->locators == 0) return 0;
    if (!ensure_packfile_open(storage)) return 0;
    for (i = 0; i < m->num_parts; ++i) {
        if (!load_part_from_open_packfile(m, storage, i, &cursor)) return 0;
    }
    if (m->num_shards == 0) return set_shards_to_parts(m);
    return 1;
}

__host__ __forceinline__ int fetch_all_parts(sharded<sparse::compressed> *m, const shard_storage *s) {
    unsigned long i = 0;
    shard_storage *storage = const_cast<shard_storage *>(s);

    if (storage == 0 || storage->packfile_path == 0) return 0;
    if (storage->backend == shard_storage_backend_series_h5) {
        for (i = 0; i < m->num_parts; ++i) {
            if (!fetch_series_compressed_h5_part(m, s, i)) return 0;
        }
        if (m->num_shards == 0) return set_shards_to_parts(m);
        return 1;
    }

    {
        std::uint64_t cursor = 0;
        if (storage->locators == 0) return 0;
        if (!ensure_packfile_open(storage)) return 0;
        for (i = 0; i < m->num_parts; ++i) {
            if (!load_part_from_open_packfile(m, storage, i, &cursor)) return 0;
        }
    }
    if (m->num_shards == 0) return set_shards_to_parts(m);
    return 1;
}

__host__ __forceinline__ int fetch_all_parts(sharded<sparse::blocked_ell> *m, const shard_storage *s) {
    unsigned long i = 0;
    shard_storage *storage = const_cast<shard_storage *>(s);

    if (storage == 0 || storage->packfile_path == 0) return 0;
    if (storage->backend == shard_storage_backend_series_h5) {
        for (i = 0; i < m->num_parts; ++i) {
            if (!fetch_series_blocked_ell_h5_part(m, s, i)) return 0;
        }
        if (m->num_shards == 0) return set_shards_to_parts(m);
        return 1;
    }

    {
        std::uint64_t cursor = 0;
        if (storage->locators == 0) return 0;
        if (!ensure_packfile_open(storage)) return 0;
        for (i = 0; i < m->num_parts; ++i) {
            if (!load_part_from_open_packfile(m, storage, i, &cursor)) return 0;
        }
    }
    if (m->num_shards == 0) return set_shards_to_parts(m);
    return 1;
}

template<typename MatrixT>
__host__ __forceinline__ int fetch_shard(sharded<MatrixT> *m, const shard_storage *s, unsigned long shardId) {
    unsigned long begin = 0;
    unsigned long end = 0;
    unsigned long i = 0;
    std::uint64_t cursor = 0;
    shard_storage *storage = const_cast<shard_storage *>(s);

    if (shardId >= m->num_shards || storage == 0 || storage->packfile_path == 0 || storage->locators == 0) return 0;
    if (!ensure_packfile_open(storage)) return 0;
    // Shard fetch is a simple loop over part fetches. Cost scales with parts
    // per shard plus packfile seek/read behavior.
    begin = first_part_in_shard(m, shardId);
    end = last_part_in_shard(m, shardId);
    for (i = begin; i < end; ++i) {
        if (!load_part_from_open_packfile(m, storage, i, &cursor)) return 0;
    }
    return 1;
}

__host__ __forceinline__ int fetch_shard(sharded<sparse::compressed> *m, const shard_storage *s, unsigned long shardId) {
    unsigned long begin = 0;
    unsigned long end = 0;
    unsigned long i = 0;
    std::uint64_t cursor = 0;
    shard_storage *storage = const_cast<shard_storage *>(s);

    if (shardId >= m->num_shards || storage == 0 || storage->packfile_path == 0) return 0;
    if (storage->backend == shard_storage_backend_series_h5) {
        return fetch_series_compressed_h5_shard(m, s, shardId);
    }
    if (storage->locators == 0) return 0;
    if (!ensure_packfile_open(storage)) return 0;
    begin = first_part_in_shard(m, shardId);
    end = last_part_in_shard(m, shardId);
    for (i = begin; i < end; ++i) {
        if (!load_part_from_open_packfile(m, storage, i, &cursor)) return 0;
    }
    return 1;
}

__host__ __forceinline__ int fetch_shard(sharded<sparse::blocked_ell> *m, const shard_storage *s, unsigned long shardId) {
    unsigned long begin = 0;
    unsigned long end = 0;
    unsigned long i = 0;
    std::uint64_t cursor = 0;
    shard_storage *storage = const_cast<shard_storage *>(s);

    if (shardId >= m->num_shards || storage == 0 || storage->packfile_path == 0) return 0;
    if (storage->backend == shard_storage_backend_series_h5) {
        begin = first_part_in_shard(m, shardId);
        end = last_part_in_shard(m, shardId);
        for (i = begin; i < end; ++i) {
            if (!fetch_series_blocked_ell_h5_part(m, s, i)) return 0;
        }
        return 1;
    }
    if (storage->locators == 0) return 0;
    if (!ensure_packfile_open(storage)) return 0;
    begin = first_part_in_shard(m, shardId);
    end = last_part_in_shard(m, shardId);
    for (i = begin; i < end; ++i) {
        if (!load_part_from_open_packfile(m, storage, i, &cursor)) return 0;
    }
    return 1;
}

template<typename MatrixT>
__host__ __forceinline__ int drop_part(sharded<MatrixT> *m, unsigned long partId) {
    if (partId >= m->num_parts) return 0;
    // Drop only releases the host materialization for this part. Packfile bytes
    // remain on disk, and any device residency is managed separately.
    destroy(m->parts[partId]);
    m->parts[partId] = 0;
    return 1;
}

template<typename MatrixT>
__host__ __forceinline__ int drop_all_parts(sharded<MatrixT> *m) {
    unsigned long i = 0;
    for (i = 0; i < m->num_parts; ++i) {
        if (!drop_part(m, i)) return 0;
    }
    return 1;
}

template<typename MatrixT>
__host__ __forceinline__ int drop_shard(sharded<MatrixT> *m, unsigned long shardId) {
    unsigned long begin = 0;
    unsigned long end = 0;
    unsigned long i = 0;

    if (shardId >= m->num_shards) return 0;
    begin = first_part_in_shard(m, shardId);
    end = last_part_in_shard(m, shardId);
    for (i = begin; i < end; ++i) {
        if (!drop_part(m, i)) return 0;
    }
    return 1;
}

} // namespace cellshard
