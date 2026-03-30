#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>

#include "../../matrix.cuh"

namespace cellshard {

struct shard_storage {
    unsigned int capacity;
    char **paths;
};

namespace detail {

inline int sharded_write_block(std::FILE *fp, const void *ptr, std::size_t elem_size, std::size_t count) {
    if (count == 0) return 1;
    return std::fwrite(ptr, elem_size, count, fp) == count;
}

inline int sharded_read_block(std::FILE *fp, void *ptr, std::size_t elem_size, std::size_t count) {
    if (count == 0) return 1;
    return std::fread(ptr, elem_size, count, fp) == count;
}

inline int sharded_to_u64(unsigned long value, std::uint64_t *out, const char *label, const char *filename) {
    *out = (std::uint64_t) value;
    if ((unsigned long) *out != value) {
        std::fprintf(stderr, "Error: %s out of disk u64 range in %s\n", label, filename);
        return 0;
    }
    return 1;
}

inline int sharded_from_u64(std::uint64_t value, unsigned long *out, const char *label, const char *filename) {
    *out = (unsigned long) value;
    if ((std::uint64_t) *out != value) {
        std::fprintf(stderr, "Error: %s does not fit target sharded index type in %s\n", label, filename);
        return 0;
    }
    return 1;
}

inline int store_sharded_index_array(std::FILE *fp, const unsigned long *src, std::size_t count, const char *label, const char *filename) {
    std::uint64_t value = 0;
    std::size_t i = 0;

    for (i = 0; i < count; ++i) {
        if (!sharded_to_u64(src[i], &value, label, filename)) return 0;
        if (!sharded_write_block(fp, &value, sizeof(value), 1)) return 0;
    }
    return 1;
}

inline int load_sharded_index_array(std::FILE *fp, unsigned long *dst, std::size_t count, const char *label, const char *filename) {
    std::uint64_t value = 0;
    std::size_t i = 0;

    for (i = 0; i < count; ++i) {
        if (!sharded_read_block(fp, &value, sizeof(value), 1)) return 0;
        if (!sharded_from_u64(value, dst + i, label, filename)) return 0;
    }
    return 1;
}

template<typename MatrixT>
inline int store_sharded_header_typed(const char *filename, const sharded<MatrixT> *m) {
    static const unsigned char magic[8] = { 'C', 'S', 'H', 'R', 'D', '0', '1', '\0' };
    std::FILE *fp = 0;
    std::uint64_t rows = 0;
    std::uint64_t cols = 0;
    std::uint64_t nnz = 0;
    std::uint64_t num_parts = 0;
    std::uint64_t num_shards = 0;
    int ok = 0;

    fp = std::fopen(filename, "wb");
    if (fp == 0) return 0;
    if (!sharded_to_u64(m->rows, &rows, "rows", filename)) goto done;
    if (!sharded_to_u64(m->cols, &cols, "cols", filename)) goto done;
    if (!sharded_to_u64(m->nnz, &nnz, "nnz", filename)) goto done;
    if (!sharded_to_u64(m->num_parts, &num_parts, "num_parts", filename)) goto done;
    if (!sharded_to_u64(m->num_shards, &num_shards, "num_shards", filename)) goto done;
    if (!sharded_write_block(fp, magic, sizeof(magic), 1)) goto done;
    if (!sharded_write_block(fp, &m->format, sizeof(m->format), 1)) goto done;
    if (!sharded_write_block(fp, &rows, sizeof(rows), 1)) goto done;
    if (!sharded_write_block(fp, &cols, sizeof(cols), 1)) goto done;
    if (!sharded_write_block(fp, &nnz, sizeof(nnz), 1)) goto done;
    if (!sharded_write_block(fp, &num_parts, sizeof(num_parts), 1)) goto done;
    if (!sharded_write_block(fp, &num_shards, sizeof(num_shards), 1)) goto done;
    if (!store_sharded_index_array(fp, m->part_rows, (std::size_t) m->num_parts, "part_rows", filename)) goto done;
    if (!store_sharded_index_array(fp, m->part_nnz, (std::size_t) m->num_parts, "part_nnz", filename)) goto done;
    if (!store_sharded_index_array(fp, m->part_aux, (std::size_t) m->num_parts, "part_aux", filename)) goto done;
    if (!store_sharded_index_array(fp, m->shard_offsets, (std::size_t) (m->num_shards + 1), "shard_offsets", filename)) goto done;
    ok = 1;

done:
    std::fclose(fp);
    return ok;
}

template<typename MatrixT>
inline int load_sharded_header_typed(const char *filename, sharded<MatrixT> *m) {
    static const unsigned char magic[8] = { 'C', 'S', 'H', 'R', 'D', '0', '1', '\0' };
    unsigned char got_magic[8];
    std::FILE *fp = 0;
    std::uint64_t rows = 0;
    std::uint64_t cols = 0;
    std::uint64_t nnz = 0;
    std::uint64_t num_parts = 0;
    std::uint64_t num_shards = 0;
    int ok = 0;

    fp = std::fopen(filename, "rb");
    if (fp == 0) return 0;
    if (!sharded_read_block(fp, got_magic, sizeof(got_magic), 1)) goto done;
    if (std::memcmp(got_magic, magic, sizeof(magic)) != 0) goto done;
    clear(m);
    init(m);
    if (!sharded_read_block(fp, &m->format, sizeof(m->format), 1)) goto done;
    if (!sharded_read_block(fp, &rows, sizeof(rows), 1)) goto done;
    if (!sharded_read_block(fp, &cols, sizeof(cols), 1)) goto done;
    if (!sharded_read_block(fp, &nnz, sizeof(nnz), 1)) goto done;
    if (!sharded_read_block(fp, &num_parts, sizeof(num_parts), 1)) goto done;
    if (!sharded_read_block(fp, &num_shards, sizeof(num_shards), 1)) goto done;
    if (!sharded_from_u64(rows, &m->rows, "rows", filename)) goto done;
    if (!sharded_from_u64(cols, &m->cols, "cols", filename)) goto done;
    if (!sharded_from_u64(nnz, &m->nnz, "nnz", filename)) goto done;
    if (!sharded_from_u64(num_parts, &m->num_parts, "num_parts", filename)) goto done;
    if (!sharded_from_u64(num_shards, &m->num_shards, "num_shards", filename)) goto done;
    m->part_capacity = m->num_parts;
    m->shard_capacity = m->num_shards;

    if (m->part_capacity != 0) {
        m->parts = (MatrixT **) std::calloc((std::size_t) m->part_capacity, sizeof(MatrixT *));
        m->part_offsets = (unsigned long *) std::calloc((std::size_t) (m->part_capacity + 1), sizeof(unsigned long));
        m->part_rows = (unsigned long *) std::calloc((std::size_t) m->part_capacity, sizeof(unsigned long));
        m->part_nnz = (unsigned long *) std::calloc((std::size_t) m->part_capacity, sizeof(unsigned long));
        m->part_aux = (unsigned long *) std::calloc((std::size_t) m->part_capacity, sizeof(unsigned long));
        if (m->parts == 0 || m->part_offsets == 0 || m->part_rows == 0 || m->part_nnz == 0 || m->part_aux == 0) {
            clear(m);
            goto done;
        }
    }
    if (m->shard_capacity != 0) {
        m->shard_offsets = (unsigned long *) std::calloc((std::size_t) (m->shard_capacity + 1), sizeof(unsigned long));
        if (m->shard_offsets == 0) {
            clear(m);
            goto done;
        }
    }

    if (!load_sharded_index_array(fp, m->part_rows, (std::size_t) m->num_parts, "part_rows", filename)) goto done;
    if (!load_sharded_index_array(fp, m->part_nnz, (std::size_t) m->num_parts, "part_nnz", filename)) goto done;
    if (!load_sharded_index_array(fp, m->part_aux, (std::size_t) m->num_parts, "part_aux", filename)) goto done;
    if (!load_sharded_index_array(fp, m->shard_offsets, (std::size_t) (m->num_shards + 1), "shard_offsets", filename)) goto done;
    rebuild_part_offsets(m);
    ok = 1;

done:
    if (!ok) clear(m);
    std::fclose(fp);
    return ok;
}

struct dense_load_result {
    header h;
    void *val;
};

struct csr_load_result {
    header h;
    unsigned int *rowPtr;
    unsigned int *colIdx;
    void *val;
};

struct coo_load_result {
    header h;
    unsigned int *rowIdx;
    unsigned int *colIdx;
    void *val;
};

struct dia_load_result {
    header h;
    unsigned int num_diagonals;
    int *offsets;
    void *val;
};

struct sharded_header_load_result {
    header h;
    unsigned int num_parts;
    unsigned int num_shards;
    unsigned int *part_rows;
    unsigned int *part_nnz;
    unsigned int *part_aux;
    unsigned int *shard_offsets;
};

void init(shard_storage *s);
void clear(shard_storage *s);
int reserve(shard_storage *s, unsigned int capacity);
int bind(shard_storage *s, unsigned int partId, const char *path);
int bind_sequential(shard_storage *s, unsigned int count, const char *prefix);

int store_dense_raw(const char *filename, unsigned int rows, unsigned int cols, unsigned int nnz, const void *val, std::size_t value_size);
int load_dense_raw(const char *filename, std::size_t value_size, dense_load_result *out);

int store_csr_raw(const char *filename, unsigned int rows, unsigned int cols, unsigned int nnz, const unsigned int *rowPtr, const unsigned int *colIdx, const void *val, std::size_t value_size);
int load_csr_raw(const char *filename, std::size_t value_size, csr_load_result *out);

int store_coo_raw(const char *filename, unsigned int rows, unsigned int cols, unsigned int nnz, const unsigned int *rowIdx, const unsigned int *colIdx, const void *val, std::size_t value_size);
int load_coo_raw(const char *filename, std::size_t value_size, coo_load_result *out);

int store_dia_raw(const char *filename, unsigned int rows, unsigned int cols, unsigned int nnz, unsigned int num_diagonals, const int *offsets, const void *val, std::size_t value_size);
int load_dia_raw(const char *filename, std::size_t value_size, dia_load_result *out);

int store_sharded_header_raw(const char *filename,
                             unsigned char format,
                             unsigned int rows,
                             unsigned int cols,
                             unsigned int nnz,
                             unsigned int num_parts,
                             unsigned int num_shards,
                             const unsigned int *part_rows,
                             const unsigned int *part_nnz,
                             const unsigned int *part_aux,
                             const unsigned int *shard_offsets);

int load_sharded_header_raw(const char *filename, sharded_header_load_result *out);

} // namespace detail

inline void init(shard_storage *s) {
    detail::init(s);
}

inline void clear(shard_storage *s) {
    detail::clear(s);
}

inline int reserve(shard_storage *s, unsigned int capacity) {
    return detail::reserve(s, capacity);
}

inline int bind(shard_storage *s, unsigned int partId, const char *path) {
    return detail::bind(s, partId, path);
}

inline int bind_sequential(shard_storage *s, unsigned int count, const char *prefix) {
    return detail::bind_sequential(s, count, prefix);
}

inline int store(const char *filename, const dense *m) {
    return detail::store_dense_raw(filename, m->rows, m->cols, m->nnz, m->val, sizeof(__half));
}

inline int load(const char *filename, dense *m) {
    detail::dense_load_result tmp;

    tmp.val = 0;
    if (!detail::load_dense_raw(filename, sizeof(__half), &tmp)) return 0;
    clear(m);
    init(m, tmp.h.rows, tmp.h.cols);
    m->nnz = tmp.h.nnz;
    m->val = (__half *) tmp.val;
    return 1;
}

inline int store(const char *filename, const sparse::csr *m) {
    return detail::store_csr_raw(filename, m->rows, m->cols, m->nnz, m->rowPtr, m->colIdx, m->val, sizeof(__half));
}

inline int load(const char *filename, sparse::csr *m) {
    detail::csr_load_result tmp;

    tmp.rowPtr = 0;
    tmp.colIdx = 0;
    tmp.val = 0;
    if (!detail::load_csr_raw(filename, sizeof(__half), &tmp)) return 0;
    sparse::clear(m);
    sparse::init(m, tmp.h.rows, tmp.h.cols, tmp.h.nnz);
    m->rowPtr = tmp.rowPtr;
    m->colIdx = tmp.colIdx;
    m->val = (__half *) tmp.val;
    return 1;
}

inline int store(const char *filename, const sparse::coo *m) {
    return detail::store_coo_raw(filename, m->rows, m->cols, m->nnz, m->rowIdx, m->colIdx, m->val, sizeof(__half));
}

inline int load(const char *filename, sparse::coo *m) {
    detail::coo_load_result tmp;

    tmp.rowIdx = 0;
    tmp.colIdx = 0;
    tmp.val = 0;
    if (!detail::load_coo_raw(filename, sizeof(__half), &tmp)) return 0;
    sparse::clear(m);
    sparse::init(m, tmp.h.rows, tmp.h.cols, tmp.h.nnz);
    m->rowIdx = tmp.rowIdx;
    m->colIdx = tmp.colIdx;
    m->val = (__half *) tmp.val;
    return 1;
}

inline int store(const char *filename, const sparse::dia *m) {
    return detail::store_dia_raw(filename, m->rows, m->cols, m->nnz, m->num_diagonals, m->offsets, m->val, sizeof(__half));
}

inline int load(const char *filename, sparse::dia *m) {
    detail::dia_load_result tmp;

    tmp.num_diagonals = 0;
    tmp.offsets = 0;
    tmp.val = 0;
    if (!detail::load_dia_raw(filename, sizeof(__half), &tmp)) return 0;
    sparse::clear(m);
    sparse::init(m, tmp.h.rows, tmp.h.cols, tmp.h.nnz);
    m->num_diagonals = tmp.num_diagonals;
    m->offsets = tmp.offsets;
    m->val = (__half *) tmp.val;
    return 1;
}

template<typename MatrixT>
inline int store_header(const char *filename, const sharded<MatrixT> *m) {
    return detail::store_sharded_header_typed(filename, m);
}

template<typename MatrixT>
inline int store(const char *filename, const sharded<MatrixT> *m, const shard_storage *s) {
    unsigned long i = 0;

    if (s == 0 || s->capacity < m->num_parts) return 0;
    if (!store_header(filename, m)) return 0;

    for (i = 0; i < m->num_parts; ++i) {
        if (m->parts[i] == 0 || s->paths[i] == 0) return 0;
        if (!store(s->paths[i], m->parts[i])) return 0;
    }
    return 1;
}

template<typename MatrixT>
inline int load_header(const char *filename, sharded<MatrixT> *m) {
    return detail::load_sharded_header_typed(filename, m);
}

template<typename MatrixT>
inline int fetch_part(sharded<MatrixT> *m, const shard_storage *s, unsigned long partId) {
    MatrixT *part = 0;
    int ok = 0;

    if (partId >= m->num_parts || s == 0 || partId >= s->capacity || s->paths[partId] == 0) return 0;
    if (m->parts[partId] != 0) destroy(m->parts[partId]);
    m->parts[partId] = 0;

    part = new MatrixT;
    init(part);
    if (!load(s->paths[partId], part)) {
        destroy(part);
        return 0;
    }
    if (part->rows != m->part_rows[partId]) goto fail;
    if (part->nnz != m->part_nnz[partId]) goto fail;
    if (m->cols != 0 && part->cols != m->cols) goto fail;
    if (part->format != m->format) goto fail;
    if (::matrix::part_aux(part) != m->part_aux[partId]) goto fail;
    m->parts[partId] = part;
    ok = 1;

fail:
    if (!ok) destroy(part);
    return ok;
}

template<typename MatrixT>
inline int fetch_all_parts(sharded<MatrixT> *m, const shard_storage *s) {
    unsigned long i = 0;
    for (i = 0; i < m->num_parts; ++i) {
        if (!fetch_part(m, s, i)) return 0;
    }
    if (m->num_shards == 0) return set_shards_to_parts(m);
    return 1;
}

template<typename MatrixT>
inline int fetch_shard(sharded<MatrixT> *m, const shard_storage *s, unsigned long shardId) {
    unsigned long begin = 0;
    unsigned long end = 0;
    unsigned long i = 0;

    if (shardId >= m->num_shards) return 0;
    begin = first_part_in_shard(m, shardId);
    end = last_part_in_shard(m, shardId);
    for (i = begin; i < end; ++i) {
        if (!fetch_part(m, s, i)) return 0;
    }
    return 1;
}

template<typename MatrixT>
inline int drop_part(sharded<MatrixT> *m, unsigned long partId) {
    if (partId >= m->num_parts) return 0;
    destroy(m->parts[partId]);
    m->parts[partId] = 0;
    return 1;
}

template<typename MatrixT>
inline int drop_shard(sharded<MatrixT> *m, unsigned long shardId) {
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
