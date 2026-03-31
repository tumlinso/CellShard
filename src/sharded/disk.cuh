#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "../formats/dense.cuh"
#include "../formats/compressed.cuh"
#include "../formats/triplet.cuh"
#include "../formats/diagonal.cuh"
#include "sharded.cuh"
#include "sharded_host.cuh"
#include "../disk/matrix.cuh"
#include "shard_paths.cuh"

namespace cellshard {

template<typename MatrixT>
inline int check_sharded_disk_format(unsigned char actual) {
    return check_disk_format((unsigned char) disk_format_code<MatrixT>::value, actual, disk_format_code<MatrixT>::name());
}

struct sharded_header_load_result {
    disk_header h;
    unsigned int num_parts;
    unsigned int num_shards;
    unsigned int *part_rows;
    unsigned int *part_nnz;
    unsigned int *part_aux;
    unsigned int *shard_offsets;
};

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

inline int write_sharded_block(std::FILE *fp, const void *ptr, std::size_t elem_size, std::size_t count) {
    if (count == 0) return 1;
    return std::fwrite(ptr, elem_size, count, fp) == count;
}

inline int read_sharded_block(std::FILE *fp, void *ptr, std::size_t elem_size, std::size_t count) {
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
        if (!write_sharded_block(fp, &value, sizeof(value), 1)) return 0;
    }
    return 1;
}

inline int load_sharded_index_array(std::FILE *fp, unsigned long *dst, std::size_t count, const char *label, const char *filename) {
    std::uint64_t value = 0;
    std::size_t i = 0;

    for (i = 0; i < count; ++i) {
        if (!read_sharded_block(fp, &value, sizeof(value), 1)) return 0;
        if (!sharded_from_u64(value, dst + i, label, filename)) return 0;
    }
    return 1;
}

template<typename MatrixT>
inline int store_header(const char *filename, const sharded<MatrixT> *m) {
    static const unsigned char magic[8] = { 'C', 'S', 'H', 'R', 'D', '0', '1', '\0' };
    std::FILE *fp = 0;
    const unsigned char format = (unsigned char) disk_format_code<MatrixT>::value;
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
    if (!write_sharded_block(fp, magic, sizeof(magic), 1)) goto done;
    if (!write_sharded_block(fp, &format, sizeof(format), 1)) goto done;
    if (!write_sharded_block(fp, &rows, sizeof(rows), 1)) goto done;
    if (!write_sharded_block(fp, &cols, sizeof(cols), 1)) goto done;
    if (!write_sharded_block(fp, &nnz, sizeof(nnz), 1)) goto done;
    if (!write_sharded_block(fp, &num_parts, sizeof(num_parts), 1)) goto done;
    if (!write_sharded_block(fp, &num_shards, sizeof(num_shards), 1)) goto done;
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
inline int load_header(const char *filename, sharded<MatrixT> *m) {
    static const unsigned char magic[8] = { 'C', 'S', 'H', 'R', 'D', '0', '1', '\0' };
    unsigned char got_magic[8];
    std::FILE *fp = 0;
    unsigned char format = 0;
    std::uint64_t rows = 0;
    std::uint64_t cols = 0;
    std::uint64_t nnz = 0;
    std::uint64_t num_parts = 0;
    std::uint64_t num_shards = 0;
    int ok = 0;

    fp = std::fopen(filename, "rb");
    if (fp == 0) return 0;
    if (!read_sharded_block(fp, got_magic, sizeof(got_magic), 1)) goto done;
    if (std::memcmp(got_magic, magic, sizeof(magic)) != 0) goto done;
    clear(m);
    init(m);
    if (!read_sharded_block(fp, &format, sizeof(format), 1)) goto done;
    if (!check_sharded_disk_format<MatrixT>(format)) goto done;
    if (!read_sharded_block(fp, &rows, sizeof(rows), 1)) goto done;
    if (!read_sharded_block(fp, &cols, sizeof(cols), 1)) goto done;
    if (!read_sharded_block(fp, &nnz, sizeof(nnz), 1)) goto done;
    if (!read_sharded_block(fp, &num_parts, sizeof(num_parts), 1)) goto done;
    if (!read_sharded_block(fp, &num_shards, sizeof(num_shards), 1)) goto done;
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

} // namespace cellshard
