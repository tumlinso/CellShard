#include "matrix_io.cuh"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace cellshard {
namespace detail {

namespace {

inline int write_block(std::FILE *fp, const void *ptr, std::size_t elem_size, std::size_t count) {
    if (count == 0) return 1;
    return std::fwrite(ptr, elem_size, count, fp) == count;
}

inline int read_block(std::FILE *fp, void *ptr, std::size_t elem_size, std::size_t count) {
    if (count == 0) return 1;
    return std::fread(ptr, elem_size, count, fp) == count;
}

inline int write_header(std::FILE *fp, unsigned char format, unsigned int rows, unsigned int cols, unsigned int nnz) {
    if (!write_block(fp, &format, sizeof(format), 1)) return 0;
    if (!write_block(fp, &rows, sizeof(rows), 1)) return 0;
    if (!write_block(fp, &cols, sizeof(cols), 1)) return 0;
    if (!write_block(fp, &nnz, sizeof(nnz), 1)) return 0;
    return 1;
}

inline int read_header(std::FILE *fp, header *out) {
    if (!read_block(fp, &out->format, sizeof(out->format), 1)) return 0;
    if (!read_block(fp, &out->rows, sizeof(out->rows), 1)) return 0;
    if (!read_block(fp, &out->cols, sizeof(out->cols), 1)) return 0;
    if (!read_block(fp, &out->nnz, sizeof(out->nnz), 1)) return 0;
    return 1;
}

inline void *alloc_bytes(std::size_t bytes) {
    if (bytes == 0) return 0;
    return std::malloc(bytes);
}

inline void free_csr_result(csr_load_result *out) {
    std::free(out->rowPtr);
    std::free(out->colIdx);
    std::free(out->val);
    out->rowPtr = 0;
    out->colIdx = 0;
    out->val = 0;
}

inline void free_coo_result(coo_load_result *out) {
    std::free(out->rowIdx);
    std::free(out->colIdx);
    std::free(out->val);
    out->rowIdx = 0;
    out->colIdx = 0;
    out->val = 0;
}

inline void free_dia_result(dia_load_result *out) {
    std::free(out->offsets);
    std::free(out->val);
    out->offsets = 0;
    out->val = 0;
    out->num_diagonals = 0;
}

inline void free_sharded_header_result(sharded_header_load_result *out) {
    std::free(out->part_rows);
    std::free(out->part_nnz);
    std::free(out->part_aux);
    std::free(out->shard_offsets);
    out->part_rows = 0;
    out->part_nnz = 0;
    out->part_aux = 0;
    out->shard_offsets = 0;
    out->num_parts = 0;
    out->num_shards = 0;
}

} // namespace

void init(shard_storage *s) {
    s->capacity = 0;
    s->paths = 0;
}

void clear(shard_storage *s) {
    unsigned int i = 0;

    if (s->paths != 0) {
        for (i = 0; i < s->capacity; ++i) std::free(s->paths[i]);
    }
    std::free(s->paths);
    s->capacity = 0;
    s->paths = 0;
}

int reserve(shard_storage *s, unsigned int capacity) {
    char **paths = 0;
    unsigned int i = 0;

    if (capacity <= s->capacity) return 1;
    paths = (char **) std::calloc((std::size_t) capacity, sizeof(char *));
    if (paths == 0) return 0;
    for (i = 0; i < s->capacity; ++i) paths[i] = s->paths[i];
    std::free(s->paths);
    s->paths = paths;
    s->capacity = capacity;
    return 1;
}

int bind(shard_storage *s, unsigned int partId, const char *path) {
    std::size_t len = 0;
    char *copy = 0;

    if (partId >= s->capacity) return 0;
    std::free(s->paths[partId]);
    s->paths[partId] = 0;
    if (path == 0) return 1;

    len = std::strlen(path);
    copy = (char *) std::malloc(len + 1);
    if (copy == 0) return 0;
    std::memcpy(copy, path, len + 1);
    s->paths[partId] = copy;
    return 1;
}

int bind_sequential(shard_storage *s, unsigned int count, const char *prefix) {
    char path[1024];
    unsigned int i = 0;

    if (!::matrix::detail::reserve(s, count)) return 0;
    for (i = 0; i < count; ++i) {
        if (std::snprintf(path, sizeof(path), "%s.%u", prefix, (unsigned int) i) <= 0) return 0;
        if (!::matrix::detail::bind(s, i, path)) return 0;
    }
    return 1;
}

int store_dense_raw(const char *filename, unsigned int rows, unsigned int cols, unsigned int nnz, const void *val, std::size_t value_size) {
    std::FILE *fp = 0;
    int ok = 0;

    fp = std::fopen(filename, "wb");
    if (fp == 0) return 0;
    if (!write_header(fp, format_dense, rows, cols, nnz)) goto done;
    if (!write_block(fp, val, value_size, nnz)) goto done;
    ok = 1;

done:
    std::fclose(fp);
    return ok;
}

int load_dense_raw(const char *filename, std::size_t value_size, dense_load_result *out) {
    std::FILE *fp = 0;
    int ok = 0;

    out->val = 0;
    fp = std::fopen(filename, "rb");
    if (fp == 0) return 0;
    if (!read_header(fp, &out->h)) goto done;
    if (!checkformat(format_dense, out->h.format, "dense matrix")) goto done;
    out->val = alloc_bytes((std::size_t) out->h.nnz * value_size);
    if (out->h.nnz != 0 && out->val == 0) goto done;
    if (!read_block(fp, out->val, value_size, out->h.nnz)) goto done;
    ok = 1;

done:
    if (!ok) {
        std::free(out->val);
        out->val = 0;
    }
    std::fclose(fp);
    return ok;
}

int store_csr_raw(const char *filename, unsigned int rows, unsigned int cols, unsigned int nnz, const unsigned int *rowPtr, const unsigned int *colIdx, const void *val, std::size_t value_size) {
    std::FILE *fp = 0;
    int ok = 0;

    fp = std::fopen(filename, "wb");
    if (fp == 0) return 0;
    if (!write_header(fp, format_csr, rows, cols, nnz)) goto done;
    if (!write_block(fp, val, value_size, nnz)) goto done;
    if (rows != 0 && !write_block(fp, rowPtr, sizeof(unsigned int), (std::size_t) rows + 1)) goto done;
    if (!write_block(fp, colIdx, sizeof(unsigned int), nnz)) goto done;
    ok = 1;

done:
    std::fclose(fp);
    return ok;
}

int load_csr_raw(const char *filename, std::size_t value_size, csr_load_result *out) {
    std::FILE *fp = 0;
    int ok = 0;

    out->rowPtr = 0;
    out->colIdx = 0;
    out->val = 0;
    fp = std::fopen(filename, "rb");
    if (fp == 0) return 0;
    if (!read_header(fp, &out->h)) goto done;
    if (!checkformat(format_csr, out->h.format, "csr matrix")) goto done;
    if (out->h.rows != 0) out->rowPtr = (unsigned int *) alloc_bytes((std::size_t) (out->h.rows + 1) * sizeof(unsigned int));
    out->colIdx = (unsigned int *) alloc_bytes((std::size_t) out->h.nnz * sizeof(unsigned int));
    out->val = alloc_bytes((std::size_t) out->h.nnz * value_size);
    if (out->h.rows != 0 && out->rowPtr == 0) goto done;
    if (out->h.nnz != 0 && (out->colIdx == 0 || out->val == 0)) goto done;
    if (!read_block(fp, out->val, value_size, out->h.nnz)) goto done;
    if (out->h.rows != 0 && !read_block(fp, out->rowPtr, sizeof(unsigned int), (std::size_t) out->h.rows + 1)) goto done;
    if (!read_block(fp, out->colIdx, sizeof(unsigned int), out->h.nnz)) goto done;
    ok = 1;

done:
    if (!ok) free_csr_result(out);
    std::fclose(fp);
    return ok;
}

int store_coo_raw(const char *filename, unsigned int rows, unsigned int cols, unsigned int nnz, const unsigned int *rowIdx, const unsigned int *colIdx, const void *val, std::size_t value_size) {
    std::FILE *fp = 0;
    int ok = 0;

    fp = std::fopen(filename, "wb");
    if (fp == 0) return 0;
    if (!write_header(fp, format_coo, rows, cols, nnz)) goto done;
    if (!write_block(fp, val, value_size, nnz)) goto done;
    if (!write_block(fp, rowIdx, sizeof(unsigned int), nnz)) goto done;
    if (!write_block(fp, colIdx, sizeof(unsigned int), nnz)) goto done;
    ok = 1;

done:
    std::fclose(fp);
    return ok;
}

int load_coo_raw(const char *filename, std::size_t value_size, coo_load_result *out) {
    std::FILE *fp = 0;
    int ok = 0;

    out->rowIdx = 0;
    out->colIdx = 0;
    out->val = 0;
    fp = std::fopen(filename, "rb");
    if (fp == 0) return 0;
    if (!read_header(fp, &out->h)) goto done;
    if (!checkformat(format_coo, out->h.format, "coo matrix")) goto done;
    out->rowIdx = (unsigned int *) alloc_bytes((std::size_t) out->h.nnz * sizeof(unsigned int));
    out->colIdx = (unsigned int *) alloc_bytes((std::size_t) out->h.nnz * sizeof(unsigned int));
    out->val = alloc_bytes((std::size_t) out->h.nnz * value_size);
    if (out->h.nnz != 0 && (out->rowIdx == 0 || out->colIdx == 0 || out->val == 0)) goto done;
    if (!read_block(fp, out->val, value_size, out->h.nnz)) goto done;
    if (!read_block(fp, out->rowIdx, sizeof(unsigned int), out->h.nnz)) goto done;
    if (!read_block(fp, out->colIdx, sizeof(unsigned int), out->h.nnz)) goto done;
    ok = 1;

done:
    if (!ok) free_coo_result(out);
    std::fclose(fp);
    return ok;
}

int store_dia_raw(const char *filename, unsigned int rows, unsigned int cols, unsigned int nnz, unsigned int num_diagonals, const int *offsets, const void *val, std::size_t value_size) {
    std::FILE *fp = 0;
    int ok = 0;

    fp = std::fopen(filename, "wb");
    if (fp == 0) return 0;
    if (!write_header(fp, format_dia, rows, cols, nnz)) goto done;
    if (!write_block(fp, &num_diagonals, sizeof(unsigned int), 1)) goto done;
    if (!write_block(fp, offsets, sizeof(int), num_diagonals)) goto done;
    if (!write_block(fp, val, value_size, nnz)) goto done;
    ok = 1;

done:
    std::fclose(fp);
    return ok;
}

int load_dia_raw(const char *filename, std::size_t value_size, dia_load_result *out) {
    std::FILE *fp = 0;
    int ok = 0;

    out->num_diagonals = 0;
    out->offsets = 0;
    out->val = 0;
    fp = std::fopen(filename, "rb");
    if (fp == 0) return 0;
    if (!read_header(fp, &out->h)) goto done;
    if (!checkformat(format_dia, out->h.format, "dia matrix")) goto done;
    if (!read_block(fp, &out->num_diagonals, sizeof(unsigned int), 1)) goto done;
    out->offsets = (int *) alloc_bytes((std::size_t) out->num_diagonals * sizeof(int));
    out->val = alloc_bytes((std::size_t) out->h.nnz * value_size);
    if (out->num_diagonals != 0 && out->offsets == 0) goto done;
    if (out->h.nnz != 0 && out->val == 0) goto done;
    if (!read_block(fp, out->offsets, sizeof(int), out->num_diagonals)) goto done;
    if (!read_block(fp, out->val, value_size, out->h.nnz)) goto done;
    ok = 1;

done:
    if (!ok) free_dia_result(out);
    std::fclose(fp);
    return ok;
}

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
                             const unsigned int *shard_offsets) {
    std::FILE *fp = 0;
    int ok = 0;

    fp = std::fopen(filename, "wb");
    if (fp == 0) return 0;
    if (!write_header(fp, format, rows, cols, nnz)) goto done;
    if (!write_block(fp, &num_parts, sizeof(unsigned int), 1)) goto done;
    if (!write_block(fp, &num_shards, sizeof(unsigned int), 1)) goto done;
    if (!write_block(fp, part_rows, sizeof(unsigned int), num_parts)) goto done;
    if (!write_block(fp, part_nnz, sizeof(unsigned int), num_parts)) goto done;
    if (!write_block(fp, part_aux, sizeof(unsigned int), num_parts)) goto done;
    if (num_shards != 0 && !write_block(fp, shard_offsets, sizeof(unsigned int), (std::size_t) num_shards + 1)) goto done;
    ok = 1;

done:
    std::fclose(fp);
    return ok;
}

int load_sharded_header_raw(const char *filename, sharded_header_load_result *out) {
    std::FILE *fp = 0;
    int ok = 0;

    out->num_parts = 0;
    out->num_shards = 0;
    out->part_rows = 0;
    out->part_nnz = 0;
    out->part_aux = 0;
    out->shard_offsets = 0;
    fp = std::fopen(filename, "rb");
    if (fp == 0) return 0;
    if (!read_header(fp, &out->h)) goto done;
    if (!read_block(fp, &out->num_parts, sizeof(unsigned int), 1)) goto done;
    if (!read_block(fp, &out->num_shards, sizeof(unsigned int), 1)) goto done;

    out->part_rows = (unsigned int *) alloc_bytes((std::size_t) out->num_parts * sizeof(unsigned int));
    out->part_nnz = (unsigned int *) alloc_bytes((std::size_t) out->num_parts * sizeof(unsigned int));
    out->part_aux = (unsigned int *) alloc_bytes((std::size_t) out->num_parts * sizeof(unsigned int));
    if (out->num_shards != 0) out->shard_offsets = (unsigned int *) alloc_bytes((std::size_t) (out->num_shards + 1) * sizeof(unsigned int));
    if (out->num_parts != 0 && (out->part_rows == 0 || out->part_nnz == 0 || out->part_aux == 0)) goto done;
    if (out->num_shards != 0 && out->shard_offsets == 0) goto done;

    if (!read_block(fp, out->part_rows, sizeof(unsigned int), out->num_parts)) goto done;
    if (!read_block(fp, out->part_nnz, sizeof(unsigned int), out->num_parts)) goto done;
    if (!read_block(fp, out->part_aux, sizeof(unsigned int), out->num_parts)) goto done;
    if (out->num_shards != 0 && !read_block(fp, out->shard_offsets, sizeof(unsigned int), (std::size_t) out->num_shards + 1)) goto done;
    ok = 1;

done:
    if (!ok) free_sharded_header_result(out);
    std::fclose(fp);
    return ok;
}

} // namespace detail
} // namespace cellshard
