#include "matrix.cuh"

#include <cstdio>
#include <cstdlib>

namespace cellshard {

namespace {

// Packed part layout is:
// fixed header
// followed by raw arrays in format-specific order.
inline std::size_t header_bytes() {
    return sizeof(unsigned char) + sizeof(types::dim_t) + sizeof(types::dim_t) + sizeof(types::nnz_t);
}

// Thin fwrite/fread wrappers keep the raw I/O path explicit.
inline int write_block(std::FILE *fp, const void *ptr, std::size_t elem_size, std::size_t count) {
    if (count == 0) return 1;
    return std::fwrite(ptr, elem_size, count, fp) == count;
}

inline int read_block(std::FILE *fp, void *ptr, std::size_t elem_size, std::size_t count) {
    if (count == 0) return 1;
    return std::fread(ptr, elem_size, count, fp) == count;
}

inline int write_header(std::FILE *fp, unsigned char format, types::dim_t rows, types::dim_t cols, types::nnz_t nnz) {
    if (!write_block(fp, &format, sizeof(format), 1)) return 0;
    if (!write_block(fp, &rows, sizeof(rows), 1)) return 0;
    if (!write_block(fp, &cols, sizeof(cols), 1)) return 0;
    if (!write_block(fp, &nnz, sizeof(nnz), 1)) return 0;
    return 1;
}

inline int read_header(std::FILE *fp, disk_header *out) {
    if (!read_block(fp, &out->format, sizeof(out->format), 1)) return 0;
    if (!read_block(fp, &out->rows, sizeof(out->rows), 1)) return 0;
    if (!read_block(fp, &out->cols, sizeof(out->cols), 1)) return 0;
    if (!read_block(fp, &out->nnz, sizeof(out->nnz), 1)) return 0;
    return 1;
}

// Raw load paths allocate host buffers eagerly and then fill them from disk.
inline void *alloc_bytes(std::size_t bytes) {
    if (bytes == 0) return 0;
    return std::malloc(bytes);
}

// Cleanup helpers for partially built load results.
inline void free_compressed_result(compressed_load_result *out) {
    std::free(out->majorPtr);
    std::free(out->minorIdx);
    std::free(out->val);
    out->majorPtr = 0;
    out->minorIdx = 0;
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

} // namespace

std::size_t packed_dense_bytes(types::nnz_t nnz, std::size_t value_size) {
    return header_bytes() + (std::size_t) nnz * value_size;
}

std::size_t packed_compressed_bytes(types::dim_t rows, types::dim_t cols, types::nnz_t nnz, types::u32 axis, std::size_t value_size) {
    const types::dim_t major_dim = axis == sparse::compressed_by_col ? cols : rows;
    return header_bytes()
        + sizeof(types::u32)
        + (std::size_t) nnz * value_size
        + (std::size_t) (major_dim + 1) * sizeof(types::ptr_t)
        + (std::size_t) nnz * sizeof(types::idx_t);
}

std::size_t packed_coo_bytes(types::nnz_t nnz, std::size_t value_size) {
    return header_bytes()
        + (std::size_t) nnz * value_size
        + (std::size_t) nnz * sizeof(types::idx_t)
        + (std::size_t) nnz * sizeof(types::idx_t);
}

std::size_t packed_dia_bytes(types::nnz_t nnz, types::idx_t num_diagonals, std::size_t value_size) {
    return header_bytes()
        + sizeof(types::idx_t)
        + (std::size_t) num_diagonals * sizeof(int)
        + (std::size_t) nnz * value_size;
}

// Standalone filename helpers are full synchronous host I/O operations.
int store_dense_raw(const char *filename, types::dim_t rows, types::dim_t cols, types::nnz_t nnz, const void *val, std::size_t value_size) {
    std::FILE *fp = 0;
    int ok = 0;

    fp = std::fopen(filename, "wb");
    if (fp == 0) return 0;
    ok = store_dense_raw(fp, rows, cols, nnz, val, value_size);

done:
    std::fclose(fp);
    return ok;
}

// FILE* store writes one packed dense payload at the current file position.
int store_dense_raw(std::FILE *fp, types::dim_t rows, types::dim_t cols, types::nnz_t nnz, const void *val, std::size_t value_size) {
    if (!write_header(fp, disk_format_dense, rows, cols, nnz)) return 0;
    if (!write_block(fp, val, value_size, nnz)) return 0;
    return 1;
}

int load_dense_raw(const char *filename, std::size_t value_size, dense_load_result *out) {
    std::FILE *fp = 0;
    int ok = 0;

    out->val = 0;
    fp = std::fopen(filename, "rb");
    if (fp == 0) return 0;
    ok = load_dense_raw(fp, value_size, out);

done:
    std::fclose(fp);
    return ok;
}

// FILE* load reads one packed dense payload into a fresh host allocation.
int load_dense_raw(std::FILE *fp, std::size_t value_size, dense_load_result *out) {
    int ok = 0;

    out->val = 0;
    if (!read_header(fp, &out->h)) goto done;
    if (!check_disk_format(disk_format_dense, out->h.format, "dense matrix")) goto done;
    out->val = alloc_bytes((std::size_t) out->h.nnz * value_size);
    if (out->h.nnz != 0 && out->val == 0) goto done;
    if (!read_block(fp, out->val, value_size, out->h.nnz)) goto done;
    ok = 1;

done:
    if (!ok) {
        std::free(out->val);
        out->val = 0;
    }
    return ok;
}

int store_compressed_raw(const char *filename, types::dim_t rows, types::dim_t cols, types::nnz_t nnz, types::u32 axis, types::dim_t major_dim, const types::ptr_t *majorPtr, const types::idx_t *minorIdx, const void *val, std::size_t value_size) {
    std::FILE *fp = 0;
    int ok = 0;

    fp = std::fopen(filename, "wb");
    if (fp == 0) return 0;
    ok = store_compressed_raw(fp, rows, cols, nnz, axis, major_dim, majorPtr, minorIdx, val, value_size);

done:
    std::fclose(fp);
    return ok;
}

// Compressed layout persists value payload first, then majorPtr, then minorIdx.
int store_compressed_raw(std::FILE *fp, types::dim_t rows, types::dim_t cols, types::nnz_t nnz, types::u32 axis, types::dim_t major_dim, const types::ptr_t *majorPtr, const types::idx_t *minorIdx, const void *val, std::size_t value_size) {
    if (!write_header(fp, disk_format_compressed, rows, cols, nnz)) return 0;
    if (!write_block(fp, &axis, sizeof(axis), 1)) return 0;
    if (!write_block(fp, val, value_size, nnz)) return 0;
    if (major_dim != 0 && !write_block(fp, majorPtr, sizeof(types::ptr_t), (std::size_t) major_dim + 1)) return 0;
    if (!write_block(fp, minorIdx, sizeof(types::idx_t), nnz)) return 0;
    return 1;
}

int load_compressed_raw(const char *filename, std::size_t value_size, compressed_load_result *out) {
    std::FILE *fp = 0;
    int ok = 0;

    out->axis = sparse::compressed_by_row;
    out->majorPtr = 0;
    out->minorIdx = 0;
    out->val = 0;
    fp = std::fopen(filename, "rb");
    if (fp == 0) return 0;
    ok = load_compressed_raw(fp, value_size, out);

done:
    std::fclose(fp);
    return ok;
}

int load_compressed_raw(std::FILE *fp, std::size_t value_size, compressed_load_result *out) {
    int ok = 0;
    types::dim_t major_dim = 0;

    out->axis = sparse::compressed_by_row;
    out->majorPtr = 0;
    out->minorIdx = 0;
    out->val = 0;
    if (!read_header(fp, &out->h)) goto done;
    if (!check_disk_format(disk_format_compressed, out->h.format, "compressed matrix")) goto done;
    if (!read_block(fp, &out->axis, sizeof(out->axis), 1)) goto done;
    major_dim = out->axis == sparse::compressed_by_col ? out->h.cols : out->h.rows;
    if (major_dim != 0) out->majorPtr = (types::ptr_t *) alloc_bytes((std::size_t) (major_dim + 1) * sizeof(types::ptr_t));
    out->minorIdx = (types::idx_t *) alloc_bytes((std::size_t) out->h.nnz * sizeof(types::idx_t));
    out->val = alloc_bytes((std::size_t) out->h.nnz * value_size);
    if (major_dim != 0 && out->majorPtr == 0) goto done;
    if (out->h.nnz != 0 && (out->minorIdx == 0 || out->val == 0)) goto done;
    if (!read_block(fp, out->val, value_size, out->h.nnz)) goto done;
    if (major_dim != 0 && !read_block(fp, out->majorPtr, sizeof(types::ptr_t), (std::size_t) major_dim + 1)) goto done;
    if (!read_block(fp, out->minorIdx, sizeof(types::idx_t), out->h.nnz)) goto done;
    ok = 1;

done:
    if (!ok) free_compressed_result(out);
    return ok;
}

int store_coo_raw(const char *filename, types::dim_t rows, types::dim_t cols, types::nnz_t nnz, const types::idx_t *rowIdx, const types::idx_t *colIdx, const void *val, std::size_t value_size) {
    std::FILE *fp = 0;
    int ok = 0;

    fp = std::fopen(filename, "wb");
    if (fp == 0) return 0;
    ok = store_coo_raw(fp, rows, cols, nnz, rowIdx, colIdx, val, value_size);

done:
    std::fclose(fp);
    return ok;
}

// COO layout persists values, then row indices, then column indices.
int store_coo_raw(std::FILE *fp, types::dim_t rows, types::dim_t cols, types::nnz_t nnz, const types::idx_t *rowIdx, const types::idx_t *colIdx, const void *val, std::size_t value_size) {
    if (!write_header(fp, disk_format_coo, rows, cols, nnz)) return 0;
    if (!write_block(fp, val, value_size, nnz)) return 0;
    if (!write_block(fp, rowIdx, sizeof(types::idx_t), nnz)) return 0;
    if (!write_block(fp, colIdx, sizeof(types::idx_t), nnz)) return 0;
    return 1;
}

int load_coo_raw(const char *filename, std::size_t value_size, coo_load_result *out) {
    std::FILE *fp = 0;
    int ok = 0;

    out->rowIdx = 0;
    out->colIdx = 0;
    out->val = 0;
    fp = std::fopen(filename, "rb");
    if (fp == 0) return 0;
    ok = load_coo_raw(fp, value_size, out);

done:
    std::fclose(fp);
    return ok;
}

int load_coo_raw(std::FILE *fp, std::size_t value_size, coo_load_result *out) {
    int ok = 0;

    out->rowIdx = 0;
    out->colIdx = 0;
    out->val = 0;
    if (!read_header(fp, &out->h)) goto done;
    if (!check_disk_format(disk_format_coo, out->h.format, "coo matrix")) goto done;
    out->rowIdx = (types::idx_t *) alloc_bytes((std::size_t) out->h.nnz * sizeof(types::idx_t));
    out->colIdx = (types::idx_t *) alloc_bytes((std::size_t) out->h.nnz * sizeof(types::idx_t));
    out->val = alloc_bytes((std::size_t) out->h.nnz * value_size);
    if (out->h.nnz != 0 && (out->rowIdx == 0 || out->colIdx == 0 || out->val == 0)) goto done;
    if (!read_block(fp, out->val, value_size, out->h.nnz)) goto done;
    if (!read_block(fp, out->rowIdx, sizeof(types::idx_t), out->h.nnz)) goto done;
    if (!read_block(fp, out->colIdx, sizeof(types::idx_t), out->h.nnz)) goto done;
    ok = 1;

done:
    if (!ok) free_coo_result(out);
    return ok;
}

int store_dia_raw(const char *filename, types::dim_t rows, types::dim_t cols, types::nnz_t nnz, types::idx_t num_diagonals, const int *offsets, const void *val, std::size_t value_size) {
    std::FILE *fp = 0;
    int ok = 0;

    fp = std::fopen(filename, "wb");
    if (fp == 0) return 0;
    ok = store_dia_raw(fp, rows, cols, nnz, num_diagonals, offsets, val, value_size);

done:
    std::fclose(fp);
    return ok;
}

// DIA layout writes offsets before the diagonal-value payload.
int store_dia_raw(std::FILE *fp, types::dim_t rows, types::dim_t cols, types::nnz_t nnz, types::idx_t num_diagonals, const int *offsets, const void *val, std::size_t value_size) {
    if (!write_header(fp, disk_format_dia, rows, cols, nnz)) return 0;
    if (!write_block(fp, &num_diagonals, sizeof(types::idx_t), 1)) return 0;
    if (!write_block(fp, offsets, sizeof(int), num_diagonals)) return 0;
    if (!write_block(fp, val, value_size, nnz)) return 0;
    return 1;
}

int load_dia_raw(const char *filename, std::size_t value_size, dia_load_result *out) {
    std::FILE *fp = 0;
    int ok = 0;

    out->num_diagonals = 0;
    out->offsets = 0;
    out->val = 0;
    fp = std::fopen(filename, "rb");
    if (fp == 0) return 0;
    ok = load_dia_raw(fp, value_size, out);

done:
    std::fclose(fp);
    return ok;
}

int load_dia_raw(std::FILE *fp, std::size_t value_size, dia_load_result *out) {
    int ok = 0;

    out->num_diagonals = 0;
    out->offsets = 0;
    out->val = 0;
    if (!read_header(fp, &out->h)) goto done;
    if (!check_disk_format(disk_format_dia, out->h.format, "dia matrix")) goto done;
    if (!read_block(fp, &out->num_diagonals, sizeof(types::idx_t), 1)) goto done;
    out->offsets = (int *) alloc_bytes((std::size_t) out->num_diagonals * sizeof(int));
    out->val = alloc_bytes((std::size_t) out->h.nnz * value_size);
    if (out->num_diagonals != 0 && out->offsets == 0) goto done;
    if (out->h.nnz != 0 && out->val == 0) goto done;
    if (!read_block(fp, out->offsets, sizeof(int), out->num_diagonals)) goto done;
    if (!read_block(fp, out->val, value_size, out->h.nnz)) goto done;
    ok = 1;

done:
    if (!ok) free_dia_result(out);
    return ok;
}

int store(std::FILE *fp, const dense *m) {
    return store_dense_raw(fp, m->rows, m->cols, (types::nnz_t) ((std::size_t) m->rows * (std::size_t) m->cols), m->val, sizeof(real::storage_t));
}

int load(std::FILE *fp, dense *m) {
    dense_load_result tmp;

    tmp.val = 0;
    if (!load_dense_raw(fp, sizeof(real::storage_t), &tmp)) return 0;
    clear(m);
    init(m, tmp.h.rows, tmp.h.cols);
    m->val = (real::storage_t *) tmp.val;
    return 1;
}

int store(std::FILE *fp, const sparse::compressed *m) {
    return store_compressed_raw(fp, m->rows, m->cols, m->nnz, m->axis, sparse::major_dim(m), m->majorPtr, m->minorIdx, m->val, sizeof(real::storage_t));
}

int load(std::FILE *fp, sparse::compressed *m) {
    compressed_load_result tmp;

    tmp.axis = sparse::compressed_by_row;
    tmp.majorPtr = 0;
    tmp.minorIdx = 0;
    tmp.val = 0;
    if (!load_compressed_raw(fp, sizeof(real::storage_t), &tmp)) return 0;
    sparse::clear(m);
    sparse::init(m, tmp.h.rows, tmp.h.cols, tmp.h.nnz, tmp.axis);
    m->majorPtr = tmp.majorPtr;
    m->minorIdx = tmp.minorIdx;
    m->val = (real::storage_t *) tmp.val;
    return 1;
}

int store(std::FILE *fp, const sparse::coo *m) {
    return store_coo_raw(fp, m->rows, m->cols, m->nnz, m->rowIdx, m->colIdx, m->val, sizeof(real::storage_t));
}

int load(std::FILE *fp, sparse::coo *m) {
    coo_load_result tmp;

    tmp.rowIdx = 0;
    tmp.colIdx = 0;
    tmp.val = 0;
    if (!load_coo_raw(fp, sizeof(real::storage_t), &tmp)) return 0;
    sparse::clear(m);
    sparse::init(m, tmp.h.rows, tmp.h.cols, tmp.h.nnz);
    m->rowIdx = tmp.rowIdx;
    m->colIdx = tmp.colIdx;
    m->val = (real::storage_t *) tmp.val;
    return 1;
}

int store(std::FILE *fp, const sparse::dia *m) {
    return store_dia_raw(fp, m->rows, m->cols, m->nnz, m->num_diagonals, m->offsets, m->val, sizeof(real::storage_t));
}

int load(std::FILE *fp, sparse::dia *m) {
    dia_load_result tmp;

    tmp.num_diagonals = 0;
    tmp.offsets = 0;
    tmp.val = 0;
    if (!load_dia_raw(fp, sizeof(real::storage_t), &tmp)) return 0;
    sparse::clear(m);
    sparse::init(m, tmp.h.rows, tmp.h.cols, tmp.h.nnz);
    m->num_diagonals = tmp.num_diagonals;
    m->offsets = tmp.offsets;
    m->val = (real::storage_t *) tmp.val;
    return 1;
}

} // namespace cellshard
