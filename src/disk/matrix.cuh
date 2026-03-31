#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "../formats/compressed.cuh"
#include "../formats/dense.cuh"
#include "../formats/diagonal.cuh"
#include "../formats/triplet.cuh"

namespace cellshard {

enum {
    disk_format_none       = 0,
    disk_format_dense      = 1,
    disk_format_compressed = 2,
    disk_format_coo        = 3,
    disk_format_dia        = 4,
    disk_format_ell        = 5
};

struct disk_header {
    unsigned char format;
    types::dim_t rows;
    types::dim_t cols;
    types::nnz_t nnz;
};

inline int check_disk_format(unsigned char expected, unsigned char actual, const char *name) {
    if (expected == actual) return 1;
    std::fprintf(stderr,
                 "Error: expected format %u, got %u for %s\n",
                 (unsigned int) expected,
                 (unsigned int) actual,
                 name);
    return 0;
}

template<typename MatrixT>
struct disk_format_code;

template<>
struct disk_format_code<dense> {
    enum { value = disk_format_dense };
    static inline const char *name() { return "dense matrix"; }
};

template<>
struct disk_format_code<sparse::compressed> {
    enum { value = disk_format_compressed };
    static inline const char *name() { return "compressed matrix"; }
};

template<>
struct disk_format_code<sparse::coo> {
    enum { value = disk_format_coo };
    static inline const char *name() { return "coo matrix"; }
};

template<>
struct disk_format_code<sparse::dia> {
    enum { value = disk_format_dia };
    static inline const char *name() { return "dia matrix"; }
};

struct dense_load_result {
    disk_header h;
    void *val;
};

struct compressed_load_result {
    disk_header h;
    types::u32 axis;
    types::ptr_t *majorPtr;
    types::idx_t *minorIdx;
    void *val;
};

struct coo_load_result {
    disk_header h;
    types::idx_t *rowIdx;
    types::idx_t *colIdx;
    void *val;
};

struct dia_load_result {
    disk_header h;
    types::idx_t num_diagonals;
    int *offsets;
    void *val;
};

int store_dense_raw(const char *filename, types::dim_t rows, types::dim_t cols, types::nnz_t nnz, const void *val, std::size_t value_size);
int load_dense_raw(const char *filename, std::size_t value_size, dense_load_result *out);

int store_compressed_raw(const char *filename, types::dim_t rows, types::dim_t cols, types::nnz_t nnz, types::u32 axis, types::dim_t major_dim, const types::ptr_t *majorPtr, const types::idx_t *minorIdx, const void *val, std::size_t value_size);
int load_compressed_raw(const char *filename, std::size_t value_size, compressed_load_result *out);

int store_coo_raw(const char *filename, types::dim_t rows, types::dim_t cols, types::nnz_t nnz, const types::idx_t *rowIdx, const types::idx_t *colIdx, const void *val, std::size_t value_size);
int load_coo_raw(const char *filename, std::size_t value_size, coo_load_result *out);

int store_dia_raw(const char *filename, types::dim_t rows, types::dim_t cols, types::nnz_t nnz, types::idx_t num_diagonals, const int *offsets, const void *val, std::size_t value_size);
int load_dia_raw(const char *filename, std::size_t value_size, dia_load_result *out);

inline int store(const char *filename, const dense *m) {
    return store_dense_raw(
        filename,
        m->rows,
        m->cols,
        (types::nnz_t) ((std::size_t) m->rows * (std::size_t) m->cols),
        m->val,
        sizeof(real::storage_t)
    );
}

inline int load(const char *filename, dense *m) {
    dense_load_result tmp;

    tmp.val = 0;
    if (!load_dense_raw(filename, sizeof(real::storage_t), &tmp)) return 0;
    clear(m);
    init(m, tmp.h.rows, tmp.h.cols);
    m->val = (real::storage_t *) tmp.val;
    return 1;
}

inline int store(const char *filename, const sparse::compressed *m) {
    return store_compressed_raw(
        filename,
        m->rows,
        m->cols,
        m->nnz,
        m->axis,
        sparse::major_dim(m),
        m->majorPtr,
        m->minorIdx,
        m->val,
        sizeof(real::storage_t)
    );
}

inline int load(const char *filename, sparse::compressed *m) {
    compressed_load_result tmp;

    tmp.axis = sparse::compressed_by_row;
    tmp.majorPtr = 0;
    tmp.minorIdx = 0;
    tmp.val = 0;
    if (!load_compressed_raw(filename, sizeof(real::storage_t), &tmp)) return 0;
    sparse::clear(m);
    sparse::init(m, tmp.h.rows, tmp.h.cols, tmp.h.nnz, tmp.axis);
    m->majorPtr = tmp.majorPtr;
    m->minorIdx = tmp.minorIdx;
    m->val = (real::storage_t *) tmp.val;
    return 1;
}

inline int store(const char *filename, const sparse::coo *m) {
    return store_coo_raw(filename, m->rows, m->cols, m->nnz, m->rowIdx, m->colIdx, m->val, sizeof(real::storage_t));
}

inline int load(const char *filename, sparse::coo *m) {
    coo_load_result tmp;

    tmp.rowIdx = 0;
    tmp.colIdx = 0;
    tmp.val = 0;
    if (!load_coo_raw(filename, sizeof(real::storage_t), &tmp)) return 0;
    sparse::clear(m);
    sparse::init(m, tmp.h.rows, tmp.h.cols, tmp.h.nnz);
    m->rowIdx = tmp.rowIdx;
    m->colIdx = tmp.colIdx;
    m->val = (real::storage_t *) tmp.val;
    return 1;
}

inline int store(const char *filename, const sparse::dia *m) {
    return store_dia_raw(filename, m->rows, m->cols, m->nnz, m->num_diagonals, m->offsets, m->val, sizeof(real::storage_t));
}

inline int load(const char *filename, sparse::dia *m) {
    dia_load_result tmp;

    tmp.num_diagonals = 0;
    tmp.offsets = 0;
    tmp.val = 0;
    if (!load_dia_raw(filename, sizeof(real::storage_t), &tmp)) return 0;
    sparse::clear(m);
    sparse::init(m, tmp.h.rows, tmp.h.cols, tmp.h.nnz);
    m->num_diagonals = tmp.num_diagonals;
    m->offsets = tmp.offsets;
    m->val = (real::storage_t *) tmp.val;
    return 1;
}

} // namespace cellshard
