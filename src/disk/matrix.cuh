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

// Per-part binary format tags used inside the native disk payload.
enum {
    disk_format_none       = 0,
    disk_format_dense      = 1,
    disk_format_compressed = 2,
    disk_format_coo        = 3,
    disk_format_dia        = 4,
    disk_format_ell        = 5
};

// Minimal fixed header stored at the front of every packed part.
struct disk_header {
    unsigned char format;
    types::dim_t rows;
    types::dim_t cols;
    types::nnz_t nnz;
};

// Fail loudly on disk-format mismatches instead of silently decoding garbage.
inline int check_disk_format(unsigned char expected, unsigned char actual, const char *name) {
    if (expected == actual) return 1;
    std::fprintf(stderr,
                 "Error: expected format %u, got %u for %s\n",
                 (unsigned int) expected,
                 (unsigned int) actual,
                 name);
    return 0;
}

// Compile-time map from in-memory matrix type to disk format code.
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

// Temporary raw load results own host allocations until a typed matrix adopts
// them.
struct dense_load_result {
    disk_header h;
    void *storage;
    void *val;
};

struct compressed_load_result {
    disk_header h;
    types::u32 axis;
    void *storage;
    types::ptr_t *majorPtr;
    types::idx_t *minorIdx;
    void *val;
};

struct coo_load_result {
    disk_header h;
    void *storage;
    types::idx_t *rowIdx;
    types::idx_t *colIdx;
    void *val;
};

struct dia_load_result {
    disk_header h;
    types::idx_t num_diagonals;
    void *storage;
    int *offsets;
    void *val;
};

// Disk-byte estimators for one packed part payload.
std::size_t packed_dense_bytes(types::nnz_t nnz, std::size_t value_size);
std::size_t packed_compressed_bytes(types::dim_t rows, types::dim_t cols, types::nnz_t nnz, types::u32 axis, std::size_t value_size);
std::size_t packed_coo_bytes(types::nnz_t nnz, std::size_t value_size);
std::size_t packed_dia_bytes(types::nnz_t nnz, types::idx_t num_diagonals, std::size_t value_size);

int store_dense_raw(const char *filename, types::dim_t rows, types::dim_t cols, types::nnz_t nnz, const void *val, std::size_t value_size);
int load_dense_raw(const char *filename, std::size_t value_size, dense_load_result *out);
int store_dense_raw(std::FILE *fp, types::dim_t rows, types::dim_t cols, types::nnz_t nnz, const void *val, std::size_t value_size);
int load_dense_raw(std::FILE *fp, std::size_t value_size, dense_load_result *out);

int store_compressed_raw(const char *filename, types::dim_t rows, types::dim_t cols, types::nnz_t nnz, types::u32 axis, types::dim_t major_dim, const types::ptr_t *majorPtr, const types::idx_t *minorIdx, const void *val, std::size_t value_size);
int load_compressed_raw(const char *filename, std::size_t value_size, compressed_load_result *out);
int store_compressed_raw(std::FILE *fp, types::dim_t rows, types::dim_t cols, types::nnz_t nnz, types::u32 axis, types::dim_t major_dim, const types::ptr_t *majorPtr, const types::idx_t *minorIdx, const void *val, std::size_t value_size);
int load_compressed_raw(std::FILE *fp, std::size_t value_size, compressed_load_result *out);

int store_coo_raw(const char *filename, types::dim_t rows, types::dim_t cols, types::nnz_t nnz, const types::idx_t *rowIdx, const types::idx_t *colIdx, const void *val, std::size_t value_size);
int load_coo_raw(const char *filename, std::size_t value_size, coo_load_result *out);
int store_coo_raw(std::FILE *fp, types::dim_t rows, types::dim_t cols, types::nnz_t nnz, const types::idx_t *rowIdx, const types::idx_t *colIdx, const void *val, std::size_t value_size);
int load_coo_raw(std::FILE *fp, std::size_t value_size, coo_load_result *out);

int store_dia_raw(const char *filename, types::dim_t rows, types::dim_t cols, types::nnz_t nnz, types::idx_t num_diagonals, const int *offsets, const void *val, std::size_t value_size);
int load_dia_raw(const char *filename, std::size_t value_size, dia_load_result *out);
int store_dia_raw(std::FILE *fp, types::dim_t rows, types::dim_t cols, types::nnz_t nnz, types::idx_t num_diagonals, const int *offsets, const void *val, std::size_t value_size);
int load_dia_raw(std::FILE *fp, std::size_t value_size, dia_load_result *out);

inline std::size_t packed_bytes(const dense *, types::dim_t, types::dim_t, types::nnz_t nnz, unsigned long, std::size_t value_size) {
    return packed_dense_bytes(nnz, value_size);
}

inline std::size_t packed_bytes(const sparse::compressed *, types::dim_t rows, types::dim_t cols, types::nnz_t nnz, unsigned long axis, std::size_t value_size) {
    return packed_compressed_bytes(rows, cols, nnz, (types::u32) axis, value_size);
}

inline std::size_t packed_bytes(const sparse::coo *, types::dim_t, types::dim_t, types::nnz_t nnz, unsigned long, std::size_t value_size) {
    return packed_coo_bytes(nnz, value_size);
}

inline std::size_t packed_bytes(const sparse::dia *, types::dim_t, types::dim_t, types::nnz_t nnz, unsigned long num_diagonals, std::size_t value_size) {
    return packed_dia_bytes(nnz, (types::idx_t) num_diagonals, value_size);
}

// FILE* variants let packfile code write/read many parts through one open file
// handle, which avoids reopen churn in sequential shard fetch/store loops.
int store(std::FILE *fp, const dense *m);
int load(std::FILE *fp, dense *m);
int store(std::FILE *fp, const sparse::compressed *m);
int load(std::FILE *fp, sparse::compressed *m);
int store(std::FILE *fp, const sparse::coo *m);
int load(std::FILE *fp, sparse::coo *m);
int store(std::FILE *fp, const sparse::dia *m);
int load(std::FILE *fp, sparse::dia *m);

// Filename variants open a file and do a full synchronous host I/O operation.
// Load paths allocate and take ownership of a fresh host payload every call.
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

    tmp.storage = 0;
    tmp.val = 0;
    if (!load_dense_raw(filename, sizeof(real::storage_t), &tmp)) return 0;
    clear(m);
    init(m, tmp.h.rows, tmp.h.cols);
    m->storage = tmp.storage;
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
    tmp.storage = 0;
    tmp.majorPtr = 0;
    tmp.minorIdx = 0;
    tmp.val = 0;
    if (!load_compressed_raw(filename, sizeof(real::storage_t), &tmp)) return 0;
    sparse::clear(m);
    sparse::init(m, tmp.h.rows, tmp.h.cols, tmp.h.nnz, tmp.axis);
    m->storage = tmp.storage;
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

    tmp.storage = 0;
    tmp.rowIdx = 0;
    tmp.colIdx = 0;
    tmp.val = 0;
    if (!load_coo_raw(filename, sizeof(real::storage_t), &tmp)) return 0;
    sparse::clear(m);
    sparse::init(m, tmp.h.rows, tmp.h.cols, tmp.h.nnz);
    m->storage = tmp.storage;
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
    tmp.storage = 0;
    tmp.offsets = 0;
    tmp.val = 0;
    if (!load_dia_raw(filename, sizeof(real::storage_t), &tmp)) return 0;
    sparse::clear(m);
    sparse::init(m, tmp.h.rows, tmp.h.cols, tmp.h.nnz);
    m->num_diagonals = tmp.num_diagonals;
    m->storage = tmp.storage;
    m->offsets = tmp.offsets;
    m->val = (real::storage_t *) tmp.val;
    return 1;
}

} // namespace cellshard
