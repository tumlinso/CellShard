#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "../../formats/compressed.cuh"
#include "../../formats/blocked_ell.cuh"
#include "../../formats/quantized_blocked_ell.cuh"
#include "../../formats/sliced_ell.cuh"
#include "../../formats/dense.cuh"
#include "../../formats/diagonal.cuh"
#include "../../formats/triplet.cuh"

namespace cellshard {

// Per-part binary format tags used inside the native disk payload.
enum {
    disk_format_none       = 0,
    disk_format_dense      = 1,
    disk_format_compressed = 2,
    disk_format_coo        = 3,
    disk_format_dia        = 4,
    disk_format_ell        = 5,
    disk_format_blocked_ell = 6,
    disk_format_sliced_ell = 7,
    disk_format_quantized_blocked_ell = 8
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
struct disk_format_code<sparse::blocked_ell> {
    enum { value = disk_format_blocked_ell };
    static inline const char *name() { return "blocked ell matrix"; }
};

template<>
struct disk_format_code<sparse::quantized_blocked_ell> {
    enum { value = disk_format_quantized_blocked_ell };
    static inline const char *name() { return "quantized blocked ell matrix"; }
};

template<>
struct disk_format_code<sparse::sliced_ell> {
    enum { value = disk_format_sliced_ell };
    static inline const char *name() { return "sliced ell matrix"; }
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

struct blocked_ell_load_result {
    disk_header h;
    types::u32 block_size;
    types::u32 ell_cols;
    void *storage;
    types::idx_t *blockColIdx;
    void *val;
};

struct quantized_blocked_ell_load_result {
    disk_header h;
    types::u32 block_size;
    types::u32 ell_cols;
    types::u32 bits;
    types::u32 row_stride_bytes;
    types::u32 decode_policy;
    void *storage;
    types::idx_t *blockColIdx;
    std::uint8_t *packed_values;
    float *column_scales;
    float *column_offsets;
    float *row_offsets;
};

struct sliced_ell_load_result {
    disk_header h;
    types::u32 slice_count;
    void *storage;
    types::u32 *slice_row_offsets;
    types::u32 *slice_widths;
    types::idx_t *col_idx;
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
std::size_t packed_sliced_ell_bytes(types::u32 slice_count, types::u32 total_slots, std::size_t value_size);
inline std::size_t packed_blocked_ell_bytes(types::dim_t rows, types::u32 ell_cols, types::u32 block_size, std::size_t value_size) {
    return sizeof(disk_header)
        + sizeof(types::u32)
        + sizeof(types::u32)
        + (std::size_t) ((rows + block_size - 1u) / block_size) * (std::size_t) (ell_cols / block_size) * sizeof(types::idx_t)
        + (std::size_t) rows * (std::size_t) ell_cols * value_size;
}

inline std::size_t packed_quantized_blocked_ell_bytes(
    types::dim_t rows,
    types::dim_t cols,
    types::u32 bits,
    types::u32 ell_cols,
    types::u32 block_size) {
    const std::size_t row_blocks = block_size == 0u ? 0u : (std::size_t) ((rows + block_size - 1u) / block_size);
    const std::size_t ell_width = block_size == 0u ? 0u : (std::size_t) (ell_cols / block_size);
    return sizeof(disk_header)
        + sizeof(types::u32) * 5u
        + row_blocks * ell_width * sizeof(types::idx_t)
        + (std::size_t) rows * (std::size_t) sparse::quantized_blocked_ell_aligned_row_bytes(bits, ell_cols)
        + (std::size_t) cols * sizeof(float)
        + (std::size_t) cols * sizeof(float)
        + (std::size_t) rows * sizeof(float);
}

int store_dense_raw(const char *filename, types::dim_t rows, types::dim_t cols, types::nnz_t nnz, const void *val, std::size_t value_size);
int load_dense_raw(const char *filename, std::size_t value_size, dense_load_result *out);
int store_dense_raw(std::FILE *fp, types::dim_t rows, types::dim_t cols, types::nnz_t nnz, const void *val, std::size_t value_size);
int load_dense_raw(std::FILE *fp, std::size_t value_size, dense_load_result *out);

int store_compressed_raw(const char *filename, types::dim_t rows, types::dim_t cols, types::nnz_t nnz, types::u32 axis, types::dim_t major_dim, const types::ptr_t *majorPtr, const types::idx_t *minorIdx, const void *val, std::size_t value_size);
int load_compressed_raw(const char *filename, std::size_t value_size, compressed_load_result *out);
int store_compressed_raw(std::FILE *fp, types::dim_t rows, types::dim_t cols, types::nnz_t nnz, types::u32 axis, types::dim_t major_dim, const types::ptr_t *majorPtr, const types::idx_t *minorIdx, const void *val, std::size_t value_size);
int load_compressed_raw(std::FILE *fp, std::size_t value_size, compressed_load_result *out);

int store_blocked_ell_raw(const char *filename, types::dim_t rows, types::dim_t cols, types::nnz_t nnz, types::u32 block_size, types::u32 ell_cols, const types::idx_t *blockColIdx, const void *val, std::size_t value_size);
int load_blocked_ell_raw(const char *filename, std::size_t value_size, blocked_ell_load_result *out);
int store_blocked_ell_raw(std::FILE *fp, types::dim_t rows, types::dim_t cols, types::nnz_t nnz, types::u32 block_size, types::u32 ell_cols, const types::idx_t *blockColIdx, const void *val, std::size_t value_size);
int load_blocked_ell_raw(std::FILE *fp, std::size_t value_size, blocked_ell_load_result *out);

int store_quantized_blocked_ell_raw(const char *filename,
                                    types::dim_t rows,
                                    types::dim_t cols,
                                    types::nnz_t nnz,
                                    types::u32 block_size,
                                    types::u32 ell_cols,
                                    types::u32 bits,
                                    types::u32 row_stride_bytes,
                                    types::u32 decode_policy,
                                    const types::idx_t *blockColIdx,
                                    const std::uint8_t *packed_values,
                                    const float *column_scales,
                                    const float *column_offsets,
                                    const float *row_offsets);
int load_quantized_blocked_ell_raw(const char *filename, quantized_blocked_ell_load_result *out);
int store_quantized_blocked_ell_raw(std::FILE *fp,
                                    types::dim_t rows,
                                    types::dim_t cols,
                                    types::nnz_t nnz,
                                    types::u32 block_size,
                                    types::u32 ell_cols,
                                    types::u32 bits,
                                    types::u32 row_stride_bytes,
                                    types::u32 decode_policy,
                                    const types::idx_t *blockColIdx,
                                    const std::uint8_t *packed_values,
                                    const float *column_scales,
                                    const float *column_offsets,
                                    const float *row_offsets);
int load_quantized_blocked_ell_raw(std::FILE *fp, quantized_blocked_ell_load_result *out);

int store_sliced_ell_raw(const char *filename,
                         types::dim_t rows,
                         types::dim_t cols,
                         types::nnz_t nnz,
                         types::u32 slice_count,
                         const types::u32 *slice_row_offsets,
                         const types::u32 *slice_widths,
                         const types::idx_t *col_idx,
                         const void *val,
                         std::size_t value_size);
int load_sliced_ell_raw(const char *filename, std::size_t value_size, sliced_ell_load_result *out);
int store_sliced_ell_raw(std::FILE *fp,
                         types::dim_t rows,
                         types::dim_t cols,
                         types::nnz_t nnz,
                         types::u32 slice_count,
                         const types::u32 *slice_row_offsets,
                         const types::u32 *slice_widths,
                         const types::idx_t *col_idx,
                         const void *val,
                         std::size_t value_size);
int load_sliced_ell_raw(std::FILE *fp, std::size_t value_size, sliced_ell_load_result *out);

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

inline std::size_t packed_bytes(const sparse::blocked_ell *, types::dim_t rows, types::dim_t, types::nnz_t, unsigned long aux, std::size_t value_size) {
    return packed_blocked_ell_bytes(rows, sparse::unpack_blocked_ell_cols(aux), sparse::unpack_blocked_ell_block_size(aux), value_size);
}

inline std::size_t packed_bytes(const sparse::quantized_blocked_ell *, types::dim_t rows, types::dim_t cols, types::nnz_t, unsigned long aux, std::size_t) {
    return packed_quantized_blocked_ell_bytes(rows,
                                              cols,
                                              sparse::unpack_quantized_blocked_ell_bits(aux),
                                              sparse::unpack_quantized_blocked_ell_cols(aux),
                                              sparse::unpack_quantized_blocked_ell_block_size(aux));
}

inline std::size_t packed_bytes(const sparse::sliced_ell *, types::dim_t, types::dim_t, types::nnz_t, unsigned long aux, std::size_t value_size) {
    return packed_sliced_ell_bytes(sparse::unpack_sliced_ell_slice_count(aux), sparse::unpack_sliced_ell_total_slots(aux), value_size);
}

// FILE* variants let shard-pack cache code write/read many parts through one open file
// handle, which avoids reopen churn in sequential shard fetch/store loops.
int store(std::FILE *fp, const dense *m);
int load(std::FILE *fp, dense *m);
int store(std::FILE *fp, const sparse::compressed *m);
int load(std::FILE *fp, sparse::compressed *m);
int store(std::FILE *fp, const sparse::blocked_ell *m);
int load(std::FILE *fp, sparse::blocked_ell *m);
int store(std::FILE *fp, const sparse::quantized_blocked_ell *m);
int load(std::FILE *fp, sparse::quantized_blocked_ell *m);
int store(std::FILE *fp, const sparse::sliced_ell *m);
int load(std::FILE *fp, sparse::sliced_ell *m);
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

inline int store(const char *filename, const sparse::blocked_ell *m) {
    return store_blocked_ell_raw(
        filename,
        m->rows,
        m->cols,
        m->nnz,
        m->block_size,
        m->ell_cols,
        m->blockColIdx,
        m->val,
        sizeof(real::storage_t)
    );
}

inline int load(const char *filename, sparse::blocked_ell *m) {
    blocked_ell_load_result tmp;

    tmp.block_size = 0u;
    tmp.ell_cols = 0u;
    tmp.storage = 0;
    tmp.blockColIdx = 0;
    tmp.val = 0;
    if (!load_blocked_ell_raw(filename, sizeof(real::storage_t), &tmp)) return 0;
    sparse::clear(m);
    sparse::init(m, tmp.h.rows, tmp.h.cols, tmp.h.nnz, tmp.block_size, tmp.ell_cols);
    m->storage = tmp.storage;
    m->blockColIdx = tmp.blockColIdx;
    m->val = (real::storage_t *) tmp.val;
    return 1;
}

inline int store(const char *filename, const sparse::quantized_blocked_ell *m) {
    return store_quantized_blocked_ell_raw(filename,
                                           m->rows,
                                           m->cols,
                                           m->nnz,
                                           m->block_size,
                                           m->ell_cols,
                                           m->bits,
                                           m->row_stride_bytes,
                                           m->decode_policy,
                                           m->blockColIdx,
                                           m->packed_values,
                                           m->column_scales,
                                           m->column_offsets,
                                           m->row_offsets);
}

inline int load(const char *filename, sparse::quantized_blocked_ell *m) {
    quantized_blocked_ell_load_result tmp;

    tmp.storage = 0;
    tmp.blockColIdx = 0;
    tmp.packed_values = 0;
    tmp.column_scales = 0;
    tmp.column_offsets = 0;
    tmp.row_offsets = 0;
    if (!load_quantized_blocked_ell_raw(filename, &tmp)) return 0;
    sparse::clear(m);
    sparse::init(m,
                 tmp.h.rows,
                 tmp.h.cols,
                 tmp.h.nnz,
                 tmp.block_size,
                 tmp.ell_cols,
                 tmp.bits,
                 tmp.decode_policy,
                 tmp.row_stride_bytes);
    m->storage = tmp.storage;
    m->blockColIdx = tmp.blockColIdx;
    m->packed_values = tmp.packed_values;
    m->column_scales = tmp.column_scales;
    m->column_offsets = tmp.column_offsets;
    m->row_offsets = tmp.row_offsets;
    return 1;
}

inline int store(const char *filename, const sparse::sliced_ell *m) {
    return store_sliced_ell_raw(filename,
                                m->rows,
                                m->cols,
                                m->nnz,
                                m->slice_count,
                                m->slice_row_offsets,
                                m->slice_widths,
                                m->col_idx,
                                m->val,
                                sizeof(real::storage_t));
}

inline int load(const char *filename, sparse::sliced_ell *m) {
    sliced_ell_load_result tmp;

    tmp.slice_count = 0u;
    tmp.storage = 0;
    tmp.slice_row_offsets = 0;
    tmp.slice_widths = 0;
    tmp.col_idx = 0;
    tmp.val = 0;
    if (!load_sliced_ell_raw(filename, sizeof(real::storage_t), &tmp)) return 0;
    sparse::clear(m);
    sparse::init(m, tmp.h.rows, tmp.h.cols, tmp.h.nnz);
    m->slice_count = tmp.slice_count;
    m->storage = tmp.storage;
    m->slice_row_offsets = tmp.slice_row_offsets;
    m->slice_widths = tmp.slice_widths;
    m->col_idx = tmp.col_idx;
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
