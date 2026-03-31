#include "matrix.cuh"

#include <cstdio>
#include <cstdlib>

namespace cellshard {

namespace {

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

inline void *alloc_bytes(std::size_t bytes) {
    if (bytes == 0) return 0;
    return std::malloc(bytes);
}

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

int store_dense_raw(const char *filename, types::dim_t rows, types::dim_t cols, types::nnz_t nnz, const void *val, std::size_t value_size) {
    std::FILE *fp = 0;
    int ok = 0;

    fp = std::fopen(filename, "wb");
    if (fp == 0) return 0;
    if (!write_header(fp, disk_format_dense, rows, cols, nnz)) goto done;
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
    std::fclose(fp);
    return ok;
}

int store_compressed_raw(const char *filename, types::dim_t rows, types::dim_t cols, types::nnz_t nnz, types::u32 axis, types::dim_t major_dim, const types::ptr_t *majorPtr, const types::idx_t *minorIdx, const void *val, std::size_t value_size) {
    std::FILE *fp = 0;
    int ok = 0;

    fp = std::fopen(filename, "wb");
    if (fp == 0) return 0;
    if (!write_header(fp, disk_format_compressed, rows, cols, nnz)) goto done;
    if (!write_block(fp, &axis, sizeof(axis), 1)) goto done;
    if (!write_block(fp, val, value_size, nnz)) goto done;
    if (major_dim != 0 && !write_block(fp, majorPtr, sizeof(types::ptr_t), (std::size_t) major_dim + 1)) goto done;
    if (!write_block(fp, minorIdx, sizeof(types::idx_t), nnz)) goto done;
    ok = 1;

done:
    std::fclose(fp);
    return ok;
}

int load_compressed_raw(const char *filename, std::size_t value_size, compressed_load_result *out) {
    std::FILE *fp = 0;
    int ok = 0;
    types::dim_t major_dim = 0;

    out->axis = sparse::compressed_by_row;
    out->majorPtr = 0;
    out->minorIdx = 0;
    out->val = 0;
    fp = std::fopen(filename, "rb");
    if (fp == 0) return 0;
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
    std::fclose(fp);
    return ok;
}

int store_coo_raw(const char *filename, types::dim_t rows, types::dim_t cols, types::nnz_t nnz, const types::idx_t *rowIdx, const types::idx_t *colIdx, const void *val, std::size_t value_size) {
    std::FILE *fp = 0;
    int ok = 0;

    fp = std::fopen(filename, "wb");
    if (fp == 0) return 0;
    if (!write_header(fp, disk_format_coo, rows, cols, nnz)) goto done;
    if (!write_block(fp, val, value_size, nnz)) goto done;
    if (!write_block(fp, rowIdx, sizeof(types::idx_t), nnz)) goto done;
    if (!write_block(fp, colIdx, sizeof(types::idx_t), nnz)) goto done;
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
    std::fclose(fp);
    return ok;
}

int store_dia_raw(const char *filename, types::dim_t rows, types::dim_t cols, types::nnz_t nnz, types::idx_t num_diagonals, const int *offsets, const void *val, std::size_t value_size) {
    std::FILE *fp = 0;
    int ok = 0;

    fp = std::fopen(filename, "wb");
    if (fp == 0) return 0;
    if (!write_header(fp, disk_format_dia, rows, cols, nnz)) goto done;
    if (!write_block(fp, &num_diagonals, sizeof(types::idx_t), 1)) goto done;
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
    std::fclose(fp);
    return ok;
}

} // namespace cellshard
