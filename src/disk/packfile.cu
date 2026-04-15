#include "packfile.cuh"

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

inline void configure_stream(std::FILE *fp) {
    std::setvbuf(fp, 0, _IOFBF, (std::size_t) 8u << 20u);
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

inline std::size_t align_up_bytes(std::size_t value, std::size_t alignment) {
    return (value + alignment - 1u) & ~(alignment - 1u);
}

// Cleanup helpers for partially built load results.
inline void free_compressed_result(compressed_load_result *out) {
    if (out->storage != 0) std::free(out->storage);
    else {
        std::free(out->majorPtr);
        std::free(out->minorIdx);
        std::free(out->val);
    }
    out->storage = 0;
    out->majorPtr = 0;
    out->minorIdx = 0;
    out->val = 0;
}

inline void free_blocked_ell_result(blocked_ell_load_result *out) {
    if (out->storage != 0) std::free(out->storage);
    else {
        std::free(out->blockColIdx);
        std::free(out->val);
    }
    out->storage = 0;
    out->blockColIdx = 0;
    out->val = 0;
    out->block_size = 0u;
    out->ell_cols = 0u;
}

inline void free_quantized_blocked_ell_result(quantized_blocked_ell_load_result *out) {
    if (out->storage != 0) std::free(out->storage);
    else {
        std::free(out->blockColIdx);
        std::free(out->packed_values);
        std::free(out->column_scales);
        std::free(out->column_offsets);
        std::free(out->row_offsets);
    }
    out->storage = 0;
    out->blockColIdx = 0;
    out->packed_values = 0;
    out->column_scales = 0;
    out->column_offsets = 0;
    out->row_offsets = 0;
    out->block_size = 0u;
    out->ell_cols = 0u;
    out->bits = 0u;
    out->row_stride_bytes = 0u;
    out->decode_policy = 0u;
}

inline void free_sliced_ell_result(sliced_ell_load_result *out) {
    if (out->storage != 0) std::free(out->storage);
    else {
        std::free(out->slice_row_offsets);
        std::free(out->slice_widths);
        std::free(out->col_idx);
        std::free(out->val);
    }
    out->storage = 0;
    out->slice_row_offsets = 0;
    out->slice_widths = 0;
    out->col_idx = 0;
    out->val = 0;
    out->slice_count = 0u;
}

inline void free_coo_result(coo_load_result *out) {
    if (out->storage != 0) std::free(out->storage);
    else {
        std::free(out->rowIdx);
        std::free(out->colIdx);
        std::free(out->val);
    }
    out->storage = 0;
    out->rowIdx = 0;
    out->colIdx = 0;
    out->val = 0;
}

inline void free_dia_result(dia_load_result *out) {
    if (out->storage != 0) std::free(out->storage);
    else {
        std::free(out->offsets);
        std::free(out->val);
    }
    out->storage = 0;
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

std::size_t packed_sliced_ell_bytes(types::u32 slice_count, types::u32 total_slots, std::size_t value_size) {
    const std::size_t offsets_bytes = ((std::size_t) slice_count + 1u) * sizeof(types::u32);
    const std::size_t widths_bytes = (std::size_t) slice_count * sizeof(types::u32);
    return header_bytes()
        + sizeof(types::u32)
        + offsets_bytes
        + widths_bytes
        + (std::size_t) total_slots * sizeof(types::idx_t)
        + (std::size_t) total_slots * value_size;
}

// Standalone filename helpers are full synchronous host I/O operations.
int store_dense_raw(const char *filename, types::dim_t rows, types::dim_t cols, types::nnz_t nnz, const void *val, std::size_t value_size) {
    std::FILE *fp = 0;
    int ok = 0;

    fp = std::fopen(filename, "wb");
    if (fp == 0) return 0;
    configure_stream(fp);
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
    configure_stream(fp);
    ok = load_dense_raw(fp, value_size, out);

done:
    std::fclose(fp);
    return ok;
}

// FILE* load reads one packed dense payload into a fresh host allocation.
int load_dense_raw(std::FILE *fp, std::size_t value_size, dense_load_result *out) {
    int ok = 0;

    out->storage = 0;
    out->val = 0;
    if (!read_header(fp, &out->h)) goto done;
    if (!check_disk_format(disk_format_dense, out->h.format, "dense matrix")) goto done;
    out->storage = alloc_bytes((std::size_t) out->h.nnz * value_size);
    out->val = out->storage;
    if (out->h.nnz != 0 && out->storage == 0) goto done;
    if (!read_block(fp, out->val, value_size, out->h.nnz)) goto done;
    ok = 1;

done:
    if (!ok) {
        std::free(out->storage);
        out->storage = 0;
        out->val = 0;
    }
    return ok;
}

int store_compressed_raw(const char *filename, types::dim_t rows, types::dim_t cols, types::nnz_t nnz, types::u32 axis, types::dim_t major_dim, const types::ptr_t *majorPtr, const types::idx_t *minorIdx, const void *val, std::size_t value_size) {
    std::FILE *fp = 0;
    int ok = 0;

    fp = std::fopen(filename, "wb");
    if (fp == 0) return 0;
    configure_stream(fp);
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
    out->storage = 0;
    out->majorPtr = 0;
    out->minorIdx = 0;
    out->val = 0;
    fp = std::fopen(filename, "rb");
    if (fp == 0) return 0;
    configure_stream(fp);
    ok = load_compressed_raw(fp, value_size, out);

done:
    std::fclose(fp);
    return ok;
}

int load_compressed_raw(std::FILE *fp, std::size_t value_size, compressed_load_result *out) {
    int ok = 0;
    types::dim_t major_dim = 0;
    std::size_t major_bytes = 0;
    std::size_t minor_offset = 0;
    std::size_t val_offset = 0;
    std::size_t total_bytes = 0;

    out->axis = sparse::compressed_by_row;
    out->storage = 0;
    out->majorPtr = 0;
    out->minorIdx = 0;
    out->val = 0;
    if (!read_header(fp, &out->h)) goto done;
    if (!check_disk_format(disk_format_compressed, out->h.format, "compressed matrix")) goto done;
    if (!read_block(fp, &out->axis, sizeof(out->axis), 1)) goto done;
    major_dim = out->axis == sparse::compressed_by_col ? out->h.cols : out->h.rows;
    major_bytes = ((std::size_t) major_dim + 1u) * sizeof(types::ptr_t);
    minor_offset = align_up_bytes(major_bytes, alignof(types::idx_t));
    val_offset = align_up_bytes(minor_offset + (std::size_t) out->h.nnz * sizeof(types::idx_t), alignof(real::storage_t));
    total_bytes = val_offset + (std::size_t) out->h.nnz * value_size;
    out->storage = alloc_bytes(total_bytes);
    if (total_bytes != 0 && out->storage == 0) goto done;
    out->majorPtr = major_dim != 0 ? (types::ptr_t *) out->storage : 0;
    out->minorIdx = out->h.nnz != 0 ? (types::idx_t *) ((char *) out->storage + minor_offset) : 0;
    out->val = out->h.nnz != 0 ? (void *) ((char *) out->storage + val_offset) : 0;
    if (!read_block(fp, out->val, value_size, out->h.nnz)) goto done;
    if (major_dim != 0 && !read_block(fp, out->majorPtr, sizeof(types::ptr_t), (std::size_t) major_dim + 1)) goto done;
    if (!read_block(fp, out->minorIdx, sizeof(types::idx_t), out->h.nnz)) goto done;
    ok = 1;

done:
    if (!ok) free_compressed_result(out);
    return ok;
}

int store_blocked_ell_raw(const char *filename,
                          types::dim_t rows,
                          types::dim_t cols,
                          types::nnz_t nnz,
                          types::u32 block_size,
                          types::u32 ell_cols,
                          const types::idx_t *blockColIdx,
                          const void *val,
                          std::size_t value_size) {
    std::FILE *fp = 0;
    int ok = 0;

    fp = std::fopen(filename, "wb");
    if (fp == 0) return 0;
    configure_stream(fp);
    ok = store_blocked_ell_raw(fp, rows, cols, nnz, block_size, ell_cols, blockColIdx, val, value_size);

done:
    std::fclose(fp);
    return ok;
}

int store_blocked_ell_raw(std::FILE *fp,
                          types::dim_t rows,
                          types::dim_t cols,
                          types::nnz_t nnz,
                          types::u32 block_size,
                          types::u32 ell_cols,
                          const types::idx_t *blockColIdx,
                          const void *val,
                          std::size_t value_size) {
    const std::size_t row_blocks = block_size == 0u ? 0u : ((std::size_t) rows + block_size - 1u) / block_size;
    const std::size_t ell_width = block_size == 0u ? 0u : (std::size_t) ell_cols / block_size;

    if (!write_header(fp, disk_format_blocked_ell, rows, cols, nnz)) return 0;
    if (!write_block(fp, &block_size, sizeof(block_size), 1)) return 0;
    if (!write_block(fp, &ell_cols, sizeof(ell_cols), 1)) return 0;
    if (!write_block(fp, blockColIdx, sizeof(types::idx_t), row_blocks * ell_width)) return 0;
    if (!write_block(fp, val, value_size, (std::size_t) rows * (std::size_t) ell_cols)) return 0;
    return 1;
}

int load_blocked_ell_raw(const char *filename, std::size_t value_size, blocked_ell_load_result *out) {
    std::FILE *fp = 0;
    int ok = 0;

    out->block_size = 0u;
    out->ell_cols = 0u;
    out->storage = 0;
    out->blockColIdx = 0;
    out->val = 0;
    fp = std::fopen(filename, "rb");
    if (fp == 0) return 0;
    configure_stream(fp);
    ok = load_blocked_ell_raw(fp, value_size, out);

done:
    std::fclose(fp);
    return ok;
}

int load_blocked_ell_raw(std::FILE *fp, std::size_t value_size, blocked_ell_load_result *out) {
    int ok = 0;
    const std::size_t idx_bytes_alignment = alignof(types::idx_t);
    const std::size_t val_bytes_alignment = alignof(real::storage_t);
    std::size_t idx_bytes = 0u;
    std::size_t val_offset = 0u;
    std::size_t total_bytes = 0u;
    std::size_t row_blocks = 0u;
    std::size_t ell_width = 0u;

    out->block_size = 0u;
    out->ell_cols = 0u;
    out->storage = 0;
    out->blockColIdx = 0;
    out->val = 0;
    if (!read_header(fp, &out->h)) goto done;
    if (!check_disk_format(disk_format_blocked_ell, out->h.format, "blocked ell matrix")) goto done;
    if (!read_block(fp, &out->block_size, sizeof(out->block_size), 1)) goto done;
    if (!read_block(fp, &out->ell_cols, sizeof(out->ell_cols), 1)) goto done;
    row_blocks = out->block_size == 0u ? 0u : ((std::size_t) out->h.rows + out->block_size - 1u) / out->block_size;
    ell_width = out->block_size == 0u ? 0u : (std::size_t) out->ell_cols / out->block_size;
    idx_bytes = row_blocks * ell_width * sizeof(types::idx_t);
    val_offset = align_up_bytes(idx_bytes, val_bytes_alignment > idx_bytes_alignment ? val_bytes_alignment : idx_bytes_alignment);
    total_bytes = val_offset + (std::size_t) out->h.rows * (std::size_t) out->ell_cols * value_size;
    out->storage = alloc_bytes(total_bytes);
    if (total_bytes != 0u && out->storage == 0) goto done;
    out->blockColIdx = idx_bytes != 0u ? (types::idx_t *) out->storage : 0;
    out->val = out->h.rows != 0u && out->ell_cols != 0u ? (void *) ((char *) out->storage + val_offset) : 0;
    if (!read_block(fp, out->blockColIdx, sizeof(types::idx_t), row_blocks * ell_width)) goto done;
    if (!read_block(fp, out->val, value_size, (std::size_t) out->h.rows * (std::size_t) out->ell_cols)) goto done;
    ok = 1;

done:
    if (!ok) free_blocked_ell_result(out);
    return ok;
}

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
                                    const float *row_offsets) {
    std::FILE *fp = 0;
    int ok = 0;

    fp = std::fopen(filename, "wb");
    if (fp == 0) return 0;
    configure_stream(fp);
    ok = store_quantized_blocked_ell_raw(fp,
                                         rows,
                                         cols,
                                         nnz,
                                         block_size,
                                         ell_cols,
                                         bits,
                                         row_stride_bytes,
                                         decode_policy,
                                         blockColIdx,
                                         packed_values,
                                         column_scales,
                                         column_offsets,
                                         row_offsets);

done:
    std::fclose(fp);
    return ok;
}

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
                                    const float *row_offsets) {
    const std::size_t row_blocks = block_size == 0u ? 0u : (std::size_t) ((rows + block_size - 1u) / block_size);
    const std::size_t ell_width = block_size == 0u ? 0u : (std::size_t) (ell_cols / block_size);
    const std::size_t packed_bytes = (std::size_t) rows * (std::size_t) row_stride_bytes;

    if (!write_header(fp, disk_format_quantized_blocked_ell, rows, cols, nnz)) return 0;
    if (!write_block(fp, &block_size, sizeof(block_size), 1)) return 0;
    if (!write_block(fp, &ell_cols, sizeof(ell_cols), 1)) return 0;
    if (!write_block(fp, &bits, sizeof(bits), 1)) return 0;
    if (!write_block(fp, &row_stride_bytes, sizeof(row_stride_bytes), 1)) return 0;
    if (!write_block(fp, &decode_policy, sizeof(decode_policy), 1)) return 0;
    if (row_blocks * ell_width != 0u && !write_block(fp, blockColIdx, sizeof(types::idx_t), row_blocks * ell_width)) return 0;
    if (packed_bytes != 0u && !write_block(fp, packed_values, sizeof(std::uint8_t), packed_bytes)) return 0;
    if (cols != 0 && !write_block(fp, column_scales, sizeof(float), (std::size_t) cols)) return 0;
    if (cols != 0 && !write_block(fp, column_offsets, sizeof(float), (std::size_t) cols)) return 0;
    if (rows != 0 && !write_block(fp, row_offsets, sizeof(float), (std::size_t) rows)) return 0;
    return 1;
}

int load_quantized_blocked_ell_raw(const char *filename, quantized_blocked_ell_load_result *out) {
    std::FILE *fp = 0;
    int ok = 0;

    out->storage = 0;
    out->blockColIdx = 0;
    out->packed_values = 0;
    out->column_scales = 0;
    out->column_offsets = 0;
    out->row_offsets = 0;
    fp = std::fopen(filename, "rb");
    if (fp == 0) return 0;
    configure_stream(fp);
    ok = load_quantized_blocked_ell_raw(fp, out);

done:
    std::fclose(fp);
    return ok;
}

int load_quantized_blocked_ell_raw(std::FILE *fp, quantized_blocked_ell_load_result *out) {
    int ok = 0;
    std::size_t storage_bytes = 0u;
    std::size_t idx_bytes = 0u;
    std::size_t packed_offset = 0u;
    std::size_t packed_bytes = 0u;
    std::size_t column_scales_offset = 0u;
    std::size_t column_offsets_offset = 0u;
    std::size_t row_offsets_offset = 0u;
    std::size_t row_blocks = 0u;
    std::size_t ell_width = 0u;

    out->storage = 0;
    out->blockColIdx = 0;
    out->packed_values = 0;
    out->column_scales = 0;
    out->column_offsets = 0;
    out->row_offsets = 0;
    if (!read_header(fp, &out->h)) goto done;
    if (!check_disk_format(disk_format_quantized_blocked_ell, out->h.format, "quantized blocked ell matrix")) goto done;
    if (!read_block(fp, &out->block_size, sizeof(out->block_size), 1)) goto done;
    if (!read_block(fp, &out->ell_cols, sizeof(out->ell_cols), 1)) goto done;
    if (!read_block(fp, &out->bits, sizeof(out->bits), 1)) goto done;
    if (!read_block(fp, &out->row_stride_bytes, sizeof(out->row_stride_bytes), 1)) goto done;
    if (!read_block(fp, &out->decode_policy, sizeof(out->decode_policy), 1)) goto done;
    row_blocks = out->block_size == 0u ? 0u : (std::size_t) ((out->h.rows + out->block_size - 1u) / out->block_size);
    ell_width = out->block_size == 0u ? 0u : (std::size_t) (out->ell_cols / out->block_size);
    idx_bytes = row_blocks * ell_width * sizeof(types::idx_t);
    packed_offset = align_up_bytes(idx_bytes, alignof(std::uint8_t));
    packed_bytes = (std::size_t) out->h.rows * (std::size_t) out->row_stride_bytes;
    column_scales_offset = align_up_bytes(packed_offset + packed_bytes, alignof(float));
    column_offsets_offset = column_scales_offset + (std::size_t) out->h.cols * sizeof(float);
    row_offsets_offset = column_offsets_offset + (std::size_t) out->h.cols * sizeof(float);
    storage_bytes = row_offsets_offset + (std::size_t) out->h.rows * sizeof(float);
    out->storage = alloc_bytes(storage_bytes);
    if (storage_bytes != 0u && out->storage == 0) goto done;
    out->blockColIdx = idx_bytes != 0u ? (types::idx_t *) out->storage : 0;
    out->packed_values = packed_bytes != 0u ? (std::uint8_t *) ((char *) out->storage + packed_offset) : 0;
    out->column_scales = out->h.cols != 0u ? (float *) ((char *) out->storage + column_scales_offset) : 0;
    out->column_offsets = out->h.cols != 0u ? (float *) ((char *) out->storage + column_offsets_offset) : 0;
    out->row_offsets = out->h.rows != 0u ? (float *) ((char *) out->storage + row_offsets_offset) : 0;
    if (idx_bytes != 0u && !read_block(fp, out->blockColIdx, sizeof(types::idx_t), row_blocks * ell_width)) goto done;
    if (packed_bytes != 0u && !read_block(fp, out->packed_values, sizeof(std::uint8_t), packed_bytes)) goto done;
    if (out->h.cols != 0u && !read_block(fp, out->column_scales, sizeof(float), (std::size_t) out->h.cols)) goto done;
    if (out->h.cols != 0u && !read_block(fp, out->column_offsets, sizeof(float), (std::size_t) out->h.cols)) goto done;
    if (out->h.rows != 0u && !read_block(fp, out->row_offsets, sizeof(float), (std::size_t) out->h.rows)) goto done;
    ok = 1;

done:
    if (!ok) free_quantized_blocked_ell_result(out);
    return ok;
}

int store_sliced_ell_raw(const char *filename,
                         types::dim_t rows,
                         types::dim_t cols,
                         types::nnz_t nnz,
                         types::u32 slice_count,
                         const types::u32 *slice_row_offsets,
                         const types::u32 *slice_widths,
                         const types::idx_t *col_idx,
                         const void *val,
                         std::size_t value_size) {
    std::FILE *fp = 0;
    int ok = 0;

    fp = std::fopen(filename, "wb");
    if (fp == 0) return 0;
    configure_stream(fp);
    ok = store_sliced_ell_raw(fp, rows, cols, nnz, slice_count, slice_row_offsets, slice_widths, col_idx, val, value_size);

done:
    std::fclose(fp);
    return ok;
}

int store_sliced_ell_raw(std::FILE *fp,
                         types::dim_t rows,
                         types::dim_t cols,
                         types::nnz_t nnz,
                         types::u32 slice_count,
                         const types::u32 *slice_row_offsets,
                         const types::u32 *slice_widths,
                         const types::idx_t *col_idx,
                         const void *val,
                         std::size_t value_size) {
    types::u32 total_slots = 0u;
    types::u32 slice = 0u;

    if (!write_header(fp, disk_format_sliced_ell, rows, cols, nnz)) return 0;
    if (!write_block(fp, &slice_count, sizeof(slice_count), 1)) return 0;
    if (slice_count != 0u && (slice_row_offsets == 0 || slice_widths == 0)) return 0;
    for (slice = 0u; slice < slice_count; ++slice) {
        total_slots += (slice_row_offsets[slice + 1u] - slice_row_offsets[slice]) * slice_widths[slice];
    }
    if (!write_block(fp, slice_row_offsets, sizeof(types::u32), (std::size_t) slice_count + 1u)) return 0;
    if (!write_block(fp, slice_widths, sizeof(types::u32), slice_count)) return 0;
    if (!write_block(fp, col_idx, sizeof(types::idx_t), total_slots)) return 0;
    if (!write_block(fp, val, value_size, total_slots)) return 0;
    return 1;
}

int load_sliced_ell_raw(const char *filename, std::size_t value_size, sliced_ell_load_result *out) {
    std::FILE *fp = 0;
    int ok = 0;

    out->slice_count = 0u;
    out->storage = 0;
    out->slice_row_offsets = 0;
    out->slice_widths = 0;
    out->col_idx = 0;
    out->val = 0;
    fp = std::fopen(filename, "rb");
    if (fp == 0) return 0;
    configure_stream(fp);
    ok = load_sliced_ell_raw(fp, value_size, out);

done:
    std::fclose(fp);
    return ok;
}

int load_sliced_ell_raw(std::FILE *fp, std::size_t value_size, sliced_ell_load_result *out) {
    int ok = 0;
    types::u32 total_slots = 0u;
    types::u32 slice = 0u;
    std::size_t offsets_bytes = 0u;
    std::size_t widths_offset = 0u;
    std::size_t widths_bytes = 0u;
    std::size_t col_offset = 0u;
    std::size_t col_bytes = 0u;
    std::size_t val_offset = 0u;
    std::size_t total_bytes = 0u;

    out->slice_count = 0u;
    out->storage = 0;
    out->slice_row_offsets = 0;
    out->slice_widths = 0;
    out->col_idx = 0;
    out->val = 0;
    if (!read_header(fp, &out->h)) goto done;
    if (!check_disk_format(disk_format_sliced_ell, out->h.format, "sliced ell matrix")) goto done;
    if (!read_block(fp, &out->slice_count, sizeof(out->slice_count), 1)) goto done;
    offsets_bytes = ((std::size_t) out->slice_count + 1u) * sizeof(types::u32);
    widths_offset = align_up_bytes(offsets_bytes, alignof(types::u32));
    widths_bytes = (std::size_t) out->slice_count * sizeof(types::u32);
    total_bytes = align_up_bytes(widths_offset + widths_bytes, alignof(types::idx_t));
    out->storage = alloc_bytes(total_bytes);
    if (total_bytes != 0u && out->storage == 0) goto done;
    out->slice_row_offsets = out->slice_count != 0u || out->h.rows == 0u ? (types::u32 *) out->storage : 0;
    out->slice_widths = out->slice_count != 0u ? (types::u32 *) ((char *) out->storage + widths_offset) : 0;
    if (!read_block(fp, out->slice_row_offsets, sizeof(types::u32), (std::size_t) out->slice_count + 1u)) goto done;
    if (!read_block(fp, out->slice_widths, sizeof(types::u32), out->slice_count)) goto done;
    if (out->slice_count != 0u && out->slice_row_offsets[out->slice_count] != out->h.rows) goto done;
    for (slice = 0u; slice < out->slice_count; ++slice) {
        total_slots += (out->slice_row_offsets[slice + 1u] - out->slice_row_offsets[slice]) * out->slice_widths[slice];
    }
    col_offset = align_up_bytes(widths_offset + widths_bytes, alignof(types::idx_t));
    col_bytes = (std::size_t) total_slots * sizeof(types::idx_t);
    val_offset = align_up_bytes(col_offset + col_bytes, alignof(real::storage_t));
    total_bytes = val_offset + (std::size_t) total_slots * value_size;
    if (total_bytes != 0u) {
        void *storage = std::realloc(out->storage, total_bytes);
        if (storage == 0) goto done;
        out->storage = storage;
    }
    out->slice_row_offsets = (types::u32 *) out->storage;
    out->slice_widths = (types::u32 *) ((char *) out->storage + widths_offset);
    out->col_idx = total_slots != 0u ? (types::idx_t *) ((char *) out->storage + col_offset) : 0;
    out->val = total_slots != 0u ? (void *) ((char *) out->storage + val_offset) : 0;
    if (!read_block(fp, out->col_idx, sizeof(types::idx_t), total_slots)) goto done;
    if (!read_block(fp, out->val, value_size, total_slots)) goto done;
    ok = 1;

done:
    if (!ok) free_sliced_ell_result(out);
    return ok;
}

int store_coo_raw(const char *filename, types::dim_t rows, types::dim_t cols, types::nnz_t nnz, const types::idx_t *rowIdx, const types::idx_t *colIdx, const void *val, std::size_t value_size) {
    std::FILE *fp = 0;
    int ok = 0;

    fp = std::fopen(filename, "wb");
    if (fp == 0) return 0;
    configure_stream(fp);
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

    out->storage = 0;
    out->rowIdx = 0;
    out->colIdx = 0;
    out->val = 0;
    fp = std::fopen(filename, "rb");
    if (fp == 0) return 0;
    configure_stream(fp);
    ok = load_coo_raw(fp, value_size, out);

done:
    std::fclose(fp);
    return ok;
}

int load_coo_raw(std::FILE *fp, std::size_t value_size, coo_load_result *out) {
    int ok = 0;
    std::size_t row_bytes = 0;
    std::size_t col_offset = 0;
    std::size_t val_offset = 0;
    std::size_t total_bytes = 0;

    out->storage = 0;
    out->rowIdx = 0;
    out->colIdx = 0;
    out->val = 0;
    if (!read_header(fp, &out->h)) goto done;
    if (!check_disk_format(disk_format_coo, out->h.format, "coo matrix")) goto done;
    row_bytes = (std::size_t) out->h.nnz * sizeof(types::idx_t);
    col_offset = align_up_bytes(row_bytes, alignof(types::idx_t));
    val_offset = align_up_bytes(col_offset + (std::size_t) out->h.nnz * sizeof(types::idx_t), alignof(real::storage_t));
    total_bytes = val_offset + (std::size_t) out->h.nnz * value_size;
    out->storage = alloc_bytes(total_bytes);
    if (total_bytes != 0 && out->storage == 0) goto done;
    out->rowIdx = out->h.nnz != 0 ? (types::idx_t *) out->storage : 0;
    out->colIdx = out->h.nnz != 0 ? (types::idx_t *) ((char *) out->storage + col_offset) : 0;
    out->val = out->h.nnz != 0 ? (void *) ((char *) out->storage + val_offset) : 0;
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
    configure_stream(fp);
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
    out->storage = 0;
    out->offsets = 0;
    out->val = 0;
    fp = std::fopen(filename, "rb");
    if (fp == 0) return 0;
    configure_stream(fp);
    ok = load_dia_raw(fp, value_size, out);

done:
    std::fclose(fp);
    return ok;
}

int load_dia_raw(std::FILE *fp, std::size_t value_size, dia_load_result *out) {
    int ok = 0;
    std::size_t offsets_bytes = 0;
    std::size_t val_offset = 0;
    std::size_t total_bytes = 0;

    out->num_diagonals = 0;
    out->storage = 0;
    out->offsets = 0;
    out->val = 0;
    if (!read_header(fp, &out->h)) goto done;
    if (!check_disk_format(disk_format_dia, out->h.format, "dia matrix")) goto done;
    if (!read_block(fp, &out->num_diagonals, sizeof(types::idx_t), 1)) goto done;
    offsets_bytes = (std::size_t) out->num_diagonals * sizeof(int);
    val_offset = align_up_bytes(offsets_bytes, alignof(real::storage_t));
    total_bytes = val_offset + (std::size_t) out->h.nnz * value_size;
    out->storage = alloc_bytes(total_bytes);
    if (total_bytes != 0 && out->storage == 0) goto done;
    out->offsets = out->num_diagonals != 0 ? (int *) out->storage : 0;
    out->val = out->h.nnz != 0 ? (void *) ((char *) out->storage + val_offset) : 0;
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

    tmp.storage = 0;
    tmp.val = 0;
    if (!load_dense_raw(fp, sizeof(real::storage_t), &tmp)) return 0;
    clear(m);
    init(m, tmp.h.rows, tmp.h.cols);
    m->storage = tmp.storage;
    m->val = (real::storage_t *) tmp.val;
    return 1;
}

int store(std::FILE *fp, const sparse::compressed *m) {
    return store_compressed_raw(fp, m->rows, m->cols, m->nnz, m->axis, sparse::major_dim(m), m->majorPtr, m->minorIdx, m->val, sizeof(real::storage_t));
}

int load(std::FILE *fp, sparse::compressed *m) {
    compressed_load_result tmp;

    tmp.axis = sparse::compressed_by_row;
    tmp.storage = 0;
    tmp.majorPtr = 0;
    tmp.minorIdx = 0;
    tmp.val = 0;
    if (!load_compressed_raw(fp, sizeof(real::storage_t), &tmp)) return 0;
    sparse::clear(m);
    sparse::init(m, tmp.h.rows, tmp.h.cols, tmp.h.nnz, tmp.axis);
    m->storage = tmp.storage;
    m->majorPtr = tmp.majorPtr;
    m->minorIdx = tmp.minorIdx;
    m->val = (real::storage_t *) tmp.val;
    return 1;
}

int store(std::FILE *fp, const sparse::blocked_ell *m) {
    return store_blocked_ell_raw(fp, m->rows, m->cols, m->nnz, m->block_size, m->ell_cols, m->blockColIdx, m->val, sizeof(real::storage_t));
}

int load(std::FILE *fp, sparse::blocked_ell *m) {
    blocked_ell_load_result tmp;

    tmp.block_size = 0u;
    tmp.ell_cols = 0u;
    tmp.storage = 0;
    tmp.blockColIdx = 0;
    tmp.val = 0;
    if (!load_blocked_ell_raw(fp, sizeof(real::storage_t), &tmp)) return 0;
    sparse::clear(m);
    sparse::init(m, tmp.h.rows, tmp.h.cols, tmp.h.nnz, tmp.block_size, tmp.ell_cols);
    m->storage = tmp.storage;
    m->blockColIdx = tmp.blockColIdx;
    m->val = (real::storage_t *) tmp.val;
    return 1;
}

int store(std::FILE *fp, const sparse::quantized_blocked_ell *m) {
    return store_quantized_blocked_ell_raw(fp,
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

int load(std::FILE *fp, sparse::quantized_blocked_ell *m) {
    quantized_blocked_ell_load_result tmp;

    tmp.storage = 0;
    tmp.blockColIdx = 0;
    tmp.packed_values = 0;
    tmp.column_scales = 0;
    tmp.column_offsets = 0;
    tmp.row_offsets = 0;
    if (!load_quantized_blocked_ell_raw(fp, &tmp)) return 0;
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

int store(std::FILE *fp, const sparse::sliced_ell *m) {
    return store_sliced_ell_raw(fp,
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

int load(std::FILE *fp, sparse::sliced_ell *m) {
    sliced_ell_load_result tmp;

    tmp.slice_count = 0u;
    tmp.storage = 0;
    tmp.slice_row_offsets = 0;
    tmp.slice_widths = 0;
    tmp.col_idx = 0;
    tmp.val = 0;
    if (!load_sliced_ell_raw(fp, sizeof(real::storage_t), &tmp)) return 0;
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

int store(std::FILE *fp, const sparse::coo *m) {
    return store_coo_raw(fp, m->rows, m->cols, m->nnz, m->rowIdx, m->colIdx, m->val, sizeof(real::storage_t));
}

int load(std::FILE *fp, sparse::coo *m) {
    coo_load_result tmp;

    tmp.storage = 0;
    tmp.rowIdx = 0;
    tmp.colIdx = 0;
    tmp.val = 0;
    if (!load_coo_raw(fp, sizeof(real::storage_t), &tmp)) return 0;
    sparse::clear(m);
    sparse::init(m, tmp.h.rows, tmp.h.cols, tmp.h.nnz);
    m->storage = tmp.storage;
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
    tmp.storage = 0;
    tmp.offsets = 0;
    tmp.val = 0;
    if (!load_dia_raw(fp, sizeof(real::storage_t), &tmp)) return 0;
    sparse::clear(m);
    sparse::init(m, tmp.h.rows, tmp.h.cols, tmp.h.nnz);
    m->num_diagonals = tmp.num_diagonals;
    m->storage = tmp.storage;
    m->offsets = tmp.offsets;
    m->val = (real::storage_t *) tmp.val;
    return 1;
}

} // namespace cellshard
