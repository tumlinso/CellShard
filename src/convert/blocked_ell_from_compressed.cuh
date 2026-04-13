#pragma once

#include "../formats/blocked_ell.cuh"
#include "../formats/compressed.cuh"
#include "../formats/triplet.cuh"
#include "../sharded/sharded.cuh"
#include "../sharded/sharded_host.cuh"

#include <cstdlib>
#include <cstring>
#include <memory>

namespace cellshard {
namespace convert {

struct blocked_ell_tune_result {
    unsigned int block_size;
    double fill_ratio;
    std::size_t padded_bytes;
};

namespace detail {

inline int compare_u32_asc_(const void *lhs, const void *rhs) {
    const types::u32 a = *(const types::u32 *) lhs;
    const types::u32 b = *(const types::u32 *) rhs;
    if (a < b) return -1;
    if (a > b) return 1;
    return 0;
}

template<typename RowFn>
inline int count_blocked_ell_shape_(
    types::dim_t rows,
    types::dim_t cols,
    types::u32 block_size,
    RowFn &&row_accessor,
    types::u32 *out_ell_width,
    double *out_fill_ratio
) {
    const types::u32 col_blocks = block_size == 0u ? 0u : (cols + block_size - 1u) / block_size;
    const types::u32 row_blocks = block_size == 0u ? 0u : (rows + block_size - 1u) / block_size;
    std::unique_ptr<types::u32[]> marks;
    std::unique_ptr<types::u32[]> counts;
    types::u32 ell_width = 0u;
    unsigned long total_blocks = 0ul;
    types::u32 epoch = 1u;

    if (out_ell_width == 0 || out_fill_ratio == 0) return 0;
    *out_ell_width = 0u;
    *out_fill_ratio = 0.0;
    if (rows == 0u || cols == 0u || block_size == 0u) return 1;
    if (col_blocks != 0u) marks.reset(new types::u32[col_blocks]());
    if (row_blocks != 0u) counts.reset(new types::u32[row_blocks]());
    if (col_blocks != 0u && !marks) return 0;
    if (row_blocks != 0u && !counts) return 0;

    for (types::u32 rb = 0u; rb < row_blocks; ++rb) {
        types::u32 unique = 0u;
        const types::u32 row_begin = rb * block_size;
        const types::u32 row_end = row_begin + block_size < rows ? row_begin + block_size : rows;
        for (types::u32 row = row_begin; row < row_end; ++row) {
            const types::ptr_t begin = row_accessor.major_ptr(row)[0];
            const types::ptr_t end = row_accessor.major_ptr(row)[1];
            const types::idx_t *minor = row_accessor.minor_idx(row);
            for (types::ptr_t i = begin; i < end; ++i) {
                const types::u32 block_col = minor[i - begin] / block_size;
                if (marks[block_col] != epoch) {
                    marks[block_col] = epoch;
                    ++unique;
                }
            }
        }
        counts[rb] = unique;
        total_blocks += unique;
        if (unique > ell_width) ell_width = unique;
        ++epoch;
        if (epoch == 0u) {
            std::memset(marks.get(), 0, (std::size_t) col_blocks * sizeof(types::u32));
            epoch = 1u;
        }
    }

    *out_ell_width = ell_width;
    if (row_blocks == 0u || ell_width == 0u) {
        *out_fill_ratio = 1.0;
        return 1;
    }
    *out_fill_ratio = (double) total_blocks / (double) ((unsigned long) row_blocks * (unsigned long) ell_width);
    return 1;
}

struct compressed_row_accessor {
    const sparse::compressed *src;

    __host__ __device__ __forceinline__ const types::ptr_t *major_ptr(types::u32 row) const {
        return src->majorPtr + row;
    }

    __host__ __device__ __forceinline__ const types::idx_t *minor_idx(types::u32 row) const {
        return src->minorIdx + src->majorPtr[row];
    }
};

inline types::u32 mapped_col_(types::idx_t col,
                              const types::u32 *feature_to_global) {
    return feature_to_global != 0 ? feature_to_global[col] : (types::u32) col;
}

inline int bucket_coo_block_cols_(
    const sparse::coo *src,
    types::dim_t cols,
    types::u32 block_size,
    const types::u32 *feature_to_global,
    std::unique_ptr<types::ptr_t[]> *out_row_block_offsets,
    std::unique_ptr<types::u32[]> *out_block_cols
) {
    const types::u32 row_blocks = block_size == 0u ? 0u : (src->rows + block_size - 1u) / block_size;
    std::unique_ptr<types::ptr_t[]> row_block_offsets;
    std::unique_ptr<types::u32[]> block_cols;
    std::unique_ptr<types::ptr_t[]> write_ptr;

    if (src == 0 || out_row_block_offsets == 0 || out_block_cols == 0 || block_size == 0u) return 0;
    (void) cols;
    if (row_blocks != 0u) {
        row_block_offsets.reset(new types::ptr_t[(std::size_t) row_blocks + 1u]());
        write_ptr.reset(new types::ptr_t[(std::size_t) row_blocks]());
        if (!row_block_offsets || !write_ptr) return 0;
    }
    if (src->nnz != 0u) {
        block_cols.reset(new types::u32[(std::size_t) src->nnz]());
        if (!block_cols) return 0;
    }

    for (types::nnz_t i = 0; i < src->nnz; ++i) {
        const types::u32 row_block = src->rowIdx[i] / block_size;
        ++row_block_offsets[row_block + 1u];
    }
    for (types::u32 rb = 0u; rb < row_blocks; ++rb) {
        row_block_offsets[rb + 1u] += row_block_offsets[rb];
        write_ptr[rb] = row_block_offsets[rb];
    }
    for (types::nnz_t i = 0; i < src->nnz; ++i) {
        const types::u32 row_block = src->rowIdx[i] / block_size;
        const types::u32 block_col = mapped_col_(src->colIdx[i], feature_to_global) / block_size;
        const types::ptr_t dst = write_ptr[row_block]++;
        block_cols[dst] = block_col;
    }

    *out_row_block_offsets = std::move(row_block_offsets);
    *out_block_cols = std::move(block_cols);
    return 1;
}

} // namespace detail

inline int choose_blocked_ell_block_size(
    const sparse::compressed *src,
    const unsigned int *candidates,
    unsigned int candidate_count,
    blocked_ell_tune_result *out
) {
    blocked_ell_tune_result best = { 0u, 0.0, 0u };
    detail::compressed_row_accessor accessor = { src };

    if (src == 0 || out == 0 || candidates == 0 || candidate_count == 0u) return 0;
    if (src->axis != sparse::compressed_by_row) return 0;

    for (unsigned int i = 0; i < candidate_count; ++i) {
        const unsigned int block_size = candidates[i];
        types::u32 ell_width = 0u;
        double fill_ratio = 0.0;
        std::size_t padded_bytes = 0u;
        if (block_size == 0u) continue;
        if (!detail::count_blocked_ell_shape_(src->rows, src->cols, block_size, accessor, &ell_width, &fill_ratio)) return 0;
        padded_bytes = (std::size_t) src->rows * (std::size_t) ell_width * (std::size_t) block_size * sizeof(real::storage_t);
        if (best.block_size == 0u ||
            fill_ratio > best.fill_ratio + 1.0e-9 ||
            (fill_ratio + 1.0e-9 >= best.fill_ratio && padded_bytes < best.padded_bytes) ||
            (fill_ratio + 1.0e-9 >= best.fill_ratio && padded_bytes == best.padded_bytes && block_size > best.block_size)) {
            best.block_size = block_size;
            best.fill_ratio = fill_ratio;
            best.padded_bytes = padded_bytes;
        }
    }

    if (best.block_size == 0u) return 0;
    *out = best;
    return 1;
}

inline int choose_blocked_ell_block_size_from_coo(
    const sparse::coo *src,
    types::dim_t cols,
    const types::u32 *feature_to_global,
    const unsigned int *candidates,
    unsigned int candidate_count,
    blocked_ell_tune_result *out
) {
    blocked_ell_tune_result best = { 0u, 0.0, 0u };

    if (src == 0 || out == 0 || candidates == 0 || candidate_count == 0u || cols == 0u) return 0;

    for (unsigned int i = 0; i < candidate_count; ++i) {
        const unsigned int block_size = candidates[i];
        const types::u32 row_blocks = block_size == 0u ? 0u : (src->rows + block_size - 1u) / block_size;
        std::unique_ptr<types::ptr_t[]> row_block_offsets;
        std::unique_ptr<types::u32[]> block_cols;
        std::size_t padded_bytes = 0u;
        unsigned long total_blocks = 0ul;
        types::u32 ell_width = 0u;
        double fill_ratio = 0.0;

        if (block_size == 0u) continue;
        if (!detail::bucket_coo_block_cols_(src,
                                            cols,
                                            block_size,
                                            feature_to_global,
                                            &row_block_offsets,
                                            &block_cols)) return 0;

        for (types::u32 rb = 0u; rb < row_blocks; ++rb) {
            const types::ptr_t begin = row_block_offsets[rb];
            const types::ptr_t end = row_block_offsets[rb + 1u];
            types::u32 unique = 0u;

            if (end > begin) {
                std::qsort(block_cols.get() + begin, (std::size_t) (end - begin), sizeof(types::u32), detail::compare_u32_asc_);
                unique = 1u;
                for (types::ptr_t j = begin + 1u; j < end; ++j) {
                    if (block_cols[j] != block_cols[j - 1u]) ++unique;
                }
            }
            total_blocks += unique;
            if (unique > ell_width) ell_width = unique;
        }

        padded_bytes = (std::size_t) src->rows * (std::size_t) ell_width * (std::size_t) block_size * sizeof(real::storage_t);
        fill_ratio = (row_blocks == 0u || ell_width == 0u)
            ? 1.0
            : (double) total_blocks / (double) ((unsigned long) row_blocks * (unsigned long) ell_width);

        if (best.block_size == 0u ||
            fill_ratio > best.fill_ratio + 1.0e-9 ||
            (fill_ratio + 1.0e-9 >= best.fill_ratio && padded_bytes < best.padded_bytes) ||
            (fill_ratio + 1.0e-9 >= best.fill_ratio && padded_bytes == best.padded_bytes && block_size > best.block_size)) {
            best.block_size = block_size;
            best.fill_ratio = fill_ratio;
            best.padded_bytes = padded_bytes;
        }
    }

    if (best.block_size == 0u) return 0;
    *out = best;
    return 1;
}

inline int blocked_ell_from_compressed(const sparse::compressed *src, unsigned int block_size, sparse::blocked_ell *dst) {
    detail::compressed_row_accessor accessor = { src };
    types::u32 ell_width = 0u;
    double fill_ratio = 0.0;
    const types::u32 row_blocks = block_size == 0u ? 0u : (src->rows + block_size - 1u) / block_size;
    const types::u32 col_blocks = block_size == 0u ? 0u : (src->cols + block_size - 1u) / block_size;
    std::unique_ptr<types::u32[]> marks;
    std::unique_ptr<types::u32[]> slot_map;
    std::unique_ptr<types::u32[]> block_cols;
    types::u32 epoch = 1u;

    if (src == 0 || dst == 0 || block_size == 0u) return 0;
    if (src->axis != sparse::compressed_by_row) return 0;
    if (!detail::count_blocked_ell_shape_(src->rows, src->cols, block_size, accessor, &ell_width, &fill_ratio)) return 0;

    sparse::clear(dst);
    sparse::init(dst, src->rows, src->cols, src->nnz, block_size, ell_width * block_size);
    if (!sparse::allocate(dst)) return 0;
    if (dst->blockColIdx != 0 && row_blocks != 0u && ell_width != 0u) {
        for (std::size_t i = 0; i < (std::size_t) row_blocks * ell_width; ++i) dst->blockColIdx[i] = sparse::blocked_ell_invalid_col;
    }
    if (dst->val != 0 && dst->rows != 0u && dst->ell_cols != 0u) {
        std::memset(dst->val, 0, (std::size_t) dst->rows * (std::size_t) dst->ell_cols * sizeof(real::storage_t));
    }

    if (col_blocks != 0u) {
        marks.reset(new types::u32[col_blocks]());
        slot_map.reset(new types::u32[col_blocks]());
        block_cols.reset(new types::u32[ell_width != 0u ? ell_width : 1u]());
        if (!marks || !slot_map || !block_cols) {
            sparse::clear(dst);
            return 0;
        }
    }

    for (types::u32 rb = 0u; rb < row_blocks; ++rb) {
        types::u32 unique = 0u;
        const types::u32 row_begin = rb * block_size;
        const types::u32 row_end = row_begin + block_size < src->rows ? row_begin + block_size : src->rows;
        ++epoch;
        if (epoch == 0u) {
            std::memset(marks.get(), 0, (std::size_t) col_blocks * sizeof(types::u32));
            epoch = 1u;
        }

        for (types::u32 row = row_begin; row < row_end; ++row) {
            const types::ptr_t begin = src->majorPtr[row];
            const types::ptr_t end = src->majorPtr[row + 1u];
            for (types::ptr_t i = begin; i < end; ++i) {
                const types::u32 block_col = src->minorIdx[i] / block_size;
                if (marks[block_col] != epoch) {
                    marks[block_col] = epoch;
                    block_cols[unique++] = block_col;
                }
            }
        }

        std::qsort(block_cols.get(), unique, sizeof(types::u32), detail::compare_u32_asc_);
        for (types::u32 slot = 0u; slot < unique; ++slot) {
            const types::u32 block_col = block_cols[slot];
            dst->blockColIdx[(std::size_t) rb * ell_width + slot] = block_col;
            slot_map[block_col] = slot;
        }

        for (types::u32 row = row_begin; row < row_end; ++row) {
            const types::ptr_t begin = src->majorPtr[row];
            const types::ptr_t end = src->majorPtr[row + 1u];
            for (types::ptr_t i = begin; i < end; ++i) {
                const types::u32 block_col = src->minorIdx[i] / block_size;
                const types::u32 slot = slot_map[block_col];
                const types::u32 col_in_block = src->minorIdx[i] % block_size;
                dst->val[(std::size_t) row * dst->ell_cols + (std::size_t) slot * block_size + col_in_block] = src->val[i];
            }
        }
    }

    return 1;
}

inline int blocked_ell_from_coo(
    const sparse::coo *src,
    types::dim_t cols,
    const types::u32 *feature_to_global,
    unsigned int block_size,
    sparse::blocked_ell *dst
) {
    const types::u32 row_blocks = block_size == 0u ? 0u : (src->rows + block_size - 1u) / block_size;
    const types::u32 col_blocks = block_size == 0u ? 0u : (cols + block_size - 1u) / block_size;
    std::unique_ptr<types::ptr_t[]> row_block_offsets;
    std::unique_ptr<types::u32[]> block_cols;
    std::unique_ptr<types::u32[]> slot_map;
    blocked_ell_tune_result tune = { 0u, 0.0, 0u };
    types::u32 ell_width = 0u;

    if (src == 0 || dst == 0 || cols == 0u || block_size == 0u) return 0;
    if (!choose_blocked_ell_block_size_from_coo(src, cols, feature_to_global, &block_size, 1u, &tune)) return 0;
    if (!detail::bucket_coo_block_cols_(src,
                                        cols,
                                        block_size,
                                        feature_to_global,
                                        &row_block_offsets,
                                        &block_cols)) return 0;
    ell_width = (src->rows == 0u || block_size == 0u)
        ? 0u
        : (types::u32) (tune.padded_bytes / ((std::size_t) src->rows * (std::size_t) block_size * sizeof(real::storage_t)));
    if (row_blocks != 0u && col_blocks != 0u) {
        slot_map.reset(new types::u32[col_blocks]());
        if (!slot_map) return 0;
    }

    sparse::clear(dst);
    sparse::init(dst, src->rows, cols, src->nnz, block_size, ell_width * block_size);
    if (!sparse::allocate(dst)) return 0;
    if (dst->blockColIdx != 0 && row_blocks != 0u && ell_width != 0u) {
        for (std::size_t i = 0; i < (std::size_t) row_blocks * ell_width; ++i) dst->blockColIdx[i] = sparse::blocked_ell_invalid_col;
    }
    if (dst->val != 0 && dst->rows != 0u && dst->ell_cols != 0u) {
        std::memset(dst->val, 0, (std::size_t) dst->rows * (std::size_t) dst->ell_cols * sizeof(real::storage_t));
    }

    for (types::u32 rb = 0u; rb < row_blocks; ++rb) {
        const types::ptr_t begin = row_block_offsets[rb];
        const types::ptr_t end = row_block_offsets[rb + 1u];
        types::u32 unique = 0u;

        if (end > begin) {
            std::qsort(block_cols.get() + begin, (std::size_t) (end - begin), sizeof(types::u32), detail::compare_u32_asc_);
            for (types::ptr_t j = begin; j < end; ++j) {
                if (j == begin || block_cols[j] != block_cols[j - 1u]) {
                    dst->blockColIdx[(std::size_t) rb * ell_width + unique] = block_cols[j];
                    slot_map[block_cols[j]] = unique;
                    ++unique;
                }
            }
        }
    }

    for (types::nnz_t i = 0; i < src->nnz; ++i) {
        const types::u32 row = src->rowIdx[i];
        const types::u32 col = detail::mapped_col_(src->colIdx[i], feature_to_global);
        const types::u32 block_col = col / block_size;
        const types::u32 slot = slot_map[block_col];
        const types::u32 col_in_block = col % block_size;
        dst->val[(std::size_t) row * dst->ell_cols + (std::size_t) slot * block_size + col_in_block] = src->val[i];
    }

    return 1;
}

inline int blocked_ell_from_compressed_auto(
    const sparse::compressed *src,
    const unsigned int *candidates,
    unsigned int candidate_count,
    sparse::blocked_ell *dst,
    blocked_ell_tune_result *picked = 0
) {
    blocked_ell_tune_result tune = { 0u, 0.0, 0u };
    if (!choose_blocked_ell_block_size(src, candidates, candidate_count, &tune)) return 0;
    if (!blocked_ell_from_compressed(src, tune.block_size, dst)) return 0;
    if (picked != 0) *picked = tune;
    return 1;
}

inline int blocked_ell_from_coo_auto(
    const sparse::coo *src,
    types::dim_t cols,
    const types::u32 *feature_to_global,
    const unsigned int *candidates,
    unsigned int candidate_count,
    sparse::blocked_ell *dst,
    blocked_ell_tune_result *picked = 0
) {
    blocked_ell_tune_result tune = { 0u, 0.0, 0u };

    if (!choose_blocked_ell_block_size_from_coo(src, cols, feature_to_global, candidates, candidate_count, &tune)) return 0;
    if (!blocked_ell_from_coo(src, cols, feature_to_global, tune.block_size, dst)) return 0;
    if (picked != 0) *picked = tune;
    return 1;
}

inline int repack_sharded_compressed_to_blocked_ell(
    const sharded<sparse::compressed> *src,
    unsigned int block_size,
    unsigned long target_block_rows,
    sharded<sparse::blocked_ell> *dst
) {
    const unsigned long rows_per_part = target_block_rows == 0ul ? src->rows : target_block_rows * (unsigned long) block_size;
    unsigned long row_begin = 0ul;

    if (src == 0 || dst == 0 || block_size == 0u) return 0;
    clear(dst);
    init(dst);
    dst->cols = src->cols;

    while (row_begin < src->rows) {
        const unsigned long row_end = rows_per_part == 0ul || row_begin + rows_per_part >= src->rows ? src->rows : row_begin + rows_per_part;
        sparse::compressed stitched;
        sparse::blocked_ell *part = 0;
        std::unique_ptr<types::ptr_t[]> row_ptr;
        std::unique_ptr<types::idx_t[]> col_idx;
        std::unique_ptr<real::storage_t[]> values;
        types::nnz_t nnz = 0u;

        sparse::init(&stitched, (types::dim_t) (row_end - row_begin), (types::dim_t) src->cols, 0u, sparse::compressed_by_row);

        row_ptr.reset(new types::ptr_t[(std::size_t) stitched.rows + 1u]());
        if (!row_ptr) return 0;

        for (unsigned long global_row = row_begin; global_row < row_end; ++global_row) {
            const unsigned long part_id = find_part(src, global_row);
            const sparse::compressed *src_part = part_id < src->num_parts ? src->parts[part_id] : 0;
            const unsigned long local_row = global_row - src->part_offsets[part_id];
            if (src_part == 0 || src_part->axis != sparse::compressed_by_row) return 0;
            nnz += src_part->majorPtr[local_row + 1u] - src_part->majorPtr[local_row];
            row_ptr[(global_row - row_begin) + 1u] = nnz;
        }

        col_idx.reset(nnz != 0u ? new types::idx_t[nnz] : 0);
        values.reset(nnz != 0u ? new real::storage_t[nnz] : 0);
        if (nnz != 0u && (!col_idx || !values)) return 0;

        for (unsigned long global_row = row_begin; global_row < row_end; ++global_row) {
            const unsigned long part_id = find_part(src, global_row);
            const sparse::compressed *src_part = part_id < src->num_parts ? src->parts[part_id] : 0;
            const unsigned long local_row = global_row - src->part_offsets[part_id];
            const types::ptr_t begin = src_part->majorPtr[local_row];
            const types::ptr_t end = src_part->majorPtr[local_row + 1u];
            const types::ptr_t out_begin = row_ptr[global_row - row_begin];
            const types::ptr_t count = end - begin;
            if (count != 0u) {
                std::memcpy(col_idx.get() + out_begin, src_part->minorIdx + begin, (std::size_t) count * sizeof(types::idx_t));
                std::memcpy(values.get() + out_begin, src_part->val + begin, (std::size_t) count * sizeof(real::storage_t));
            }
        }

        stitched.nnz = nnz;
        stitched.majorPtr = row_ptr.get();
        stitched.minorIdx = col_idx.get();
        stitched.val = values.get();

        part = new sparse::blocked_ell();
        sparse::init(part);
        if (!blocked_ell_from_compressed(&stitched, block_size, part)) {
            delete part;
            return 0;
        }
        if (!append_part(dst, part)) {
            destroy(part);
            return 0;
        }
        row_begin = row_end;
    }

    return 1;
}

inline int repack_sharded_compressed_to_blocked_ell_auto(
    const sharded<sparse::compressed> *src,
    const unsigned int *candidates,
    unsigned int candidate_count,
    unsigned long target_block_rows,
    sharded<sparse::blocked_ell> *dst,
    blocked_ell_tune_result *picked = 0
) {
    blocked_ell_tune_result tune = { 0u, 0.0, 0u };
    sparse::compressed whole;
    std::unique_ptr<types::ptr_t[]> row_ptr;
    std::unique_ptr<types::idx_t[]> col_idx;
    std::unique_ptr<real::storage_t[]> values;
    types::nnz_t cursor = 0u;

    if (src == 0 || dst == 0 || candidates == 0 || candidate_count == 0u) return 0;
    sparse::init(&whole, (types::dim_t) src->rows, (types::dim_t) src->cols, (types::nnz_t) src->nnz, sparse::compressed_by_row);
    row_ptr.reset(new types::ptr_t[(std::size_t) src->rows + 1u]());
    col_idx.reset(src->nnz != 0u ? new types::idx_t[src->nnz] : 0);
    values.reset(src->nnz != 0u ? new real::storage_t[src->nnz] : 0);
    if (!row_ptr || (src->nnz != 0u && (!col_idx || !values))) return 0;
    row_ptr[0] = 0u;
    for (unsigned long part = 0; part < src->num_parts; ++part) {
        const sparse::compressed *p = src->parts[part];
        if (p == 0 || p->axis != sparse::compressed_by_row) return 0;
        for (types::u32 row = 0u; row < p->rows; ++row) {
            const types::ptr_t begin = p->majorPtr[row];
            const types::ptr_t end = p->majorPtr[row + 1u];
            const types::ptr_t count = end - begin;
            if (count != 0u) {
                std::memcpy(col_idx.get() + cursor, p->minorIdx + begin, (std::size_t) count * sizeof(types::idx_t));
                std::memcpy(values.get() + cursor, p->val + begin, (std::size_t) count * sizeof(real::storage_t));
            }
            cursor += count;
            row_ptr[src->part_offsets[part] + row + 1u] = cursor;
        }
    }
    whole.majorPtr = row_ptr.get();
    whole.minorIdx = col_idx.get();
    whole.val = values.get();
    if (!choose_blocked_ell_block_size(&whole, candidates, candidate_count, &tune)) return 0;
    if (!repack_sharded_compressed_to_blocked_ell(src, tune.block_size, target_block_rows, dst)) return 0;
    if (picked != 0) *picked = tune;
    return 1;
}

} // namespace convert
} // namespace cellshard
