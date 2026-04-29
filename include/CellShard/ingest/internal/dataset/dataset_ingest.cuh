#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "dataset_manifest.cuh"
#include "dataset_partition.cuh"
#include "../common/barcode_table.cuh"
#include "../common/feature_table.cuh"
#include "../common/metadata_table.cuh"
#include "../h5ad/h5ad_reader.cuh"
#include "../mtx/mtx_reader.cuh"
#include "../mtx/compressed_parts.cuh"
#include "../../../../../src/convert/blocked_ell_from_compressed.cuh"
#include "../../../../../src/convert/blocked_ell_from_coo_cuda.cuh"
#include "../../../../../src/convert/sliced_ell_from_compressed.cuh"
#include "../../../../../src/bucket/operators/blocked_ell_bucket.cuh"
#include <CellShard/io/csh5/api.cuh>

namespace cellshard {
namespace ingest {
namespace dataset {

using ::cellshard::clear;
using ::cellshard::find_offset_span;
using ::cellshard::init;
using ::cellshard::sharded;
namespace sparse = ::cellshard::sparse;

// Conversion knobs for MTX-dataset ingest.
struct mtx_convert_options {
    unsigned long max_part_nnz;
    unsigned long convert_window_bytes;
    unsigned long target_shard_bytes;
    std::size_t reader_bytes;
    std::string cache_root;
    std::string working_root;
};

// Metadata-only init with throughput-oriented defaults.
static inline void init(mtx_convert_options *opts) {
    opts->max_part_nnz = 1ul << 26ul;
    opts->convert_window_bytes = 1ul << 30ul;
    opts->target_shard_bytes = 1ul << 30ul;
    opts->reader_bytes = (std::size_t) 8u << 20u;
    opts->cache_root.clear();
    opts->working_root.clear();
}

// Small manifest access helpers.
static inline const char *dataset_id_at(const manifest *m, unsigned int idx) {
    return common::get(&m->dataset_ids, idx);
}

static inline const char *matrix_path_at(const manifest *m, unsigned int idx) {
    return common::get(&m->matrix_paths, idx);
}

static inline unsigned int format_at(const manifest *m, unsigned int idx) {
    return idx < m->count ? m->formats[idx] : source_unknown;
}

static inline const char *feature_path_at(const manifest *m, unsigned int idx) {
    return common::get(&m->feature_paths, idx);
}

static inline const char *barcode_path_at(const manifest *m, unsigned int idx) {
    return common::get(&m->barcode_paths, idx);
}

static inline const char *metadata_path_at(const manifest *m, unsigned int idx) {
    return common::get(&m->metadata_paths, idx);
}

static inline const char *matrix_source_at(const manifest *m, unsigned int idx) {
    return common::get(&m->matrix_sources, idx);
}

static inline int allow_processed_at(const manifest *m, unsigned int idx) {
    return idx < m->count && m->allow_processed != 0 && m->allow_processed[idx] != 0u;
}

static inline int build_dataset_h5_output_path(const char *out_dir,
                                              char *out_path,
                                              std::size_t out_cap) {
    if (std::snprintf(out_path, out_cap, "%s/dataset.csh5", out_dir) <= 0) return 0;
    return 1;
}

struct dataset_h5_convert_options {
    unsigned long max_part_nnz;
    unsigned long convert_window_bytes;
    unsigned long target_shard_bytes;
    std::size_t reader_bytes;
    std::string cache_root;
    std::string working_root;
    int device;
    cudaStream_t stream;
};

static inline void init(dataset_h5_convert_options *opts) {
    opts->max_part_nnz = 1ul << 26ul;
    opts->convert_window_bytes = 1ul << 30ul;
    opts->target_shard_bytes = 1ul << 30ul;
    opts->reader_bytes = (std::size_t) 8u << 20u;
    opts->cache_root.clear();
    opts->working_root.clear();
    opts->device = 0;
    opts->stream = (cudaStream_t) 0;
}

template<typename T>
class owned_buffer {
public:
    owned_buffer() = default;

    explicit owned_buffer(std::size_t count) {
        resize(count);
    }

    owned_buffer(const owned_buffer &other) {
        assign_copy(other.data(), other.size());
    }

    owned_buffer(owned_buffer &&other) noexcept
        : data_(std::move(other.data_)),
          size_(other.size_),
          capacity_(other.capacity_) {
        other.size_ = 0u;
        other.capacity_ = 0u;
    }

    owned_buffer &operator=(const owned_buffer &other) {
        if (this != &other) assign_copy(other.data(), other.size());
        return *this;
    }

    owned_buffer &operator=(owned_buffer &&other) noexcept {
        if (this == &other) return *this;
        data_ = std::move(other.data_);
        size_ = other.size_;
        capacity_ = other.capacity_;
        other.size_ = 0u;
        other.capacity_ = 0u;
        return *this;
    }

    void clear() {
        size_ = 0u;
    }

    void reserve(std::size_t capacity) {
        if (capacity <= capacity_) return;
        std::unique_ptr<T[]> next(new T[capacity]);
        if constexpr (std::is_trivially_copyable_v<T>) {
            if (size_ != 0u) std::memcpy(next.get(), data_.get(), size_ * sizeof(T));
        } else {
            for (std::size_t i = 0; i < size_; ++i) next[i] = std::move(data_[i]);
        }
        data_ = std::move(next);
        capacity_ = capacity;
    }

    void resize(std::size_t count) {
        if (count > capacity_) reserve(count);
        size_ = count;
    }

    void resize(std::size_t count, const T &value) {
        const std::size_t old_size = size_;
        resize(count);
        for (std::size_t i = old_size; i < count; ++i) data_[i] = value;
    }

    void assign_copy(const T *src, std::size_t count) {
        resize(count);
        if (count == 0u || src == nullptr) return;
        if constexpr (std::is_trivially_copyable_v<T>) {
            std::memcpy(data_.get(), src, count * sizeof(T));
        } else {
            for (std::size_t i = 0; i < count; ++i) data_[i] = src[i];
        }
    }

    void assign_fill(std::size_t count, const T &value) {
        resize(count);
        for (std::size_t i = 0; i < count; ++i) data_[i] = value;
    }

    void push_back(const T &value) {
        if (size_ == capacity_) reserve(capacity_ != 0u ? capacity_ * 2u : 1u);
        data_[size_++] = value;
    }

    void push_back(T &&value) {
        if (size_ == capacity_) reserve(capacity_ != 0u ? capacity_ * 2u : 1u);
        data_[size_++] = std::move(value);
    }

    T *data() {
        return data_.get();
    }

    const T *data() const {
        return data_.get();
    }

    std::size_t size() const {
        return size_;
    }

    bool empty() const {
        return size_ == 0u;
    }

    T &operator[](std::size_t idx) {
        return data_[idx];
    }

    const T &operator[](std::size_t idx) const {
        return data_[idx];
    }

    T *begin() {
        return data_.get();
    }

    const T *begin() const {
        return data_.get();
    }

    T *end() {
        return data_.get() + size_;
    }

    const T *end() const {
        return data_.get() + size_;
    }

private:
    std::unique_ptr<T[]> data_;
    std::size_t size_ = 0u;
    std::size_t capacity_ = 0u;
};

struct dataset_dataset_plan {
    unsigned int manifest_idx;
    unsigned int dataset_idx;
    mtx::header header;
    owned_buffer<unsigned long> row_offsets;
    owned_buffer<unsigned long> part_rows;
    owned_buffer<unsigned long> part_nnz;
    owned_buffer<unsigned long> part_bytes;
    owned_buffer<unsigned long> part_aux;
    std::vector<std::string> spool_paths;
    owned_buffer<std::uint32_t> feature_to_global;
    unsigned long global_row_begin;
    unsigned long global_part_begin;
};

static inline std::size_t standard_csr_bytes(unsigned long rows, unsigned long nnz) {
    return (std::size_t) (rows + 1ul) * sizeof(cellshard::types::ptr_t)
        + (std::size_t) nnz * sizeof(cellshard::types::idx_t)
        + (std::size_t) nnz * sizeof(::real::storage_t);
}

static inline int convert_coo_part_to_blocked_ell_auto(const sparse::coo *src,
                                                       std::uint32_t global_cols,
                                                       const std::uint32_t *feature_to_global,
                                                       int device,
                                                       sparse::blocked_ell *dst,
                                                       cellshard::convert::blocked_ell_tune_result *tune) {
    static constexpr unsigned int candidates[] = { 4u, 8u, 16u, 32u };
    static thread_local cellshard::convert::blocked_ell_from_coo_cuda_workspace cuda_ws;
    static thread_local int cuda_ws_ready = 0;
    sparse::clear(dst);
    sparse::init(dst);
    if (!cuda_ws_ready) {
        cellshard::convert::init(&cuda_ws);
        cuda_ws_ready = 1;
    }
    if (cuda_ws.device != device && !cellshard::convert::setup(&cuda_ws, device)) {
        cuda_ws.device = -1;
    }
    if (cuda_ws.device == device
        && cellshard::convert::blocked_ell_from_coo_cuda_auto(src,
                                                              global_cols,
                                                              feature_to_global,
                                                              candidates,
                                                              sizeof(candidates) / sizeof(candidates[0]),
                                                              dst,
                                                              device,
                                                              tune,
                                                              &cuda_ws,
                                                              cuda_ws.stream)) {
        return 1;
    }
    sparse::clear(dst);
    sparse::init(dst);
    return cellshard::convert::blocked_ell_from_coo_auto(src,
                                                         global_cols,
                                                         feature_to_global,
                                                         candidates,
                                                         sizeof(candidates) / sizeof(candidates[0]),
                                                         dst,
                                                         tune);
}

static inline cellshard::dataset_text_column_view as_text_view(const common::text_column *c) {
    cellshard::dataset_text_column_view view;
    view.count = c != 0 ? c->count : 0u;
    view.bytes = c != 0 ? c->bytes : 0u;
    view.offsets = c != 0 ? c->offsets : 0;
    view.data = c != 0 ? c->data : 0;
    return view;
}

#include "internal/source_load_part.hh"

static inline int convert_compressed_part_to_sliced_ell_auto(const sparse::compressed *src,
                                                             int device,
                                                             sparse::sliced_ell *dst,
                                                             cellshard::convert::sliced_ell_tune_result *tune) {
    static constexpr unsigned int candidates[] = { 8u, 16u, 32u, 64u };
    sparse::clear(dst);
    sparse::init(dst);
    if (src == nullptr) return 0;
    if (src->axis == sparse::compressed_by_row
        && cellshard::convert::sliced_ell_from_compressed_cuda_auto(src,
                                                                    candidates,
                                                                    sizeof(candidates) / sizeof(candidates[0]),
                                                                    dst,
                                                                    device,
                                                                    (cudaStream_t) 0,
                                                                    tune)) {
        return 1;
    }
    return cellshard::convert::sliced_ell_from_compressed_auto(src,
                                                               candidates,
                                                               sizeof(candidates) / sizeof(candidates[0]),
                                                               dst,
                                                               tune);
}

static inline int convert_coo_part_to_sliced_ell_auto(const sparse::coo *src,
                                                      sparse::sliced_ell *dst,
                                                      cellshard::convert::sliced_ell_tune_result *tune) {
    static constexpr unsigned int candidates[] = { 8u, 16u, 32u, 64u };
    std::unique_ptr<cellshard::types::u32[]> row_nnz;
    std::unique_ptr<cellshard::types::u32[]> slice_row_offsets;
    std::unique_ptr<cellshard::types::u32[]> slice_widths;
    std::unique_ptr<cellshard::types::u32[]> row_write_cursor;
    cellshard::convert::sliced_ell_tune_result local_tune{};
    cellshard::types::u32 slice_count = 0u;

    if (src == nullptr || dst == nullptr) return 0;
    row_nnz.reset(src->rows != 0u ? new cellshard::types::u32[src->rows]() : nullptr);
    if (src->rows != 0u && !row_nnz) return 0;
    for (cellshard::types::nnz_t i = 0u; i < src->nnz; ++i) {
        const cellshard::types::u32 row = src->rowIdx[i];
        if (row >= src->rows) return 0;
        row_nnz[row] += 1u;
    }
    if (!cellshard::convert::choose_sliced_ell_slice_rows(row_nnz.get(),
                                                          src->rows,
                                                          src->nnz,
                                                          candidates,
                                                          sizeof(candidates) / sizeof(candidates[0]),
                                                          &local_tune)) return 0;
    if (!cellshard::convert::detail::build_uniform_sliced_layout_(row_nnz.get(),
                                                                  src->rows,
                                                                  local_tune.slice_rows == 0u ? 1u : local_tune.slice_rows,
                                                                  &slice_row_offsets,
                                                                  &slice_widths,
                                                                  &slice_count)) return 0;
    sparse::clear(dst);
    sparse::init(dst, src->rows, src->cols, src->nnz);
    if (!sparse::allocate(dst, slice_count, slice_row_offsets.get(), slice_widths.get())) return 0;
    row_write_cursor.reset(dst->rows != 0u ? new cellshard::types::u32[dst->rows]() : nullptr);
    if (dst->rows != 0u && !row_write_cursor) {
        sparse::clear(dst);
        return 0;
    }
    for (cellshard::types::u32 row = 0u; row < dst->rows; ++row) {
        const cellshard::types::u32 slice = sparse::find_slice(dst, row);
        const cellshard::types::u32 row_begin = slice < dst->slice_count ? dst->slice_row_offsets[slice] : 0u;
        row_write_cursor[row] = (cellshard::types::u32) (sparse::slice_slot_base(dst, slice)
            + (std::size_t) (row - row_begin) * (std::size_t) dst->slice_widths[slice]);
    }
    for (cellshard::types::nnz_t i = 0u; i < src->nnz; ++i) {
        const cellshard::types::u32 row = src->rowIdx[i];
        dst->col_idx[row_write_cursor[row]] = src->colIdx[i];
        dst->val[row_write_cursor[row]] = src->val[i];
        row_write_cursor[row] += 1u;
    }
    if (tune != nullptr) *tune = local_tune;
    return 1;
}

static inline int blocked_ell_to_canonical_coo(const sparse::blocked_ell *part,
                                               sparse::coo *out) {
    const std::uint32_t block_size = part != nullptr ? part->block_size : 0u;
    const std::uint32_t width_blocks = part != nullptr ? cellshard::sparse::ell_width_blocks(part) : 0u;
    std::size_t actual_nnz = 0u;
    std::size_t emitted = 0u;
    if (part == nullptr || out == nullptr) return 0;
    sparse::clear(out);
    for (std::uint32_t row = 0u; row < part->rows; ++row) {
        const std::uint32_t row_block = row / block_size;
        for (std::uint32_t slot = 0u; slot < width_blocks; ++slot) {
            const cellshard::types::idx_t block_col = part->blockColIdx[(std::size_t) row_block * width_blocks + slot];
            if (block_col == cellshard::sparse::blocked_ell_invalid_col) continue;
            for (std::uint32_t col_in_block = 0u; col_in_block < block_size; ++col_in_block) {
                const std::uint32_t col = (std::uint32_t) block_col * block_size + col_in_block;
                const ::real::storage_t value =
                    part->val[(std::size_t) row * part->ell_cols + (std::size_t) slot * block_size + col_in_block];
                if (__half2float(value) == 0.0f || col >= part->cols) continue;
                ++actual_nnz;
            }
        }
    }
    sparse::init(out, part->rows, part->cols, (cellshard::types::nnz_t) actual_nnz);
    if (!sparse::allocate(out)) return 0;
    for (std::uint32_t row = 0u; row < part->rows; ++row) {
        const std::uint32_t row_block = row / block_size;
        for (std::uint32_t slot = 0u; slot < width_blocks; ++slot) {
            const cellshard::types::idx_t block_col = part->blockColIdx[(std::size_t) row_block * width_blocks + slot];
            if (block_col == cellshard::sparse::blocked_ell_invalid_col) continue;
            for (std::uint32_t col_in_block = 0u; col_in_block < block_size; ++col_in_block) {
                const std::uint32_t col = (std::uint32_t) block_col * block_size + col_in_block;
                const ::real::storage_t value =
                    part->val[(std::size_t) row * part->ell_cols + (std::size_t) slot * block_size + col_in_block];
                if (__half2float(value) == 0.0f || col >= part->cols) continue;
                out->rowIdx[emitted] = row;
                out->colIdx[emitted] = col;
                out->val[emitted] = value;
                ++emitted;
            }
        }
    }
    if (emitted != out->nnz) {
        sparse::clear(out);
        return 0;
    }
    return 1;
}

#include "internal/convert_layout_support_part.hh"

#include "internal/layout_build_part.hh"

#include "internal/dataset_convert_part.hh"

// Default MTX-dataset conversion path. This emits one portable HDF5-backed
// dataset container in out_dir/dataset.csh5.
static inline int convert_manifest_mtx_dataset_to_hdf5(const manifest *m,
                                                      const char *out_path,
                                                      const dataset_h5_convert_options *opts) {
    return convert_manifest_dataset_to_hdf5(m, out_path, opts);
}

static inline int convert_manifest_mtx_series(const manifest *m,
                                              const char *out_dir,
                                              const mtx_convert_options *opts) {
    dataset_h5_convert_options h5_opts;
    char out_path[4096];

    if (m == 0 || out_dir == 0 || opts == 0) return 0;
    init(&h5_opts);
    h5_opts.max_part_nnz = opts->max_part_nnz;
    h5_opts.convert_window_bytes = opts->convert_window_bytes;
    h5_opts.target_shard_bytes = opts->target_shard_bytes;
    h5_opts.reader_bytes = opts->reader_bytes;
    h5_opts.cache_root = opts->cache_root;
    h5_opts.working_root = opts->working_root;
    if (!build_dataset_h5_output_path(out_dir, out_path, sizeof(out_path))) return 0;
    return convert_manifest_dataset_to_hdf5(m, out_path, &h5_opts);
}

} // namespace dataset
} // namespace ingest
} // namespace cellshard
