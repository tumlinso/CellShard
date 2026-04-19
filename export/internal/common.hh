#pragma once

#include "../../include/CellShard/export/dataset_export.hh"
#include "../../include/CellShard/core/real.cuh"
#include "../../include/CellShard/formats/blocked_ell.cuh"
#include "../../include/CellShard/formats/compressed.cuh"
#include "../../include/CellShard/formats/sliced_ell.cuh"
#include "../../include/CellShard/runtime/layout/sharded.cuh"
#include "../../include/CellShard/runtime/host/sharded_host.cuh"
#include "../../include/CellShard/io/csh5/api.cuh"
#include "../../include/CellShard/runtime/storage/disk.cuh"

#include <hdf5.h>

#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace cellshard::exporting::detail {

inline void set_error(std::string *error, const std::string &message) {
    if (error != nullptr) *error = message;
}

inline void warn_cpu_materialize_once(const char *label) {
#if !CELLSHARD_ENABLE_CUDA
    static bool warned = false;
    if (!warned) {
        std::fprintf(stderr,
                     "Warning: CellShard was built without CUDA; %s is running in a slow host-only inspect/materialize mode.\n",
                     label != nullptr ? label : "this operation");
        warned = true;
    }
#else
    (void) label;
#endif
}

inline hid_t open_group(hid_t parent, const char *path) {
    return H5Gopen2(parent, path, H5P_DEFAULT);
}

inline hid_t open_optional_group(hid_t parent, const char *path) {
    hid_t group = (hid_t) -1;
    H5E_BEGIN_TRY {
        group = H5Gopen2(parent, path, H5P_DEFAULT);
    } H5E_END_TRY;
    return group;
}

inline std::string build_execution_pack_relative_path(const runtime_service_metadata &runtime,
                                                      std::uint64_t shard_id) {
    return "packs/execution/plan."
        + std::to_string(runtime.execution_plan_generation)
        + "-pack."
        + std::to_string(runtime.pack_generation)
        + "-epoch."
        + std::to_string(runtime.service_epoch)
        + "/shard."
        + std::to_string(shard_id)
        + ".exec.pack";
}

inline bool read_attr_u64(hid_t obj, const char *name, std::uint64_t *value) {
    hid_t attr = H5Aopen(obj, name, H5P_DEFAULT);
    const bool ok = attr >= 0 && H5Aread(attr, H5T_NATIVE_UINT64, value) >= 0;
    if (attr >= 0) H5Aclose(attr);
    return ok;
}

inline bool read_attr_u32(hid_t obj, const char *name, std::uint32_t *value) {
    hid_t attr = H5Aopen(obj, name, H5P_DEFAULT);
    const bool ok = attr >= 0 && H5Aread(attr, H5T_NATIVE_UINT32, value) >= 0;
    if (attr >= 0) H5Aclose(attr);
    return ok;
}

inline bool read_attr_string(hid_t obj, const char *name, std::string *value) {
    hid_t attr = (hid_t) -1;
    hid_t type = (hid_t) -1;
    std::size_t size = 0u;
    std::vector<char> buffer;

    if (value == nullptr) return false;
    attr = H5Aopen(obj, name, H5P_DEFAULT);
    if (attr < 0) return false;
    type = H5Aget_type(attr);
    if (type < 0) goto done;
    size = H5Tget_size(type);
    if (size == 0u) goto done;
    buffer.assign(size + 1u, '\0');
    if (H5Aread(attr, type, buffer.data()) < 0) goto done;
    *value = buffer.data();
    H5Tclose(type);
    H5Aclose(attr);
    return true;

done:
    if (type >= 0) H5Tclose(type);
    if (attr >= 0) H5Aclose(attr);
    return false;
}

template<typename T>
inline bool read_dataset_vector(hid_t parent, const char *name, hid_t dtype, std::vector<T> *out) {
    hid_t dset = H5Dopen2(parent, name, H5P_DEFAULT);
    hid_t space = (hid_t) -1;
    hsize_t dims[1] = {0};
    int ndims = 0;
    if (dset < 0 || out == nullptr) {
        if (dset >= 0) H5Dclose(dset);
        return false;
    }
    space = H5Dget_space(dset);
    if (space < 0) {
        H5Dclose(dset);
        return false;
    }
    ndims = H5Sget_simple_extent_dims(space, dims, nullptr);
    if (ndims != 1) {
        H5Sclose(space);
        H5Dclose(dset);
        return false;
    }
    out->assign((std::size_t) dims[0], T{});
    const bool ok = out->empty() || H5Dread(dset, dtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, out->data()) >= 0;
    H5Sclose(space);
    H5Dclose(dset);
    return ok;
}

template<typename T>
inline bool read_optional_dataset_vector(hid_t parent, const char *name, hid_t dtype, std::vector<T> *out) {
    hid_t dset = (hid_t) -1;
    H5E_BEGIN_TRY {
        dset = H5Dopen2(parent, name, H5P_DEFAULT);
    } H5E_END_TRY;
    if (dset < 0) {
        if (out != nullptr) out->clear();
        return true;
    }
    H5Dclose(dset);
    return read_dataset_vector(parent, name, dtype, out);
}

inline bool read_text_column_strings(hid_t parent, const char *name, std::vector<std::string> *out) {
    hid_t group = H5Gopen2(parent, name, H5P_DEFAULT);
    std::uint32_t count = 0;
    std::uint32_t bytes = 0;
    std::vector<std::uint32_t> offsets;
    std::vector<char> data;
    if (out == nullptr) return false;
    out->clear();
    if (group < 0) return false;
    if (!read_attr_u32(group, "count", &count) || !read_attr_u32(group, "bytes", &bytes)) {
        H5Gclose(group);
        return false;
    }
    offsets.assign((std::size_t) count + 1u, 0u);
    data.assign((std::size_t) bytes, '\0');
    if (!read_dataset_vector(group, "offsets", H5T_NATIVE_UINT32, &offsets)
        || !read_dataset_vector(group, "data", H5T_NATIVE_CHAR, &data)) {
        H5Gclose(group);
        return false;
    }
    out->reserve(count);
    for (std::uint32_t i = 0; i < count; ++i) {
        const std::uint32_t begin = offsets[i];
        const std::uint32_t end = offsets[i + 1u];
        if (end <= begin || end > data.size()) {
            out->push_back(std::string());
            continue;
        }
        out->emplace_back(data.data() + begin);
    }
    H5Gclose(group);
    return true;
}

inline bool read_optional_text_column_strings(hid_t parent, const char *name, std::vector<std::string> *out) {
    hid_t group = (hid_t) -1;
    H5E_BEGIN_TRY {
        group = H5Gopen2(parent, name, H5P_DEFAULT);
    } H5E_END_TRY;
    if (group < 0) {
        if (out != nullptr) out->clear();
        return true;
    }
    H5Gclose(group);
    return read_text_column_strings(parent, name, out);
}

inline unsigned long find_partition_index_for_row(const std::vector<std::uint64_t> &partition_row_offsets,
                                                  std::uint64_t row_begin) {
    if (partition_row_offsets.size() < 2u) return 0ul;
    const auto it = std::upper_bound(partition_row_offsets.begin(), partition_row_offsets.end(), row_begin);
    if (it == partition_row_offsets.begin()) return 0ul;
    return (unsigned long) std::distance(partition_row_offsets.begin(), it - 1);
}

inline unsigned long find_partition_end_for_row(const std::vector<std::uint64_t> &partition_row_offsets,
                                                std::uint64_t row_end) {
    if (partition_row_offsets.size() < 2u || row_end == 0u) return 0ul;
    const auto it = std::lower_bound(partition_row_offsets.begin(), partition_row_offsets.end(), row_end);
    return (unsigned long) std::distance(partition_row_offsets.begin(), it);
}

template<typename Part>
inline void clear_loaded_view(cellshard::sharded<Part> *view, cellshard::shard_storage *storage) {
    if (storage != nullptr) cellshard::clear(storage);
    if (view != nullptr) cellshard::clear(view);
}

inline void append_blocked_ell_row(const cellshard::sparse::blocked_ell *part,
                                   std::uint32_t row,
                                   std::vector<std::int64_t> *indices,
                                   std::vector<float> *data) {
    const std::uint32_t block_size = part != nullptr ? part->block_size : 0u;
    const std::uint32_t width = part != nullptr ? cellshard::sparse::ell_width_blocks(part) : 0u;
    const std::uint32_t row_block = block_size == 0u ? 0u : row / block_size;
    if (part == nullptr || indices == nullptr || data == nullptr || block_size == 0u) return;

    for (std::uint32_t slot = 0u; slot < width; ++slot) {
        const std::uint32_t stored = part->blockColIdx[(std::size_t) row_block * width + slot];
        if (stored == cellshard::sparse::blocked_ell_invalid_col) continue;
        for (std::uint32_t col_in_block = 0u; col_in_block < block_size; ++col_in_block) {
            const std::uint32_t col = stored * block_size + col_in_block;
            const float value = __half2float(
                part->val[(std::size_t) row * part->ell_cols + (std::size_t) slot * block_size + col_in_block]);
            if (col >= part->cols || value == 0.0f) continue;
            indices->push_back((std::int64_t) col);
            data->push_back(value);
        }
    }
}

inline void append_sliced_ell_row(const cellshard::sparse::sliced_ell *part,
                                  std::uint32_t row,
                                  std::vector<std::int64_t> *indices,
                                  std::vector<float> *data) {
    if (part == nullptr || indices == nullptr || data == nullptr || row >= part->rows) return;

    const std::uint32_t slice = cellshard::sparse::find_slice(part, row);
    if (slice >= part->slice_count) return;
    const std::uint32_t row_begin = part->slice_row_offsets[slice];
    const std::uint32_t width = part->slice_widths[slice];
    const std::size_t slot_base = cellshard::sparse::slice_slot_base(part, slice)
        + (std::size_t) (row - row_begin) * (std::size_t) width;

    for (std::uint32_t slot = 0u; slot < width; ++slot) {
        const std::uint32_t col = part->col_idx[slot_base + slot];
        const float value = __half2float(part->val[slot_base + slot]);
        if (col == cellshard::sparse::sliced_ell_invalid_col || col >= part->cols || value == 0.0f) continue;
        indices->push_back((std::int64_t) col);
        data->push_back(value);
    }
}

inline bool load_matrix_format(const char *path, std::string *matrix_format, std::string *error) {
    hid_t file = (hid_t) -1;
    bool ok = false;
    if (matrix_format == nullptr || path == nullptr || *path == '\0') {
        set_error(error, "invalid dataset path");
        return false;
    }
    file = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0) {
        set_error(error, "failed to open dataset file");
        return false;
    }
    ok = read_attr_string(file, "matrix_format", matrix_format);
    H5Fclose(file);
    if (!ok) set_error(error, "failed to read matrix_format");
    return ok;
}

} // namespace cellshard::exporting::detail
