#include "dataset_export.hh"

#include "../src/real.cuh"
#include "../src/formats/compressed.cuh"
#include "../src/formats/blocked_ell.cuh"
#include "../src/sharded/sharded.cuh"
#include "../src/sharded/sharded_host.cuh"
#include "../src/sharded/shard_paths.cuh"
#include "../src/disk/csh5.cuh"
#include "../src/sharded/disk.cuh"

#include <hdf5.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace cellshard::exporting {

namespace {

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
bool read_dataset_vector(hid_t parent, const char *name, hid_t dtype, std::vector<T> *out) {
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
bool read_optional_dataset_vector(hid_t parent, const char *name, hid_t dtype, std::vector<T> *out) {
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

inline bool load_observation_metadata(const char *path,
                                      std::vector<observation_metadata_column> *columns,
                                      std::string *error);

struct byte_reader {
    const std::uint8_t *cur = nullptr;
    const std::uint8_t *end = nullptr;
};

inline bool checked_size_from_u64(std::uint64_t value, std::size_t *out) {
    if (out == nullptr || value > (std::uint64_t) std::numeric_limits<std::size_t>::max()) return false;
    *out = (std::size_t) value;
    return true;
}

inline void append_bytes(std::vector<std::uint8_t> *out, const void *data, std::size_t bytes) {
    const auto *begin = static_cast<const std::uint8_t *>(data);
    if (out == nullptr || bytes == 0u) return;
    out->insert(out->end(), begin, begin + bytes);
}

template<typename T>
inline void append_pod(std::vector<std::uint8_t> *out, const T &value) {
    append_bytes(out, &value, sizeof(T));
}

template<typename T>
inline void append_pod_vector(std::vector<std::uint8_t> *out, const std::vector<T> &values) {
    const std::uint64_t count = (std::uint64_t) values.size();
    append_pod(out, count);
    if (!values.empty()) append_bytes(out, values.data(), values.size() * sizeof(T));
}

inline void append_string(std::vector<std::uint8_t> *out, const std::string &value) {
    const std::uint64_t bytes = (std::uint64_t) value.size();
    append_pod(out, bytes);
    if (!value.empty()) append_bytes(out, value.data(), value.size());
}

inline void append_string_vector(std::vector<std::uint8_t> *out, const std::vector<std::string> &values) {
    const std::uint64_t count = (std::uint64_t) values.size();
    append_pod(out, count);
    for (const std::string &value : values) append_string(out, value);
}

template<typename T>
inline bool read_pod(byte_reader *reader, T *out, std::string *error, const char *label) {
    if (reader == nullptr || out == nullptr || reader->cur == nullptr || reader->end == nullptr
        || (std::size_t) (reader->end - reader->cur) < sizeof(T)) {
        set_error(error, std::string("truncated metadata snapshot while reading ") + (label != nullptr ? label : "field"));
        return false;
    }
    std::memcpy(out, reader->cur, sizeof(T));
    reader->cur += sizeof(T);
    return true;
}

inline bool read_string(byte_reader *reader, std::string *out, std::string *error, const char *label) {
    std::uint64_t size = 0u;
    std::size_t bytes = 0u;
    if (out == nullptr || !read_pod(reader, &size, error, label)) return false;
    if (!checked_size_from_u64(size, &bytes)) {
        set_error(error, std::string("metadata snapshot string is too large for ") + (label != nullptr ? label : "field"));
        return false;
    }
    if ((std::size_t) (reader->end - reader->cur) < bytes) {
        set_error(error, std::string("truncated metadata snapshot while reading ") + (label != nullptr ? label : "string"));
        return false;
    }
    out->assign(reinterpret_cast<const char *>(reader->cur), bytes);
    reader->cur += bytes;
    return true;
}

inline bool read_string_vector(byte_reader *reader,
                               std::vector<std::string> *out,
                               std::string *error,
                               const char *label) {
    std::uint64_t count = 0u;
    std::size_t size = 0u;
    if (out == nullptr || !read_pod(reader, &count, error, label)) return false;
    if (!checked_size_from_u64(count, &size)) {
        set_error(error, std::string("metadata snapshot string vector is too large for ") + (label != nullptr ? label : "field"));
        return false;
    }
    out->clear();
    out->reserve(size);
    for (std::size_t i = 0; i < size; ++i) {
        std::string value;
        if (!read_string(reader, &value, error, label)) return false;
        out->push_back(std::move(value));
    }
    return true;
}

template<typename T>
inline bool read_pod_vector(byte_reader *reader,
                            std::vector<T> *out,
                            std::string *error,
                            const char *label) {
    std::uint64_t count = 0u;
    std::size_t size = 0u;
    if (out == nullptr || !read_pod(reader, &count, error, label)) return false;
    if (!checked_size_from_u64(count, &size)) {
        set_error(error, std::string("metadata snapshot vector is too large for ") + (label != nullptr ? label : "field"));
        return false;
    }
    const std::size_t bytes = size * sizeof(T);
    if (size != 0u && bytes / sizeof(T) != size) {
        set_error(error, std::string("metadata snapshot vector size overflow for ") + (label != nullptr ? label : "field"));
        return false;
    }
    if ((std::size_t) (reader->end - reader->cur) < bytes) {
        set_error(error, std::string("truncated metadata snapshot while reading ") + (label != nullptr ? label : "vector"));
        return false;
    }
    out->assign(size, T{});
    if (bytes != 0u) {
        std::memcpy(out->data(), reader->cur, bytes);
        reader->cur += bytes;
    }
    return true;
}

inline std::uint64_t fnv1a64(const std::vector<std::uint8_t> &bytes) {
    std::uint64_t hash = 1469598103934665603ull;
    for (std::uint8_t value : bytes) {
        hash ^= (std::uint64_t) value;
        hash *= 1099511628211ull;
    }
    return hash;
}

inline bool load_embedded_metadata_tables(const char *path,
                                          std::vector<embedded_metadata_table> *tables,
                                          std::string *error) {
    hid_t file = (hid_t) -1;
    hid_t root = (hid_t) -1;
    std::uint32_t count = 0u;
    std::vector<std::uint32_t> dataset_indices;
    std::vector<std::uint64_t> row_begin;
    std::vector<std::uint64_t> row_end;

    if (tables == nullptr || path == nullptr || *path == '\0') {
        set_error(error, "invalid embedded metadata request");
        return false;
    }
    tables->clear();
    file = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0) {
        set_error(error, "failed to open dataset file for embedded metadata");
        return false;
    }
    root = open_optional_group(file, "/embedded_metadata");
    if (root < 0) {
        H5Fclose(file);
        return true;
    }
    if (!read_attr_u32(root, "count", &count)
        || !read_dataset_vector(root, "dataset_indices", H5T_NATIVE_UINT32, &dataset_indices)
        || !read_dataset_vector(root, "global_row_begin", H5T_NATIVE_UINT64, &row_begin)
        || !read_dataset_vector(root, "global_row_end", H5T_NATIVE_UINT64, &row_end)) {
        set_error(error, "failed to read embedded metadata directory");
        H5Gclose(root);
        H5Fclose(file);
        return false;
    }

    tables->reserve(count);
    for (std::uint32_t i = 0u; i < count; ++i) {
        char name[64];
        hid_t table = (hid_t) -1;
        embedded_metadata_table out;
        if (std::snprintf(name, sizeof(name), "table_%u", i) <= 0) {
            set_error(error, "failed to format embedded metadata table name");
            H5Gclose(root);
            H5Fclose(file);
            return false;
        }
        table = H5Gopen2(root, name, H5P_DEFAULT);
        if (table < 0) {
            set_error(error, "failed to open embedded metadata table");
            H5Gclose(root);
            H5Fclose(file);
            return false;
        }
        if (!read_attr_u32(table, "rows", &out.rows)
            || !read_attr_u32(table, "cols", &out.cols)
            || !read_text_column_strings(table, "column_names", &out.column_names)
            || !read_text_column_strings(table, "field_values", &out.field_values)
            || !read_dataset_vector(table, "row_offsets", H5T_NATIVE_UINT32, &out.row_offsets)) {
            H5Gclose(table);
            set_error(error, "failed to read embedded metadata payload");
            H5Gclose(root);
            H5Fclose(file);
            return false;
        }
        H5Gclose(table);
        out.dataset_index = i < dataset_indices.size() ? dataset_indices[i] : i;
        out.row_begin = i < row_begin.size() ? row_begin[i] : 0u;
        out.row_end = i < row_end.size() ? row_end[i] : 0u;
        tables->push_back(std::move(out));
    }

    H5Gclose(root);
    H5Fclose(file);
    return true;
}

inline bool load_observation_metadata_table(const char *path,
                                            std::uint64_t *rows_out,
                                            std::vector<observation_metadata_column> *columns,
                                            std::string *error) {
    hid_t file = (hid_t) -1;
    hid_t metadata = (hid_t) -1;
    std::uint32_t cols = 0u;
    std::uint64_t rows = 0u;

    if (rows_out == nullptr || columns == nullptr || path == nullptr || *path == '\0') {
        set_error(error, "invalid observation metadata request");
        return false;
    }
    *rows_out = 0u;
    columns->clear();
    file = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0) {
        set_error(error, "failed to open dataset file for observation metadata");
        return false;
    }
    metadata = open_optional_group(file, "/observation_metadata");
    if (metadata < 0) {
        H5Fclose(file);
        return true;
    }
    if (!read_attr_u64(metadata, "rows", &rows) || !read_attr_u32(metadata, "cols", &cols)) {
        set_error(error, "failed to read observation metadata header");
        H5Gclose(metadata);
        H5Fclose(file);
        return false;
    }
    H5Gclose(metadata);
    H5Fclose(file);
    if (!load_observation_metadata(path, columns, error)) return false;
    *rows_out = rows;
    return true;
}

inline bool load_observation_metadata(const char *path,
                                      std::vector<observation_metadata_column> *columns,
                                      std::string *error) {
    hid_t file = (hid_t) -1;
    hid_t metadata = (hid_t) -1;
    std::uint32_t cols = 0u;
    std::uint64_t rows = 0u;

    if (columns == nullptr || path == nullptr || *path == '\0') {
        set_error(error, "invalid observation metadata request");
        return false;
    }
    columns->clear();

    file = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0) {
        set_error(error, "failed to open dataset file for observation metadata");
        return false;
    }

    metadata = open_optional_group(file, "/observation_metadata");
    if (metadata < 0) {
        H5Fclose(file);
        return true;
    }

    if (!read_attr_u64(metadata, "rows", &rows) || !read_attr_u32(metadata, "cols", &cols)) {
        set_error(error, "failed to read observation metadata header");
        H5Gclose(metadata);
        H5Fclose(file);
        return false;
    }

    columns->reserve(cols);
    for (std::uint32_t i = 0; i < cols; ++i) {
        char name[64];
        hid_t column = (hid_t) -1;
        observation_metadata_column out;
        if (std::snprintf(name, sizeof(name), "column_%u", i) <= 0) {
            set_error(error, "failed to format observation metadata column name");
            H5Gclose(metadata);
            H5Fclose(file);
            return false;
        }
        column = H5Gopen2(metadata, name, H5P_DEFAULT);
        if (column < 0) {
            set_error(error, "failed to open observation metadata column");
            H5Gclose(metadata);
            H5Fclose(file);
            return false;
        }
        if (!read_attr_string(column, "name", &out.name) || !read_attr_u32(column, "type", &out.type)) {
            H5Gclose(column);
            set_error(error, "failed to read observation metadata column header");
            H5Gclose(metadata);
            H5Fclose(file);
            return false;
        }
        if (out.type == cellshard::dataset_observation_metadata_type_text) {
            if (!read_text_column_strings(column, "values", &out.text_values)) {
                H5Gclose(column);
                set_error(error, "failed to read observation metadata text values");
                H5Gclose(metadata);
                H5Fclose(file);
                return false;
            }
        } else if (out.type == cellshard::dataset_observation_metadata_type_float32) {
            if (!read_dataset_vector(column, "values", H5T_NATIVE_FLOAT, &out.float32_values)) {
                H5Gclose(column);
                set_error(error, "failed to read observation metadata float values");
                H5Gclose(metadata);
                H5Fclose(file);
                return false;
            }
        } else if (out.type == cellshard::dataset_observation_metadata_type_uint8) {
            if (!read_dataset_vector(column, "values", H5T_NATIVE_UINT8, &out.uint8_values)) {
                H5Gclose(column);
                set_error(error, "failed to read observation metadata uint8 values");
                H5Gclose(metadata);
                H5Fclose(file);
                return false;
            }
        } else {
            H5Gclose(column);
            set_error(error, "observation metadata contains an unknown column type");
            H5Gclose(metadata);
            H5Fclose(file);
            return false;
        }
        H5Gclose(column);
        columns->push_back(std::move(out));
    }

    H5Gclose(metadata);
    H5Fclose(file);
    return true;
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
            const float value = __half2float(part->val[(std::size_t) row * part->ell_cols + (std::size_t) slot * block_size + col_in_block]);
            if (col >= part->cols) continue;
            if (value == 0.0f) continue;
            indices->push_back((std::int64_t) col);
            data->push_back(value);
        }
    }
}

inline void append_compressed_row(const cellshard::sparse::compressed *part,
                                  std::uint32_t row,
                                  std::vector<std::int64_t> *indices,
                                  std::vector<float> *data) {
    if (part == nullptr || indices == nullptr || data == nullptr || row >= part->rows) return;
    for (cellshard::types::ptr_t i = part->majorPtr[row]; i < part->majorPtr[row + 1u]; ++i) {
        indices->push_back((std::int64_t) part->minorIdx[i]);
        data->push_back(__half2float(part->val[i]));
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
        if (col == cellshard::sparse::sliced_ell_invalid_col || col >= part->cols) continue;
        if (value == 0.0f) continue;
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

inline void append_source_dataset_summary(std::vector<std::uint8_t> *out, const source_dataset_summary &value) {
    append_string(out, value.dataset_id);
    append_string(out, value.matrix_path);
    append_string(out, value.feature_path);
    append_string(out, value.barcode_path);
    append_string(out, value.metadata_path);
    append_pod(out, value.format);
    append_pod(out, value.row_begin);
    append_pod(out, value.row_end);
    append_pod(out, value.rows);
    append_pod(out, value.cols);
    append_pod(out, value.nnz);
}

inline bool read_source_dataset_summary(byte_reader *reader,
                                        source_dataset_summary *out,
                                        std::string *error) {
    return out != nullptr
        && read_string(reader, &out->dataset_id, error, "dataset_id")
        && read_string(reader, &out->matrix_path, error, "matrix_path")
        && read_string(reader, &out->feature_path, error, "feature_path")
        && read_string(reader, &out->barcode_path, error, "barcode_path")
        && read_string(reader, &out->metadata_path, error, "metadata_path")
        && read_pod(reader, &out->format, error, "dataset_format")
        && read_pod(reader, &out->row_begin, error, "dataset_row_begin")
        && read_pod(reader, &out->row_end, error, "dataset_row_end")
        && read_pod(reader, &out->rows, error, "dataset_rows")
        && read_pod(reader, &out->cols, error, "dataset_cols")
        && read_pod(reader, &out->nnz, error, "dataset_nnz");
}

inline void append_dataset_codec_summary(std::vector<std::uint8_t> *out, const dataset_codec_summary &value) {
    append_pod(out, value.codec_id);
    append_pod(out, value.family);
    append_pod(out, value.value_code);
    append_pod(out, value.scale_value_code);
    append_pod(out, value.bits);
    append_pod(out, value.flags);
}

inline bool read_dataset_codec_summary(byte_reader *reader,
                                       dataset_codec_summary *out,
                                       std::string *error) {
    return out != nullptr
        && read_pod(reader, &out->codec_id, error, "codec_id")
        && read_pod(reader, &out->family, error, "codec_family")
        && read_pod(reader, &out->value_code, error, "codec_value_code")
        && read_pod(reader, &out->scale_value_code, error, "codec_scale_value_code")
        && read_pod(reader, &out->bits, error, "codec_bits")
        && read_pod(reader, &out->flags, error, "codec_flags");
}

inline void append_dataset_partition_summary(std::vector<std::uint8_t> *out, const dataset_partition_summary &value) {
    append_pod(out, value.partition_id);
    append_pod(out, value.row_begin);
    append_pod(out, value.row_end);
    append_pod(out, value.rows);
    append_pod(out, value.nnz);
    append_pod(out, value.aux);
    append_pod(out, value.dataset_id);
    append_pod(out, value.axis);
    append_pod(out, value.codec_id);
}

inline bool read_dataset_partition_summary(byte_reader *reader,
                                           dataset_partition_summary *out,
                                           std::string *error) {
    return out != nullptr
        && read_pod(reader, &out->partition_id, error, "partition_id")
        && read_pod(reader, &out->row_begin, error, "partition_row_begin")
        && read_pod(reader, &out->row_end, error, "partition_row_end")
        && read_pod(reader, &out->rows, error, "partition_rows")
        && read_pod(reader, &out->nnz, error, "partition_nnz")
        && read_pod(reader, &out->aux, error, "partition_aux")
        && read_pod(reader, &out->dataset_id, error, "partition_dataset_id")
        && read_pod(reader, &out->axis, error, "partition_axis")
        && read_pod(reader, &out->codec_id, error, "partition_codec_id");
}

inline void append_dataset_shard_summary(std::vector<std::uint8_t> *out, const dataset_shard_summary &value) {
    append_pod(out, value.shard_id);
    append_pod(out, value.partition_begin);
    append_pod(out, value.partition_end);
    append_pod(out, value.row_begin);
    append_pod(out, value.row_end);
}

inline bool read_dataset_shard_summary(byte_reader *reader,
                                       dataset_shard_summary *out,
                                       std::string *error) {
    return out != nullptr
        && read_pod(reader, &out->shard_id, error, "shard_id")
        && read_pod(reader, &out->partition_begin, error, "shard_partition_begin")
        && read_pod(reader, &out->partition_end, error, "shard_partition_end")
        && read_pod(reader, &out->row_begin, error, "shard_row_begin")
        && read_pod(reader, &out->row_end, error, "shard_row_end");
}

inline void append_dataset_summary(std::vector<std::uint8_t> *out,
                                   const dataset_summary &value,
                                   bool include_path) {
    if (include_path) append_string(out, value.path);
    append_string(out, value.matrix_format);
    append_string(out, value.payload_layout);
    append_pod(out, value.rows);
    append_pod(out, value.cols);
    append_pod(out, value.nnz);
    append_pod(out, value.num_partitions);
    append_pod(out, value.num_shards);
    append_pod(out, value.num_datasets);
    append_pod(out, (std::uint64_t) value.datasets.size());
    for (const source_dataset_summary &dataset : value.datasets) append_source_dataset_summary(out, dataset);
    append_pod(out, (std::uint64_t) value.partitions.size());
    for (const dataset_partition_summary &partition : value.partitions) append_dataset_partition_summary(out, partition);
    append_pod(out, (std::uint64_t) value.shards.size());
    for (const dataset_shard_summary &shard : value.shards) append_dataset_shard_summary(out, shard);
    append_pod(out, (std::uint64_t) value.codecs.size());
    for (const dataset_codec_summary &codec : value.codecs) append_dataset_codec_summary(out, codec);
    append_string_vector(out, value.obs_names);
    append_string_vector(out, value.var_ids);
    append_string_vector(out, value.var_names);
    append_string_vector(out, value.var_types);
}

inline bool read_dataset_summary(byte_reader *reader,
                                 dataset_summary *out,
                                 std::string *error,
                                 bool include_path) {
    std::uint64_t count = 0u;
    std::size_t size = 0u;
    if (out == nullptr) return false;
    *out = dataset_summary{};
    if (include_path && !read_string(reader, &out->path, error, "summary_path")) return false;
    if (!read_string(reader, &out->matrix_format, error, "matrix_format")
        || !read_string(reader, &out->payload_layout, error, "payload_layout")
        || !read_pod(reader, &out->rows, error, "rows")
        || !read_pod(reader, &out->cols, error, "cols")
        || !read_pod(reader, &out->nnz, error, "nnz")
        || !read_pod(reader, &out->num_partitions, error, "num_partitions")
        || !read_pod(reader, &out->num_shards, error, "num_shards")
        || !read_pod(reader, &out->num_datasets, error, "num_datasets")) return false;

    if (!read_pod(reader, &count, error, "dataset_count") || !checked_size_from_u64(count, &size)) {
        set_error(error, "metadata snapshot dataset_count is too large");
        return false;
    }
    out->datasets.assign(size, source_dataset_summary{});
    for (std::size_t i = 0; i < size; ++i) {
        if (!read_source_dataset_summary(reader, out->datasets.data() + i, error)) return false;
    }

    if (!read_pod(reader, &count, error, "partition_count") || !checked_size_from_u64(count, &size)) {
        set_error(error, "metadata snapshot partition_count is too large");
        return false;
    }
    out->partitions.assign(size, dataset_partition_summary{});
    for (std::size_t i = 0; i < size; ++i) {
        if (!read_dataset_partition_summary(reader, out->partitions.data() + i, error)) return false;
    }

    if (!read_pod(reader, &count, error, "shard_count") || !checked_size_from_u64(count, &size)) {
        set_error(error, "metadata snapshot shard_count is too large");
        return false;
    }
    out->shards.assign(size, dataset_shard_summary{});
    for (std::size_t i = 0; i < size; ++i) {
        if (!read_dataset_shard_summary(reader, out->shards.data() + i, error)) return false;
    }

    if (!read_pod(reader, &count, error, "codec_count") || !checked_size_from_u64(count, &size)) {
        set_error(error, "metadata snapshot codec_count is too large");
        return false;
    }
    out->codecs.assign(size, dataset_codec_summary{});
    for (std::size_t i = 0; i < size; ++i) {
        if (!read_dataset_codec_summary(reader, out->codecs.data() + i, error)) return false;
    }

    return read_string_vector(reader, &out->obs_names, error, "obs_names")
        && read_string_vector(reader, &out->var_ids, error, "var_ids")
        && read_string_vector(reader, &out->var_names, error, "var_names")
        && read_string_vector(reader, &out->var_types, error, "var_types");
}

inline void append_observation_metadata_column(std::vector<std::uint8_t> *out,
                                               const observation_metadata_column &value) {
    append_string(out, value.name);
    append_pod(out, value.type);
    append_string_vector(out, value.text_values);
    append_pod_vector(out, value.float32_values);
    append_pod_vector(out, value.uint8_values);
}

inline bool read_observation_metadata_column(byte_reader *reader,
                                             observation_metadata_column *out,
                                             std::string *error) {
    return out != nullptr
        && read_string(reader, &out->name, error, "observation_metadata_name")
        && read_pod(reader, &out->type, error, "observation_metadata_type")
        && read_string_vector(reader, &out->text_values, error, "observation_text_values")
        && read_pod_vector(reader, &out->float32_values, error, "observation_float32_values")
        && read_pod_vector(reader, &out->uint8_values, error, "observation_uint8_values");
}

inline void append_embedded_metadata_table(std::vector<std::uint8_t> *out,
                                           const embedded_metadata_table &value) {
    append_pod(out, value.dataset_index);
    append_pod(out, value.row_begin);
    append_pod(out, value.row_end);
    append_pod(out, value.rows);
    append_pod(out, value.cols);
    append_string_vector(out, value.column_names);
    append_string_vector(out, value.field_values);
    append_pod_vector(out, value.row_offsets);
}

inline bool read_embedded_metadata_table(byte_reader *reader,
                                         embedded_metadata_table *out,
                                         std::string *error) {
    return out != nullptr
        && read_pod(reader, &out->dataset_index, error, "embedded_dataset_index")
        && read_pod(reader, &out->row_begin, error, "embedded_row_begin")
        && read_pod(reader, &out->row_end, error, "embedded_row_end")
        && read_pod(reader, &out->rows, error, "embedded_rows")
        && read_pod(reader, &out->cols, error, "embedded_cols")
        && read_string_vector(reader, &out->column_names, error, "embedded_column_names")
        && read_string_vector(reader, &out->field_values, error, "embedded_field_values")
        && read_pod_vector(reader, &out->row_offsets, error, "embedded_row_offsets");
}

inline void append_execution_partition_metadata(std::vector<std::uint8_t> *out,
                                                const execution_partition_metadata &value) {
    append_pod(out, value.partition_id);
    append_pod(out, value.row_begin);
    append_pod(out, value.row_end);
    append_pod(out, value.rows);
    append_pod(out, value.nnz);
    append_pod(out, value.aux);
    append_pod(out, value.dataset_id);
    append_pod(out, value.axis);
    append_pod(out, value.codec_id);
    append_pod(out, value.execution_format);
    append_pod(out, value.blocked_ell_block_size);
    append_pod(out, value.blocked_ell_bucket_count);
    append_pod(out, value.blocked_ell_fill_ratio);
    append_pod(out, value.execution_bytes);
    append_pod(out, value.blocked_ell_bytes);
    append_pod(out, value.bucketed_blocked_ell_bytes);
}

inline bool read_execution_partition_metadata(byte_reader *reader,
                                              execution_partition_metadata *out,
                                              std::string *error) {
    return out != nullptr
        && read_pod(reader, &out->partition_id, error, "exec_partition_id")
        && read_pod(reader, &out->row_begin, error, "exec_partition_row_begin")
        && read_pod(reader, &out->row_end, error, "exec_partition_row_end")
        && read_pod(reader, &out->rows, error, "exec_partition_rows")
        && read_pod(reader, &out->nnz, error, "exec_partition_nnz")
        && read_pod(reader, &out->aux, error, "exec_partition_aux")
        && read_pod(reader, &out->dataset_id, error, "exec_partition_dataset_id")
        && read_pod(reader, &out->axis, error, "exec_partition_axis")
        && read_pod(reader, &out->codec_id, error, "exec_partition_codec_id")
        && read_pod(reader, &out->execution_format, error, "exec_partition_format")
        && read_pod(reader, &out->blocked_ell_block_size, error, "exec_partition_block_size")
        && read_pod(reader, &out->blocked_ell_bucket_count, error, "exec_partition_bucket_count")
        && read_pod(reader, &out->blocked_ell_fill_ratio, error, "exec_partition_fill_ratio")
        && read_pod(reader, &out->execution_bytes, error, "exec_partition_execution_bytes")
        && read_pod(reader, &out->blocked_ell_bytes, error, "exec_partition_blocked_ell_bytes")
        && read_pod(reader, &out->bucketed_blocked_ell_bytes, error, "exec_partition_bucketed_blocked_ell_bytes");
}

inline void append_execution_shard_metadata(std::vector<std::uint8_t> *out,
                                            const execution_shard_metadata &value) {
    append_pod(out, value.shard_id);
    append_pod(out, value.partition_begin);
    append_pod(out, value.partition_end);
    append_pod(out, value.row_begin);
    append_pod(out, value.row_end);
    append_pod(out, value.execution_format);
    append_pod(out, value.blocked_ell_block_size);
    append_pod(out, value.bucketed_partition_count);
    append_pod(out, value.bucketed_segment_count);
    append_pod(out, value.blocked_ell_fill_ratio);
    append_pod(out, value.execution_bytes);
    append_pod(out, value.bucketed_blocked_ell_bytes);
    append_pod(out, value.preferred_pair);
    append_pod(out, value.owner_node_id);
    append_pod(out, value.owner_rank_id);
}

inline bool read_execution_shard_metadata(byte_reader *reader,
                                          execution_shard_metadata *out,
                                          std::string *error) {
    return out != nullptr
        && read_pod(reader, &out->shard_id, error, "exec_shard_id")
        && read_pod(reader, &out->partition_begin, error, "exec_shard_partition_begin")
        && read_pod(reader, &out->partition_end, error, "exec_shard_partition_end")
        && read_pod(reader, &out->row_begin, error, "exec_shard_row_begin")
        && read_pod(reader, &out->row_end, error, "exec_shard_row_end")
        && read_pod(reader, &out->execution_format, error, "exec_shard_format")
        && read_pod(reader, &out->blocked_ell_block_size, error, "exec_shard_block_size")
        && read_pod(reader, &out->bucketed_partition_count, error, "exec_shard_bucketed_partition_count")
        && read_pod(reader, &out->bucketed_segment_count, error, "exec_shard_bucketed_segment_count")
        && read_pod(reader, &out->blocked_ell_fill_ratio, error, "exec_shard_fill_ratio")
        && read_pod(reader, &out->execution_bytes, error, "exec_shard_execution_bytes")
        && read_pod(reader, &out->bucketed_blocked_ell_bytes, error, "exec_shard_bucketed_blocked_ell_bytes")
        && read_pod(reader, &out->preferred_pair, error, "exec_shard_preferred_pair")
        && read_pod(reader, &out->owner_node_id, error, "exec_shard_owner_node_id")
        && read_pod(reader, &out->owner_rank_id, error, "exec_shard_owner_rank_id");
}

inline void append_runtime_service_metadata(std::vector<std::uint8_t> *out,
                                            const runtime_service_metadata &value) {
    append_pod(out, value.service_mode);
    append_pod(out, value.live_write_mode);
    append_pod(out, value.prefer_pack_delivery);
    append_pod(out, value.remote_pack_delivery);
    append_pod(out, value.single_reader_coordinator);
    append_pod(out, value.maintenance_lock_blocks_overwrite);
    append_pod(out, value.canonical_generation);
    append_pod(out, value.execution_plan_generation);
    append_pod(out, value.pack_generation);
    append_pod(out, value.service_epoch);
    append_pod(out, value.active_read_generation);
    append_pod(out, value.staged_write_generation);
}

inline bool read_runtime_service_metadata(byte_reader *reader,
                                          runtime_service_metadata *out,
                                          std::string *error) {
    return out != nullptr
        && read_pod(reader, &out->service_mode, error, "runtime_service_mode")
        && read_pod(reader, &out->live_write_mode, error, "runtime_live_write_mode")
        && read_pod(reader, &out->prefer_pack_delivery, error, "runtime_prefer_pack_delivery")
        && read_pod(reader, &out->remote_pack_delivery, error, "runtime_remote_pack_delivery")
        && read_pod(reader, &out->single_reader_coordinator, error, "runtime_single_reader_coordinator")
        && read_pod(reader, &out->maintenance_lock_blocks_overwrite, error, "runtime_maintenance_lock_blocks_overwrite")
        && read_pod(reader, &out->canonical_generation, error, "runtime_canonical_generation")
        && read_pod(reader, &out->execution_plan_generation, error, "runtime_execution_plan_generation")
        && read_pod(reader, &out->pack_generation, error, "runtime_pack_generation")
        && read_pod(reader, &out->service_epoch, error, "runtime_service_epoch")
        && read_pod(reader, &out->active_read_generation, error, "runtime_active_read_generation")
        && read_pod(reader, &out->staged_write_generation, error, "runtime_staged_write_generation");
}

inline bool serialize_global_metadata_snapshot_payload(const global_metadata_snapshot &snapshot,
                                                       std::vector<std::uint8_t> *out,
                                                       bool include_snapshot_id,
                                                       bool include_path,
                                                       std::string *error) {
    static const char magic[] = {'C', 'S', 'G', 'M'};
    const std::uint32_t version = 1u;
    if (out == nullptr) {
        set_error(error, "metadata snapshot output buffer is null");
        return false;
    }
    out->clear();
    append_bytes(out, magic, sizeof(magic));
    append_pod(out, version);
    if (include_snapshot_id) append_pod(out, snapshot.snapshot_id);
    append_dataset_summary(out, snapshot.summary, include_path);
    append_pod(out, (std::uint64_t) snapshot.embedded_metadata.size());
    for (const embedded_metadata_table &table : snapshot.embedded_metadata) append_embedded_metadata_table(out, table);
    append_pod(out, snapshot.observation_metadata_rows);
    append_pod(out, (std::uint64_t) snapshot.observation_metadata.size());
    for (const observation_metadata_column &column : snapshot.observation_metadata) append_observation_metadata_column(out, column);
    append_pod(out, (std::uint64_t) snapshot.execution_partitions.size());
    for (const execution_partition_metadata &partition : snapshot.execution_partitions) append_execution_partition_metadata(out, partition);
    append_pod(out, (std::uint64_t) snapshot.execution_shards.size());
    for (const execution_shard_metadata &shard : snapshot.execution_shards) append_execution_shard_metadata(out, shard);
    append_runtime_service_metadata(out, snapshot.runtime_service);
    return true;
}

inline std::uint64_t compute_snapshot_id(const global_metadata_snapshot &snapshot) {
    global_metadata_snapshot normalized = snapshot;
    std::vector<std::uint8_t> bytes;
    normalized.snapshot_id = 0u;
    (void) serialize_global_metadata_snapshot_payload(normalized, &bytes, false, false, nullptr);
    return fnv1a64(bytes);
}

inline client_snapshot_ref build_client_snapshot_ref(const global_metadata_snapshot &snapshot) {
    client_snapshot_ref ref;
    ref.snapshot_id = compute_snapshot_id(snapshot);
    ref.canonical_generation = snapshot.runtime_service.canonical_generation;
    ref.execution_plan_generation = snapshot.runtime_service.execution_plan_generation;
    ref.pack_generation = snapshot.runtime_service.pack_generation;
    ref.service_epoch = snapshot.runtime_service.service_epoch;
    return ref;
}

inline void copy_execution_partition_metadata(const dataset_summary &summary,
                                              const cellshard::dataset_execution_view &execution,
                                              std::vector<execution_partition_metadata> *out) {
    const std::size_t count = summary.partitions.size();
    if (out == nullptr) return;
    out->assign(count, execution_partition_metadata{});
    for (std::size_t i = 0; i < count; ++i) {
        (*out)[i] = execution_partition_metadata{
            summary.partitions[i].partition_id,
            summary.partitions[i].row_begin,
            summary.partitions[i].row_end,
            summary.partitions[i].rows,
            summary.partitions[i].nnz,
            summary.partitions[i].aux,
            summary.partitions[i].dataset_id,
            summary.partitions[i].axis,
            summary.partitions[i].codec_id,
            i < execution.partition_count && execution.partition_execution_formats != nullptr ? execution.partition_execution_formats[i] : 0u,
            i < execution.partition_count && execution.partition_blocked_ell_block_sizes != nullptr ? execution.partition_blocked_ell_block_sizes[i] : 0u,
            i < execution.partition_count && execution.partition_blocked_ell_bucket_counts != nullptr ? execution.partition_blocked_ell_bucket_counts[i] : 0u,
            i < execution.partition_count && execution.partition_blocked_ell_fill_ratios != nullptr ? execution.partition_blocked_ell_fill_ratios[i] : 0.0f,
            i < execution.partition_count && execution.partition_execution_bytes != nullptr ? execution.partition_execution_bytes[i] : 0u,
            i < execution.partition_count && execution.partition_blocked_ell_bytes != nullptr ? execution.partition_blocked_ell_bytes[i] : 0u,
            i < execution.partition_count && execution.partition_bucketed_blocked_ell_bytes != nullptr ? execution.partition_bucketed_blocked_ell_bytes[i] : 0u
        };
    }
}

inline void copy_execution_shard_metadata(const dataset_summary &summary,
                                          const cellshard::dataset_execution_view &execution,
                                          std::vector<execution_shard_metadata> *out) {
    const std::size_t count = summary.shards.size();
    if (out == nullptr) return;
    out->assign(count, execution_shard_metadata{});
    for (std::size_t i = 0; i < count; ++i) {
        (*out)[i] = execution_shard_metadata{
            summary.shards[i].shard_id,
            summary.shards[i].partition_begin,
            summary.shards[i].partition_end,
            summary.shards[i].row_begin,
            summary.shards[i].row_end,
            i < execution.shard_count && execution.shard_execution_formats != nullptr ? execution.shard_execution_formats[i] : 0u,
            i < execution.shard_count && execution.shard_blocked_ell_block_sizes != nullptr ? execution.shard_blocked_ell_block_sizes[i] : 0u,
            i < execution.shard_count && execution.shard_bucketed_partition_counts != nullptr ? execution.shard_bucketed_partition_counts[i] : 0u,
            i < execution.shard_count && execution.shard_bucketed_segment_counts != nullptr ? execution.shard_bucketed_segment_counts[i] : 0u,
            i < execution.shard_count && execution.shard_blocked_ell_fill_ratios != nullptr ? execution.shard_blocked_ell_fill_ratios[i] : 0.0f,
            i < execution.shard_count && execution.shard_execution_bytes != nullptr ? execution.shard_execution_bytes[i] : 0u,
            i < execution.shard_count && execution.shard_bucketed_blocked_ell_bytes != nullptr ? execution.shard_bucketed_blocked_ell_bytes[i] : 0u,
            i < execution.shard_count && execution.shard_preferred_pair_ids != nullptr ? execution.shard_preferred_pair_ids[i] : 0u,
            i < execution.shard_count && execution.shard_owner_node_ids != nullptr ? execution.shard_owner_node_ids[i] : 0u,
            i < execution.shard_count && execution.shard_owner_rank_ids != nullptr ? execution.shard_owner_rank_ids[i] : 0u
        };
    }
}

inline void copy_runtime_service_metadata(const cellshard::dataset_runtime_service_view &view,
                                          runtime_service_metadata *out) {
    if (out == nullptr) return;
    *out = runtime_service_metadata{
        view.service_mode,
        view.live_write_mode,
        view.prefer_pack_delivery,
        view.remote_pack_delivery,
        view.single_reader_coordinator,
        view.maintenance_lock_blocks_overwrite,
        view.canonical_generation,
        view.execution_plan_generation,
        view.pack_generation,
        view.service_epoch,
        view.active_read_generation,
        view.staged_write_generation
    };
}

} // namespace

bool load_dataset_summary(const char *path, dataset_summary *out, std::string *error) {
    hid_t file = (hid_t) -1;
    hid_t datasets = (hid_t) -1;
    hid_t matrix = (hid_t) -1;
    hid_t provenance = (hid_t) -1;
    hid_t codecs = (hid_t) -1;
    std::vector<std::string> dataset_ids;
    std::vector<std::string> matrix_paths;
    std::vector<std::string> feature_paths;
    std::vector<std::string> barcode_paths;
    std::vector<std::string> metadata_paths;
    std::vector<std::uint32_t> dataset_formats;
    std::vector<std::uint64_t> dataset_row_begin;
    std::vector<std::uint64_t> dataset_row_end;
    std::vector<std::uint64_t> dataset_rows;
    std::vector<std::uint64_t> dataset_cols;
    std::vector<std::uint64_t> dataset_nnz;
    std::vector<std::uint64_t> partition_rows;
    std::vector<std::uint64_t> partition_nnz;
    std::vector<std::uint32_t> partition_axes;
    std::vector<std::uint64_t> partition_aux;
    std::vector<std::uint64_t> partition_row_offsets;
    std::vector<std::uint32_t> partition_dataset_ids;
    std::vector<std::uint32_t> partition_codec_ids;
    std::vector<std::uint64_t> shard_offsets;
    std::vector<std::uint32_t> codec_ids;
    std::vector<std::uint32_t> codec_families;
    std::vector<std::uint32_t> codec_value_codes;
    std::vector<std::uint32_t> codec_scale_codes;
    std::vector<std::uint32_t> codec_bits;
    std::vector<std::uint32_t> codec_flags;

    if (out == nullptr || path == nullptr || *path == '\0') {
        set_error(error, "dataset path is empty");
        return false;
    }

    *out = dataset_summary{};
    out->path = path;

    file = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0) {
        set_error(error, "failed to open dataset file");
        return false;
    }

    if (!read_attr_string(file, "matrix_format", &out->matrix_format)
        || !read_attr_u64(file, "rows", &out->rows)
        || !read_attr_u64(file, "cols", &out->cols)
        || !read_attr_u64(file, "nnz", &out->nnz)
        || !read_attr_u64(file, "num_partitions", &out->num_partitions)
        || !read_attr_u64(file, "num_shards", &out->num_shards)
        || !read_attr_u64(file, "num_datasets", &out->num_datasets)) {
        H5Fclose(file);
        set_error(error, "failed to read top-level dataset attributes");
        return false;
    }
    read_attr_string(file, "payload_layout", &out->payload_layout);

    datasets = open_group(file, "/datasets");
    matrix = open_group(file, "/matrix");
    provenance = open_optional_group(file, "/provenance");
    codecs = open_optional_group(file, "/codecs");
    if (datasets < 0 || matrix < 0) {
        if (codecs >= 0) H5Gclose(codecs);
        if (provenance >= 0) H5Gclose(provenance);
        if (matrix >= 0) H5Gclose(matrix);
        if (datasets >= 0) H5Gclose(datasets);
        H5Fclose(file);
        set_error(error, "dataset file is missing required groups");
        return false;
    }

    if (out->num_datasets != 0u) {
        if (!read_text_column_strings(datasets, "dataset_ids", &dataset_ids)
            || !read_text_column_strings(datasets, "matrix_paths", &matrix_paths)
            || !read_text_column_strings(datasets, "feature_paths", &feature_paths)
            || !read_text_column_strings(datasets, "barcode_paths", &barcode_paths)
            || !read_text_column_strings(datasets, "metadata_paths", &metadata_paths)
            || !read_dataset_vector(datasets, "formats", H5T_NATIVE_UINT32, &dataset_formats)
            || !read_dataset_vector(datasets, "row_begin", H5T_NATIVE_UINT64, &dataset_row_begin)
            || !read_dataset_vector(datasets, "row_end", H5T_NATIVE_UINT64, &dataset_row_end)
            || !read_dataset_vector(datasets, "rows", H5T_NATIVE_UINT64, &dataset_rows)
            || !read_dataset_vector(datasets, "cols", H5T_NATIVE_UINT64, &dataset_cols)
            || !read_dataset_vector(datasets, "nnz", H5T_NATIVE_UINT64, &dataset_nnz)) {
            if (codecs >= 0) H5Gclose(codecs);
            if (provenance >= 0) H5Gclose(provenance);
            H5Gclose(matrix);
            H5Gclose(datasets);
            H5Fclose(file);
            set_error(error, "failed to read dataset metadata");
            return false;
        }
    }

    if (!read_dataset_vector(matrix, "partition_rows", H5T_NATIVE_UINT64, &partition_rows)
        || !read_dataset_vector(matrix, "partition_nnz", H5T_NATIVE_UINT64, &partition_nnz)
        || !read_optional_dataset_vector(matrix, "partition_axes", H5T_NATIVE_UINT32, &partition_axes)
        || !read_dataset_vector(matrix, "partition_aux", H5T_NATIVE_UINT64, &partition_aux)
        || !read_dataset_vector(matrix, "partition_row_offsets", H5T_NATIVE_UINT64, &partition_row_offsets)
        || !read_dataset_vector(matrix, "partition_dataset_ids", H5T_NATIVE_UINT32, &partition_dataset_ids)
        || !read_dataset_vector(matrix, "partition_codec_ids", H5T_NATIVE_UINT32, &partition_codec_ids)
        || !read_dataset_vector(matrix, "shard_offsets", H5T_NATIVE_UINT64, &shard_offsets)) {
        if (codecs >= 0) H5Gclose(codecs);
        if (provenance >= 0) H5Gclose(provenance);
        H5Gclose(matrix);
        H5Gclose(datasets);
        H5Fclose(file);
        set_error(error, "failed to read matrix layout metadata");
        return false;
    }

    out->datasets.reserve(dataset_ids.size());
    for (std::size_t i = 0; i < dataset_ids.size(); ++i) {
        out->datasets.push_back(source_dataset_summary{
            dataset_ids[i],
            i < matrix_paths.size() ? matrix_paths[i] : std::string(),
            i < feature_paths.size() ? feature_paths[i] : std::string(),
            i < barcode_paths.size() ? barcode_paths[i] : std::string(),
            i < metadata_paths.size() ? metadata_paths[i] : std::string(),
            i < dataset_formats.size() ? dataset_formats[i] : 0u,
            i < dataset_row_begin.size() ? dataset_row_begin[i] : 0u,
            i < dataset_row_end.size() ? dataset_row_end[i] : 0u,
            i < dataset_rows.size() ? dataset_rows[i] : 0u,
            i < dataset_cols.size() ? dataset_cols[i] : 0u,
            i < dataset_nnz.size() ? dataset_nnz[i] : 0u
        });
    }

    out->partitions.reserve(partition_rows.size());
    for (std::size_t i = 0; i < partition_rows.size(); ++i) {
        out->partitions.push_back(dataset_partition_summary{
            (std::uint64_t) i,
            i < partition_row_offsets.size() ? partition_row_offsets[i] : 0u,
            i + 1u < partition_row_offsets.size() ? partition_row_offsets[i + 1u] : 0u,
            i < partition_rows.size() ? partition_rows[i] : 0u,
            i < partition_nnz.size() ? partition_nnz[i] : 0u,
            i < partition_aux.size() ? partition_aux[i] : 0u,
            i < partition_dataset_ids.size() ? partition_dataset_ids[i] : 0u,
            i < partition_axes.size() ? partition_axes[i] : 0u,
            i < partition_codec_ids.size() ? partition_codec_ids[i] : 0u
        });
    }

    out->shards.reserve(shard_offsets.size() > 0u ? shard_offsets.size() - 1u : 0u);
    for (std::size_t i = 0; i + 1u < shard_offsets.size(); ++i) {
        const std::uint64_t row_begin = shard_offsets[i];
        const std::uint64_t row_end = shard_offsets[i + 1u];
        out->shards.push_back(dataset_shard_summary{
            (std::uint64_t) i,
            (std::uint64_t) find_partition_index_for_row(partition_row_offsets, row_begin),
            (std::uint64_t) (row_end == row_begin
                ? find_partition_index_for_row(partition_row_offsets, row_begin)
                : std::max<unsigned long>(
                    find_partition_index_for_row(partition_row_offsets, row_begin),
                    find_partition_end_for_row(partition_row_offsets, row_end))),
            row_begin,
            row_end
        });
    }

    if (codecs >= 0) {
        if (read_dataset_vector(codecs, "codec_id", H5T_NATIVE_UINT32, &codec_ids)
            && read_dataset_vector(codecs, "family", H5T_NATIVE_UINT32, &codec_families)
            && read_dataset_vector(codecs, "value_code", H5T_NATIVE_UINT32, &codec_value_codes)
            && read_dataset_vector(codecs, "scale_value_code", H5T_NATIVE_UINT32, &codec_scale_codes)
            && read_dataset_vector(codecs, "bits", H5T_NATIVE_UINT32, &codec_bits)
            && read_dataset_vector(codecs, "flags", H5T_NATIVE_UINT32, &codec_flags)) {
            out->codecs.reserve(codec_ids.size());
            for (std::size_t i = 0; i < codec_ids.size(); ++i) {
                out->codecs.push_back(dataset_codec_summary{
                    codec_ids[i],
                    i < codec_families.size() ? codec_families[i] : 0u,
                    i < codec_value_codes.size() ? codec_value_codes[i] : 0u,
                    i < codec_scale_codes.size() ? codec_scale_codes[i] : 0u,
                    i < codec_bits.size() ? codec_bits[i] : 0u,
                    i < codec_flags.size() ? codec_flags[i] : 0u
                });
            }
        }
    }

    if (provenance >= 0) {
        read_optional_text_column_strings(provenance, "global_barcodes", &out->obs_names);
        read_optional_text_column_strings(provenance, "feature_ids", &out->var_ids);
        read_optional_text_column_strings(provenance, "feature_names", &out->var_names);
        read_optional_text_column_strings(provenance, "feature_types", &out->var_types);
    }

    if (codecs >= 0) H5Gclose(codecs);
    if (provenance >= 0) H5Gclose(provenance);
    H5Gclose(matrix);
    H5Gclose(datasets);
    H5Fclose(file);
    return true;
}

bool load_dataset_as_csr(const char *path, csr_matrix_export *out, std::string *error) {
    std::string matrix_format;

    if (out == nullptr || path == nullptr || *path == '\0') {
        set_error(error, "dataset path is empty");
        return false;
    }
    *out = csr_matrix_export{};
    warn_cpu_materialize_once("CSR materialization");

    if (!load_matrix_format(path, &matrix_format, error)) return false;

    if (matrix_format == "blocked_ell") {
        cellshard::sharded<cellshard::sparse::blocked_ell> view;
        cellshard::shard_storage storage;
        unsigned long global_row = 0ul;
        cellshard::init(&view);
        cellshard::init(&storage);
        if (!cellshard::load_header(path, &view, &storage)) {
            set_error(error, "failed to load blocked-ELL dataset header");
            return false;
        }

        out->rows = view.rows;
        out->cols = view.cols;
        out->indptr.assign((std::size_t) view.rows + 1u, 0);
        out->indices.reserve(view.nnz);
        out->data.reserve(view.nnz);

        for (unsigned long partition_id = 0; partition_id < view.num_partitions; ++partition_id) {
            const cellshard::sparse::blocked_ell *part = nullptr;
            if (!cellshard::fetch_partition(&view, &storage, partition_id)) {
                cellshard::clear(&storage);
                cellshard::clear(&view);
                set_error(error, "failed to materialize blocked-ELL dataset part");
                return false;
            }
            part = view.parts[partition_id];
            if (part == nullptr) {
                cellshard::clear(&storage);
                cellshard::clear(&view);
                set_error(error, "blocked-ELL dataset part is null after fetch");
                return false;
            }
            for (std::uint32_t row = 0u; row < part->rows; ++row) {
                append_blocked_ell_row(part, row, &out->indices, &out->data);
                ++global_row;
                out->indptr[global_row] = (std::int64_t) out->data.size();
            }
            if (!cellshard::drop_partition(&view, partition_id)) {
                cellshard::clear(&storage);
                cellshard::clear(&view);
                set_error(error, "failed to release blocked-ELL dataset part");
                return false;
            }
        }

        cellshard::clear(&storage);
        cellshard::clear(&view);
        return true;
    }

    if (matrix_format == "sliced_ell") {
        cellshard::sharded<cellshard::sparse::sliced_ell> view;
        cellshard::shard_storage storage;
        unsigned long global_row = 0ul;
        cellshard::init(&view);
        cellshard::init(&storage);
        if (!cellshard::load_header(path, &view, &storage)) {
            set_error(error, "failed to load sliced-ELL dataset header");
            return false;
        }

        out->rows = view.rows;
        out->cols = view.cols;
        out->indptr.assign((std::size_t) view.rows + 1u, 0);
        out->indices.reserve(view.nnz);
        out->data.reserve(view.nnz);

        for (unsigned long partition_id = 0; partition_id < view.num_partitions; ++partition_id) {
            const cellshard::sparse::sliced_ell *part = nullptr;
            if (!cellshard::fetch_partition(&view, &storage, partition_id)) {
                cellshard::clear(&storage);
                cellshard::clear(&view);
                set_error(error, "failed to materialize sliced-ELL dataset partition");
                return false;
            }
            part = view.parts[partition_id];
            if (part == nullptr) {
                cellshard::clear(&storage);
                cellshard::clear(&view);
                set_error(error, "sliced-ELL dataset partition is null after fetch");
                return false;
            }
            for (std::uint32_t row = 0u; row < part->rows; ++row) {
                append_sliced_ell_row(part, row, &out->indices, &out->data);
                ++global_row;
                out->indptr[global_row] = (std::int64_t) out->data.size();
            }
            if (!cellshard::drop_partition(&view, partition_id)) {
                cellshard::clear(&storage);
                cellshard::clear(&view);
                set_error(error, "failed to release sliced-ELL dataset partition");
                return false;
            }
        }

        cellshard::clear(&storage);
        cellshard::clear(&view);
        return true;
    }

    set_error(error, "unsupported matrix_format for CSR export: " + matrix_format);
    return false;
}

bool load_dataset_rows_as_csr(const char *path,
                              const std::uint64_t *row_indices,
                              std::size_t row_count,
                              csr_matrix_export *out,
                              std::string *error) {
    std::string matrix_format;

    if (out == nullptr || path == nullptr || *path == '\0') {
        set_error(error, "dataset path is empty");
        return false;
    }
    if (row_count != 0u && row_indices == nullptr) {
        set_error(error, "row_indices is null");
        return false;
    }
    *out = csr_matrix_export{};
    warn_cpu_materialize_once("row-subset CSR materialization");

    if (!load_matrix_format(path, &matrix_format, error)) return false;

    if (matrix_format == "blocked_ell") {
        cellshard::sharded<cellshard::sparse::blocked_ell> view;
        cellshard::shard_storage storage;
        std::vector<unsigned long> fetched_partitions;
        cellshard::init(&view);
        cellshard::init(&storage);
        if (!cellshard::load_header(path, &view, &storage)) {
            set_error(error, "failed to load blocked-ELL dataset header");
            return false;
        }

        out->rows = (std::uint64_t) row_count;
        out->cols = view.cols;
        out->indptr.assign(row_count + 1u, 0);
        fetched_partitions.reserve(std::min<std::size_t>(row_count, (std::size_t) view.num_partitions));

        for (std::size_t i = 0; i < row_count; ++i) {
            const std::uint64_t global_row = row_indices[i];
            const cellshard::sparse::blocked_ell *part = nullptr;
            unsigned long partition_id = 0ul;
            unsigned long partition_row_begin = 0ul;
            std::uint64_t local_row = 0u;

            if (global_row >= view.rows) {
                for (unsigned long fetched : fetched_partitions) (void) cellshard::drop_partition(&view, fetched);
                cellshard::clear(&storage);
                cellshard::clear(&view);
                set_error(error, "row index is out of range");
                return false;
            }

            partition_id = cellshard::find_partition(&view, (unsigned long) global_row);
            if (partition_id >= view.num_partitions) {
                for (unsigned long fetched : fetched_partitions) (void) cellshard::drop_partition(&view, fetched);
                cellshard::clear(&storage);
                cellshard::clear(&view);
                set_error(error, "failed to resolve row index to a partition");
                return false;
            }
            if (!cellshard::partition_loaded(&view, partition_id)) {
                if (!cellshard::fetch_partition(&view, &storage, partition_id)) {
                    for (unsigned long fetched : fetched_partitions) (void) cellshard::drop_partition(&view, fetched);
                    cellshard::clear(&storage);
                    cellshard::clear(&view);
                    set_error(error, "failed to fetch blocked-ELL partition for row subset");
                    return false;
                }
                fetched_partitions.push_back(partition_id);
            }

            part = view.parts[partition_id];
            if (part == nullptr) {
                for (unsigned long fetched : fetched_partitions) (void) cellshard::drop_partition(&view, fetched);
                cellshard::clear(&storage);
                cellshard::clear(&view);
                set_error(error, "blocked-ELL partition is null after row subset fetch");
                return false;
            }

            partition_row_begin = cellshard::first_row_in_partition(&view, partition_id);
            local_row = global_row - (std::uint64_t) partition_row_begin;
            if (local_row >= part->rows) {
                for (unsigned long fetched : fetched_partitions) (void) cellshard::drop_partition(&view, fetched);
                cellshard::clear(&storage);
                cellshard::clear(&view);
                set_error(error, "resolved local row is outside the fetched blocked-ELL partition");
                return false;
            }

            append_blocked_ell_row(part, (std::uint32_t) local_row, &out->indices, &out->data);
            out->indptr[i + 1u] = (std::int64_t) out->data.size();
        }

        for (unsigned long fetched : fetched_partitions) {
            if (!cellshard::drop_partition(&view, fetched)) {
                cellshard::clear(&storage);
                cellshard::clear(&view);
                set_error(error, "failed to release blocked-ELL partition after row subset export");
                return false;
            }
        }
        cellshard::clear(&storage);
        cellshard::clear(&view);
        return true;
    }

    if (matrix_format == "sliced_ell") {
        cellshard::sharded<cellshard::sparse::sliced_ell> view;
        cellshard::shard_storage storage;
        std::vector<unsigned long> fetched_partitions;
        cellshard::init(&view);
        cellshard::init(&storage);
        if (!cellshard::load_header(path, &view, &storage)) {
            set_error(error, "failed to load sliced-ELL dataset header");
            return false;
        }

        out->rows = (std::uint64_t) row_count;
        out->cols = view.cols;
        out->indptr.assign(row_count + 1u, 0);
        fetched_partitions.reserve(std::min<std::size_t>(row_count, (std::size_t) view.num_partitions));

        for (std::size_t i = 0; i < row_count; ++i) {
            const std::uint64_t global_row = row_indices[i];
            const cellshard::sparse::sliced_ell *part = nullptr;
            unsigned long partition_id = 0ul;
            unsigned long partition_row_begin = 0ul;
            std::uint64_t local_row = 0u;

            if (global_row >= view.rows) {
                for (unsigned long fetched : fetched_partitions) (void) cellshard::drop_partition(&view, fetched);
                cellshard::clear(&storage);
                cellshard::clear(&view);
                set_error(error, "row index is out of range");
                return false;
            }

            partition_id = cellshard::find_partition(&view, (unsigned long) global_row);
            if (partition_id >= view.num_partitions) {
                for (unsigned long fetched : fetched_partitions) (void) cellshard::drop_partition(&view, fetched);
                cellshard::clear(&storage);
                cellshard::clear(&view);
                set_error(error, "failed to resolve row index to a partition");
                return false;
            }
            if (!cellshard::partition_loaded(&view, partition_id)) {
                if (!cellshard::fetch_partition(&view, &storage, partition_id)) {
                    for (unsigned long fetched : fetched_partitions) (void) cellshard::drop_partition(&view, fetched);
                    cellshard::clear(&storage);
                    cellshard::clear(&view);
                    set_error(error, "failed to fetch sliced-ELL partition for row subset");
                    return false;
                }
                fetched_partitions.push_back(partition_id);
            }

            part = view.parts[partition_id];
            if (part == nullptr) {
                for (unsigned long fetched : fetched_partitions) (void) cellshard::drop_partition(&view, fetched);
                cellshard::clear(&storage);
                cellshard::clear(&view);
                set_error(error, "sliced-ELL partition is null after row subset fetch");
                return false;
            }

            partition_row_begin = cellshard::first_row_in_partition(&view, partition_id);
            local_row = global_row - (std::uint64_t) partition_row_begin;
            if (local_row >= part->rows) {
                for (unsigned long fetched : fetched_partitions) (void) cellshard::drop_partition(&view, fetched);
                cellshard::clear(&storage);
                cellshard::clear(&view);
                set_error(error, "resolved local row is outside the fetched sliced-ELL partition");
                return false;
            }

            append_sliced_ell_row(part, (std::uint32_t) local_row, &out->indices, &out->data);
            out->indptr[i + 1u] = (std::int64_t) out->data.size();
        }

        for (unsigned long fetched : fetched_partitions) {
            if (!cellshard::drop_partition(&view, fetched)) {
                cellshard::clear(&storage);
                cellshard::clear(&view);
                set_error(error, "failed to release sliced-ELL partition after row subset export");
                return false;
            }
        }
        cellshard::clear(&storage);
        cellshard::clear(&view);
        return true;
    }

    set_error(error, "unsupported matrix_format for row-subset CSR export: " + matrix_format);
    return false;
}

bool load_dataset_for_anndata(const char *path, anndata_export *out, std::string *error) {
    if (out == nullptr) {
        set_error(error, "anndata export output is null");
        return false;
    }
    *out = anndata_export{};
    if (!load_dataset_summary(path, &out->summary, error)) return false;
    if (!load_observation_metadata(path, &out->obs_columns, error)) return false;
    if (!load_dataset_as_csr(path, &out->x, error)) return false;
    return true;
}

bool load_dataset_global_metadata_snapshot(const char *path,
                                           global_metadata_snapshot *out,
                                           std::string *error) {
    if (out == nullptr || path == nullptr || *path == '\0') {
        set_error(error, "dataset path is empty");
        return false;
    }
    *out = global_metadata_snapshot{};
    if (!load_dataset_summary(path, &out->summary, error)) return false;
    if (!load_embedded_metadata_tables(path, &out->embedded_metadata, error)) return false;
    if (!load_observation_metadata_table(path, &out->observation_metadata_rows, &out->observation_metadata, error)) return false;

    if (out->summary.matrix_format == "blocked_ell") {
        cellshard::sharded<cellshard::sparse::blocked_ell> matrix;
        cellshard::shard_storage storage;
        cellshard::dataset_execution_view execution{};
        cellshard::dataset_runtime_service_view runtime{};
        cellshard::init(&matrix);
        cellshard::init(&storage);
        if (!cellshard::load_header(path, &matrix, &storage)
            || !cellshard::get_dataset_h5_execution_metadata(&storage, &execution)
            || !cellshard::get_dataset_h5_runtime_service(&storage, &runtime)) {
            cellshard::clear(&storage);
            cellshard::clear(&matrix);
            set_error(error, "failed to load blocked-ELL owner metadata");
            return false;
        }
        copy_execution_partition_metadata(out->summary, execution, &out->execution_partitions);
        copy_execution_shard_metadata(out->summary, execution, &out->execution_shards);
        copy_runtime_service_metadata(runtime, &out->runtime_service);
        cellshard::clear(&storage);
        cellshard::clear(&matrix);
    } else if (out->summary.matrix_format == "sliced_ell") {
        cellshard::sharded<cellshard::sparse::sliced_ell> matrix;
        cellshard::shard_storage storage;
        cellshard::dataset_execution_view execution{};
        cellshard::dataset_runtime_service_view runtime{};
        cellshard::init(&matrix);
        cellshard::init(&storage);
        if (!cellshard::load_header(path, &matrix, &storage)
            || !cellshard::get_dataset_h5_execution_metadata(&storage, &execution)
            || !cellshard::get_dataset_h5_runtime_service(&storage, &runtime)) {
            cellshard::clear(&storage);
            cellshard::clear(&matrix);
            set_error(error, "failed to load sliced-ELL owner metadata");
            return false;
        }
        copy_execution_partition_metadata(out->summary, execution, &out->execution_partitions);
        copy_execution_shard_metadata(out->summary, execution, &out->execution_shards);
        copy_runtime_service_metadata(runtime, &out->runtime_service);
        cellshard::clear(&storage);
        cellshard::clear(&matrix);
    } else {
        set_error(error, "unsupported matrix_format for owner metadata bootstrap: " + out->summary.matrix_format);
        return false;
    }

    out->snapshot_id = compute_snapshot_id(*out);
    return true;
}

client_snapshot_ref make_client_snapshot_ref(const global_metadata_snapshot &snapshot) {
    return build_client_snapshot_ref(snapshot);
}

bool validate_client_snapshot_ref(const global_metadata_snapshot &owner_snapshot,
                                  const client_snapshot_ref &request,
                                  std::string *error) {
    const client_snapshot_ref expected = build_client_snapshot_ref(owner_snapshot);
    if (request.snapshot_id != expected.snapshot_id) {
        set_error(error, "client snapshot_id does not match the owner snapshot");
        return false;
    }
    if (request.canonical_generation != expected.canonical_generation) {
        set_error(error, "client canonical_generation is stale");
        return false;
    }
    if (request.execution_plan_generation != expected.execution_plan_generation) {
        set_error(error, "client execution_plan_generation is stale");
        return false;
    }
    if (request.pack_generation != expected.pack_generation) {
        set_error(error, "client pack_generation is stale");
        return false;
    }
    if (request.service_epoch != expected.service_epoch) {
        set_error(error, "client service_epoch is stale");
        return false;
    }
    return true;
}

bool stage_append_only_runtime_service(const runtime_service_metadata &current,
                                       runtime_service_metadata *staged,
                                       std::string *error) {
    if (staged == nullptr) {
        set_error(error, "staged runtime_service output is null");
        return false;
    }
    if (current.live_write_mode != cellshard::dataset_live_write_mode_append_only) {
        set_error(error, "runtime service is not in append-only mode");
        return false;
    }

    const std::uint64_t next_generation = std::max({
        current.canonical_generation,
        current.execution_plan_generation,
        current.pack_generation,
        current.active_read_generation,
        current.staged_write_generation
    }) + 1u;

    *staged = current;
    staged->canonical_generation = next_generation;
    staged->execution_plan_generation = next_generation;
    staged->pack_generation = next_generation;
    staged->staged_write_generation = next_generation;
    return true;
}

bool publish_runtime_service_cutover(const runtime_service_metadata &current,
                                     const runtime_service_metadata &staged,
                                     runtime_service_metadata *published,
                                     std::string *error) {
    if (published == nullptr) {
        set_error(error, "published runtime_service output is null");
        return false;
    }
    if (current.live_write_mode != cellshard::dataset_live_write_mode_append_only
        || staged.live_write_mode != cellshard::dataset_live_write_mode_append_only) {
        set_error(error, "runtime service cutover requires append-only mode");
        return false;
    }
    if (staged.staged_write_generation == 0u) {
        set_error(error, "staged runtime service does not define a staged_write_generation");
        return false;
    }
    if (staged.staged_write_generation < current.active_read_generation) {
        set_error(error, "staged runtime generation is older than the active read generation");
        return false;
    }

    *published = staged;
    published->active_read_generation = staged.staged_write_generation;
    published->staged_write_generation = published->active_read_generation;
    published->service_epoch = std::max(current.service_epoch, staged.service_epoch) + 1u;
    return true;
}

bool describe_pack_delivery(const global_metadata_snapshot &owner_snapshot,
                            const pack_delivery_request &request,
                            pack_delivery_descriptor *out,
                            std::string *error) {
    if (out == nullptr) {
        set_error(error, "pack delivery descriptor output is null");
        return false;
    }
    if (!validate_client_snapshot_ref(owner_snapshot, request.request, error)) return false;
    if (request.shard_id >= owner_snapshot.summary.shards.size()) {
        set_error(error, "pack delivery shard_id is out of range");
        return false;
    }

    *out = pack_delivery_descriptor{};
    out->snapshot_id = owner_snapshot.snapshot_id;
    out->shard_id = request.shard_id;
    out->canonical_generation = owner_snapshot.runtime_service.canonical_generation;
    out->execution_plan_generation = owner_snapshot.runtime_service.execution_plan_generation;
    out->pack_generation = owner_snapshot.runtime_service.pack_generation;
    out->service_epoch = owner_snapshot.runtime_service.service_epoch;
    out->prefer_execution_pack = request.prefer_execution_pack != 0u ? 1u : 0u;

    if (request.shard_id < owner_snapshot.execution_shards.size()) {
        const execution_shard_metadata &shard = owner_snapshot.execution_shards[(std::size_t) request.shard_id];
        out->owner_node_id = shard.owner_node_id;
        out->owner_rank_id = shard.owner_rank_id;
        out->execution_format = shard.execution_format;
    }

    if (out->prefer_execution_pack != 0u) {
        out->pack_kind = "execution";
        out->relative_pack_path = "packs/execution/shard." + std::to_string(request.shard_id) + ".exec.pack";
    } else {
        out->pack_kind = "canonical";
        out->relative_pack_path = "packs/canonical/shard." + std::to_string(request.shard_id) + ".pack";
    }
    return true;
}

bool serialize_global_metadata_snapshot(const global_metadata_snapshot &snapshot,
                                        std::vector<std::uint8_t> *out,
                                        std::string *error) {
    global_metadata_snapshot normalized = snapshot;
    normalized.snapshot_id = compute_snapshot_id(snapshot);
    return serialize_global_metadata_snapshot_payload(normalized, out, true, true, error);
}

bool deserialize_global_metadata_snapshot(const void *data,
                                          std::size_t bytes,
                                          global_metadata_snapshot *out,
                                          std::string *error) {
    static const char magic[] = {'C', 'S', 'G', 'M'};
    byte_reader reader{};
    std::uint32_t version = 0u;
    std::uint64_t count = 0u;
    std::size_t size = 0u;
    std::uint64_t encoded_snapshot_id = 0u;

    if (out == nullptr || data == nullptr) {
        set_error(error, "metadata snapshot input is null");
        return false;
    }
    *out = global_metadata_snapshot{};
    reader.cur = static_cast<const std::uint8_t *>(data);
    reader.end = reader.cur + bytes;
    if ((std::size_t) (reader.end - reader.cur) < sizeof(magic) || std::memcmp(reader.cur, magic, sizeof(magic)) != 0) {
        set_error(error, "metadata snapshot magic mismatch");
        return false;
    }
    reader.cur += sizeof(magic);
    if (!read_pod(&reader, &version, error, "snapshot_version")) return false;
    if (version != 1u) {
        set_error(error, "unsupported metadata snapshot version");
        return false;
    }
    if (!read_pod(&reader, &encoded_snapshot_id, error, "snapshot_id")) return false;
    if (!read_dataset_summary(&reader, &out->summary, error, true)) return false;
    if (!read_pod(&reader, &count, error, "embedded_metadata_count") || !checked_size_from_u64(count, &size)) {
        set_error(error, "embedded_metadata_count is too large");
        return false;
    }
    out->embedded_metadata.assign(size, embedded_metadata_table{});
    for (std::size_t i = 0; i < size; ++i) {
        if (!read_embedded_metadata_table(&reader, out->embedded_metadata.data() + i, error)) return false;
    }
    if (!read_pod(&reader, &out->observation_metadata_rows, error, "observation_metadata_rows")) return false;
    if (!read_pod(&reader, &count, error, "observation_metadata_count") || !checked_size_from_u64(count, &size)) {
        set_error(error, "observation_metadata_count is too large");
        return false;
    }
    out->observation_metadata.assign(size, observation_metadata_column{});
    for (std::size_t i = 0; i < size; ++i) {
        if (!read_observation_metadata_column(&reader, out->observation_metadata.data() + i, error)) return false;
    }
    if (!read_pod(&reader, &count, error, "execution_partition_count") || !checked_size_from_u64(count, &size)) {
        set_error(error, "execution_partition_count is too large");
        return false;
    }
    out->execution_partitions.assign(size, execution_partition_metadata{});
    for (std::size_t i = 0; i < size; ++i) {
        if (!read_execution_partition_metadata(&reader, out->execution_partitions.data() + i, error)) return false;
    }
    if (!read_pod(&reader, &count, error, "execution_shard_count") || !checked_size_from_u64(count, &size)) {
        set_error(error, "execution_shard_count is too large");
        return false;
    }
    out->execution_shards.assign(size, execution_shard_metadata{});
    for (std::size_t i = 0; i < size; ++i) {
        if (!read_execution_shard_metadata(&reader, out->execution_shards.data() + i, error)) return false;
    }
    if (!read_runtime_service_metadata(&reader, &out->runtime_service, error)) return false;
    if (reader.cur != reader.end) {
        set_error(error, "metadata snapshot contains trailing bytes");
        return false;
    }

    out->snapshot_id = compute_snapshot_id(*out);
    if (out->snapshot_id != encoded_snapshot_id) {
        set_error(error, "metadata snapshot_id does not match the payload contents");
        return false;
    }
    return true;
}

} // namespace cellshard::exporting
