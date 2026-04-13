#include "series_export.hh"

#include "../src/real.cuh"
#include "../src/formats/compressed.cuh"
#include "../src/formats/blocked_ell.cuh"
#include "../src/sharded/sharded.cuh"
#include "../src/sharded/sharded_host.cuh"
#include "../src/sharded/shard_paths.cuh"
#include "../src/sharded/series_h5.cuh"
#include "../src/sharded/disk.cuh"

#include <hdf5.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
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

inline unsigned long find_partition_index_for_row(const std::vector<std::uint64_t> &part_row_offsets,
                                                  std::uint64_t row_begin) {
    if (part_row_offsets.size() < 2u) return 0ul;
    const auto it = std::upper_bound(part_row_offsets.begin(), part_row_offsets.end(), row_begin);
    if (it == part_row_offsets.begin()) return 0ul;
    return (unsigned long) std::distance(part_row_offsets.begin(), it - 1);
}

inline unsigned long find_partition_end_for_row(const std::vector<std::uint64_t> &part_row_offsets,
                                                std::uint64_t row_end) {
    if (part_row_offsets.size() < 2u || row_end == 0u) return 0ul;
    const auto it = std::lower_bound(part_row_offsets.begin(), part_row_offsets.end(), row_end);
    return (unsigned long) std::distance(part_row_offsets.begin(), it);
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
        set_error(error, "failed to open series file for observation metadata");
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
        if (out.type == cellshard::series_observation_metadata_type_text) {
            if (!read_text_column_strings(column, "values", &out.text_values)) {
                H5Gclose(column);
                set_error(error, "failed to read observation metadata text values");
                H5Gclose(metadata);
                H5Fclose(file);
                return false;
            }
        } else if (out.type == cellshard::series_observation_metadata_type_float32) {
            if (!read_dataset_vector(column, "values", H5T_NATIVE_FLOAT, &out.float32_values)) {
                H5Gclose(column);
                set_error(error, "failed to read observation metadata float values");
                H5Gclose(metadata);
                H5Fclose(file);
                return false;
            }
        } else if (out.type == cellshard::series_observation_metadata_type_uint8) {
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

inline bool load_matrix_format(const char *path, std::string *matrix_format, std::string *error) {
    hid_t file = (hid_t) -1;
    bool ok = false;
    if (matrix_format == nullptr || path == nullptr || *path == '\0') {
        set_error(error, "invalid series path");
        return false;
    }
    file = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0) {
        set_error(error, "failed to open series file");
        return false;
    }
    ok = read_attr_string(file, "matrix_format", matrix_format);
    H5Fclose(file);
    if (!ok) set_error(error, "failed to read matrix_format");
    return ok;
}

} // namespace

bool load_series_summary(const char *path, series_summary *out, std::string *error) {
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
    std::vector<std::uint64_t> part_rows;
    std::vector<std::uint64_t> part_nnz;
    std::vector<std::uint32_t> part_axes;
    std::vector<std::uint64_t> part_aux;
    std::vector<std::uint64_t> part_row_offsets;
    std::vector<std::uint32_t> part_dataset_ids;
    std::vector<std::uint32_t> part_codec_ids;
    std::vector<std::uint64_t> shard_offsets;
    std::vector<std::uint32_t> codec_ids;
    std::vector<std::uint32_t> codec_families;
    std::vector<std::uint32_t> codec_value_codes;
    std::vector<std::uint32_t> codec_scale_codes;
    std::vector<std::uint32_t> codec_bits;
    std::vector<std::uint32_t> codec_flags;

    if (out == nullptr || path == nullptr || *path == '\0') {
        set_error(error, "series path is empty");
        return false;
    }

    *out = series_summary{};
    out->path = path;

    file = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0) {
        set_error(error, "failed to open series file");
        return false;
    }

    if (!read_attr_string(file, "matrix_format", &out->matrix_format)
        || !read_attr_u64(file, "rows", &out->rows)
        || !read_attr_u64(file, "cols", &out->cols)
        || !read_attr_u64(file, "nnz", &out->nnz)
        || !read_attr_u64(file, "num_parts", &out->num_partitions)
        || !read_attr_u64(file, "num_shards", &out->num_shards)
        || !read_attr_u64(file, "num_datasets", &out->num_datasets)) {
        H5Fclose(file);
        set_error(error, "failed to read top-level series attributes");
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
        set_error(error, "series file is missing required groups");
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

    if (!read_dataset_vector(matrix, "part_rows", H5T_NATIVE_UINT64, &part_rows)
        || !read_dataset_vector(matrix, "part_nnz", H5T_NATIVE_UINT64, &part_nnz)
        || !read_optional_dataset_vector(matrix, "part_axes", H5T_NATIVE_UINT32, &part_axes)
        || !read_dataset_vector(matrix, "part_aux", H5T_NATIVE_UINT64, &part_aux)
        || !read_dataset_vector(matrix, "part_row_offsets", H5T_NATIVE_UINT64, &part_row_offsets)
        || !read_dataset_vector(matrix, "part_dataset_ids", H5T_NATIVE_UINT32, &part_dataset_ids)
        || !read_dataset_vector(matrix, "part_codec_ids", H5T_NATIVE_UINT32, &part_codec_ids)
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
        out->datasets.push_back(series_dataset_summary{
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

    out->partitions.reserve(part_rows.size());
    for (std::size_t i = 0; i < part_rows.size(); ++i) {
        out->partitions.push_back(series_partition_summary{
            (std::uint64_t) i,
            i < part_row_offsets.size() ? part_row_offsets[i] : 0u,
            i + 1u < part_row_offsets.size() ? part_row_offsets[i + 1u] : 0u,
            i < part_rows.size() ? part_rows[i] : 0u,
            i < part_nnz.size() ? part_nnz[i] : 0u,
            i < part_aux.size() ? part_aux[i] : 0u,
            i < part_dataset_ids.size() ? part_dataset_ids[i] : 0u,
            i < part_axes.size() ? part_axes[i] : 0u,
            i < part_codec_ids.size() ? part_codec_ids[i] : 0u
        });
    }

    out->shards.reserve(shard_offsets.size() > 0u ? shard_offsets.size() - 1u : 0u);
    for (std::size_t i = 0; i + 1u < shard_offsets.size(); ++i) {
        const std::uint64_t row_begin = shard_offsets[i];
        const std::uint64_t row_end = shard_offsets[i + 1u];
        out->shards.push_back(series_shard_summary{
            (std::uint64_t) i,
            (std::uint64_t) find_partition_index_for_row(part_row_offsets, row_begin),
            (std::uint64_t) (row_end == row_begin
                ? find_partition_index_for_row(part_row_offsets, row_begin)
                : std::max<unsigned long>(
                    find_partition_index_for_row(part_row_offsets, row_begin),
                    find_partition_end_for_row(part_row_offsets, row_end))),
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
                out->codecs.push_back(series_codec_summary{
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

bool load_series_as_csr(const char *path, csr_matrix_export *out, std::string *error) {
    std::string matrix_format;

    if (out == nullptr || path == nullptr || *path == '\0') {
        set_error(error, "series path is empty");
        return false;
    }
    *out = csr_matrix_export{};
    warn_cpu_materialize_once("CSR materialization");

    if (!load_matrix_format(path, &matrix_format, error)) return false;

    if (matrix_format == "compressed") {
        cellshard::sharded<cellshard::sparse::compressed> view;
        cellshard::shard_storage storage;
        unsigned long global_row = 0ul;
        cellshard::init(&view);
        cellshard::init(&storage);
        if (!cellshard::load_header(path, &view, &storage)) {
            set_error(error, "failed to load compressed series header");
            return false;
        }

        out->rows = view.rows;
        out->cols = view.cols;
        out->indptr.assign((std::size_t) view.rows + 1u, 0);
        out->indices.reserve(view.nnz);
        out->data.reserve(view.nnz);

        for (unsigned long part_id = 0; part_id < view.num_parts; ++part_id) {
            const cellshard::sparse::compressed *part = nullptr;
            if (!cellshard::fetch_part(&view, &storage, part_id)) {
                cellshard::clear(&storage);
                cellshard::clear(&view);
                set_error(error, "failed to materialize compressed series part");
                return false;
            }
            part = view.parts[part_id];
            if (part == nullptr || part->axis != cellshard::sparse::compressed_by_row) {
                cellshard::clear(&storage);
                cellshard::clear(&view);
                set_error(error, "compressed series export requires row-compressed parts");
                return false;
            }
            for (std::uint32_t row = 0u; row < part->rows; ++row) {
                for (cellshard::types::ptr_t i = part->majorPtr[row]; i < part->majorPtr[row + 1u]; ++i) {
                    out->indices.push_back((std::int64_t) part->minorIdx[i]);
                    out->data.push_back(__half2float(part->val[i]));
                }
                ++global_row;
                out->indptr[global_row] = (std::int64_t) out->data.size();
            }
            if (!cellshard::drop_part(&view, part_id)) {
                cellshard::clear(&storage);
                cellshard::clear(&view);
                set_error(error, "failed to release compressed series part");
                return false;
            }
        }

        cellshard::clear(&storage);
        cellshard::clear(&view);
        return true;
    }

    if (matrix_format == "blocked_ell") {
        cellshard::sharded<cellshard::sparse::blocked_ell> view;
        cellshard::shard_storage storage;
        unsigned long global_row = 0ul;
        cellshard::init(&view);
        cellshard::init(&storage);
        if (!cellshard::load_header(path, &view, &storage)) {
            set_error(error, "failed to load blocked-ELL series header");
            return false;
        }

        out->rows = view.rows;
        out->cols = view.cols;
        out->indptr.assign((std::size_t) view.rows + 1u, 0);
        out->indices.reserve(view.nnz);
        out->data.reserve(view.nnz);

        for (unsigned long part_id = 0; part_id < view.num_parts; ++part_id) {
            const cellshard::sparse::blocked_ell *part = nullptr;
            if (!cellshard::fetch_part(&view, &storage, part_id)) {
                cellshard::clear(&storage);
                cellshard::clear(&view);
                set_error(error, "failed to materialize blocked-ELL series part");
                return false;
            }
            part = view.parts[part_id];
            if (part == nullptr) {
                cellshard::clear(&storage);
                cellshard::clear(&view);
                set_error(error, "blocked-ELL series part is null after fetch");
                return false;
            }
            for (std::uint32_t row = 0u; row < part->rows; ++row) {
                append_blocked_ell_row(part, row, &out->indices, &out->data);
                ++global_row;
                out->indptr[global_row] = (std::int64_t) out->data.size();
            }
            if (!cellshard::drop_part(&view, part_id)) {
                cellshard::clear(&storage);
                cellshard::clear(&view);
                set_error(error, "failed to release blocked-ELL series part");
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

bool load_series_for_anndata(const char *path, anndata_export *out, std::string *error) {
    if (out == nullptr) {
        set_error(error, "anndata export output is null");
        return false;
    }
    *out = anndata_export{};
    if (!load_series_summary(path, &out->summary, error)) return false;
    if (!load_observation_metadata(path, &out->obs_columns, error)) return false;
    if (!load_series_as_csr(path, &out->x, error)) return false;
    return true;
}

} // namespace cellshard::exporting
