#include "../internal/common.hh"

#include <cstdio>
#include <utility>

namespace cellshard::exporting {

namespace {

using namespace detail;

bool load_annotation_metadata_impl(const char *path,
                                   const char *group_path,
                                   std::vector<annotation_column> *columns,
                                   std::string *error) {
    hid_t file = (hid_t) -1;
    hid_t metadata = (hid_t) -1;
    std::uint32_t cols = 0u;
    std::uint64_t extent = 0u;

    if (columns == nullptr || path == nullptr || *path == '\0' || group_path == nullptr) {
        set_error(error, "invalid annotation metadata request");
        return false;
    }
    columns->clear();

    file = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0) {
        set_error(error, "failed to open dataset file for annotation metadata");
        return false;
    }

    metadata = open_optional_group(file, group_path);
    if (metadata < 0) {
        H5Fclose(file);
        return true;
    }

    if ((!read_attr_u64(metadata, "extent", &extent) && !read_attr_u64(metadata, "rows", &extent))
        || !read_attr_u32(metadata, "cols", &cols)) {
        set_error(error, "failed to read annotation metadata header");
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
            set_error(error, "failed to format annotation metadata column name");
            H5Gclose(metadata);
            H5Fclose(file);
            return false;
        }
        column = H5Gopen2(metadata, name, H5P_DEFAULT);
        if (column < 0) {
            set_error(error, "failed to open annotation metadata column");
            H5Gclose(metadata);
            H5Fclose(file);
            return false;
        }
        if (!read_attr_string(column, "name", &out.name) || !read_attr_u32(column, "type", &out.type)) {
            H5Gclose(column);
            set_error(error, "failed to read annotation metadata column header");
            H5Gclose(metadata);
            H5Fclose(file);
            return false;
        }
        if (out.type == cellshard::dataset_observation_metadata_type_text) {
            if (!read_text_column_strings(column, "values", &out.text_values)) {
                H5Gclose(column);
                set_error(error, "failed to read annotation metadata text values");
                H5Gclose(metadata);
                H5Fclose(file);
                return false;
            }
        } else if (out.type == cellshard::dataset_observation_metadata_type_float32) {
            if (!read_dataset_vector(column, "values", H5T_NATIVE_FLOAT, &out.float32_values)) {
                H5Gclose(column);
                set_error(error, "failed to read annotation metadata float values");
                H5Gclose(metadata);
                H5Fclose(file);
                return false;
            }
        } else if (out.type == cellshard::dataset_observation_metadata_type_uint8) {
            if (!read_dataset_vector(column, "values", H5T_NATIVE_UINT8, &out.uint8_values)) {
                H5Gclose(column);
                set_error(error, "failed to read annotation metadata uint8 values");
                H5Gclose(metadata);
                H5Fclose(file);
                return false;
            }
        } else {
            H5Gclose(column);
            set_error(error, "annotation metadata contains an unknown column type");
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

void load_annotation_summary(hid_t file, const char *group_path, annotation_summary *summary) {
    hid_t group = open_optional_group(file, group_path);
    if (group < 0 || summary == nullptr) return;

    std::uint32_t cols = 0u;
    std::uint64_t extent = 0u;
    if ((!read_attr_u64(group, "extent", &extent) && !read_attr_u64(group, "rows", &extent))
        || !read_attr_u32(group, "cols", &cols)) {
        H5Gclose(group);
        return;
    }

    summary->available = true;
    summary->extent = extent;
    summary->names.reserve(cols);
    summary->types.reserve(cols);
    for (std::uint32_t i = 0; i < cols; ++i) {
        char name[64];
        hid_t column = (hid_t) -1;
        std::string column_name;
        std::uint32_t type = 0u;
        if (std::snprintf(name, sizeof(name), "column_%u", i) <= 0) continue;
        column = H5Gopen2(group, name, H5P_DEFAULT);
        if (column < 0) continue;
        if (read_attr_string(column, "name", &column_name) && read_attr_u32(column, "type", &type)) {
            summary->names.push_back(std::move(column_name));
            summary->types.push_back(type);
        }
        H5Gclose(column);
    }

    H5Gclose(group);
}

bool load_dataset_attributes_impl(const char *path,
                                  std::vector<dataset_attribute> *attributes,
                                  std::string *error) {
    hid_t file = (hid_t) -1;
    hid_t group = (hid_t) -1;
    std::vector<std::string> keys;
    std::vector<std::string> values;

    if (attributes == nullptr || path == nullptr || *path == '\0') {
        set_error(error, "invalid dataset attribute request");
        return false;
    }
    attributes->clear();
    file = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0) {
        set_error(error, "failed to open dataset file for dataset attributes");
        return false;
    }
    group = open_optional_group(file, "/dataset_attributes");
    if (group < 0) {
        H5Fclose(file);
        return true;
    }
    H5Gclose(group);
    if (!read_text_column_strings(file, "/dataset_attributes/keys", &keys)
        || !read_text_column_strings(file, "/dataset_attributes/values", &values)) {
        H5Fclose(file);
        set_error(error, "failed to read dataset attributes");
        return false;
    }
    H5Fclose(file);
    if (keys.size() != values.size()) {
        set_error(error, "dataset attribute keys and values are not aligned");
        return false;
    }
    attributes->reserve(keys.size());
    for (std::size_t i = 0; i < keys.size(); ++i) {
        attributes->push_back(dataset_attribute{keys[i], values[i]});
    }
    return true;
}

} // namespace

bool load_observation_metadata(const char *path,
                               std::vector<observation_metadata_column> *columns,
                               std::string *error) {
    return load_annotation_metadata_impl(path, "/observation_metadata", columns, error);
}

bool load_feature_metadata(const char *path,
                           std::vector<annotation_column> *columns,
                           std::string *error) {
    return load_annotation_metadata_impl(path, "/feature_metadata", columns, error);
}

bool load_dataset_attributes(const char *path,
                             std::vector<dataset_attribute> *attributes,
                             std::string *error) {
    return load_dataset_attributes_impl(path, attributes, error);
}

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

    load_annotation_summary(file, "/observation_metadata", &out->observation_annotations);
    load_annotation_summary(file, "/feature_metadata", &out->feature_annotations);

    {
        hid_t group = open_optional_group(file, "/dataset_attributes");
        if (group >= 0) {
            out->dataset_attributes.available =
                read_text_column_strings(file, "/dataset_attributes/keys", &out->dataset_attributes.keys);
            H5Gclose(group);
        }
    }

    if (codecs >= 0) H5Gclose(codecs);
    if (provenance >= 0) H5Gclose(provenance);
    H5Gclose(matrix);
    H5Gclose(datasets);
    H5Fclose(file);
    return true;
}

} // namespace cellshard::exporting
