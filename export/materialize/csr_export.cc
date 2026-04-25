#include "../internal/common.hh"

#include "../../include/CellShard/io/pack/packfile.cuh"
#include "../../src/convert/blocked_ell_from_compressed.cuh"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <filesystem>
#include <limits>
#include <memory>
#include <unordered_set>

namespace cellshard::exporting {

namespace {

using namespace detail;
namespace fs = std::filesystem;

struct owned_text_column {
    std::vector<std::uint32_t> offsets;
    std::vector<char> data;

    dataset_text_column_view view() const {
        dataset_text_column_view out{};
        out.count = offsets.empty() ? 0u : (std::uint32_t) offsets.size() - 1u;
        out.bytes = (std::uint32_t) data.size();
        out.offsets = offsets.empty() ? nullptr : offsets.data();
        out.data = data.empty() ? nullptr : data.data();
        return out;
    }
};

struct owned_annotation_column {
    std::string name;
    std::uint32_t type = 0u;
    owned_text_column text_values;
    std::vector<float> float32_values;
    std::vector<std::uint8_t> uint8_values;

    dataset_observation_metadata_column_view view() const {
        dataset_observation_metadata_column_view out{};
        out.name = name.c_str();
        out.type = type;
        out.text_values = text_values.view();
        out.float32_values = float32_values.empty() ? nullptr : float32_values.data();
        out.uint8_values = uint8_values.empty() ? nullptr : uint8_values.data();
        return out;
    }
};

inline owned_text_column make_text_column(const std::vector<std::string> &values) {
    owned_text_column out;
    std::uint32_t cursor = 0u;
    out.offsets.resize(values.size() + 1u, 0u);
    for (std::size_t i = 0; i < values.size(); ++i) {
        out.offsets[i] = cursor;
        out.data.insert(out.data.end(), values[i].begin(), values[i].end());
        out.data.push_back('\0');
        cursor += (std::uint32_t) values[i].size() + 1u;
    }
    out.offsets[values.size()] = cursor;
    return out;
}

inline std::string sanitize_label(std::string value, const char *fallback) {
    for (char &c : value) {
        if (std::isalnum((unsigned char) c) || c == '_' || c == '-' || c == '.') continue;
        c = '_';
    }
    while (!value.empty() && value.front() == '_') value.erase(value.begin());
    while (!value.empty() && value.back() == '_') value.pop_back();
    if (!value.empty()) return value;
    return fallback != nullptr ? std::string(fallback) : std::string("derived");
}

inline bool validate_unique_indices(const std::vector<std::uint64_t> &indices,
                                    std::uint64_t upper_bound,
                                    const char *label,
                                    std::string *error) {
    std::unordered_set<std::uint64_t> seen;
    if (indices.empty()) {
        set_error(error, std::string(label != nullptr ? label : "selection") + " is empty");
        return false;
    }
    seen.reserve(indices.size());
    for (std::uint64_t value : indices) {
        if (value >= upper_bound) {
            set_error(error, std::string(label != nullptr ? label : "selection") + " index is out of range");
            return false;
        }
        if (!seen.insert(value).second) {
            set_error(error, std::string(label != nullptr ? label : "selection") + " contains duplicate indices");
            return false;
        }
    }
    return true;
}

inline bool validate_group_spans(const std::vector<derivation_group_span> &groups,
                                 std::uint64_t extent,
                                 const char *label,
                                 std::string *error) {
    std::uint64_t previous_end = 0u;
    for (const derivation_group_span &group : groups) {
        if (group.end < group.begin || group.end > extent) {
            set_error(error, std::string(label != nullptr ? label : "group") + " span is out of range");
            return false;
        }
        if (group.begin < previous_end) {
            set_error(error, std::string(label != nullptr ? label : "group") + " spans overlap");
            return false;
        }
        previous_end = group.end;
    }
    return true;
}

inline std::string default_derived_dataset_path(const char *source_path,
                                                const derived_materialization_request &request) {
    const std::string pack_name = sanitize_label(request.derived_pack_name, "derived");
    if (!request.output_path.empty()) return request.output_path;
    if (!request.cache_root.empty()) {
        fs::path path = fs::path(request.cache_root) / "derived" / (pack_name + ".csh5");
        return path.string();
    }
    return std::string(source_path != nullptr ? source_path : "derived.csh5") + "." + pack_name + ".derived.csh5";
}

inline std::string default_execution_cache_root(const std::string &dataset_path,
                                                const derived_materialization_request &request) {
    const std::string pack_name = sanitize_label(request.derived_pack_name, "derived");
    if (!request.cache_root.empty()) {
        fs::path path = fs::path(request.cache_root) / "derived" / (pack_name + ".cache");
        return path.string();
    }
    return dataset_path + ".cache";
}

inline bool load_runtime_service_metadata(const char *path,
                                          runtime_service_metadata *runtime,
                                          std::string *error) {
    global_metadata_snapshot snapshot;
    if (runtime == nullptr) {
        set_error(error, "runtime metadata output is null");
        return false;
    }
    *runtime = runtime_service_metadata{};
    if (!load_dataset_global_metadata_snapshot(path, &snapshot, error)) return false;
    *runtime = snapshot.runtime_service;
    return true;
}

struct source_provenance_vectors {
    std::vector<std::uint32_t> cell_dataset_ids;
    std::vector<std::uint64_t> cell_local_indices;
    std::vector<std::uint32_t> feature_dataset_ids;
    std::vector<std::uint64_t> feature_local_indices;
};

inline bool load_source_provenance_vectors(const char *path,
                                           std::uint64_t rows,
                                           std::uint64_t cols,
                                           source_provenance_vectors *out,
                                           std::string *error) {
    hid_t file = (hid_t) -1;
    hid_t provenance = (hid_t) -1;

    if (out == nullptr || path == nullptr || *path == '\0') {
        set_error(error, "invalid provenance request");
        return false;
    }

    out->cell_dataset_ids.assign((std::size_t) rows, 0u);
    out->cell_local_indices.resize((std::size_t) rows);
    out->feature_dataset_ids.assign((std::size_t) cols, 0u);
    out->feature_local_indices.resize((std::size_t) cols);
    for (std::uint64_t i = 0u; i < rows; ++i) out->cell_local_indices[(std::size_t) i] = i;
    for (std::uint64_t i = 0u; i < cols; ++i) out->feature_local_indices[(std::size_t) i] = i;

    file = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0) return true;
    provenance = open_optional_group(file, "/provenance");
    if (provenance < 0) {
        H5Fclose(file);
        return true;
    }
    (void) read_optional_dataset_vector(provenance, "cell_dataset_ids", H5T_NATIVE_UINT32, &out->cell_dataset_ids);
    (void) read_optional_dataset_vector(provenance, "cell_local_indices", H5T_NATIVE_UINT64, &out->cell_local_indices);
    (void) read_optional_dataset_vector(provenance, "feature_dataset_ids", H5T_NATIVE_UINT32, &out->feature_dataset_ids);
    (void) read_optional_dataset_vector(provenance, "feature_local_indices", H5T_NATIVE_UINT64, &out->feature_local_indices);
    H5Gclose(provenance);
    H5Fclose(file);
    if (out->cell_dataset_ids.size() != (std::size_t) rows) out->cell_dataset_ids.assign((std::size_t) rows, 0u);
    if (out->cell_local_indices.size() != (std::size_t) rows) {
        out->cell_local_indices.resize((std::size_t) rows);
        for (std::uint64_t i = 0u; i < rows; ++i) out->cell_local_indices[(std::size_t) i] = i;
    }
    if (out->feature_dataset_ids.size() != (std::size_t) cols) out->feature_dataset_ids.assign((std::size_t) cols, 0u);
    if (out->feature_local_indices.size() != (std::size_t) cols) {
        out->feature_local_indices.resize((std::size_t) cols);
        for (std::uint64_t i = 0u; i < cols; ++i) out->feature_local_indices[(std::size_t) i] = i;
    }
    return true;
}

inline void ensure_default_names(std::vector<std::string> *values,
                                 std::uint64_t count,
                                 const char *prefix) {
    if (values == nullptr) return;
    if (values->size() < (std::size_t) count) values->resize((std::size_t) count);
    for (std::uint64_t i = 0u; i < count; ++i) {
        if (!(*values)[(std::size_t) i].empty()) continue;
        (*values)[(std::size_t) i] = std::string(prefix != nullptr ? prefix : "item") + std::to_string(i);
    }
}

inline bool remap_feature_subset(const csr_matrix_export &src,
                                 const std::vector<std::uint64_t> &feature_indices,
                                 csr_matrix_export *out,
                                 std::string *error) {
    std::vector<std::int64_t> remap;
    std::vector<std::pair<std::int64_t, float>> row_entries;

    if (out == nullptr) {
        set_error(error, "feature-remap output is null");
        return false;
    }

    remap.assign((std::size_t) src.cols, -1);
    for (std::size_t i = 0; i < feature_indices.size(); ++i) {
        remap[(std::size_t) feature_indices[i]] = (std::int64_t) i;
    }

    out->rows = src.rows;
    out->cols = (std::uint64_t) feature_indices.size();
    out->indptr.assign((std::size_t) src.rows + 1u, 0);
    out->indices.clear();
    out->data.clear();
    row_entries.reserve(64u);

    for (std::uint64_t row = 0u; row < src.rows; ++row) {
        const std::size_t begin = (std::size_t) src.indptr[(std::size_t) row];
        const std::size_t end = (std::size_t) src.indptr[(std::size_t) row + 1u];
        row_entries.clear();
        for (std::size_t i = begin; i < end; ++i) {
            const std::int64_t source_col = src.indices[i];
            if (source_col < 0 || (std::uint64_t) source_col >= src.cols) {
                set_error(error, "CSR export contained an out-of-range column index");
                return false;
            }
            const std::int64_t mapped = remap[(std::size_t) source_col];
            if (mapped < 0) continue;
            row_entries.emplace_back(mapped, src.data[i]);
        }
        std::sort(row_entries.begin(), row_entries.end(),
                  [](const auto &lhs, const auto &rhs) { return lhs.first < rhs.first; });
        for (const auto &entry : row_entries) {
            out->indices.push_back(entry.first);
            out->data.push_back(entry.second);
        }
        out->indptr[(std::size_t) row + 1u] = (std::int64_t) out->data.size();
    }

    return true;
}

inline bool build_compressed_from_csr(const csr_matrix_export &csr,
                                      sparse::compressed *out,
                                      std::string *error) {
    if (out == nullptr) {
        set_error(error, "compressed output is null");
        return false;
    }
    sparse::clear(out);
    sparse::init(out,
                 (types::dim_t) csr.rows,
                 (types::dim_t) csr.cols,
                 (types::nnz_t) csr.data.size(),
                 sparse::compressed_by_row);
    if (!sparse::allocate(out)) {
        set_error(error, "failed to allocate compressed subset matrix");
        return false;
    }
    for (std::size_t i = 0; i < csr.indptr.size(); ++i) {
        out->majorPtr[i] = (types::ptr_t) csr.indptr[i];
    }
    for (std::size_t i = 0; i < csr.indices.size(); ++i) {
        out->minorIdx[i] = (types::idx_t) csr.indices[i];
        out->val[i] = __float2half(csr.data[i]);
    }
    return true;
}

inline bool build_group_text_values(std::uint64_t extent,
                                    const std::vector<derivation_group_span> &groups,
                                    std::vector<std::string> *out,
                                    std::string *error) {
    if (out == nullptr) {
        set_error(error, "group text output is null");
        return false;
    }
    out->assign((std::size_t) extent, std::string());
    if (!validate_group_spans(groups, extent, "derived group", error)) return false;
    for (const derivation_group_span &group : groups) {
        for (std::uint64_t i = group.begin; i < group.end; ++i) {
            (*out)[(std::size_t) i] = group.name;
        }
    }
    return true;
}

inline bool build_selected_observation_columns(const std::vector<observation_metadata_column> &source_columns,
                                               const std::vector<std::uint64_t> &row_indices,
                                               const std::vector<derivation_group_span> &row_groups,
                                               std::vector<owned_annotation_column> *out,
                                               std::vector<dataset_observation_metadata_column_view> *views,
                                               std::string *error) {
    std::vector<std::string> group_values;
    if (out == nullptr || views == nullptr) {
        set_error(error, "observation metadata output is null");
        return false;
    }
    out->clear();
    views->clear();
    out->reserve(source_columns.size() + (row_groups.empty() ? 0u : 1u));
    for (const observation_metadata_column &column : source_columns) {
        owned_annotation_column selected;
        selected.name = column.name;
        selected.type = column.type;
        if (column.type == dataset_observation_metadata_type_text) {
            std::vector<std::string> values;
            values.reserve(row_indices.size());
            for (std::uint64_t row : row_indices) values.push_back(column.text_values[(std::size_t) row]);
            selected.text_values = make_text_column(values);
        } else if (column.type == dataset_observation_metadata_type_float32) {
            selected.float32_values.reserve(row_indices.size());
            for (std::uint64_t row : row_indices) selected.float32_values.push_back(column.float32_values[(std::size_t) row]);
        } else if (column.type == dataset_observation_metadata_type_uint8) {
            selected.uint8_values.reserve(row_indices.size());
            for (std::uint64_t row : row_indices) selected.uint8_values.push_back(column.uint8_values[(std::size_t) row]);
        }
        out->push_back(std::move(selected));
    }
    if (!row_groups.empty()) {
        owned_annotation_column grouped;
        grouped.name = "derived.row_group";
        grouped.type = dataset_observation_metadata_type_text;
        if (!build_group_text_values((std::uint64_t) row_indices.size(), row_groups, &group_values, error)) return false;
        grouped.text_values = make_text_column(group_values);
        out->push_back(std::move(grouped));
    }
    views->reserve(out->size());
    for (const owned_annotation_column &column : *out) views->push_back(column.view());
    return true;
}

inline bool build_selected_feature_columns(const std::vector<annotation_column> &source_columns,
                                          const std::vector<std::uint64_t> &feature_indices,
                                          const std::vector<derivation_group_span> &feature_groups,
                                          std::vector<owned_annotation_column> *out,
                                          std::vector<dataset_observation_metadata_column_view> *views,
                                          std::string *error) {
    std::vector<std::string> group_values;
    if (out == nullptr || views == nullptr) {
        set_error(error, "feature metadata output is null");
        return false;
    }
    out->clear();
    views->clear();
    out->reserve(source_columns.size() + (feature_groups.empty() ? 0u : 1u));
    for (const annotation_column &column : source_columns) {
        owned_annotation_column selected;
        selected.name = column.name;
        selected.type = column.type;
        if (column.type == dataset_observation_metadata_type_text) {
            std::vector<std::string> values;
            values.reserve(feature_indices.size());
            for (std::uint64_t feature : feature_indices) values.push_back(column.text_values[(std::size_t) feature]);
            selected.text_values = make_text_column(values);
        } else if (column.type == dataset_observation_metadata_type_float32) {
            selected.float32_values.reserve(feature_indices.size());
            for (std::uint64_t feature : feature_indices) selected.float32_values.push_back(column.float32_values[(std::size_t) feature]);
        } else if (column.type == dataset_observation_metadata_type_uint8) {
            selected.uint8_values.reserve(feature_indices.size());
            for (std::uint64_t feature : feature_indices) selected.uint8_values.push_back(column.uint8_values[(std::size_t) feature]);
        }
        out->push_back(std::move(selected));
    }
    if (!feature_groups.empty()) {
        owned_annotation_column grouped;
        grouped.name = "derived.feature_group";
        grouped.type = dataset_observation_metadata_type_text;
        if (!build_group_text_values((std::uint64_t) feature_indices.size(), feature_groups, &group_values, error)) return false;
        grouped.text_values = make_text_column(group_values);
        out->push_back(std::move(grouped));
    }
    views->reserve(out->size());
    for (const owned_annotation_column &column : *out) views->push_back(column.view());
    return true;
}

inline bool build_identity_bucketed_shard(const sparse::blocked_ell &part,
                                          std::uint32_t bucket_count,
                                          bucketed_blocked_ell_shard *out,
                                          std::string *error) {
    bucketed_blocked_ell_partition bucketed_part;
    init(&bucketed_part);
    if (out == nullptr) {
        set_error(error, "bucketed shard output is null");
        return false;
    }
    init(out);
    if (!build_bucketed_blocked_ell_partition(&bucketed_part, &part, bucket_count, nullptr)) {
        set_error(error, "failed to build bucketed blocked-ELL partition");
        return false;
    }
    out->partition_count = 1u;
    out->rows = part.rows;
    out->cols = part.cols;
    out->nnz = part.nnz;
    out->partitions = (bucketed_blocked_ell_partition *) std::calloc(1u, sizeof(bucketed_blocked_ell_partition));
    out->partition_row_offsets = (std::uint32_t *) std::calloc(2u, sizeof(std::uint32_t));
    out->exec_to_canonical_cols = (std::uint32_t *) std::calloc((std::size_t) part.cols, sizeof(std::uint32_t));
    out->canonical_to_exec_cols = (std::uint32_t *) std::calloc((std::size_t) part.cols, sizeof(std::uint32_t));
    bucketed_part.exec_to_canonical_cols = (std::uint32_t *) std::calloc((std::size_t) part.cols, sizeof(std::uint32_t));
    bucketed_part.canonical_to_exec_cols = (std::uint32_t *) std::calloc((std::size_t) part.cols, sizeof(std::uint32_t));
    if (out->partitions == nullptr
        || out->partition_row_offsets == nullptr
        || out->exec_to_canonical_cols == nullptr
        || out->canonical_to_exec_cols == nullptr
        || bucketed_part.exec_to_canonical_cols == nullptr
        || bucketed_part.canonical_to_exec_cols == nullptr) {
        clear(&bucketed_part);
        clear(out);
        set_error(error, "failed to allocate derived bucketed blocked-ELL shard metadata");
        return false;
    }
    out->partitions[0] = bucketed_part;
    std::memset(&bucketed_part, 0, sizeof(bucketed_part));
    out->partition_row_offsets[0] = 0u;
    out->partition_row_offsets[1] = part.rows;
    for (std::uint32_t col = 0u; col < part.cols; ++col) {
        out->partitions[0].exec_to_canonical_cols[col] = col;
        out->partitions[0].canonical_to_exec_cols[col] = col;
        out->exec_to_canonical_cols[col] = col;
        out->canonical_to_exec_cols[col] = col;
    }
    return true;
}

template<typename Part>
using append_row_fn = void (*)(const Part *, std::uint32_t, std::vector<std::int64_t> *, std::vector<float> *);

template<typename Part>
void release_fetched_partitions(cellshard::sharded<Part> *view, const std::vector<unsigned long> &partition_ids) {
    if (view == nullptr) return;
    for (unsigned long partition_id : partition_ids) (void) cellshard::drop_partition(view, partition_id);
}

template<typename Part, append_row_fn<Part> append_row>
bool load_dataset_as_csr_impl(const char *path,
                              const char *format_label,
                              csr_matrix_export *out,
                              std::string *error) {
    cellshard::sharded<Part> view;
    cellshard::shard_storage storage;
    unsigned long global_row = 0ul;
    cellshard::init(&view);
    cellshard::init(&storage);

    const auto fail = [&](const std::string &message) {
        clear_loaded_view(&view, &storage);
        set_error(error, message);
        return false;
    };

    if (!cellshard::load_header(path, &view, &storage)) {
        return fail(std::string("failed to load ") + format_label + " dataset header");
    }

    out->rows = view.rows;
    out->cols = view.cols;
    out->indptr.assign((std::size_t) view.rows + 1u, 0);
    out->indices.reserve((std::size_t) view.nnz);
    out->data.reserve((std::size_t) view.nnz);

    for (unsigned long partition_id = 0; partition_id < view.num_partitions; ++partition_id) {
        const Part *part = nullptr;
        if (!cellshard::fetch_partition(&view, &storage, partition_id)) {
            return fail(std::string("failed to materialize ") + format_label + " dataset partition");
        }
        part = view.parts[partition_id];
        if (part == nullptr) {
            return fail(std::string(format_label) + " dataset partition is null after fetch");
        }
        for (std::uint32_t row = 0u; row < part->rows; ++row) {
            append_row(part, row, &out->indices, &out->data);
            ++global_row;
            out->indptr[global_row] = (std::int64_t) out->data.size();
        }
        if (!cellshard::drop_partition(&view, partition_id)) {
            return fail(std::string("failed to release ") + format_label + " dataset partition");
        }
    }

    clear_loaded_view(&view, &storage);
    return true;
}

template<typename Part, append_row_fn<Part> append_row>
bool load_dataset_rows_as_csr_impl(const char *path,
                                   const char *format_label,
                                   const std::uint64_t *row_indices,
                                   std::size_t row_count,
                                   csr_matrix_export *out,
                                   std::string *error) {
    cellshard::sharded<Part> view;
    cellshard::shard_storage storage;
    std::vector<unsigned long> fetched_partitions;
    cellshard::init(&view);
    cellshard::init(&storage);

    const auto fail = [&](const std::string &message) {
        release_fetched_partitions(&view, fetched_partitions);
        clear_loaded_view(&view, &storage);
        set_error(error, message);
        return false;
    };

    if (!cellshard::load_header(path, &view, &storage)) {
        return fail(std::string("failed to load ") + format_label + " dataset header");
    }

    out->rows = (std::uint64_t) row_count;
    out->cols = view.cols;
    out->indptr.assign(row_count + 1u, 0);
    fetched_partitions.reserve(std::min<std::size_t>(row_count, (std::size_t) view.num_partitions));

    for (std::size_t i = 0; i < row_count; ++i) {
        const std::uint64_t global_row = row_indices[i];
        const Part *part = nullptr;
        unsigned long partition_id = 0ul;
        unsigned long partition_row_begin = 0ul;
        std::uint64_t local_row = 0u;

        if (global_row >= view.rows) return fail("row index is out of range");

        partition_id = cellshard::find_partition(&view, (unsigned long) global_row);
        if (partition_id >= view.num_partitions) {
            return fail("failed to resolve row index to a partition");
        }
        if (!cellshard::partition_loaded(&view, partition_id)) {
            if (!cellshard::fetch_partition(&view, &storage, partition_id)) {
                return fail(std::string("failed to fetch ") + format_label + " partition for row subset");
            }
            fetched_partitions.push_back(partition_id);
        }

        part = view.parts[partition_id];
        if (part == nullptr) {
            return fail(std::string(format_label) + " partition is null after row subset fetch");
        }

        partition_row_begin = cellshard::first_row_in_partition(&view, partition_id);
        local_row = global_row - (std::uint64_t) partition_row_begin;
        if (local_row >= part->rows) {
            return fail(std::string("resolved local row is outside the fetched ") + format_label + " partition");
        }

        append_row(part, (std::uint32_t) local_row, &out->indices, &out->data);
        out->indptr[i + 1u] = (std::int64_t) out->data.size();
    }

    for (unsigned long partition_id : fetched_partitions) {
        if (!cellshard::drop_partition(&view, partition_id)) {
            return fail(std::string("failed to release ") + format_label + " partition after row subset export");
        }
    }

    clear_loaded_view(&view, &storage);
    return true;
}

} // namespace

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
        return load_dataset_as_csr_impl<cellshard::sparse::blocked_ell, append_blocked_ell_row>(
            path, "blocked-ELL", out, error);
    }
    if (matrix_format == "sliced_ell") {
        return load_dataset_as_csr_impl<cellshard::sparse::sliced_ell, append_sliced_ell_row>(
            path, "sliced-ELL", out, error);
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
        return load_dataset_rows_as_csr_impl<cellshard::sparse::blocked_ell, append_blocked_ell_row>(
            path, "blocked-ELL", row_indices, row_count, out, error);
    }
    if (matrix_format == "sliced_ell") {
        return load_dataset_rows_as_csr_impl<cellshard::sparse::sliced_ell, append_sliced_ell_row>(
            path, "sliced-ELL", row_indices, row_count, out, error);
    }

    set_error(error, "unsupported matrix_format for row-subset CSR export: " + matrix_format);
    return false;
}

bool materialize_derived_dataset(const char *source_path,
                                 const derived_materialization_request &request,
                                 derived_materialization_result *out,
                                 std::string *error) {
    static const unsigned int blocked_ell_candidates[] = {4u, 8u, 16u, 32u};
    dataset_summary source_summary;
    runtime_service_metadata source_runtime;
    source_provenance_vectors source_provenance;
    csr_matrix_export row_subset;
    csr_matrix_export feature_subset;
    sparse::compressed compressed;
    sparse::blocked_ell blocked;
    bucketed_blocked_ell_shard optimized_shard;
    std::vector<observation_metadata_column> observation_columns;
    std::vector<annotation_column> feature_columns;
    std::vector<dataset_attribute> source_attributes;
    std::vector<std::string> selected_barcodes;
    std::vector<std::string> selected_feature_ids;
    std::vector<std::string> selected_feature_names;
    std::vector<std::string> selected_feature_types;
    std::vector<std::uint32_t> selected_cell_dataset_ids;
    std::vector<std::uint64_t> selected_cell_local_indices;
    std::vector<std::uint32_t> selected_feature_dataset_ids;
    std::vector<std::uint64_t> selected_feature_local_indices;
    std::vector<std::uint64_t> dataset_feature_offsets = {0u, 0u};
    std::vector<std::uint32_t> dataset_feature_to_global;
    std::vector<std::string> dataset_ids;
    std::vector<std::string> dataset_paths;
    owned_text_column dataset_ids_column;
    owned_text_column matrix_paths_column;
    owned_text_column feature_paths_column;
    owned_text_column barcode_paths_column;
    owned_text_column metadata_paths_column;
    owned_text_column global_barcodes_column;
    owned_text_column feature_ids_column;
    owned_text_column feature_names_column;
    owned_text_column feature_types_column;
    std::vector<std::uint32_t> dataset_formats(1u, 0u);
    std::vector<std::uint64_t> dataset_row_begin(1u, 0u);
    std::vector<std::uint64_t> dataset_row_end(1u, 0u);
    std::vector<std::uint64_t> dataset_rows(1u, 0u);
    std::vector<std::uint64_t> dataset_cols(1u, 0u);
    std::vector<std::uint64_t> dataset_nnz(1u, 0u);
    dataset_dataset_table_view dataset_view{};
    dataset_provenance_view provenance_view{};
    dataset_codec_descriptor codec{};
    dataset_layout_view layout{};
    std::vector<std::uint64_t> partition_rows(1u, 0u);
    std::vector<std::uint64_t> partition_nnz(1u, 0u);
    std::vector<std::uint64_t> partition_aux(1u, 0u);
    std::vector<std::uint32_t> partition_axes(1u, 0u);
    std::vector<std::uint64_t> partition_row_offsets = {0u, 0u};
    std::vector<std::uint32_t> partition_dataset_ids(1u, 0u);
    std::vector<std::uint32_t> partition_codec_ids(1u, 0u);
    std::vector<std::uint64_t> shard_offsets = {0u, 0u};
    std::vector<owned_annotation_column> owned_observation_columns;
    std::vector<dataset_observation_metadata_column_view> observation_views;
    std::vector<owned_annotation_column> owned_feature_columns;
    std::vector<dataset_observation_metadata_column_view> feature_views;
    dataset_annotation_view observation_view{};
    dataset_feature_metadata_view feature_view{};
    std::vector<std::string> attribute_keys;
    std::vector<std::string> attribute_values;
    owned_text_column attribute_keys_column;
    owned_text_column attribute_values_column;
    dataset_user_attribute_view attribute_view{};
    std::uint32_t part_execution_format = dataset_execution_format_bucketed_blocked_ell;
    std::uint32_t part_block_size = 0u;
    std::uint32_t part_bucket_count = 1u;
    float part_fill_ratio = 0.0f;
    std::uint64_t part_execution_bytes = 0u;
    std::uint64_t part_blocked_ell_bytes = 0u;
    std::uint64_t part_bucketed_blocked_ell_bytes = 0u;
    std::uint32_t shard_execution_format = dataset_execution_format_bucketed_blocked_ell;
    std::uint32_t shard_block_size = 0u;
    std::uint32_t shard_bucketed_partition_count = 1u;
    std::uint32_t shard_bucketed_segment_count = 0u;
    float shard_fill_ratio = 0.0f;
    std::uint64_t shard_execution_bytes = 0u;
    std::uint64_t shard_bucketed_blocked_ell_bytes = 0u;
    std::uint32_t shard_preferred_pair_id = 0u;
    std::uint32_t shard_owner_node_id = 0u;
    std::uint32_t shard_owner_rank_id = 0u;
    dataset_execution_view execution_view{};
    dataset_runtime_service_view runtime_service{};
    ::cellshard::convert::blocked_ell_tune_result tune{};
    std::string output_path;
    std::string cache_root;
    std::string pack_name;

    if (out == nullptr || source_path == nullptr || *source_path == '\0') {
        set_error(error, "source dataset path is empty");
        return false;
    }
    sparse::init(&compressed);
    sparse::init(&blocked);
    init(&optimized_shard);
    *out = derived_materialization_result{};
    if (!request.materialize_dataset && !request.materialize_execution_pack) {
        set_error(error, "derived materialization requested neither dataset nor execution pack output");
        return false;
    }
    if (!load_dataset_summary(source_path, &source_summary, error)) return false;
    if (!validate_unique_indices(request.row_indices, source_summary.rows, "row selection", error)
        || !validate_unique_indices(request.feature_indices, source_summary.cols, "feature selection", error)
        || !validate_group_spans(request.row_groups, request.row_indices.size(), "row group", error)
        || !validate_group_spans(request.feature_groups, request.feature_indices.size(), "feature group", error)) {
        return false;
    }
    output_path = default_derived_dataset_path(source_path, request);
    cache_root = default_execution_cache_root(output_path, request);
    pack_name = sanitize_label(request.derived_pack_name, "derived");

    if (!load_runtime_service_metadata(source_path, &source_runtime, error)
        || !load_source_provenance_vectors(source_path,
                                          source_summary.rows,
                                          source_summary.cols,
                                          &source_provenance,
                                          error)
        || !load_dataset_rows_as_csr(source_path,
                                     request.row_indices.data(),
                                     request.row_indices.size(),
                                     &row_subset,
                                     error)
        || !remap_feature_subset(row_subset, request.feature_indices, &feature_subset, error)
        || !build_compressed_from_csr(feature_subset, &compressed, error)) {
        sparse::clear(&compressed);
        return false;
    }

    if (!(::cellshard::convert::choose_blocked_ell_block_size)(&compressed,
                                                               blocked_ell_candidates,
                                                               4u,
                                                               &tune)
        || !(::cellshard::convert::blocked_ell_from_compressed)(&compressed, tune.block_size, &blocked)) {
        sparse::clear(&compressed);
        sparse::clear(&blocked);
        set_error(error, "failed to rebuild derived blocked-ELL matrix");
        return false;
    }
    sparse::clear(&compressed);

    if (!build_identity_bucketed_shard(blocked, 1u, &optimized_shard, error)) {
        sparse::clear(&blocked);
        return false;
    }

    selected_barcodes.reserve(request.row_indices.size());
    selected_cell_dataset_ids.reserve(request.row_indices.size());
    selected_cell_local_indices.reserve(request.row_indices.size());
    selected_feature_ids.reserve(request.feature_indices.size());
    selected_feature_names.reserve(request.feature_indices.size());
    selected_feature_types.reserve(request.feature_indices.size());
    selected_feature_dataset_ids.reserve(request.feature_indices.size());
    selected_feature_local_indices.reserve(request.feature_indices.size());
    ensure_default_names(&source_summary.obs_names, source_summary.rows, "cell");
    ensure_default_names(&source_summary.var_ids, source_summary.cols, "feature");
    ensure_default_names(&source_summary.var_names, source_summary.cols, "feature");
    ensure_default_names(&source_summary.var_types, source_summary.cols, "feature");
    for (std::uint64_t row : request.row_indices) {
        selected_barcodes.push_back(source_summary.obs_names[(std::size_t) row]);
        selected_cell_dataset_ids.push_back(0u);
        selected_cell_local_indices.push_back(source_provenance.cell_local_indices[(std::size_t) row]);
    }
    for (std::uint64_t feature : request.feature_indices) {
        selected_feature_ids.push_back(source_summary.var_ids[(std::size_t) feature]);
        selected_feature_names.push_back(source_summary.var_names[(std::size_t) feature]);
        selected_feature_types.push_back(source_summary.var_types[(std::size_t) feature]);
        selected_feature_dataset_ids.push_back(0u);
        selected_feature_local_indices.push_back(source_provenance.feature_local_indices[(std::size_t) feature]);
        dataset_feature_to_global.push_back((std::uint32_t) (dataset_feature_to_global.size()));
    }
    dataset_feature_offsets[1] = (std::uint64_t) dataset_feature_to_global.size();

    dataset_ids = { pack_name.empty() ? std::string("derived") : pack_name };
    dataset_paths = { source_path };
    dataset_ids_column = make_text_column(dataset_ids);
    matrix_paths_column = make_text_column(dataset_paths);
    feature_paths_column = make_text_column(dataset_paths);
    barcode_paths_column = make_text_column(dataset_paths);
    metadata_paths_column = make_text_column(dataset_paths);
    dataset_formats[0] = 0u;
    dataset_row_begin[0] = 0u;
    dataset_row_end[0] = feature_subset.rows;
    dataset_rows[0] = feature_subset.rows;
    dataset_cols[0] = feature_subset.cols;
    dataset_nnz[0] = feature_subset.data.size();
    dataset_view.count = 1u;
    dataset_view.dataset_ids = dataset_ids_column.view();
    dataset_view.matrix_paths = matrix_paths_column.view();
    dataset_view.feature_paths = feature_paths_column.view();
    dataset_view.barcode_paths = barcode_paths_column.view();
    dataset_view.metadata_paths = metadata_paths_column.view();
    dataset_view.formats = dataset_formats.data();
    dataset_view.row_begin = dataset_row_begin.data();
    dataset_view.row_end = dataset_row_end.data();
    dataset_view.rows = dataset_rows.data();
    dataset_view.cols = dataset_cols.data();
    dataset_view.nnz = dataset_nnz.data();

    global_barcodes_column = make_text_column(selected_barcodes);
    feature_ids_column = make_text_column(selected_feature_ids);
    feature_names_column = make_text_column(selected_feature_names);
    feature_types_column = make_text_column(selected_feature_types);
    provenance_view.global_barcodes = global_barcodes_column.view();
    provenance_view.cell_dataset_ids = selected_cell_dataset_ids.data();
    provenance_view.cell_local_indices = selected_cell_local_indices.data();
    provenance_view.feature_ids = feature_ids_column.view();
    provenance_view.feature_names = feature_names_column.view();
    provenance_view.feature_types = feature_types_column.view();
    provenance_view.feature_dataset_ids = selected_feature_dataset_ids.data();
    provenance_view.feature_local_indices = selected_feature_local_indices.data();
    provenance_view.dataset_feature_offsets = dataset_feature_offsets.data();
    provenance_view.dataset_feature_to_global = dataset_feature_to_global.data();

    codec.codec_id = 0u;
    codec.family = dataset_codec_family_blocked_ell;
    codec.value_code = (std::uint32_t) ::real::code_of< ::real::storage_t>::code;
    codec.scale_value_code = 0u;
    codec.bits = (std::uint32_t) (sizeof(::real::storage_t) * 8u);
    codec.flags = 0u;

    partition_rows[0] = blocked.rows;
    partition_nnz[0] = blocked.nnz;
    partition_aux[0] = (std::uint64_t) sparse::pack_blocked_ell_aux(blocked.block_size, sparse::ell_width_blocks(&blocked));
    partition_row_offsets[1] = blocked.rows;
    shard_offsets[1] = blocked.rows;
    layout.rows = blocked.rows;
    layout.cols = blocked.cols;
    layout.nnz = blocked.nnz;
    layout.num_partitions = 1u;
    layout.num_shards = 1u;
    layout.partition_rows = partition_rows.data();
    layout.partition_nnz = partition_nnz.data();
    layout.partition_axes = partition_axes.data();
    layout.partition_aux = partition_aux.data();
    layout.partition_row_offsets = partition_row_offsets.data();
    layout.partition_dataset_ids = partition_dataset_ids.data();
    layout.partition_codec_ids = partition_codec_ids.data();
    layout.shard_offsets = shard_offsets.data();
    layout.codecs = &codec;
    layout.num_codecs = 1u;

    part_block_size = blocked.block_size;
    part_fill_ratio = (float) tune.fill_ratio;
    part_blocked_ell_bytes = (std::uint64_t) packed_blocked_ell_bytes(blocked.rows,
                                                                      blocked.ell_cols,
                                                                      blocked.block_size,
                                                                      sizeof(::real::storage_t));
    part_bucketed_blocked_ell_bytes = part_blocked_ell_bytes;
    part_execution_bytes = part_bucketed_blocked_ell_bytes;
    shard_block_size = part_block_size;
    shard_fill_ratio = part_fill_ratio;
    shard_bucketed_segment_count = optimized_shard.partitions[0].segment_count;
    shard_bucketed_blocked_ell_bytes = part_bucketed_blocked_ell_bytes;
    shard_execution_bytes = shard_bucketed_blocked_ell_bytes;
    execution_view.partition_count = 1u;
    execution_view.partition_execution_formats = &part_execution_format;
    execution_view.partition_blocked_ell_block_sizes = &part_block_size;
    execution_view.partition_blocked_ell_bucket_counts = &part_bucket_count;
    execution_view.partition_blocked_ell_fill_ratios = &part_fill_ratio;
    execution_view.partition_execution_bytes = &part_execution_bytes;
    execution_view.partition_blocked_ell_bytes = &part_blocked_ell_bytes;
    execution_view.partition_bucketed_blocked_ell_bytes = &part_bucketed_blocked_ell_bytes;
    execution_view.shard_count = 1u;
    execution_view.shard_execution_formats = &shard_execution_format;
    execution_view.shard_blocked_ell_block_sizes = &shard_block_size;
    execution_view.shard_bucketed_partition_counts = &shard_bucketed_partition_count;
    execution_view.shard_bucketed_segment_counts = &shard_bucketed_segment_count;
    execution_view.shard_blocked_ell_fill_ratios = &shard_fill_ratio;
    execution_view.shard_execution_bytes = &shard_execution_bytes;
    execution_view.shard_bucketed_blocked_ell_bytes = &shard_bucketed_blocked_ell_bytes;
    execution_view.shard_preferred_pair_ids = &shard_preferred_pair_id;
    execution_view.shard_owner_node_ids = &shard_owner_node_id;
    execution_view.shard_owner_rank_ids = &shard_owner_rank_id;
    execution_view.preferred_base_format = dataset_execution_format_bucketed_blocked_ell;

    init(&runtime_service);
    runtime_service.service_mode = dataset_runtime_service_mode_owner_hosted;
    runtime_service.live_write_mode = dataset_live_write_mode_append_only;
    runtime_service.prefer_pack_delivery = request.materialize_execution_pack ? 1u : 0u;
    runtime_service.single_reader_coordinator = 1u;
    runtime_service.maintenance_lock_blocks_overwrite = 1u;
    runtime_service.canonical_generation = 1u;
    runtime_service.execution_plan_generation = 1u;
    runtime_service.pack_generation = 1u;
    runtime_service.service_epoch = 1u;
    runtime_service.active_read_generation = 1u;
    runtime_service.staged_write_generation = 1u;

    (void) load_observation_metadata(source_path, &observation_columns, error);
    (void) load_feature_metadata(source_path, &feature_columns, error);
    (void) load_dataset_attributes(source_path, &source_attributes, error);

    if (!build_selected_observation_columns(observation_columns,
                                            request.row_indices,
                                            request.row_groups,
                                            &owned_observation_columns,
                                            &observation_views,
                                            error)
        || !build_selected_feature_columns(feature_columns,
                                           request.feature_indices,
                                           request.feature_groups,
                                           &owned_feature_columns,
                                           &feature_views,
                                           error)) {
        clear(&optimized_shard);
        sparse::clear(&blocked);
        return false;
    }
    observation_view.extent = (std::uint64_t) request.row_indices.size();
    observation_view.cols = (std::uint32_t) observation_views.size();
    observation_view.columns = observation_views.empty() ? nullptr : observation_views.data();
    feature_view.cols = (std::uint64_t) request.feature_indices.size();
    feature_view.annotation_count = (std::uint32_t) feature_views.size();
    feature_view.annotations = feature_views.empty() ? nullptr : feature_views.data();

    attribute_keys.reserve(source_attributes.size() + 8u);
    attribute_values.reserve(source_attributes.size() + 8u);
    for (const dataset_attribute &entry : source_attributes) {
        attribute_keys.push_back(entry.key);
        attribute_values.push_back(entry.value);
    }
    attribute_keys.push_back("derived.parent_path");
    attribute_values.push_back(source_path);
    attribute_keys.push_back("derived.parent_canonical_generation");
    attribute_values.push_back(std::to_string(source_runtime.canonical_generation));
    attribute_keys.push_back("derived.parent_execution_plan_generation");
    attribute_values.push_back(std::to_string(source_runtime.execution_plan_generation));
    attribute_keys.push_back("derived.parent_pack_generation");
    attribute_values.push_back(std::to_string(source_runtime.pack_generation));
    attribute_keys.push_back("derived.parent_service_epoch");
    attribute_values.push_back(std::to_string(source_runtime.service_epoch));
    attribute_keys.push_back("derived.pack_name");
    attribute_values.push_back(pack_name);
    attribute_keys.push_back("derived.requested_dataset_output");
    attribute_values.push_back(request.materialize_dataset ? "1" : "0");
    attribute_keys.push_back("derived.requested_pack_output");
    attribute_values.push_back(request.materialize_execution_pack ? "1" : "0");
    attribute_keys_column = make_text_column(attribute_keys);
    attribute_values_column = make_text_column(attribute_values);
    attribute_view.count = (std::uint32_t) attribute_keys.size();
    attribute_view.keys = attribute_keys_column.view();
    attribute_view.values = attribute_values_column.view();

    {
        const fs::path parent = fs::path(output_path).parent_path();
        std::error_code ec;
        if (!parent.empty()) fs::create_directories(parent, ec);
    }
    std::remove(output_path.c_str());
    if (!create_dataset_optimized_blocked_ell_h5(output_path.c_str(), &layout, &dataset_view, &provenance_view)
        || !append_bucketed_blocked_ell_shard_h5(output_path.c_str(), 0u, &optimized_shard)
        || !append_dataset_execution_h5(output_path.c_str(), &execution_view)
        || !append_dataset_runtime_service_h5(output_path.c_str(), &runtime_service)
        || !append_dataset_observation_annotations_h5(output_path.c_str(), &observation_view)
        || !append_dataset_feature_metadata_h5(output_path.c_str(), &feature_view)
        || !append_dataset_user_attributes_h5(output_path.c_str(), &attribute_view)) {
        clear(&optimized_shard);
        sparse::clear(&blocked);
        set_error(error, "failed to write derived dataset");
        return false;
    }

    clear(&optimized_shard);
    sparse::clear(&blocked);

    if (request.materialize_execution_pack) {
        std::error_code ec;
        fs::create_directories(cache_root, ec);
        if (!warm_dataset_blocked_ell_h5_cache(output_path.c_str(), cache_root.c_str())
            || !warm_dataset_blocked_ell_h5_execution_cache(output_path.c_str(), cache_root.c_str())) {
            set_error(error, "failed to warm derived execution packs");
            return false;
        }
    }

    out->materialized_dataset = true;
    out->materialized_execution_pack = request.materialize_execution_pack;
    out->materialized_dataset_path = output_path;
    out->execution_cache_root = request.materialize_execution_pack ? cache_root : std::string();
    out->derived_pack_name = pack_name;
    out->rows = feature_subset.rows;
    out->cols = feature_subset.cols;
    out->nnz = feature_subset.data.size();
    out->runtime_service.service_mode = runtime_service.service_mode;
    out->runtime_service.live_write_mode = runtime_service.live_write_mode;
    out->runtime_service.prefer_pack_delivery = runtime_service.prefer_pack_delivery;
    out->runtime_service.remote_pack_delivery = runtime_service.remote_pack_delivery;
    out->runtime_service.single_reader_coordinator = runtime_service.single_reader_coordinator;
    out->runtime_service.maintenance_lock_blocks_overwrite = runtime_service.maintenance_lock_blocks_overwrite;
    out->runtime_service.canonical_generation = runtime_service.canonical_generation;
    out->runtime_service.execution_plan_generation = runtime_service.execution_plan_generation;
    out->runtime_service.pack_generation = runtime_service.pack_generation;
    out->runtime_service.service_epoch = runtime_service.service_epoch;
    out->runtime_service.active_read_generation = runtime_service.active_read_generation;
    out->runtime_service.staged_write_generation = runtime_service.staged_write_generation;
    return true;
}

bool load_dataset_for_anndata(const char *path, anndata_export *out, std::string *error) {
    if (out == nullptr) {
        set_error(error, "anndata export output is null");
        return false;
    }
    *out = anndata_export{};
    if (!load_dataset_summary(path, &out->summary, error)) return false;
    if (!(::cellshard::exporting::load_observation_metadata)(path, &out->obs_columns, error)) return false;
    if (!(::cellshard::exporting::load_feature_metadata)(path, &out->var_columns, error)) return false;
    if (!(::cellshard::exporting::load_dataset_attributes)(path, &out->uns, error)) return false;
    if (!load_dataset_as_csr(path, &out->x, error)) return false;
    return true;
}

} // namespace cellshard::exporting
