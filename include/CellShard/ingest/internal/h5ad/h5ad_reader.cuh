#pragma once

#include "../common/barcode_table.cuh"
#include "../common/feature_table.cuh"
#include "../common/metadata_table.cuh"
#include "../mtx/mtx_reader.cuh"

#include <CellShard/formats/triplet.cuh>
#include <CellShard/formats/compressed.cuh>
#include <CellShard/runtime/layout/sharded.cuh>
#include <CellShard/runtime/host/sharded_host.cuh>

#include <hdf5.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace cellshard {
namespace ingest {
namespace h5ad {

using ::cellshard::clear;
using ::cellshard::find_offset_span;
using ::cellshard::init;
using ::cellshard::reserve_partitions;
using ::cellshard::set_shards_to_partitions;
using ::cellshard::sharded;
namespace sparse = ::cellshard::sparse;

enum {
    observation_column_none = 0u,
    observation_column_text = 1u,
    observation_column_float32 = 2u,
    observation_column_uint8 = 3u
};

struct observation_column {
    std::string name;
    std::uint32_t type = observation_column_none;
    common::text_column text_values;
    std::vector<float> float32_values;
    std::vector<std::uint8_t> uint8_values;
};

struct selected_matrix_info {
    bool available = false;
    bool sparse = false;
    bool csr = false;
    bool csc = false;
    bool processed_like = false;
    bool count_like = false;
    bool dense = false;
    std::string matrix_path;
    std::string var_path;
    unsigned long rows = 0ul;
    unsigned long cols = 0ul;
    unsigned long nnz = 0ul;
};

static inline bool probe_selected_matrix(const char *path,
                                         const char *matrix_source,
                                         selected_matrix_info *out,
                                         std::string *error);
static inline bool load_dataframe_columns(const char *path,
                                          const char *group_path,
                                          std::vector<observation_column> *columns,
                                          unsigned long *rows_out,
                                          std::string *error);

static inline void init(observation_column *column) {
    if (column == nullptr) return;
    column->name.clear();
    column->type = observation_column_none;
    common::init(&column->text_values);
    column->float32_values.clear();
    column->uint8_values.clear();
}

static inline void clear(observation_column *column) {
    if (column == nullptr) return;
    common::clear(&column->text_values);
    column->float32_values.clear();
    column->uint8_values.clear();
    column->name.clear();
    column->type = observation_column_none;
}

inline void set_error(std::string *error, const char *message) {
    if (error != nullptr) *error = message != nullptr ? message : "unknown h5ad error";
}

inline void set_error(std::string *error, const std::string &message) {
    if (error != nullptr) *error = message;
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

inline hid_t open_optional_dataset(hid_t parent, const char *path) {
    hid_t dset = (hid_t) -1;
    H5E_BEGIN_TRY {
        dset = H5Dopen2(parent, path, H5P_DEFAULT);
    } H5E_END_TRY;
    return dset;
}

inline bool ends_with(const std::string &text, const char *suffix) {
    const std::size_t suffix_len = std::strlen(suffix);
    return text.size() >= suffix_len
        && text.compare(text.size() - suffix_len, suffix_len, suffix) == 0;
}

inline std::string trim_copy(std::string value) {
    std::size_t begin = 0u;
    std::size_t end = value.size();
    while (begin < end && std::isspace((unsigned char) value[begin])) ++begin;
    while (end > begin && std::isspace((unsigned char) value[end - 1u])) --end;
    return value.substr(begin, end - begin);
}

inline std::string lower_copy(std::string value) {
    for (char &ch : value) ch = (char) std::tolower((unsigned char) ch);
    return value;
}

inline bool parse_bool_text(const std::string &value, bool *out) {
    const std::string lowered = lower_copy(trim_copy(value));
    if (lowered.empty()) {
        *out = false;
        return true;
    }
    if (lowered == "1" || lowered == "true" || lowered == "yes" || lowered == "y" || lowered == "on") {
        *out = true;
        return true;
    }
    if (lowered == "0" || lowered == "false" || lowered == "no" || lowered == "n" || lowered == "off") {
        *out = false;
        return true;
    }
    return false;
}

inline bool build_matrix_paths(const char *matrix_source,
                               std::string *matrix_path,
                               std::string *var_path,
                               std::string *normalized_source,
                               std::string *error) {
    const std::string source = matrix_source != nullptr ? trim_copy(matrix_source) : "";
    if (source.empty() || source == "x") {
        if (matrix_path != nullptr) *matrix_path = "/X";
        if (var_path != nullptr) *var_path = "/var";
        if (normalized_source != nullptr) *normalized_source = "x";
        return true;
    }
    if (source == "raw_x") {
        if (matrix_path != nullptr) *matrix_path = "/raw/X";
        if (var_path != nullptr) *var_path = "/raw/var";
        if (normalized_source != nullptr) *normalized_source = "raw_x";
        return true;
    }
    if (source.rfind("layer:", 0u) == 0u && source.size() > 6u) {
        const std::string name = source.substr(6u);
        if (matrix_path != nullptr) *matrix_path = "/layers/" + name;
        if (var_path != nullptr) *var_path = "/var";
        if (normalized_source != nullptr) *normalized_source = "layer:" + name;
        return true;
    }
    set_error(error, "unsupported h5ad matrix_source; expected x, raw_x, or layer:<name>");
    return false;
}

inline bool read_attr_string(hid_t obj, const char *name, std::string *out) {
    hid_t attr = (hid_t) -1;
    hid_t type = (hid_t) -1;
    hid_t mem = (hid_t) -1;
    hid_t space = (hid_t) -1;
    hssize_t count = 0;
    char **values = nullptr;
    std::vector<char> fixed_buffer;
    bool ok = false;

    if (out == nullptr) return false;
    out->clear();
    attr = H5Aopen(obj, name, H5P_DEFAULT);
    if (attr < 0) goto done;
    type = H5Aget_type(attr);
    if (type < 0 || H5Tget_class(type) != H5T_STRING) goto done;
    space = H5Aget_space(attr);
    if (space < 0) goto done;
    count = H5Sget_simple_extent_npoints(space);
    if (count <= 0) goto done;
    if (H5Tis_variable_str(type) > 0) {
        mem = H5Tcopy(H5T_C_S1);
        if (mem < 0) goto done;
        if (H5Tset_size(mem, H5T_VARIABLE) < 0) goto done;
        if (H5Tset_cset(mem, H5Tget_cset(type)) < 0) goto done;
        values = (char **) std::calloc((std::size_t) count, sizeof(char *));
        if (values == nullptr) goto done;
        if (H5Aread(attr, mem, values) < 0) goto done;
        *out = values[0] != nullptr ? values[0] : "";
    } else {
        const std::size_t size = H5Tget_size(type);
        if (size == 0u) goto done;
        fixed_buffer.assign(size + 1u, '\0');
        if (H5Aread(attr, type, fixed_buffer.data()) < 0) goto done;
        *out = fixed_buffer.data();
    }
    ok = true;

done:
    if (values != nullptr) {
        H5Dvlen_reclaim(mem >= 0 ? mem : type, space, H5P_DEFAULT, values);
        std::free(values);
    }
    if (space >= 0) H5Sclose(space);
    if (mem >= 0) H5Tclose(mem);
    if (type >= 0) H5Tclose(type);
    if (attr >= 0) H5Aclose(attr);
    return ok;
}

inline bool read_attr_u64_vector(hid_t obj, const char *name, std::vector<std::uint64_t> *out) {
    hid_t attr = (hid_t) -1;
    hid_t type = (hid_t) -1;
    hid_t space = (hid_t) -1;
    hssize_t count = 0;
    bool ok = false;

    if (out == nullptr) return false;
    out->clear();
    attr = H5Aopen(obj, name, H5P_DEFAULT);
    if (attr < 0) goto done;
    type = H5Aget_type(attr);
    if (type < 0 || H5Tget_class(type) != H5T_INTEGER) goto done;
    space = H5Aget_space(attr);
    if (space < 0) goto done;
    count = H5Sget_simple_extent_npoints(space);
    if (count <= 0) goto done;
    out->resize((std::size_t) count, 0u);
    if (H5Tget_sign(type) == H5T_SGN_2) {
        std::vector<long long> signed_values((std::size_t) count, 0ll);
        if (H5Aread(attr, H5T_NATIVE_LLONG, signed_values.data()) < 0) goto done;
        for (std::size_t i = 0; i < signed_values.size(); ++i) {
            if (signed_values[i] < 0) goto done;
            (*out)[i] = (std::uint64_t) signed_values[i];
        }
    } else {
        if (H5Aread(attr, H5T_NATIVE_UINT64, out->data()) < 0) goto done;
    }
    ok = true;

done:
    if (space >= 0) H5Sclose(space);
    if (type >= 0) H5Tclose(type);
    if (attr >= 0) H5Aclose(attr);
    if (!ok) out->clear();
    return ok;
}

inline bool dataset_length(hid_t dset, hsize_t *length) {
    hid_t space = (hid_t) -1;
    int rank = 0;
    hsize_t dims[4] = {0u, 0u, 0u, 0u};
    bool ok = false;

    if (length == nullptr) return false;
    *length = 0u;
    space = H5Dget_space(dset);
    if (space < 0) goto done;
    rank = H5Sget_simple_extent_ndims(space);
    if (rank != 1) goto done;
    if (H5Sget_simple_extent_dims(space, dims, nullptr) != 1) goto done;
    *length = dims[0];
    ok = true;

done:
    if (space >= 0) H5Sclose(space);
    return ok;
}

inline bool read_dataset_u64_range(hid_t parent,
                                   const char *name,
                                   std::uint64_t offset,
                                   std::uint64_t count,
                                   std::vector<std::uint64_t> *out) {
    hid_t dset = (hid_t) -1;
    hid_t filespace = (hid_t) -1;
    hid_t memspace = (hid_t) -1;
    hsize_t dims[1];
    hsize_t start[1];
    bool ok = false;

    if (out == nullptr) return false;
    out->clear();
    dset = H5Dopen2(parent, name, H5P_DEFAULT);
    if (dset < 0) goto done;
    if (count == 0u) {
        ok = true;
        goto done;
    }
    out->resize((std::size_t) count, 0u);
    filespace = H5Dget_space(dset);
    if (filespace < 0) goto done;
    start[0] = (hsize_t) offset;
    dims[0] = (hsize_t) count;
    if (H5Sselect_hyperslab(filespace, H5S_SELECT_SET, start, nullptr, dims, nullptr) < 0) goto done;
    memspace = H5Screate_simple(1, dims, nullptr);
    if (memspace < 0) goto done;
    if (H5Dread(dset, H5T_NATIVE_UINT64, memspace, filespace, H5P_DEFAULT, out->data()) < 0) goto done;
    ok = true;

done:
    if (memspace >= 0) H5Sclose(memspace);
    if (filespace >= 0) H5Sclose(filespace);
    if (dset >= 0) H5Dclose(dset);
    if (!ok) out->clear();
    return ok;
}

inline bool read_dataset_i64_range(hid_t parent,
                                   const char *name,
                                   std::uint64_t offset,
                                   std::uint64_t count,
                                   std::vector<long long> *out) {
    hid_t dset = (hid_t) -1;
    hid_t filespace = (hid_t) -1;
    hid_t memspace = (hid_t) -1;
    hsize_t dims[1];
    hsize_t start[1];
    bool ok = false;

    if (out == nullptr) return false;
    out->clear();
    dset = H5Dopen2(parent, name, H5P_DEFAULT);
    if (dset < 0) goto done;
    if (count == 0u) {
        ok = true;
        goto done;
    }
    out->resize((std::size_t) count, 0);
    filespace = H5Dget_space(dset);
    if (filespace < 0) goto done;
    start[0] = (hsize_t) offset;
    dims[0] = (hsize_t) count;
    if (H5Sselect_hyperslab(filespace, H5S_SELECT_SET, start, nullptr, dims, nullptr) < 0) goto done;
    memspace = H5Screate_simple(1, dims, nullptr);
    if (memspace < 0) goto done;
    if (H5Dread(dset, H5T_NATIVE_LLONG, memspace, filespace, H5P_DEFAULT, out->data()) < 0) goto done;
    ok = true;

done:
    if (memspace >= 0) H5Sclose(memspace);
    if (filespace >= 0) H5Sclose(filespace);
    if (dset >= 0) H5Dclose(dset);
    if (!ok) out->clear();
    return ok;
}

inline bool read_dataset_double_range(hid_t parent,
                                      const char *name,
                                      std::uint64_t offset,
                                      std::uint64_t count,
                                      std::vector<double> *out) {
    hid_t dset = (hid_t) -1;
    hid_t filespace = (hid_t) -1;
    hid_t memspace = (hid_t) -1;
    hsize_t dims[1];
    hsize_t start[1];
    bool ok = false;

    if (out == nullptr) return false;
    out->clear();
    dset = H5Dopen2(parent, name, H5P_DEFAULT);
    if (dset < 0) goto done;
    if (count == 0u) {
        ok = true;
        goto done;
    }
    out->resize((std::size_t) count, 0.0);
    filespace = H5Dget_space(dset);
    if (filespace < 0) goto done;
    start[0] = (hsize_t) offset;
    dims[0] = (hsize_t) count;
    if (H5Sselect_hyperslab(filespace, H5S_SELECT_SET, start, nullptr, dims, nullptr) < 0) goto done;
    memspace = H5Screate_simple(1, dims, nullptr);
    if (memspace < 0) goto done;
    if (H5Dread(dset, H5T_NATIVE_DOUBLE, memspace, filespace, H5P_DEFAULT, out->data()) < 0) goto done;
    ok = true;

done:
    if (memspace >= 0) H5Sclose(memspace);
    if (filespace >= 0) H5Sclose(filespace);
    if (dset >= 0) H5Dclose(dset);
    if (!ok) out->clear();
    return ok;
}

inline bool read_matrix_indptr(hid_t group,
                               const selected_matrix_info &info,
                               std::vector<std::uint64_t> *indptr,
                               std::string *error) {
    const std::uint64_t count = (std::uint64_t) (info.csr ? info.rows : info.cols) + 1u;
    if (!read_dataset_u64_range(group, "indptr", 0u, count, indptr)) {
        set_error(error, info.csr ? "failed to read h5ad csr indptr" : "failed to read h5ad csc indptr");
        return false;
    }
    return indptr->size() == count;
}

inline bool read_dataset_string_column(hid_t parent, const char *name, common::text_column *out) {
    hid_t dset = (hid_t) -1;
    hid_t type = (hid_t) -1;
    hid_t mem = (hid_t) -1;
    hid_t space = (hid_t) -1;
    hssize_t count = 0;
    char **values = nullptr;
    std::vector<char> fixed_values;
    bool ok = false;

    if (out == nullptr) return false;
    common::clear(out);
    common::init(out);
    dset = H5Dopen2(parent, name, H5P_DEFAULT);
    if (dset < 0) goto done;
    type = H5Dget_type(dset);
    if (type < 0 || H5Tget_class(type) != H5T_STRING) goto done;
    space = H5Dget_space(dset);
    if (space < 0 || H5Sget_simple_extent_ndims(space) != 1) goto done;
    count = H5Sget_simple_extent_npoints(space);
    if (count < 0) goto done;
    if (H5Tis_variable_str(type) > 0) {
        mem = H5Tcopy(H5T_C_S1);
        if (mem < 0) goto done;
        if (H5Tset_size(mem, H5T_VARIABLE) < 0) goto done;
        if (H5Tset_cset(mem, H5Tget_cset(type)) < 0) goto done;
        values = (char **) std::calloc((std::size_t) count, sizeof(char *));
        if (count != 0 && values == nullptr) goto done;
        if (count != 0 && H5Dread(dset, mem, H5S_ALL, H5S_ALL, H5P_DEFAULT, values) < 0) goto done;
        for (hssize_t i = 0; i < count; ++i) {
            const char *value = values[i] != nullptr ? values[i] : "";
            if (!common::append(out, value, std::strlen(value))) goto done;
        }
    } else {
        const std::size_t item_size = H5Tget_size(type);
        if (item_size == 0u) goto done;
        fixed_values.assign((std::size_t) count * item_size, '\0');
        if (count != 0 && H5Dread(dset, type, H5S_ALL, H5S_ALL, H5P_DEFAULT, fixed_values.data()) < 0) goto done;
        for (hssize_t i = 0; i < count; ++i) {
            const char *value = fixed_values.data() + (std::size_t) i * item_size;
            std::size_t value_len = 0u;
            while (value_len < item_size && value[value_len] != '\0') ++value_len;
            while (value_len > 0u && std::isspace((unsigned char) value[value_len - 1u])) --value_len;
            if (!common::append(out, value, value_len)) goto done;
        }
    }
    ok = true;

done:
    if (values != nullptr) {
        H5Dvlen_reclaim(mem >= 0 ? mem : type, space, H5P_DEFAULT, values);
        std::free(values);
    }
    if (space >= 0) H5Sclose(space);
    if (mem >= 0) H5Tclose(mem);
    if (type >= 0) H5Tclose(type);
    if (dset >= 0) H5Dclose(dset);
    if (!ok) {
        common::clear(out);
        common::init(out);
    }
    return ok;
}

inline bool read_dataset_float32(hid_t parent, const char *name, std::vector<float> *out) {
    hid_t dset = (hid_t) -1;
    hsize_t length = 0u;
    bool ok = false;

    if (out == nullptr) return false;
    out->clear();
    dset = H5Dopen2(parent, name, H5P_DEFAULT);
    if (dset < 0) goto done;
    if (!dataset_length(dset, &length)) goto done;
    out->resize((std::size_t) length, 0.0f);
    if (length != 0u && H5Dread(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, out->data()) < 0) goto done;
    ok = true;

done:
    if (dset >= 0) H5Dclose(dset);
    if (!ok) out->clear();
    return ok;
}

inline bool read_dataset_uint8(hid_t parent, const char *name, std::vector<std::uint8_t> *out) {
    hid_t dset = (hid_t) -1;
    hsize_t length = 0u;
    bool ok = false;

    if (out == nullptr) return false;
    out->clear();
    dset = H5Dopen2(parent, name, H5P_DEFAULT);
    if (dset < 0) goto done;
    if (!dataset_length(dset, &length)) goto done;
    out->resize((std::size_t) length, (std::uint8_t) 0u);
    if (length != 0u && H5Dread(dset, H5T_NATIVE_UCHAR, H5S_ALL, H5S_ALL, H5P_DEFAULT, out->data()) < 0) goto done;
    ok = true;

done:
    if (dset >= 0) H5Dclose(dset);
    if (!ok) out->clear();
    return ok;
}

inline herr_t collect_child_names_cb(hid_t,
                                     const char *name,
                                     const H5L_info_t *,
                                     void *op_data) {
    if (op_data == nullptr || name == nullptr) return 0;
    ((std::vector<std::string> *) op_data)->push_back(name);
    return 0;
}

inline bool list_child_names(hid_t group, std::vector<std::string> *names) {
    if (names == nullptr) return false;
    names->clear();
    if (H5Literate(group, H5_INDEX_NAME, H5_ITER_NATIVE, nullptr, collect_child_names_cb, names) < 0) return false;
    std::sort(names->begin(), names->end());
    return true;
}

inline bool load_dataframe_index(hid_t file,
                                 const std::string &group_path,
                                 common::text_column *out,
                                 std::string *error) {
    hid_t group = (hid_t) -1;
    std::string index_name;
    bool ok = false;

    if (out == nullptr) return false;
    common::clear(out);
    common::init(out);
    group = H5Gopen2(file, group_path.c_str(), H5P_DEFAULT);
    if (group < 0) {
        set_error(error, "failed to open h5ad dataframe group");
        goto done;
    }
    if (!read_attr_string(group, "_index", &index_name) || index_name.empty()) index_name = "_index";
    if (!read_dataset_string_column(group, index_name.c_str(), out)) {
        set_error(error, "failed to read h5ad dataframe index");
        goto done;
    }
    ok = true;

done:
    if (group >= 0) H5Gclose(group);
    if (!ok) {
        common::clear(out);
        common::init(out);
    }
    return ok;
}

inline bool load_categorical_text_column(hid_t group,
                                         unsigned long expected_rows,
                                         common::text_column *out) {
    std::vector<long long> codes;
    common::text_column categories;
    bool ok = false;

    if (out == nullptr) return false;
    common::init(&categories);
    common::clear(out);
    common::init(out);
    if (!read_dataset_i64_range(group, "codes", 0u, (std::uint64_t) expected_rows, &codes)) goto done;
    if (!read_dataset_string_column(group, "categories", &categories)) goto done;
    if ((unsigned long) codes.size() != expected_rows) goto done;
    for (unsigned long i = 0; i < expected_rows; ++i) {
        const long long code = codes[(std::size_t) i];
        const char *value = "";
        if (code >= 0 && (unsigned long long) code < (unsigned long long) categories.count) {
            value = common::get(&categories, (unsigned int) code);
        }
        if (!common::append(out, value != nullptr ? value : "", std::strlen(value != nullptr ? value : ""))) goto done;
    }
    ok = true;

done:
    common::clear(&categories);
    if (!ok) {
        common::clear(out);
        common::init(out);
    }
    return ok;
}

inline bool load_dataframe_text_column(hid_t group,
                                       const std::string &name,
                                       unsigned long expected_rows,
                                       common::text_column *out) {
    hid_t dset = open_optional_dataset(group, name.c_str());
    hid_t sub = (hid_t) -1;
    std::string encoding;
    hsize_t length = 0u;
    bool ok = false;

    if (dset >= 0) {
        if (!dataset_length(dset, &length) || (unsigned long) length != expected_rows) goto done;
        H5Dclose(dset);
        dset = (hid_t) -1;
        return read_dataset_string_column(group, name.c_str(), out);
    }

    sub = open_optional_group(group, name.c_str());
    if (sub < 0) goto done;
    if (!read_attr_string(sub, "encoding-type", &encoding)) goto done;
    if (encoding != "categorical") goto done;
    if (!load_categorical_text_column(sub, expected_rows, out)) goto done;
    ok = true;

done:
    if (sub >= 0) H5Gclose(sub);
    if (dset >= 0) H5Dclose(dset);
    return ok;
}

#include "internal/dataframe_part.hh"

inline bool load_metadata_table(const char *path, common::metadata_table *out, std::string *error) {
    std::vector<observation_column> columns;
    unsigned long rows = 0ul;
    bool ok = false;

    if (out == nullptr) return false;
    common::clear(out);
    common::init(out);
    if (!load_observation_columns(path, &columns, &rows, error)) return false;
    ok = build_metadata_table_from_observation_columns(columns, rows, out);
    for (observation_column &column : columns) clear(&column);
    columns.clear();
    if (!ok) {
        common::clear(out);
        common::init(out);
    }
    return ok;
}

inline bool probe_selected_matrix(const char *path,
                                  const char *matrix_source,
                                  selected_matrix_info *out,
                                  std::string *error) {
    hid_t file = (hid_t) -1;
    hid_t group = (hid_t) -1;
    hid_t dense = (hid_t) -1;
    std::string matrix_path;
    std::string var_path;
    std::string normalized_source;
    std::string encoding;
    std::vector<std::uint64_t> shape;
    std::vector<std::uint64_t> indptr;
    std::vector<double> sample;
    bool ok = false;

    if (out == nullptr) return false;
    *out = selected_matrix_info{};
    if (!build_matrix_paths(matrix_source, &matrix_path, &var_path, &normalized_source, error)) return false;
    file = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0) {
        set_error(error, "failed to open h5ad file");
        goto done;
    }
    group = open_optional_group(file, matrix_path.c_str());
    if (group < 0) {
        dense = open_optional_dataset(file, matrix_path.c_str());
        if (dense >= 0) {
            out->dense = true;
            set_error(error, "dense h5ad matrices are not supported by ingest");
        } else {
            set_error(error, "selected h5ad matrix source is missing");
        }
        goto done;
    }
    if (!read_attr_string(group, "encoding-type", &encoding)) {
        set_error(error, "selected h5ad matrix is missing encoding-type");
        goto done;
    }
    if (encoding == "csr_matrix") {
        out->sparse = true;
        out->csr = true;
    } else if (encoding == "csc_matrix") {
        out->sparse = true;
        out->csc = true;
    } else {
        set_error(error, "selected h5ad matrix is not a supported sparse matrix group");
        goto done;
    }
    if (!read_attr_u64_vector(group, "shape", &shape) || shape.size() < 2u) {
        set_error(error, "selected h5ad matrix is missing shape");
        goto done;
    }
    if (shape[0] > (std::uint64_t) std::numeric_limits<unsigned long>::max()
        || shape[1] > (std::uint64_t) std::numeric_limits<unsigned long>::max()) {
        set_error(error, "selected h5ad matrix exceeds ingest dimension limits");
        goto done;
    }
    out->rows = (unsigned long) shape[0];
    out->cols = (unsigned long) shape[1];
    if (!read_matrix_indptr(group, *out, &indptr, error)) {
        goto done;
    }
    if (indptr.back() > (std::uint64_t) std::numeric_limits<unsigned long>::max()) {
        set_error(error, "selected h5ad matrix exceeds ingest dimension limits");
        goto done;
    }
    out->available = true;
    out->matrix_path = matrix_path;
    out->var_path = var_path;
    out->nnz = (unsigned long) indptr.back();
    out->count_like = true;
    if (out->nnz != 0u) {
        const std::uint64_t sample_count = std::min<std::uint64_t>(out->nnz, 1024u);
        if (!read_dataset_double_range(group, "data", 0u, sample_count, &sample)) {
            set_error(error, out->csr ? "failed to sample h5ad csr values" : "failed to sample h5ad csc values");
            goto done;
        }
        for (double value : sample) {
            if (!std::isfinite(value) || value < -1.0e-6 || std::fabs(value - std::round(value)) > 1.0e-4) {
                out->count_like = false;
                break;
            }
        }
    }
    out->processed_like = !out->count_like;
    ok = true;

done:
    if (dense >= 0) H5Dclose(dense);
    if (group >= 0) H5Gclose(group);
    if (file >= 0) H5Fclose(file);
    return ok;
}

inline bool scan_row_nnz(const char *path,
                         const char *matrix_source,
                         mtx::header *h,
                         unsigned long **row_nnz_out,
                         std::string *error) {
    hid_t file = (hid_t) -1;
    hid_t group = (hid_t) -1;
    selected_matrix_info info;
    std::vector<std::uint64_t> indptr;
    std::vector<std::uint64_t> indices;
    unsigned long *row_nnz = nullptr;
    const std::uint64_t chunk_elems = (std::uint64_t) 1u << 20u;
    std::uint64_t chunk_begin = 0u;
    std::uint64_t chunk_end = 0u;
    bool ok = false;

    if (h == nullptr || row_nnz_out == nullptr) return false;
    *row_nnz_out = nullptr;
    mtx::init(h);
    if (!probe_selected_matrix(path, matrix_source, &info, error)) return false;
    file = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0) {
        set_error(error, "failed to reopen h5ad file");
        goto done;
    }
    group = H5Gopen2(file, info.matrix_path.c_str(), H5P_DEFAULT);
    if (group < 0) {
        set_error(error, "failed to reopen selected h5ad matrix");
        goto done;
    }
    if (!read_matrix_indptr(group, info, &indptr, error)) {
        goto done;
    }
    row_nnz = (unsigned long *) std::calloc((std::size_t) info.rows, sizeof(unsigned long));
    if (info.rows != 0ul && row_nnz == nullptr) goto done;
    if (info.csr) {
        for (unsigned long row = 0; row < info.rows; ++row) {
            const std::uint64_t span = indptr[(std::size_t) row + 1u] - indptr[(std::size_t) row];
            if (span > (std::uint64_t) std::numeric_limits<unsigned long>::max()) goto done;
            row_nnz[row] = (unsigned long) span;
        }
    } else {
        for (unsigned long col = 0; col < info.cols; ++col) {
            const std::uint64_t col_begin = indptr[(std::size_t) col];
            const std::uint64_t col_end = indptr[(std::size_t) col + 1u];
            for (std::uint64_t cursor = col_begin; cursor < col_end; ++cursor) {
                if (cursor < chunk_begin || cursor >= chunk_end) {
                    chunk_begin = cursor;
                    chunk_end = std::min<std::uint64_t>(chunk_begin + chunk_elems, indptr.back());
                    if (!read_dataset_u64_range(group, "indices", chunk_begin, chunk_end - chunk_begin, &indices)) {
                        set_error(error, "failed to read h5ad csc indices");
                        goto done;
                    }
                }
                const std::uint64_t row = indices[(std::size_t) (cursor - chunk_begin)];
                if (row >= info.rows) goto done;
                ++row_nnz[(std::size_t) row];
            }
        }
    }
    h->rows = info.rows;
    h->cols = info.cols;
    h->nnz_file = info.nnz;
    h->nnz_loaded = info.nnz;
    h->row_sorted = info.csr ? 1 : 0;
    ok = true;

done:
    if (group >= 0) H5Gclose(group);
    if (file >= 0) H5Fclose(file);
    if (!ok) {
        std::free(row_nnz);
        return false;
    }
    *row_nnz_out = row_nnz;
    return true;
}

inline bool load_indptr(const char *path,
                        const char *matrix_source,
                        std::vector<std::uint64_t> *indptr,
                        mtx::header *header_out,
                        std::string *error) {
    hid_t file = (hid_t) -1;
    hid_t group = (hid_t) -1;
    selected_matrix_info info;
    bool ok = false;

    if (indptr == nullptr) return false;
    indptr->clear();
    if (!probe_selected_matrix(path, matrix_source, &info, error)) return false;
    file = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0) {
        set_error(error, "failed to open h5ad file");
        goto done;
    }
    group = H5Gopen2(file, info.matrix_path.c_str(), H5P_DEFAULT);
    if (group < 0) {
        set_error(error, "failed to open selected h5ad matrix group");
        goto done;
    }
    if (!read_matrix_indptr(group, info, indptr, error)) {
        goto done;
    }
    if (header_out != nullptr) {
        mtx::init(header_out);
        header_out->rows = info.rows;
        header_out->cols = info.cols;
        header_out->nnz_file = info.nnz;
        header_out->nnz_loaded = info.nnz;
        header_out->row_sorted = info.csr ? 1 : 0;
    }
    ok = true;

done:
    if (group >= 0) H5Gclose(group);
    if (file >= 0) H5Fclose(file);
    if (!ok) indptr->clear();
    return ok;
}

inline bool read_indices_range(const char *path,
                               const char *matrix_source,
                               std::uint64_t offset,
                               std::uint64_t count,
                               std::vector<std::uint64_t> *indices,
                               std::string *error) {
    hid_t file = (hid_t) -1;
    hid_t group = (hid_t) -1;
    selected_matrix_info info;
    bool ok = false;

    if (indices == nullptr) return false;
    indices->clear();
    if (!probe_selected_matrix(path, matrix_source, &info, error)) return false;
    file = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0) {
        set_error(error, "failed to open h5ad file");
        goto done;
    }
    group = H5Gopen2(file, info.matrix_path.c_str(), H5P_DEFAULT);
    if (group < 0) {
        set_error(error, "failed to open selected h5ad matrix group");
        goto done;
    }
    if (!read_dataset_u64_range(group, "indices", offset, count, indices)) {
        set_error(error, "failed to read h5ad csr indices");
        goto done;
    }
    ok = true;

done:
    if (group >= 0) H5Gclose(group);
    if (file >= 0) H5Fclose(file);
    if (!ok) indices->clear();
    return ok;
}

inline bool read_values_range(const char *path,
                              const char *matrix_source,
                              std::uint64_t offset,
                              std::uint64_t count,
                              std::vector<double> *values,
                              std::string *error) {
    hid_t file = (hid_t) -1;
    hid_t group = (hid_t) -1;
    selected_matrix_info info;
    bool ok = false;

    if (values == nullptr) return false;
    values->clear();
    if (!probe_selected_matrix(path, matrix_source, &info, error)) return false;
    file = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0) {
        set_error(error, "failed to open h5ad file");
        goto done;
    }
    group = H5Gopen2(file, info.matrix_path.c_str(), H5P_DEFAULT);
    if (group < 0) {
        set_error(error, "failed to open selected h5ad matrix group");
        goto done;
    }
    if (!read_dataset_double_range(group, "data", offset, count, values)) {
        set_error(error, "failed to read h5ad csr values");
        goto done;
    }
    ok = true;

done:
    if (group >= 0) H5Gclose(group);
    if (file >= 0) H5Fclose(file);
    if (!ok) values->clear();
    return ok;
}

inline bool load_part_window_coo(const char *path,
                                 const char *matrix_source,
                                 const mtx::header *h,
                                 const unsigned long *row_offsets,
                                 const unsigned long *part_nnz,
                                 unsigned long num_parts,
                                 unsigned long part_begin,
                                 unsigned long part_end,
                                 sharded<sparse::coo> *out,
                                 std::string *error) {
    hid_t file = (hid_t) -1;
    hid_t group = (hid_t) -1;
    std::vector<std::uint64_t> indptr;
    std::vector<std::uint64_t> indices;
    std::vector<double> values;
    std::vector<unsigned long> write_ptr;
    selected_matrix_info info;
    mtx::header local;
    const unsigned long row_begin = row_offsets[part_begin];
    const unsigned long row_end = row_offsets[part_end];
    const std::uint64_t chunk_elems = (std::uint64_t) 1u << 20u;
    std::uint64_t chunk_begin = 0u;
    std::uint64_t chunk_end = 0u;
    bool ok = false;

    if (out == nullptr || h == nullptr) return false;
    clear(out);
    init(out);
    if (!probe_selected_matrix(path, matrix_source, &info, error)) return false;
    mtx::init(&local);
    local.rows = info.rows;
    local.cols = info.cols;
    local.nnz_file = info.nnz;
    local.nnz_loaded = info.nnz;
    local.row_sorted = info.csr ? 1 : 0;
    if (!mtx::allocate_part_window_coo(&local, row_offsets, part_nnz, num_parts, part_begin, part_end, out)) return false;

    file = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0) {
        set_error(error, "failed to open h5ad file");
        goto done;
    }
    group = H5Gopen2(file, info.matrix_path.c_str(), H5P_DEFAULT);
    if (group < 0) {
        set_error(error, "failed to open selected h5ad matrix group");
        goto done;
    }
    write_ptr.assign(out->num_partitions, 0ul);
    if (info.csr) {
        if (!read_dataset_u64_range(group, "indptr", (std::uint64_t) row_begin, (std::uint64_t) (row_end - row_begin + 1ul), &indptr)) {
            set_error(error, "failed to read h5ad csr indptr window");
            goto done;
        }
        if (indptr.empty()) goto done;
        {
            const std::uint64_t nnz_begin = indptr.front();
            const std::uint64_t nnz_count = indptr.back() - indptr.front();
            for (std::uint64_t &value : indptr) value -= nnz_begin;
            if (!read_dataset_u64_range(group, "indices", nnz_begin, nnz_count, &indices)
                || !read_dataset_double_range(group, "data", nnz_begin, nnz_count, &values)
                || indices.size() != values.size()) {
                set_error(error, "failed to read h5ad csr payload window");
                goto done;
            }
        }

        for (unsigned long row = row_begin; row < row_end; ++row) {
            const unsigned long global_part = find_offset_span(row, row_offsets, num_parts);
            const unsigned long local_part = global_part - part_begin;
            const std::uint64_t local_row_begin = indptr[(std::size_t) (row - row_begin)];
            const std::uint64_t local_row_end = indptr[(std::size_t) (row - row_begin + 1ul)];
            for (std::uint64_t cursor = local_row_begin; cursor < local_row_end; ++cursor) {
                const unsigned long idx = write_ptr[(std::size_t) local_part]++;
                out->parts[local_part]->rowIdx[idx] = (unsigned int) (row - row_offsets[global_part]);
                out->parts[local_part]->colIdx[idx] = (unsigned int) indices[(std::size_t) cursor];
                out->parts[local_part]->val[idx] = __float2half((float) values[(std::size_t) cursor]);
            }
        }
    } else {
        if (!read_matrix_indptr(group, info, &indptr, error)) goto done;
        for (unsigned long col = 0; col < info.cols; ++col) {
            const std::uint64_t col_begin = indptr[(std::size_t) col];
            const std::uint64_t col_end = indptr[(std::size_t) col + 1u];
            for (std::uint64_t cursor = col_begin; cursor < col_end; ++cursor) {
                if (cursor < chunk_begin || cursor >= chunk_end) {
                    chunk_begin = cursor;
                    chunk_end = std::min<std::uint64_t>(chunk_begin + chunk_elems, indptr.back());
                    if (!read_dataset_u64_range(group, "indices", chunk_begin, chunk_end - chunk_begin, &indices)
                        || !read_dataset_double_range(group, "data", chunk_begin, chunk_end - chunk_begin, &values)
                        || indices.size() != values.size()) {
                        set_error(error, "failed to read h5ad csc payload window");
                        goto done;
                    }
                }
                const unsigned long row = (unsigned long) indices[(std::size_t) (cursor - chunk_begin)];
                if (row < row_begin || row >= row_end) continue;
                const unsigned long global_part = find_offset_span(row, row_offsets, num_parts);
                const unsigned long local_part = global_part - part_begin;
                const unsigned long idx = write_ptr[(std::size_t) local_part]++;
                out->parts[local_part]->rowIdx[idx] = (unsigned int) (row - row_offsets[global_part]);
                out->parts[local_part]->colIdx[idx] = (unsigned int) col;
                out->parts[local_part]->val[idx] = __float2half((float) values[(std::size_t) (cursor - chunk_begin)]);
            }
        }
    }
    ok = true;

done:
    if (group >= 0) H5Gclose(group);
    if (file >= 0) H5Fclose(file);
    if (!ok) clear(out);
    return ok;
}

inline bool load_part_window_compressed(const char *path,
                                        const char *matrix_source,
                                        const mtx::header *h,
                                        const unsigned long *row_offsets,
                                        const unsigned long *part_nnz,
                                        unsigned long num_parts,
                                        unsigned long part_begin,
                                        unsigned long part_end,
                                        sharded<sparse::compressed> *out,
                                        std::string *error) {
    hid_t file = (hid_t) -1;
    hid_t group = (hid_t) -1;
    std::vector<std::uint64_t> indptr;
    std::vector<std::uint64_t> indices;
    std::vector<double> values;
    selected_matrix_info info;
    mtx::header local;
    bool ok = false;

    if (out == nullptr || h == nullptr || part_begin >= part_end || part_end > num_parts) return false;
    clear(out);
    init(out);
    if (!probe_selected_matrix(path, matrix_source, &info, error)) return false;

    mtx::init(&local);
    local.rows = info.rows;
    local.cols = info.cols;
    local.nnz_file = info.nnz;
    local.nnz_loaded = info.nnz;
    local.row_sorted = info.csr ? 1 : 0;
    if (!reserve_partitions(out, part_end - part_begin)) return false;
    out->num_partitions = part_end - part_begin;
    out->cols = info.cols;

    for (unsigned long global_part = part_begin; global_part < part_end; ++global_part) {
        const unsigned long local_part = global_part - part_begin;
        const unsigned long rows = row_offsets[global_part + 1ul] - row_offsets[global_part];
        sparse::compressed *part = new sparse::compressed;
        sparse::init(part,
                     (::cellshard::types::dim_t) rows,
                     (::cellshard::types::dim_t) info.cols,
                     (::cellshard::types::nnz_t) part_nnz[global_part],
                     info.csr ? sparse::compressed_by_row : sparse::compressed_by_col);
        if (!sparse::allocate(part)) {
            delete part;
            clear(out);
            return false;
        }
        out->parts[local_part] = part;
        out->partition_rows[local_part] = rows;
        out->partition_nnz[local_part] = part_nnz[global_part];
        out->partition_aux[local_part] = 0u;
    }
    rebuild_partition_offsets(out);
    if (!set_shards_to_partitions(out)) {
        clear(out);
        return false;
    }

    file = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0) {
        set_error(error, "failed to open h5ad file");
        goto done;
    }
    group = H5Gopen2(file, info.matrix_path.c_str(), H5P_DEFAULT);
    if (group < 0) {
        set_error(error, "failed to open selected h5ad matrix group");
        goto done;
    }

    if (info.csr) {
        const unsigned long row_begin = row_offsets[part_begin];
        const unsigned long row_end = row_offsets[part_end];
        if (!read_dataset_u64_range(group, "indptr", (std::uint64_t) row_begin, (std::uint64_t) (row_end - row_begin + 1ul), &indptr)) {
            set_error(error, "failed to read h5ad csr indptr window");
            goto done;
        }
        if (indptr.empty()) goto done;
        {
            const std::uint64_t nnz_begin = indptr.front();
            const std::uint64_t nnz_count = indptr.back() - indptr.front();
            for (std::uint64_t &value : indptr) value -= nnz_begin;
            if (!read_dataset_u64_range(group, "indices", nnz_begin, nnz_count, &indices)
                || !read_dataset_double_range(group, "data", nnz_begin, nnz_count, &values)
                || indices.size() != values.size()) {
                set_error(error, "failed to read h5ad csr payload window");
                goto done;
            }
        }
        for (unsigned long global_part = part_begin; global_part < part_end; ++global_part) {
            const unsigned long local_part = global_part - part_begin;
            sparse::compressed *part = out->parts[local_part];
            const unsigned long local_row_begin = row_offsets[global_part] - row_begin;
            const unsigned long local_row_end = row_offsets[global_part + 1ul] - row_begin;
            const std::uint64_t nnz_begin = indptr[(std::size_t) local_row_begin];
            const std::uint64_t nnz_end = indptr[(std::size_t) local_row_end];
            std::uint64_t cursor = 0u;
            part->majorPtr[0] = 0u;
            for (unsigned long row = local_row_begin; row < local_row_end; ++row) {
                const std::uint64_t row_span = indptr[(std::size_t) row + 1u] - indptr[(std::size_t) row];
                cursor += row_span;
                part->majorPtr[(row - local_row_begin) + 1ul] = (::cellshard::types::ptr_t) cursor;
            }
            if (nnz_end > nnz_begin) {
                const std::size_t count = (std::size_t) (nnz_end - nnz_begin);
                for (std::size_t i = 0u; i < count; ++i) {
                    part->minorIdx[i] = (::cellshard::types::idx_t) indices[(std::size_t) nnz_begin + i];
                    part->val[i] = __float2half((float) values[(std::size_t) nnz_begin + i]);
                }
            }
        }
        ok = true;
        goto done;
    }

    if (!read_matrix_indptr(group, info, &indptr, error)) goto done;
    for (unsigned long global_part = part_begin; global_part < part_end; ++global_part) {
        const unsigned long local_part = global_part - part_begin;
        sparse::compressed *part = out->parts[local_part];
        if (part->majorPtr != nullptr) std::memset(part->majorPtr, 0, ((std::size_t) info.cols + 1u) * sizeof(::cellshard::types::ptr_t));
    }
    {
        const unsigned long row_begin = row_offsets[part_begin];
        const unsigned long row_end = row_offsets[part_end];
        const std::uint64_t chunk_elems = (std::uint64_t) 1u << 20u;
        std::uint64_t chunk_begin = 0u;
        std::uint64_t chunk_end = 0u;
        for (unsigned long col = 0; col < info.cols; ++col) {
            const std::uint64_t col_begin = indptr[(std::size_t) col];
            const std::uint64_t col_end = indptr[(std::size_t) col + 1u];
            for (std::uint64_t cursor = col_begin; cursor < col_end; ++cursor) {
                if (cursor < chunk_begin || cursor >= chunk_end) {
                    chunk_begin = cursor;
                    chunk_end = std::min<std::uint64_t>(chunk_begin + chunk_elems, indptr.back());
                    if (!read_dataset_u64_range(group, "indices", chunk_begin, chunk_end - chunk_begin, &indices)
                        || !read_dataset_double_range(group, "data", chunk_begin, chunk_end - chunk_begin, &values)
                        || indices.size() != values.size()) {
                        set_error(error, "failed to read h5ad csc payload window");
                        goto done;
                    }
                }
                const unsigned long row = (unsigned long) indices[(std::size_t) (cursor - chunk_begin)];
                if (row < row_begin || row >= row_end) continue;
                const unsigned long global_part = find_offset_span(row, row_offsets, num_parts);
                const unsigned long local_part = global_part - part_begin;
                sparse::compressed *part = out->parts[local_part];
                part->majorPtr[col + 1u] += 1u;
            }
        }
        for (unsigned long global_part = part_begin; global_part < part_end; ++global_part) {
            const unsigned long local_part = global_part - part_begin;
            sparse::compressed *part = out->parts[local_part];
            ::cellshard::types::ptr_t running = 0u;
            for (unsigned long col = 0; col < info.cols; ++col) {
                const ::cellshard::types::ptr_t count = part->majorPtr[col + 1u];
                part->majorPtr[col] = running;
                running += count;
                part->majorPtr[col + 1u] = running;
            }
        }
        std::vector<std::vector<std::uint64_t>> write_ptr(out->num_partitions);
        for (unsigned long global_part = part_begin; global_part < part_end; ++global_part) {
            const unsigned long local_part = global_part - part_begin;
            sparse::compressed *part = out->parts[local_part];
            write_ptr[local_part].resize(info.cols, 0u);
            for (unsigned long col = 0; col < info.cols; ++col) {
                write_ptr[local_part][col] = part->majorPtr[col];
            }
        }
        chunk_begin = 0u;
        chunk_end = 0u;
        for (unsigned long col = 0; col < info.cols; ++col) {
            const std::uint64_t col_begin = indptr[(std::size_t) col];
            const std::uint64_t col_end = indptr[(std::size_t) col + 1u];
            for (std::uint64_t cursor = col_begin; cursor < col_end; ++cursor) {
                if (cursor < chunk_begin || cursor >= chunk_end) {
                    chunk_begin = cursor;
                    chunk_end = std::min<std::uint64_t>(chunk_begin + chunk_elems, indptr.back());
                    if (!read_dataset_u64_range(group, "indices", chunk_begin, chunk_end - chunk_begin, &indices)
                        || !read_dataset_double_range(group, "data", chunk_begin, chunk_end - chunk_begin, &values)
                        || indices.size() != values.size()) {
                        set_error(error, "failed to read h5ad csc payload window");
                        goto done;
                    }
                }
                const unsigned long row = (unsigned long) indices[(std::size_t) (cursor - chunk_begin)];
                if (row < row_begin || row >= row_end) continue;
                const unsigned long global_part = find_offset_span(row, row_offsets, num_parts);
                const unsigned long local_part = global_part - part_begin;
                sparse::compressed *part = out->parts[local_part];
                const unsigned long local_row = row - row_offsets[global_part];
                const std::size_t offset = (std::size_t) write_ptr[local_part][col]++;
                part->minorIdx[offset] = (::cellshard::types::idx_t) local_row;
                part->val[offset] = __float2half((float) values[(std::size_t) (cursor - chunk_begin)]);
            }
        }
    }
    ok = true;

done:
    if (group >= 0) H5Gclose(group);
    if (file >= 0) H5Fclose(file);
    if (!ok) clear(out);
    return ok;
}

} // namespace h5ad
} // namespace ingest
} // namespace cellshard
