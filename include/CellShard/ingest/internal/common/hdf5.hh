#pragma once

#include "text_column.cuh"
#include <hdf5.h>
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

// Raw HDF5 utilities shared by the 10x and Loom format readers. AnnData/H5AD
// decoding is exclusively provided by the compiled anndatapp module dependency.
namespace cellshard::ingest::hdf5 {
inline hid_t open_optional_group(hid_t parent, const char *path) {
    hid_t group = (hid_t) -1;
    H5E_BEGIN_TRY {
        group = H5Gopen2(parent, path, H5P_DEFAULT);
    } H5E_END_TRY;
    return group;
}

inline std::string trim_copy(std::string value) {
    std::size_t begin = 0u;
    std::size_t end = value.size();
    while (begin < end && std::isspace((unsigned char) value[begin])) ++begin;
    while (end > begin && std::isspace((unsigned char) value[end - 1u])) --end;
    return value.substr(begin, end - begin);
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

} // namespace cellshard::ingest::hdf5
