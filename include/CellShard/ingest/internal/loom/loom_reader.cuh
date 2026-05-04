#pragma once

#include "../common/barcode_table.cuh"
#include "../common/feature_table.cuh"
#include "../h5ad/h5ad_reader.cuh"
#include "../mtx/mtx_reader.cuh"

#include <CellShard/formats/triplet.cuh>
#include <CellShard/runtime/host/sharded_host.cuh>
#include <CellShard/runtime/layout/sharded.cuh>

#include <hdf5.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace cellshard {
namespace ingest {
namespace loom {

using ::cellshard::clear;
using ::cellshard::find_offset_span;
using ::cellshard::init;
using ::cellshard::sharded;
namespace sparse = ::cellshard::sparse;

struct matrix_info {
    bool available = false;
    bool processed_like = false;
    std::string matrix_path;
    unsigned long rows = 0ul; // CellShard rows: cells.
    unsigned long cols = 0ul; // CellShard columns: features.
    unsigned long nnz = 0ul;
};

inline void set_error(std::string *error, const char *message) {
    if (error != nullptr) *error = message != nullptr ? message : "unknown loom ingest error";
}

inline bool build_matrix_path(const char *matrix_source, std::string *matrix_path, std::string *error) {
    const std::string source = h5ad::trim_copy(matrix_source != nullptr ? matrix_source : "");
    if (matrix_path == nullptr) return false;
    if (source.empty() || source == "matrix") {
        *matrix_path = "/matrix";
        return true;
    }
    if (source.rfind("layer:", 0u) == 0u && source.size() > 6u) {
        *matrix_path = "/layers/" + source.substr(6u);
        return true;
    }
    set_error(error, "unsupported loom matrix_source; expected matrix or layer:<name>");
    return false;
}

inline bool dense_dims(hid_t dset, hsize_t *features, hsize_t *cells, std::string *error) {
    hid_t space = (hid_t) -1;
    hsize_t dims[2] = {0u, 0u};
    bool ok = false;
    if (features == nullptr || cells == nullptr) return false;
    *features = 0u;
    *cells = 0u;
    space = H5Dget_space(dset);
    if (space < 0) goto done;
    if (H5Sget_simple_extent_ndims(space) != 2) {
        set_error(error, "loom matrix must be 2D");
        goto done;
    }
    if (H5Sget_simple_extent_dims(space, dims, nullptr) != 2) goto done;
    *features = dims[0];
    *cells = dims[1];
    ok = true;

done:
    if (space >= 0) H5Sclose(space);
    return ok;
}

inline bool numeric_dataset_supported(hid_t dset) {
    hid_t type = H5Dget_type(dset);
    bool ok = false;
    if (type < 0) return false;
    {
        const H5T_class_t cls = H5Tget_class(type);
        ok = cls == H5T_INTEGER || cls == H5T_FLOAT;
    }
    H5Tclose(type);
    return ok;
}

inline bool read_dense_block_double(hid_t dset,
                                    hsize_t feature_begin,
                                    hsize_t feature_count,
                                    hsize_t cell_begin,
                                    hsize_t cell_count,
                                    std::vector<double> *out) {
    hid_t filespace = (hid_t) -1;
    hid_t memspace = (hid_t) -1;
    hsize_t start[2] = {feature_begin, cell_begin};
    hsize_t count[2] = {feature_count, cell_count};
    bool ok = false;

    if (out == nullptr) return false;
    out->clear();
    if (feature_count == 0u || cell_count == 0u) return true;
    out->resize((std::size_t) feature_count * (std::size_t) cell_count, 0.0);
    filespace = H5Dget_space(dset);
    if (filespace < 0) goto done;
    if (H5Sselect_hyperslab(filespace, H5S_SELECT_SET, start, nullptr, count, nullptr) < 0) goto done;
    memspace = H5Screate_simple(2, count, nullptr);
    if (memspace < 0) goto done;
    if (H5Dread(dset, H5T_NATIVE_DOUBLE, memspace, filespace, H5P_DEFAULT, out->data()) < 0) goto done;
    ok = true;

done:
    if (memspace >= 0) H5Sclose(memspace);
    if (filespace >= 0) H5Sclose(filespace);
    if (!ok) out->clear();
    return ok;
}

inline bool value_allowed(double value, bool allow_processed) {
    if (!std::isfinite(value)) return false;
    if (allow_processed) return true;
    if (value < -1.0e-6) return false;
    return std::fabs(value - std::round(value)) <= 1.0e-4;
}

inline bool scan_dense_nnz(hid_t dset,
                           hsize_t feature_count,
                           hsize_t cell_count,
                           bool allow_processed,
                           unsigned long *row_nnz,
                           unsigned long *nnz_out,
                           bool *processed_like,
                           std::string *error) {
    const hsize_t max_elems = (hsize_t) 1u << 20u;
    const hsize_t cell_block_max = std::min<hsize_t>(cell_count, 1024u);
    const hsize_t cell_step = std::max<hsize_t>(cell_block_max, 1u);
    std::vector<double> block;
    unsigned long nnz = 0ul;
    bool processed = false;

    if (row_nnz == nullptr || nnz_out == nullptr || processed_like == nullptr) return false;
    *nnz_out = 0ul;
    *processed_like = false;
    for (hsize_t cell_begin = 0u; cell_begin < cell_count; cell_begin += cell_step) {
        const hsize_t cells = std::min<hsize_t>(cell_step, cell_count - cell_begin);
        const hsize_t feature_step = std::max<hsize_t>(1u, std::min<hsize_t>(feature_count, max_elems / std::max<hsize_t>(cells, 1u)));
        for (hsize_t feature_begin = 0u; feature_begin < feature_count; feature_begin += feature_step) {
            const hsize_t features = std::min<hsize_t>(feature_step, feature_count - feature_begin);
            if (!read_dense_block_double(dset, feature_begin, features, cell_begin, cells, &block)) {
                set_error(error, "failed to read loom dense matrix chunk");
                return false;
            }
            for (hsize_t f = 0u; f < features; ++f) {
                for (hsize_t c = 0u; c < cells; ++c) {
                    const double value = block[(std::size_t) f * (std::size_t) cells + (std::size_t) c];
                    if (!value_allowed(value, true)) {
                        set_error(error, "loom dense matrix contains non-finite values");
                        return false;
                    }
                    if (!value_allowed(value, false)) processed = true;
                    if (!allow_processed && processed) {
                        set_error(error, "loom dense matrix contains processed-looking non-count values");
                        return false;
                    }
                    if (value != 0.0) {
                        const hsize_t cell = cell_begin + c;
                        if (cell > (hsize_t) std::numeric_limits<unsigned long>::max()) return false;
                        ++row_nnz[(std::size_t) cell];
                        ++nnz;
                    }
                }
            }
        }
    }
    *nnz_out = nnz;
    *processed_like = processed;
    return true;
}

inline bool probe_matrix(const char *path,
                         const char *matrix_source,
                         bool allow_processed,
                         matrix_info *out,
                         std::string *error) {
    hid_t file = (hid_t) -1;
    hid_t dset = (hid_t) -1;
    std::string matrix_path;
    hsize_t features = 0u;
    hsize_t cells = 0u;
    std::vector<double> sample;
    bool processed = false;
    bool ok = false;

    if (out == nullptr) return false;
    *out = matrix_info{};
    if (!build_matrix_path(matrix_source, &matrix_path, error)) return false;
    file = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0) {
        set_error(error, "failed to open loom file");
        goto done;
    }
    dset = H5Dopen2(file, matrix_path.c_str(), H5P_DEFAULT);
    if (dset < 0) {
        set_error(error, "selected loom matrix is missing");
        goto done;
    }
    if (!numeric_dataset_supported(dset)) {
        set_error(error, "selected loom matrix is not a supported numeric dataset");
        goto done;
    }
    if (!dense_dims(dset, &features, &cells, error)) goto done;
    if (features > (hsize_t) std::numeric_limits<unsigned long>::max()
        || cells > (hsize_t) std::numeric_limits<unsigned long>::max()) {
        set_error(error, "loom matrix dimensions exceed ingest limits");
        goto done;
    }
    {
        const hsize_t sample_features = std::min<hsize_t>(features, 64u);
        const hsize_t sample_cells = std::min<hsize_t>(cells, 64u);
        if (!read_dense_block_double(dset, 0u, sample_features, 0u, sample_cells, &sample)) {
            set_error(error, "failed to sample loom matrix");
            goto done;
        }
        for (double value : sample) {
            if (!value_allowed(value, true)) {
                set_error(error, "loom dense matrix contains non-finite values");
                goto done;
            }
            if (!value_allowed(value, false)) processed = true;
        }
        if (processed && !allow_processed) {
            set_error(error, "loom dense matrix contains processed-looking non-count values");
            goto done;
        }
    }
    out->available = true;
    out->processed_like = processed;
    out->matrix_path = matrix_path;
    out->rows = (unsigned long) cells;
    out->cols = (unsigned long) features;
    ok = true;

done:
    if (dset >= 0) H5Dclose(dset);
    if (file >= 0) H5Fclose(file);
    return ok;
}

inline bool scan_row_nnz(const char *path,
                         const char *matrix_source,
                         bool allow_processed,
                         mtx::header *h,
                         unsigned long **row_nnz_out,
                         std::string *error) {
    hid_t file = (hid_t) -1;
    hid_t dset = (hid_t) -1;
    matrix_info info;
    unsigned long *row_nnz = nullptr;
    unsigned long nnz = 0ul;
    bool processed = false;
    bool ok = false;

    if (h == nullptr || row_nnz_out == nullptr) return false;
    *row_nnz_out = nullptr;
    mtx::init(h);
    if (!probe_matrix(path, matrix_source, allow_processed, &info, error)) return false;
    file = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0) {
        set_error(error, "failed to reopen loom file");
        goto done;
    }
    dset = H5Dopen2(file, info.matrix_path.c_str(), H5P_DEFAULT);
    if (dset < 0) {
        set_error(error, "failed to reopen selected loom matrix");
        goto done;
    }
    row_nnz = (unsigned long *) std::calloc((std::size_t) info.rows, sizeof(unsigned long));
    if (info.rows != 0ul && row_nnz == nullptr) goto done;
    if (!scan_dense_nnz(dset, (hsize_t) info.cols, (hsize_t) info.rows, allow_processed, row_nnz, &nnz, &processed, error)) goto done;
    h->rows = info.rows;
    h->cols = info.cols;
    h->nnz_file = nnz;
    h->nnz_loaded = nnz;
    h->row_sorted = 0;
    *row_nnz_out = row_nnz;
    row_nnz = nullptr;
    ok = true;

done:
    if (dset >= 0) H5Dclose(dset);
    if (file >= 0) H5Fclose(file);
    std::free(row_nnz);
    return ok;
}

inline bool read_text_attr(hid_t group,
                           const char *name,
                           unsigned long expected_rows,
                           common::text_column *out) {
    if (!h5ad::read_dataset_string_column(group, name, out)) return false;
    return out->count == expected_rows;
}

inline bool select_text_attr(hid_t group,
                             const char *const *candidates,
                             unsigned int candidate_count,
                             unsigned long expected_rows,
                             common::text_column *out) {
    for (unsigned int i = 0u; i < candidate_count; ++i) {
        if (read_text_attr(group, candidates[i], expected_rows, out)) return true;
    }
    return false;
}

inline bool select_first_text_attr(hid_t group,
                                   unsigned long expected_rows,
                                   common::text_column *out) {
    std::vector<std::string> names;
    if (!h5ad::list_child_names(group, &names)) return false;
    for (const std::string &name : names) {
        if (read_text_attr(group, name.c_str(), expected_rows, out)) return true;
    }
    return false;
}

inline bool load_barcodes(const char *path,
                          const char *matrix_source,
                          common::barcode_table *barcodes,
                          std::string *error) {
    static const char *const candidates[] = {"CellID", "Barcode", "barcode"};
    hid_t file = (hid_t) -1;
    hid_t group = (hid_t) -1;
    common::text_column labels;
    matrix_info info;
    bool ok = false;

    if (barcodes == nullptr) return false;
    common::clear(barcodes);
    common::init(barcodes);
    common::init(&labels);
    if (!probe_matrix(path, matrix_source, true, &info, error)) return false;
    file = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0) {
        set_error(error, "failed to open loom file");
        goto done;
    }
    group = H5Gopen2(file, "/col_attrs", H5P_DEFAULT);
    if (group < 0
        || (!select_text_attr(group, candidates, 3u, info.rows, &labels)
            && !select_first_text_attr(group, info.rows, &labels))) {
        set_error(error, "failed to find usable loom cell labels in /col_attrs");
        goto done;
    }
    for (unsigned int i = 0u; i < labels.count; ++i) {
        const char *label = common::get(&labels, i);
        if (label == nullptr || !common::append(barcodes, label, std::strlen(label))) goto done;
    }
    ok = true;

done:
    common::clear(&labels);
    if (group >= 0) H5Gclose(group);
    if (file >= 0) H5Fclose(file);
    if (!ok) {
        common::clear(barcodes);
        common::init(barcodes);
    }
    return ok;
}

inline bool load_feature_table(const char *path,
                               const char *matrix_source,
                               common::feature_table *features,
                               std::string *error) {
    static const char *const id_candidates[] = {"Accession", "Gene", "id"};
    static const char *const name_candidates[] = {"Name", "Gene", "gene_short_name"};
    hid_t file = (hid_t) -1;
    hid_t group = (hid_t) -1;
    common::text_column ids;
    common::text_column names;
    matrix_info info;
    const char *feature_type = "gene";
    bool ok = false;

    if (features == nullptr) return false;
    common::clear(features);
    common::init(features);
    common::init(&ids);
    common::init(&names);
    if (!probe_matrix(path, matrix_source, true, &info, error)) return false;
    file = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0) {
        set_error(error, "failed to open loom file");
        goto done;
    }
    group = H5Gopen2(file, "/row_attrs", H5P_DEFAULT);
    if (group < 0
        || !select_text_attr(group, id_candidates, 3u, info.cols, &ids)
        || !select_text_attr(group, name_candidates, 3u, info.cols, &names)) {
        set_error(error, "failed to find usable loom feature id/name labels in /row_attrs");
        goto done;
    }
    for (unsigned int i = 0u; i < ids.count; ++i) {
        const char *id = common::get(&ids, i);
        const char *name = common::get(&names, i);
        if (!common::append(features,
                            id != nullptr ? id : "", std::strlen(id != nullptr ? id : ""),
                            name != nullptr ? name : "", std::strlen(name != nullptr ? name : ""),
                            feature_type, std::strlen(feature_type))) goto done;
    }
    ok = true;

done:
    common::clear(&ids);
    common::clear(&names);
    if (group >= 0) H5Gclose(group);
    if (file >= 0) H5Fclose(file);
    if (!ok) {
        common::clear(features);
        common::init(features);
    }
    return ok;
}

inline bool load_part_window_coo(const char *path,
                                 const char *matrix_source,
                                 bool allow_processed,
                                 const mtx::header *h,
                                 const unsigned long *row_offsets,
                                 const unsigned long *part_nnz,
                                 unsigned long num_parts,
                                 unsigned long part_begin,
                                 unsigned long part_end,
                                 sharded<sparse::coo> *out,
                                 std::string *error) {
    hid_t file = (hid_t) -1;
    hid_t dset = (hid_t) -1;
    matrix_info info;
    std::vector<double> block;
    std::vector<unsigned long> write_ptr;
    const hsize_t max_elems = (hsize_t) 1u << 20u;
    bool ok = false;

    if (h == nullptr || out == nullptr) return false;
    if (!probe_matrix(path, matrix_source, allow_processed, &info, error)) return false;
    if (h->rows != info.rows || h->cols != info.cols) {
        set_error(error, "loom header does not match matrix dimensions");
        return false;
    }
    if (!mtx::allocate_part_window_coo(h, row_offsets, part_nnz, num_parts, part_begin, part_end, out)) return false;
    file = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0) {
        set_error(error, "failed to open loom file");
        goto done;
    }
    dset = H5Dopen2(file, info.matrix_path.c_str(), H5P_DEFAULT);
    if (dset < 0) {
        set_error(error, "failed to open selected loom matrix");
        goto done;
    }
    write_ptr.assign(out->num_partitions, 0ul);
    {
        const unsigned long row_begin = row_offsets[part_begin];
        const unsigned long row_end = row_offsets[part_end];
        const hsize_t window_cells = (hsize_t) (row_end - row_begin);
        const hsize_t cell_step = std::max<hsize_t>(1u, std::min<hsize_t>(window_cells, 1024u));
        for (hsize_t cell_offset = 0u; cell_offset < window_cells; cell_offset += cell_step) {
            const hsize_t cells = std::min<hsize_t>(cell_step, window_cells - cell_offset);
            const hsize_t feature_step = std::max<hsize_t>(1u, std::min<hsize_t>((hsize_t) info.cols, max_elems / std::max<hsize_t>(cells, 1u)));
            for (hsize_t feature_begin = 0u; feature_begin < (hsize_t) info.cols; feature_begin += feature_step) {
                const hsize_t features = std::min<hsize_t>(feature_step, (hsize_t) info.cols - feature_begin);
                if (!read_dense_block_double(dset, feature_begin, features, (hsize_t) row_begin + cell_offset, cells, &block)) {
                    set_error(error, "failed to read loom dense matrix window");
                    goto done;
                }
                for (hsize_t f = 0u; f < features; ++f) {
                    for (hsize_t c = 0u; c < cells; ++c) {
                        const double value = block[(std::size_t) f * (std::size_t) cells + (std::size_t) c];
                        if (!value_allowed(value, allow_processed)) {
                            set_error(error, allow_processed
                                ? "loom dense matrix contains non-finite values"
                                : "loom dense matrix contains processed-looking non-count values");
                            goto done;
                        }
                        if (value == 0.0) continue;
                        const unsigned long global_row = row_begin + (unsigned long) cell_offset + (unsigned long) c;
                        const unsigned long global_part = find_offset_span(global_row, row_offsets, num_parts);
                        const unsigned long local_part = global_part - part_begin;
                        if (global_part < part_begin || global_part >= part_end || local_part >= out->num_partitions) goto done;
                        {
                            sparse::coo *part = out->parts[local_part];
                            const unsigned long idx = write_ptr[(std::size_t) local_part]++;
                            if (idx >= part->nnz) goto done;
                            part->rowIdx[idx] = (::cellshard::types::idx_t) (global_row - row_offsets[global_part]);
                            part->colIdx[idx] = (::cellshard::types::idx_t) (feature_begin + f);
                            part->val[idx] = __float2half((float) value);
                        }
                    }
                }
            }
        }
    }
    for (unsigned long p = 0; p < out->num_partitions; ++p) {
        if (write_ptr[(std::size_t) p] != out->parts[p]->nnz) goto done;
    }
    ok = true;

done:
    if (dset >= 0) H5Dclose(dset);
    if (file >= 0) H5Fclose(file);
    if (!ok) clear(out);
    return ok;
}

} // namespace loom
} // namespace ingest
} // namespace cellshard
