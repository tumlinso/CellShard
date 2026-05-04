#pragma once

#include "../common/barcode_table.cuh"
#include "../common/feature_table.cuh"
#include "../h5ad/h5ad_reader.cuh"
#include "../mtx/mtx_reader.cuh"

#include <CellShard/formats/compressed.cuh>
#include <CellShard/formats/triplet.cuh>
#include <CellShard/runtime/host/sharded_host.cuh>
#include <CellShard/runtime/layout/sharded.cuh>

#include <hdf5.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace cellshard {
namespace ingest {
namespace tenx_h5 {

using ::cellshard::clear;
using ::cellshard::find_offset_span;
using ::cellshard::init;
using ::cellshard::reserve_partitions;
using ::cellshard::set_shards_to_partitions;
using ::cellshard::sharded;
namespace sparse = ::cellshard::sparse;

struct matrix_info {
    bool available = false;
    unsigned long rows = 0ul; // CellShard rows: barcodes/cells.
    unsigned long cols = 0ul; // CellShard columns: features.
    unsigned long nnz = 0ul;
};

inline void set_error(std::string *error, const char *message) {
    if (error != nullptr) *error = message != nullptr ? message : "unknown 10x h5 ingest error";
}

inline bool read_shape(hid_t matrix, std::uint64_t *features, std::uint64_t *barcodes, std::string *error) {
    std::vector<std::uint64_t> shape;
    if (features == nullptr || barcodes == nullptr) return false;
    *features = 0u;
    *barcodes = 0u;
    if (!h5ad::read_dataset_u64_range(matrix, "shape", 0u, 2u, &shape) || shape.size() != 2u) {
        set_error(error, "10x h5 matrix is missing a two-entry shape dataset");
        return false;
    }
    *features = shape[0];
    *barcodes = shape[1];
    return true;
}

inline bool dataset_len(hid_t parent, const char *name, hsize_t *length) {
    hid_t dset = (hid_t) -1;
    bool ok = false;
    if (length == nullptr) return false;
    *length = 0u;
    dset = H5Dopen2(parent, name, H5P_DEFAULT);
    if (dset < 0) goto done;
    ok = h5ad::dataset_length(dset, length);
done:
    if (dset >= 0) H5Dclose(dset);
    return ok;
}

inline bool validate_indptr(const std::vector<std::uint64_t> &indptr,
                            std::uint64_t expected_nnz,
                            std::string *error) {
    if (indptr.empty() || indptr.front() != 0u) {
        set_error(error, "10x h5 indptr must start at zero");
        return false;
    }
    for (std::size_t i = 1u; i < indptr.size(); ++i) {
        if (indptr[i] < indptr[i - 1u]) {
            set_error(error, "10x h5 indptr is not monotonic");
            return false;
        }
    }
    if (indptr.back() != expected_nnz) {
        set_error(error, "10x h5 indptr terminal value does not match data length");
        return false;
    }
    return true;
}

inline bool validate_count_values(hid_t matrix, std::uint64_t nnz, std::string *error) {
    std::vector<double> values;
    const std::uint64_t sample_count = std::min<std::uint64_t>(nnz, 1024u);
    if (sample_count == 0u) return true;
    if (!h5ad::read_dataset_double_range(matrix, "data", 0u, sample_count, &values)) {
        set_error(error, "failed to sample 10x h5 matrix values");
        return false;
    }
    for (double value : values) {
        if (!std::isfinite(value) || value < -1.0e-6 || std::fabs(value - std::round(value)) > 1.0e-4) {
            set_error(error, "10x h5 matrix values are not count-like");
            return false;
        }
    }
    return true;
}

inline bool probe_matrix(const char *path, matrix_info *out, std::string *error) {
    hid_t file = (hid_t) -1;
    hid_t matrix = (hid_t) -1;
    std::uint64_t feature_count = 0u;
    std::uint64_t barcode_count = 0u;
    hsize_t data_len = 0u;
    hsize_t indices_len = 0u;
    hsize_t indptr_len = 0u;
    std::vector<std::uint64_t> indptr;
    bool ok = false;

    if (out == nullptr) return false;
    *out = matrix_info{};
    file = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0) {
        set_error(error, "failed to open 10x h5 file");
        goto done;
    }
    matrix = h5ad::open_optional_group(file, "/matrix");
    if (matrix < 0) {
        set_error(error, "10x h5 file is missing /matrix");
        goto done;
    }
    if (!read_shape(matrix, &feature_count, &barcode_count, error)) goto done;
    if (feature_count > (std::uint64_t) std::numeric_limits<unsigned long>::max()
        || barcode_count > (std::uint64_t) std::numeric_limits<unsigned long>::max()) {
        set_error(error, "10x h5 matrix dimensions exceed ingest limits");
        goto done;
    }
    if (!dataset_len(matrix, "data", &data_len)
        || !dataset_len(matrix, "indices", &indices_len)
        || !dataset_len(matrix, "indptr", &indptr_len)) {
        set_error(error, "10x h5 matrix is missing data, indices, or indptr");
        goto done;
    }
    if ((std::uint64_t) data_len != (std::uint64_t) indices_len
        || (std::uint64_t) indptr_len != barcode_count + 1u
        || (std::uint64_t) data_len > (std::uint64_t) std::numeric_limits<unsigned long>::max()) {
        set_error(error, "10x h5 sparse dataset lengths are inconsistent");
        goto done;
    }
    if (!h5ad::read_dataset_u64_range(matrix, "indptr", 0u, barcode_count + 1u, &indptr)
        || !validate_indptr(indptr, (std::uint64_t) data_len, error)
        || !validate_count_values(matrix, (std::uint64_t) data_len, error)) {
        goto done;
    }
    out->available = true;
    out->rows = (unsigned long) barcode_count;
    out->cols = (unsigned long) feature_count;
    out->nnz = (unsigned long) data_len;
    ok = true;

done:
    if (matrix >= 0) H5Gclose(matrix);
    if (file >= 0) H5Fclose(file);
    return ok;
}

inline bool scan_row_nnz(const char *path,
                         mtx::header *h,
                         unsigned long **row_nnz_out,
                         std::string *error) {
    hid_t file = (hid_t) -1;
    hid_t matrix = (hid_t) -1;
    matrix_info info;
    std::vector<std::uint64_t> indptr;
    std::vector<std::uint64_t> indices;
    std::vector<double> values;
    unsigned long *row_nnz = nullptr;
    const std::uint64_t chunk_elems = (std::uint64_t) 1u << 20u;
    std::uint64_t chunk_begin = 0u;
    std::uint64_t chunk_end = 0u;
    bool ok = false;

    if (h == nullptr || row_nnz_out == nullptr) return false;
    *row_nnz_out = nullptr;
    mtx::init(h);
    if (!probe_matrix(path, &info, error)) return false;
    file = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0) {
        set_error(error, "failed to reopen 10x h5 file");
        goto done;
    }
    matrix = H5Gopen2(file, "/matrix", H5P_DEFAULT);
    if (matrix < 0) {
        set_error(error, "failed to reopen 10x h5 /matrix group");
        goto done;
    }
    if (!h5ad::read_dataset_u64_range(matrix, "indptr", 0u, (std::uint64_t) info.rows + 1u, &indptr)
        || !validate_indptr(indptr, (std::uint64_t) info.nnz, error)) {
        goto done;
    }
    row_nnz = (unsigned long *) std::calloc((std::size_t) info.rows, sizeof(unsigned long));
    if (info.rows != 0ul && row_nnz == nullptr) goto done;
    for (unsigned long row = 0; row < info.rows; ++row) {
        const std::uint64_t begin = indptr[(std::size_t) row];
        const std::uint64_t end = indptr[(std::size_t) row + 1u];
        if (end - begin > (std::uint64_t) std::numeric_limits<unsigned long>::max()) goto done;
        row_nnz[row] = (unsigned long) (end - begin);
        for (std::uint64_t cursor = begin; cursor < end; ++cursor) {
            if (cursor < chunk_begin || cursor >= chunk_end) {
                chunk_begin = cursor;
                chunk_end = std::min<std::uint64_t>(chunk_begin + chunk_elems, indptr.back());
                if (!h5ad::read_dataset_u64_range(matrix, "indices", chunk_begin, chunk_end - chunk_begin, &indices)
                    || !h5ad::read_dataset_double_range(matrix, "data", chunk_begin, chunk_end - chunk_begin, &values)
                    || indices.size() != values.size()) {
                    set_error(error, "failed to read 10x h5 sparse payload");
                    goto done;
                }
            }
            if (indices[(std::size_t) (cursor - chunk_begin)] >= info.cols) {
                set_error(error, "10x h5 feature index is out of range");
                goto done;
            }
            {
                const double value = values[(std::size_t) (cursor - chunk_begin)];
                if (!std::isfinite(value) || value < -1.0e-6 || std::fabs(value - std::round(value)) > 1.0e-4) {
                    set_error(error, "10x h5 matrix values are not count-like");
                    goto done;
                }
            }
        }
    }
    h->rows = info.rows;
    h->cols = info.cols;
    h->nnz_file = info.nnz;
    h->nnz_loaded = info.nnz;
    h->row_sorted = 1;
    *row_nnz_out = row_nnz;
    row_nnz = nullptr;
    ok = true;

done:
    if (matrix >= 0) H5Gclose(matrix);
    if (file >= 0) H5Fclose(file);
    std::free(row_nnz);
    return ok;
}

inline bool load_barcodes(const char *path, common::barcode_table *barcodes, std::string *error) {
    hid_t file = (hid_t) -1;
    hid_t matrix = (hid_t) -1;
    common::text_column values;
    bool ok = false;

    if (barcodes == nullptr) return false;
    common::clear(barcodes);
    common::init(barcodes);
    common::init(&values);
    file = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0) {
        set_error(error, "failed to open 10x h5 file");
        goto done;
    }
    matrix = H5Gopen2(file, "/matrix", H5P_DEFAULT);
    if (matrix < 0 || !h5ad::read_dataset_string_column(matrix, "barcodes", &values)) {
        set_error(error, "failed to read 10x h5 barcodes");
        goto done;
    }
    for (unsigned int i = 0u; i < values.count; ++i) {
        const char *value = common::get(&values, i);
        if (value == nullptr || !common::append(barcodes, value, std::strlen(value))) goto done;
    }
    ok = true;

done:
    common::clear(&values);
    if (matrix >= 0) H5Gclose(matrix);
    if (file >= 0) H5Fclose(file);
    if (!ok) {
        common::clear(barcodes);
        common::init(barcodes);
    }
    return ok;
}

inline bool load_feature_table(const char *path, common::feature_table *features, std::string *error) {
    hid_t file = (hid_t) -1;
    hid_t group = (hid_t) -1;
    common::text_column ids;
    common::text_column names;
    common::text_column types;
    bool ok = false;

    if (features == nullptr) return false;
    common::clear(features);
    common::init(features);
    common::init(&ids);
    common::init(&names);
    common::init(&types);
    file = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0) {
        set_error(error, "failed to open 10x h5 file");
        goto done;
    }
    group = H5Gopen2(file, "/matrix/features", H5P_DEFAULT);
    if (group < 0
        || !h5ad::read_dataset_string_column(group, "id", &ids)
        || !h5ad::read_dataset_string_column(group, "name", &names)
        || !h5ad::read_dataset_string_column(group, "feature_type", &types)
        || ids.count != names.count
        || ids.count != types.count) {
        set_error(error, "failed to read 10x h5 feature id, name, and feature_type columns");
        goto done;
    }
    for (unsigned int i = 0u; i < ids.count; ++i) {
        const char *id = common::get(&ids, i);
        const char *name = common::get(&names, i);
        const char *type = common::get(&types, i);
        if (!common::append(features,
                            id != nullptr ? id : "", std::strlen(id != nullptr ? id : ""),
                            name != nullptr ? name : "", std::strlen(name != nullptr ? name : ""),
                            type != nullptr ? type : "", std::strlen(type != nullptr ? type : ""))) goto done;
    }
    ok = true;

done:
    common::clear(&ids);
    common::clear(&names);
    common::clear(&types);
    if (group >= 0) H5Gclose(group);
    if (file >= 0) H5Fclose(file);
    if (!ok) {
        common::clear(features);
        common::init(features);
    }
    return ok;
}

inline bool allocate_compressed_window(const matrix_info &info,
                                       const unsigned long *row_offsets,
                                       const unsigned long *part_nnz,
                                       unsigned long num_parts,
                                       unsigned long part_begin,
                                       unsigned long part_end,
                                       sharded<sparse::compressed> *out) {
    if (out == nullptr || part_begin >= part_end || part_end > num_parts) return false;
    if (info.rows > (unsigned long) std::numeric_limits<::cellshard::types::dim_t>::max()
        || info.cols > (unsigned long) std::numeric_limits<::cellshard::types::dim_t>::max()) return false;
    clear(out);
    init(out);
    if (!reserve_partitions(out, part_end - part_begin)) return false;
    out->num_partitions = part_end - part_begin;
    out->cols = info.cols;
    for (unsigned long global_part = part_begin; global_part < part_end; ++global_part) {
        const unsigned long local_part = global_part - part_begin;
        const unsigned long rows = row_offsets[global_part + 1ul] - row_offsets[global_part];
        if (rows > (unsigned long) std::numeric_limits<::cellshard::types::dim_t>::max()
            || part_nnz[global_part] > (unsigned long) std::numeric_limits<::cellshard::types::nnz_t>::max()) return false;
        sparse::compressed *part = new sparse::compressed;
        sparse::init(part,
                     (::cellshard::types::dim_t) rows,
                     (::cellshard::types::dim_t) info.cols,
                     (::cellshard::types::nnz_t) part_nnz[global_part],
                     sparse::compressed_by_row);
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
    return set_shards_to_partitions(out);
}

inline bool load_part_window_compressed(const char *path,
                                        const mtx::header *h,
                                        const unsigned long *row_offsets,
                                        const unsigned long *part_nnz,
                                        unsigned long num_parts,
                                        unsigned long part_begin,
                                        unsigned long part_end,
                                        sharded<sparse::compressed> *out,
                                        std::string *error) {
    hid_t file = (hid_t) -1;
    hid_t matrix = (hid_t) -1;
    matrix_info info;
    std::vector<std::uint64_t> indptr;
    std::vector<std::uint64_t> indices;
    std::vector<double> values;
    bool ok = false;

    if (h == nullptr || out == nullptr) return false;
    if (!probe_matrix(path, &info, error)) return false;
    if (h->rows != info.rows || h->cols != info.cols) {
        set_error(error, "10x h5 header does not match matrix dimensions");
        return false;
    }
    if (!allocate_compressed_window(info, row_offsets, part_nnz, num_parts, part_begin, part_end, out)) return false;
    file = H5Fopen(path, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0) {
        set_error(error, "failed to open 10x h5 file");
        goto done;
    }
    matrix = H5Gopen2(file, "/matrix", H5P_DEFAULT);
    if (matrix < 0) {
        set_error(error, "failed to open 10x h5 /matrix group");
        goto done;
    }
    {
        const unsigned long row_begin = row_offsets[part_begin];
        const unsigned long row_end = row_offsets[part_end];
        if (!h5ad::read_dataset_u64_range(matrix, "indptr", (std::uint64_t) row_begin, (std::uint64_t) (row_end - row_begin + 1ul), &indptr)
            || indptr.empty()) {
            set_error(error, "failed to read 10x h5 indptr window");
            goto done;
        }
        const std::uint64_t nnz_begin = indptr.front();
        const std::uint64_t nnz_count = indptr.back() - indptr.front();
        for (std::uint64_t &ptr : indptr) ptr -= nnz_begin;
        if (!h5ad::read_dataset_u64_range(matrix, "indices", nnz_begin, nnz_count, &indices)
            || !h5ad::read_dataset_double_range(matrix, "data", nnz_begin, nnz_count, &values)
            || indices.size() != values.size()) {
            set_error(error, "failed to read 10x h5 sparse payload window");
            goto done;
        }
        for (unsigned long global_part = part_begin; global_part < part_end; ++global_part) {
            const unsigned long local_part = global_part - part_begin;
            sparse::compressed *part = out->parts[local_part];
            const unsigned long local_row_begin = row_offsets[global_part] - row_begin;
            const unsigned long local_row_end = row_offsets[global_part + 1ul] - row_begin;
            std::uint64_t running = 0u;
            part->majorPtr[0] = 0u;
            for (unsigned long local_row = local_row_begin; local_row < local_row_end; ++local_row) {
                const std::uint64_t span = indptr[(std::size_t) local_row + 1u] - indptr[(std::size_t) local_row];
                running += span;
                part->majorPtr[(local_row - local_row_begin) + 1ul] = (::cellshard::types::ptr_t) running;
            }
            const std::uint64_t part_nnz_begin = indptr[(std::size_t) local_row_begin];
            const std::uint64_t part_nnz_end = indptr[(std::size_t) local_row_end];
            for (std::uint64_t i = part_nnz_begin; i < part_nnz_end; ++i) {
                const std::uint64_t feature = indices[(std::size_t) i];
                if (feature >= info.cols) {
                    set_error(error, "10x h5 feature index is out of range");
                    goto done;
                }
                part->minorIdx[(std::size_t) (i - part_nnz_begin)] = (::cellshard::types::idx_t) feature;
                part->val[(std::size_t) (i - part_nnz_begin)] = __float2half((float) values[(std::size_t) i]);
            }
        }
    }
    ok = true;

done:
    if (matrix >= 0) H5Gclose(matrix);
    if (file >= 0) H5Fclose(file);
    if (!ok) clear(out);
    return ok;
}

inline bool load_part_window_coo(const char *path,
                                 const mtx::header *h,
                                 const unsigned long *row_offsets,
                                 const unsigned long *part_nnz,
                                 unsigned long num_parts,
                                 unsigned long part_begin,
                                 unsigned long part_end,
                                 sharded<sparse::coo> *out,
                                 std::string *error) {
    sharded<sparse::compressed> compressed;
    bool ok = false;
    init(&compressed);
    if (!mtx::allocate_part_window_coo(h, row_offsets, part_nnz, num_parts, part_begin, part_end, out)) return false;
    if (!load_part_window_compressed(path, h, row_offsets, part_nnz, num_parts, part_begin, part_end, &compressed, error)) goto done;
    for (unsigned long p = 0; p < compressed.num_partitions; ++p) {
        sparse::compressed *src = compressed.parts[p];
        sparse::coo *dst = out->parts[p];
        for (::cellshard::types::dim_t row = 0; row < src->rows; ++row) {
            for (::cellshard::types::ptr_t cursor = src->majorPtr[row]; cursor < src->majorPtr[row + 1u]; ++cursor) {
                dst->rowIdx[cursor] = row;
                dst->colIdx[cursor] = src->minorIdx[cursor];
                dst->val[cursor] = src->val[cursor];
            }
        }
    }
    ok = true;

done:
    clear(&compressed);
    if (!ok) clear(out);
    return ok;
}

} // namespace tenx_h5
} // namespace ingest
} // namespace cellshard
