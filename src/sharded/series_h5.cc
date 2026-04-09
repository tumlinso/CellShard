#include "series_h5.cuh"

#include "disk.cuh"
#include "sharded_host.cuh"

#include <hdf5.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>

#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

namespace cellshard {

namespace {

static const char series_magic[] = "CSH5S1";
static const char root_group[] = "/";
static const char matrix_group[] = "/matrix";
static const char datasets_group[] = "/datasets";
static const char provenance_group[] = "/provenance";
static const char codecs_group[] = "/codecs";
static const char payload_group[] = "/payload";
static const char payload_standard_group[] = "/payload/standard_csr";

struct series_h5_state {
    hid_t file;
    std::uint64_t num_parts;
    std::uint32_t num_codecs;
    std::uint64_t *part_indptr_offsets;
    std::uint64_t *part_nnz_offsets;
    std::uint32_t *part_codec_ids;
    series_codec_descriptor *codecs;
    char *cache_dir;
};

inline void series_h5_state_init(series_h5_state *state) {
    state->file = (hid_t) -1;
    state->num_parts = 0;
    state->num_codecs = 0;
    state->part_indptr_offsets = 0;
    state->part_nnz_offsets = 0;
    state->part_codec_ids = 0;
    state->codecs = 0;
    state->cache_dir = 0;
}

inline void series_h5_state_clear(series_h5_state *state) {
    if (state->file >= 0) H5Fclose(state->file);
    state->file = (hid_t) -1;
    std::free(state->part_indptr_offsets);
    std::free(state->part_nnz_offsets);
    std::free(state->part_codec_ids);
    std::free(state->codecs);
    std::free(state->cache_dir);
    state->part_indptr_offsets = 0;
    state->part_nnz_offsets = 0;
    state->part_codec_ids = 0;
    state->codecs = 0;
    state->cache_dir = 0;
    state->num_parts = 0;
    state->num_codecs = 0;
}

inline hid_t create_group(hid_t parent, const char *path) {
    hid_t group = H5Gcreate2(parent, path, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (group >= 0) return group;
    return H5Gopen2(parent, path, H5P_DEFAULT);
}

inline int write_attr_u64(hid_t obj, const char *name, std::uint64_t value) {
    hid_t space = H5Screate(H5S_SCALAR);
    hid_t attr = (hid_t) -1;
    int ok = 0;

    if (space < 0) return 0;
    attr = H5Acreate2(obj, name, H5T_NATIVE_UINT64, space, H5P_DEFAULT, H5P_DEFAULT);
    if (attr < 0) goto done;
    ok = H5Awrite(attr, H5T_NATIVE_UINT64, &value) >= 0;

done:
    if (attr >= 0) H5Aclose(attr);
    H5Sclose(space);
    return ok;
}

inline int write_attr_u32(hid_t obj, const char *name, std::uint32_t value) {
    hid_t space = H5Screate(H5S_SCALAR);
    hid_t attr = (hid_t) -1;
    int ok = 0;

    if (space < 0) return 0;
    attr = H5Acreate2(obj, name, H5T_NATIVE_UINT32, space, H5P_DEFAULT, H5P_DEFAULT);
    if (attr < 0) goto done;
    ok = H5Awrite(attr, H5T_NATIVE_UINT32, &value) >= 0;

done:
    if (attr >= 0) H5Aclose(attr);
    H5Sclose(space);
    return ok;
}

inline int write_attr_string(hid_t obj, const char *name, const char *value) {
    hid_t type = H5Tcopy(H5T_C_S1);
    hid_t space = H5Screate(H5S_SCALAR);
    hid_t attr = (hid_t) -1;
    int ok = 0;

    if (type < 0 || space < 0) goto done;
    if (H5Tset_size(type, std::strlen(value) + 1u) < 0) goto done;
    attr = H5Acreate2(obj, name, type, space, H5P_DEFAULT, H5P_DEFAULT);
    if (attr < 0) goto done;
    ok = H5Awrite(attr, type, value) >= 0;

done:
    if (attr >= 0) H5Aclose(attr);
    if (space >= 0) H5Sclose(space);
    if (type >= 0) H5Tclose(type);
    return ok;
}

inline int read_attr_u64(hid_t obj, const char *name, std::uint64_t *value) {
    hid_t attr = H5Aopen(obj, name, H5P_DEFAULT);
    int ok = 0;
    if (attr < 0) return 0;
    ok = H5Aread(attr, H5T_NATIVE_UINT64, value) >= 0;
    H5Aclose(attr);
    return ok;
}

inline int read_attr_string(hid_t obj, const char *name, char *dst, std::size_t cap) {
    hid_t attr = H5Aopen(obj, name, H5P_DEFAULT);
    hid_t type = (hid_t) -1;
    std::size_t size = 0;
    int ok = 0;

    if (attr < 0 || dst == 0 || cap == 0) return 0;
    type = H5Aget_type(attr);
    if (type < 0) goto done;
    size = H5Tget_size(type);
    if (size + 1u > cap) goto done;
    std::memset(dst, 0, cap);
    ok = H5Aread(attr, type, dst) >= 0;

done:
    if (type >= 0) H5Tclose(type);
    H5Aclose(attr);
    return ok;
}

inline int write_dataset_1d(hid_t parent,
                            const char *name,
                            hid_t dtype,
                            hsize_t count,
                            const void *data) {
    hid_t space = (hid_t) -1;
    hid_t dset = (hid_t) -1;
    int ok = 0;

    space = H5Screate_simple(1, &count, 0);
    if (space < 0) return 0;
    dset = H5Dcreate2(parent, name, dtype, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (dset < 0) goto done;
    if (data != 0) ok = H5Dwrite(dset, dtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, data) >= 0;
    else ok = 1;

done:
    if (dset >= 0) H5Dclose(dset);
    H5Sclose(space);
    return ok;
}

inline int read_dataset_1d(hid_t parent,
                           const char *name,
                           hid_t dtype,
                           void *data) {
    hid_t dset = H5Dopen2(parent, name, H5P_DEFAULT);
    int ok = 0;
    if (dset < 0) return 0;
    ok = H5Dread(dset, dtype, H5S_ALL, H5S_ALL, H5P_DEFAULT, data) >= 0;
    H5Dclose(dset);
    return ok;
}

inline int write_text_column(hid_t group, const char *name, const series_text_column_view *column) {
    hid_t sub = (hid_t) -1;
    int ok = 0;

    if (column == 0) return 1;
    sub = create_group(group, name);
    if (sub < 0) return 0;
    if (!write_attr_u32(sub, "count", column->count)) goto done;
    if (!write_attr_u32(sub, "bytes", column->bytes)) goto done;
    if (!write_dataset_1d(sub, "offsets", H5T_NATIVE_UINT32, (hsize_t) column->count + 1u, column->offsets)) goto done;
    if (!write_dataset_1d(sub, "data", H5T_NATIVE_CHAR, (hsize_t) column->bytes, column->data)) goto done;
    ok = 1;

done:
    if (sub >= 0) H5Gclose(sub);
    return ok;
}

inline int ensure_magic(hid_t file) {
    char got[32];
    if (!read_attr_string(file, "cellshard_magic", got, sizeof(got))) return 0;
    return std::strcmp(got, series_magic) == 0;
}

inline int read_hyperslab_1d(hid_t dataset,
                             hid_t dtype,
                             std::uint64_t offset,
                             std::uint64_t count,
                             void *dst) {
    hsize_t off[1];
    hsize_t dims[1];
    hid_t filespace = (hid_t) -1;
    hid_t memspace = (hid_t) -1;
    int ok = 0;

    if (count == 0) return 1;
    off[0] = (hsize_t) offset;
    dims[0] = (hsize_t) count;
    filespace = H5Dget_space(dataset);
    if (filespace < 0) return 0;
    if (H5Sselect_hyperslab(filespace, H5S_SELECT_SET, off, 0, dims, 0) < 0) goto done;
    memspace = H5Screate_simple(1, dims, 0);
    if (memspace < 0) goto done;
    ok = H5Dread(dataset, dtype, memspace, filespace, H5P_DEFAULT, dst) >= 0;

done:
    if (memspace >= 0) H5Sclose(memspace);
    if (filespace >= 0) H5Sclose(filespace);
    return ok;
}

inline std::size_t standard_csr_part_bytes(std::uint64_t rows, std::uint64_t nnz) {
    return (std::size_t) (rows + 1u) * sizeof(types::ptr_t)
        + (std::size_t) nnz * sizeof(types::idx_t)
        + (std::size_t) nnz * sizeof(real::storage_t);
}

int open_series_h5_backend(shard_storage *s);
void close_series_h5_backend(shard_storage *s);

inline int ensure_directory_exists(const char *path) {
    struct stat st;
    if (path == 0 || *path == 0) return 0;
    if (::stat(path, &st) == 0) return S_ISDIR(st.st_mode) ? 1 : 0;
    if (::mkdir(path, 0775) == 0) return 1;
    if (errno == EEXIST) return 1;
    return 0;
}

inline int build_cache_part_path(const series_h5_state *state,
                                 unsigned long part_id,
                                 char *path,
                                 std::size_t cap) {
    if (state == 0 || state->cache_dir == 0 || *state->cache_dir == 0 || path == 0 || cap == 0) return 0;
    return std::snprintf(path, cap, "%s/part.%lu.cscache", state->cache_dir, part_id) > 0;
}

inline int load_standard_csr_part_from_cache(sharded<sparse::compressed> *m,
                                             const series_h5_state *state,
                                             unsigned long part_id) {
    char path[4096];
    sparse::compressed *part = 0;
    int ok = 0;

    if (m == 0 || state == 0 || !build_cache_part_path(state, part_id, path, sizeof(path))) return 0;
    if (::access(path, R_OK) != 0) return 0;
    part = new sparse::compressed;
    sparse::init(part);
    if (!::cellshard::load(path, part)) goto done;
    if (part->rows != m->part_rows[part_id]) goto done;
    if (part->cols != m->cols) goto done;
    if (part->nnz != m->part_nnz[part_id]) goto done;
    if ((unsigned long) part->axis != m->part_aux[part_id]) goto done;
    if (m->parts[part_id] != 0) destroy(m->parts[part_id]);
    m->parts[part_id] = part;
    part = 0;
    ok = 1;

done:
    if (part != 0) {
        sparse::clear(part);
        delete part;
    }
    return ok;
}

inline int store_standard_csr_part_to_cache(const series_h5_state *state,
                                            unsigned long part_id,
                                            const sparse::compressed *part) {
    char path[4096];
    if (state == 0 || part == 0 || state->cache_dir == 0) return 0;
    if (!ensure_directory_exists(state->cache_dir)) return 0;
    if (!build_cache_part_path(state, part_id, path, sizeof(path))) return 0;
    return ::cellshard::store(path, part);
}

inline int load_series_h5_state(hid_t file, series_h5_state *state) {
    hid_t payload = (hid_t) -1;
    hid_t codecs = (hid_t) -1;
    int ok = 0;
    std::uint64_t num_codecs = 0;

    payload = H5Gopen2(file, payload_standard_group, H5P_DEFAULT);
    codecs = H5Gopen2(file, codecs_group, H5P_DEFAULT);
    if (payload < 0 || codecs < 0) goto done;
    if (!read_attr_u64(file, "num_parts", &state->num_parts)) goto done;
    if (!read_attr_u64(file, "num_codecs", &num_codecs)) goto done;
    state->num_codecs = (std::uint32_t) num_codecs;
    if (state->num_parts != 0) {
        state->part_indptr_offsets = (std::uint64_t *) std::calloc((std::size_t) state->num_parts, sizeof(std::uint64_t));
        state->part_nnz_offsets = (std::uint64_t *) std::calloc((std::size_t) state->num_parts, sizeof(std::uint64_t));
        state->part_codec_ids = (std::uint32_t *) std::calloc((std::size_t) state->num_parts, sizeof(std::uint32_t));
        if (state->part_indptr_offsets == 0 || state->part_nnz_offsets == 0 || state->part_codec_ids == 0) goto done;
    }
    if (state->num_codecs != 0) {
        state->codecs = (series_codec_descriptor *) std::calloc((std::size_t) state->num_codecs, sizeof(series_codec_descriptor));
        if (state->codecs == 0) goto done;
    }
    if (!read_dataset_1d(payload, "part_indptr_offsets", H5T_NATIVE_UINT64, state->part_indptr_offsets)) goto done;
    if (!read_dataset_1d(payload, "part_nnz_offsets", H5T_NATIVE_UINT64, state->part_nnz_offsets)) goto done;
    if (!read_dataset_1d(H5Gopen2(file, matrix_group, H5P_DEFAULT), "part_codec_ids", H5T_NATIVE_UINT32, state->part_codec_ids)) goto done;
    if (state->num_codecs != 0) {
        if (!read_dataset_1d(codecs, "codec_id", H5T_NATIVE_UINT32, &state->codecs[0].codec_id)) goto done;
    }
    ok = 1;

done:
    if (payload >= 0) H5Gclose(payload);
    if (codecs >= 0) H5Gclose(codecs);
    return ok;
}

inline int load_codec_table(hid_t codecs, series_codec_descriptor *descs, std::uint32_t count) {
    std::uint32_t i = 0;
    std::uint32_t *codec_ids = 0;
    std::uint32_t *families = 0;
    std::uint32_t *value_codes = 0;
    std::uint32_t *scale_value_codes = 0;
    std::uint32_t *bits = 0;
    std::uint32_t *flags = 0;
    int ok = 0;

    if (count == 0) return 1;
    codec_ids = (std::uint32_t *) std::calloc((std::size_t) count, sizeof(std::uint32_t));
    families = (std::uint32_t *) std::calloc((std::size_t) count, sizeof(std::uint32_t));
    value_codes = (std::uint32_t *) std::calloc((std::size_t) count, sizeof(std::uint32_t));
    scale_value_codes = (std::uint32_t *) std::calloc((std::size_t) count, sizeof(std::uint32_t));
    bits = (std::uint32_t *) std::calloc((std::size_t) count, sizeof(std::uint32_t));
    flags = (std::uint32_t *) std::calloc((std::size_t) count, sizeof(std::uint32_t));
    if (codec_ids == 0 || families == 0 || value_codes == 0 || scale_value_codes == 0 || bits == 0 || flags == 0) goto done;
    if (!read_dataset_1d(codecs, "codec_id", H5T_NATIVE_UINT32, codec_ids)) goto done;
    if (!read_dataset_1d(codecs, "family", H5T_NATIVE_UINT32, families)) goto done;
    if (!read_dataset_1d(codecs, "value_code", H5T_NATIVE_UINT32, value_codes)) goto done;
    if (!read_dataset_1d(codecs, "scale_value_code", H5T_NATIVE_UINT32, scale_value_codes)) goto done;
    if (!read_dataset_1d(codecs, "bits", H5T_NATIVE_UINT32, bits)) goto done;
    if (!read_dataset_1d(codecs, "flags", H5T_NATIVE_UINT32, flags)) goto done;
    for (i = 0; i < count; ++i) {
        descs[i].codec_id = codec_ids[i];
        descs[i].family = families[i];
        descs[i].value_code = value_codes[i];
        descs[i].scale_value_code = scale_value_codes[i];
        descs[i].bits = bits[i];
        descs[i].flags = flags[i];
    }
    ok = 1;

done:
    std::free(codec_ids);
    std::free(families);
    std::free(value_codes);
    std::free(scale_value_codes);
    std::free(bits);
    std::free(flags);
    return ok;
}

inline const series_codec_descriptor *find_codec(const series_h5_state *state, std::uint32_t codec_id) {
    std::uint32_t i = 0;
    if (state == 0) return 0;
    for (i = 0; i < state->num_codecs; ++i) {
        if (state->codecs[i].codec_id == codec_id) return state->codecs + i;
    }
    return 0;
}

int open_series_h5_backend(shard_storage *s) {
    series_h5_state *state = 0;
    if (s == 0 || s->packfile_path == 0 || s->backend_state == 0) return 0;
    state = (series_h5_state *) s->backend_state;
    if (state->file >= 0) return 1;
    state->file = H5Fopen(s->packfile_path, H5F_ACC_RDONLY, H5P_DEFAULT);
    return state->file >= 0;
}

void close_series_h5_backend(shard_storage *s) {
    series_h5_state *state = 0;
    if (s == 0 || s->backend_state == 0) return;
    state = (series_h5_state *) s->backend_state;
    series_h5_state_clear(state);
    std::free(state);
    s->backend_state = 0;
    s->open_backend = 0;
    s->close_backend = 0;
    s->backend = shard_storage_backend_none;
}

} // namespace

int create_series_compressed_h5(const char *filename,
                                const series_layout_view *layout,
                                const series_dataset_table_view *datasets,
                                const series_provenance_view *provenance) {
    hid_t file = (hid_t) -1;
    hid_t matrix = (hid_t) -1;
    hid_t dsets = (hid_t) -1;
    hid_t prov = (hid_t) -1;
    hid_t codecs = (hid_t) -1;
    hid_t payload_root = (hid_t) -1;
    hid_t payload = (hid_t) -1;
    std::uint64_t total_indptr = 0;
    std::uint64_t total_nnz = 0;
    std::uint64_t *part_indptr_offsets = 0;
    std::uint64_t *part_nnz_offsets = 0;
    std::uint32_t i = 0;
    int ok = 0;

    if (filename == 0 || layout == 0) return 0;
    if (layout->part_rows == 0 || layout->part_nnz == 0 || layout->part_axes == 0 || layout->part_row_offsets == 0 || layout->part_dataset_ids == 0 || layout->part_codec_ids == 0 || layout->shard_offsets == 0) return 0;

    part_indptr_offsets = (std::uint64_t *) std::calloc((std::size_t) layout->num_parts, sizeof(std::uint64_t));
    part_nnz_offsets = (std::uint64_t *) std::calloc((std::size_t) layout->num_parts, sizeof(std::uint64_t));
    if ((layout->num_parts != 0) && (part_indptr_offsets == 0 || part_nnz_offsets == 0)) goto done;

    for (i = 0; i < layout->num_parts; ++i) {
        part_indptr_offsets[i] = total_indptr;
        part_nnz_offsets[i] = total_nnz;
        total_indptr += layout->part_rows[i] + 1u;
        total_nnz += layout->part_nnz[i];
    }

    file = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file < 0) goto done;
    if (!write_attr_string(file, "cellshard_magic", series_magic)) goto done;
    if (!write_attr_u32(file, "schema_version", series_h5_schema_version)) goto done;
    if (!write_attr_string(file, "matrix_format", "compressed")) goto done;
    if (!write_attr_u64(file, "rows", layout->rows)) goto done;
    if (!write_attr_u64(file, "cols", layout->cols)) goto done;
    if (!write_attr_u64(file, "nnz", layout->nnz)) goto done;
    if (!write_attr_u64(file, "num_parts", layout->num_parts)) goto done;
    if (!write_attr_u64(file, "num_shards", layout->num_shards)) goto done;
    if (!write_attr_u64(file, "num_codecs", layout->num_codecs)) goto done;
    if (!write_attr_u64(file, "num_datasets", datasets != 0 ? datasets->count : 0u)) goto done;

    matrix = create_group(file, matrix_group);
    dsets = create_group(file, datasets_group);
    prov = create_group(file, provenance_group);
    codecs = create_group(file, codecs_group);
    payload_root = create_group(file, payload_group);
    payload = payload_root >= 0 ? create_group(payload_root, "standard_csr") : (hid_t) -1;
    if (matrix < 0 || dsets < 0 || prov < 0 || codecs < 0 || payload_root < 0 || payload < 0) goto done;

    if (!write_dataset_1d(matrix, "part_rows", H5T_NATIVE_UINT64, (hsize_t) layout->num_parts, layout->part_rows)) goto done;
    if (!write_dataset_1d(matrix, "part_nnz", H5T_NATIVE_UINT64, (hsize_t) layout->num_parts, layout->part_nnz)) goto done;
    if (!write_dataset_1d(matrix, "part_axes", H5T_NATIVE_UINT32, (hsize_t) layout->num_parts, layout->part_axes)) goto done;
    if (!write_dataset_1d(matrix, "part_row_offsets", H5T_NATIVE_UINT64, (hsize_t) layout->num_parts + 1u, layout->part_row_offsets)) goto done;
    if (!write_dataset_1d(matrix, "part_dataset_ids", H5T_NATIVE_UINT32, (hsize_t) layout->num_parts, layout->part_dataset_ids)) goto done;
    if (!write_dataset_1d(matrix, "part_codec_ids", H5T_NATIVE_UINT32, (hsize_t) layout->num_parts, layout->part_codec_ids)) goto done;
    if (!write_dataset_1d(matrix, "shard_offsets", H5T_NATIVE_UINT64, (hsize_t) layout->num_shards + 1u, layout->shard_offsets)) goto done;

    if (datasets != 0) {
        if (!write_text_column(dsets, "dataset_ids", &datasets->dataset_ids)) goto done;
        if (!write_text_column(dsets, "matrix_paths", &datasets->matrix_paths)) goto done;
        if (!write_text_column(dsets, "feature_paths", &datasets->feature_paths)) goto done;
        if (!write_text_column(dsets, "barcode_paths", &datasets->barcode_paths)) goto done;
        if (!write_text_column(dsets, "metadata_paths", &datasets->metadata_paths)) goto done;
        if (!write_dataset_1d(dsets, "formats", H5T_NATIVE_UINT32, (hsize_t) datasets->count, datasets->formats)) goto done;
        if (!write_dataset_1d(dsets, "row_begin", H5T_NATIVE_UINT64, (hsize_t) datasets->count, datasets->row_begin)) goto done;
        if (!write_dataset_1d(dsets, "row_end", H5T_NATIVE_UINT64, (hsize_t) datasets->count, datasets->row_end)) goto done;
        if (!write_dataset_1d(dsets, "rows", H5T_NATIVE_UINT64, (hsize_t) datasets->count, datasets->rows)) goto done;
        if (!write_dataset_1d(dsets, "cols", H5T_NATIVE_UINT64, (hsize_t) datasets->count, datasets->cols)) goto done;
        if (!write_dataset_1d(dsets, "nnz", H5T_NATIVE_UINT64, (hsize_t) datasets->count, datasets->nnz)) goto done;
    }

    if (provenance != 0) {
        if (!write_text_column(prov, "global_barcodes", &provenance->global_barcodes)) goto done;
        if (!write_dataset_1d(prov, "cell_dataset_ids", H5T_NATIVE_UINT32, (hsize_t) layout->rows, provenance->cell_dataset_ids)) goto done;
        if (!write_dataset_1d(prov, "cell_local_indices", H5T_NATIVE_UINT64, (hsize_t) layout->rows, provenance->cell_local_indices)) goto done;
        if (!write_text_column(prov, "feature_ids", &provenance->feature_ids)) goto done;
        if (!write_text_column(prov, "feature_names", &provenance->feature_names)) goto done;
        if (!write_text_column(prov, "feature_types", &provenance->feature_types)) goto done;
        if (!write_dataset_1d(prov, "feature_dataset_ids", H5T_NATIVE_UINT32, (hsize_t) layout->cols, provenance->feature_dataset_ids)) goto done;
        if (!write_dataset_1d(prov, "feature_local_indices", H5T_NATIVE_UINT64, (hsize_t) layout->cols, provenance->feature_local_indices)) goto done;
        if (datasets != 0) {
            if (!write_dataset_1d(prov, "dataset_feature_offsets", H5T_NATIVE_UINT64, (hsize_t) datasets->count + 1u, provenance->dataset_feature_offsets)) goto done;
            if (!write_dataset_1d(prov, "dataset_feature_to_global", H5T_NATIVE_UINT32, (hsize_t) provenance->dataset_feature_offsets[datasets->count], provenance->dataset_feature_to_global)) goto done;
        }
    }

    if (layout->num_codecs != 0) {
        std::uint32_t *codec_id = (std::uint32_t *) std::calloc((std::size_t) layout->num_codecs, sizeof(std::uint32_t));
        std::uint32_t *family = (std::uint32_t *) std::calloc((std::size_t) layout->num_codecs, sizeof(std::uint32_t));
        std::uint32_t *value_code = (std::uint32_t *) std::calloc((std::size_t) layout->num_codecs, sizeof(std::uint32_t));
        std::uint32_t *scale_value_code = (std::uint32_t *) std::calloc((std::size_t) layout->num_codecs, sizeof(std::uint32_t));
        std::uint32_t *bits = (std::uint32_t *) std::calloc((std::size_t) layout->num_codecs, sizeof(std::uint32_t));
        std::uint32_t *flags = (std::uint32_t *) std::calloc((std::size_t) layout->num_codecs, sizeof(std::uint32_t));
        if (codec_id == 0 || family == 0 || value_code == 0 || scale_value_code == 0 || bits == 0 || flags == 0) {
            std::free(codec_id);
            std::free(family);
            std::free(value_code);
            std::free(scale_value_code);
            std::free(bits);
            std::free(flags);
            goto done;
        }
        for (i = 0; i < layout->num_codecs; ++i) {
            codec_id[i] = layout->codecs[i].codec_id;
            family[i] = layout->codecs[i].family;
            value_code[i] = layout->codecs[i].value_code;
            scale_value_code[i] = layout->codecs[i].scale_value_code;
            bits[i] = layout->codecs[i].bits;
            flags[i] = layout->codecs[i].flags;
        }
        if (!write_dataset_1d(codecs, "codec_id", H5T_NATIVE_UINT32, (hsize_t) layout->num_codecs, codec_id)) goto done;
        if (!write_dataset_1d(codecs, "family", H5T_NATIVE_UINT32, (hsize_t) layout->num_codecs, family)) goto done;
        if (!write_dataset_1d(codecs, "value_code", H5T_NATIVE_UINT32, (hsize_t) layout->num_codecs, value_code)) goto done;
        if (!write_dataset_1d(codecs, "scale_value_code", H5T_NATIVE_UINT32, (hsize_t) layout->num_codecs, scale_value_code)) goto done;
        if (!write_dataset_1d(codecs, "bits", H5T_NATIVE_UINT32, (hsize_t) layout->num_codecs, bits)) goto done;
        if (!write_dataset_1d(codecs, "flags", H5T_NATIVE_UINT32, (hsize_t) layout->num_codecs, flags)) goto done;
        std::free(codec_id);
        std::free(family);
        std::free(value_code);
        std::free(scale_value_code);
        std::free(bits);
        std::free(flags);
    }

    if (!write_dataset_1d(payload, "part_indptr_offsets", H5T_NATIVE_UINT64, (hsize_t) layout->num_parts, part_indptr_offsets)) goto done;
    if (!write_dataset_1d(payload, "part_nnz_offsets", H5T_NATIVE_UINT64, (hsize_t) layout->num_parts, part_nnz_offsets)) goto done;
    if (!write_dataset_1d(payload, "indptr", H5T_NATIVE_UINT32, (hsize_t) total_indptr, 0)) goto done;
    if (!write_dataset_1d(payload, "indices", H5T_NATIVE_UINT32, (hsize_t) total_nnz, 0)) goto done;
    if (!write_dataset_1d(payload, "values", H5T_NATIVE_UINT16, (hsize_t) total_nnz, 0)) goto done;

    ok = 1;

done:
    std::free(part_indptr_offsets);
    std::free(part_nnz_offsets);
    if (payload >= 0) H5Gclose(payload);
    if (payload_root >= 0) H5Gclose(payload_root);
    if (codecs >= 0) H5Gclose(codecs);
    if (prov >= 0) H5Gclose(prov);
    if (dsets >= 0) H5Gclose(dsets);
    if (matrix >= 0) H5Gclose(matrix);
    if (file >= 0) H5Fclose(file);
    return ok;
}

int append_standard_csr_part_h5(const char *filename,
                                unsigned long part_id,
                                const sparse::compressed *part) {
    hid_t file = (hid_t) -1;
    hid_t payload = (hid_t) -1;
    hid_t d_indptr = (hid_t) -1;
    hid_t d_indices = (hid_t) -1;
    hid_t d_values = (hid_t) -1;
    std::uint64_t *part_indptr_offsets = 0;
    std::uint64_t *part_nnz_offsets = 0;
    std::uint64_t num_parts = 0;
    int ok = 0;

    if (filename == 0 || part == 0 || part->axis != sparse::compressed_by_row) return 0;

    file = H5Fopen(filename, H5F_ACC_RDWR, H5P_DEFAULT);
    if (file < 0) return 0;
    if (!ensure_magic(file)) goto done;
    if (!read_attr_u64(file, "num_parts", &num_parts)) goto done;
    if (part_id >= num_parts) goto done;

    part_indptr_offsets = (std::uint64_t *) std::calloc((std::size_t) num_parts, sizeof(std::uint64_t));
    part_nnz_offsets = (std::uint64_t *) std::calloc((std::size_t) num_parts, sizeof(std::uint64_t));
    if ((num_parts != 0) && (part_indptr_offsets == 0 || part_nnz_offsets == 0)) goto done;

    payload = H5Gopen2(file, payload_standard_group, H5P_DEFAULT);
    if (payload < 0) goto done;
    if (!read_dataset_1d(payload, "part_indptr_offsets", H5T_NATIVE_UINT64, part_indptr_offsets)) goto done;
    if (!read_dataset_1d(payload, "part_nnz_offsets", H5T_NATIVE_UINT64, part_nnz_offsets)) goto done;
    d_indptr = H5Dopen2(payload, "indptr", H5P_DEFAULT);
    d_indices = H5Dopen2(payload, "indices", H5P_DEFAULT);
    d_values = H5Dopen2(payload, "values", H5P_DEFAULT);
    if (d_indptr < 0 || d_indices < 0 || d_values < 0) goto done;
    if (!read_hyperslab_1d(d_indptr, H5T_NATIVE_UINT32, 0, 0, 0)) goto done;

    {
        hsize_t off[1];
        hsize_t dims[1];
        hid_t filespace = (hid_t) -1;
        hid_t memspace = (hid_t) -1;

        off[0] = (hsize_t) part_indptr_offsets[part_id];
        dims[0] = (hsize_t) part->rows + 1u;
        filespace = H5Dget_space(d_indptr);
        if (filespace < 0) goto done;
        if (H5Sselect_hyperslab(filespace, H5S_SELECT_SET, off, 0, dims, 0) < 0) {
            H5Sclose(filespace);
            goto done;
        }
        memspace = H5Screate_simple(1, dims, 0);
        if (memspace < 0) {
            H5Sclose(filespace);
            goto done;
        }
        if (H5Dwrite(d_indptr, H5T_NATIVE_UINT32, memspace, filespace, H5P_DEFAULT, part->majorPtr) < 0) {
            H5Sclose(memspace);
            H5Sclose(filespace);
            goto done;
        }
        H5Sclose(memspace);
        H5Sclose(filespace);
    }

    {
        hsize_t off[1];
        hsize_t dims[1];
        hid_t filespace = (hid_t) -1;
        hid_t memspace = (hid_t) -1;

        off[0] = (hsize_t) part_nnz_offsets[part_id];
        dims[0] = (hsize_t) part->nnz;
        filespace = H5Dget_space(d_indices);
        if (filespace < 0) goto done;
        if (H5Sselect_hyperslab(filespace, H5S_SELECT_SET, off, 0, dims, 0) < 0) {
            H5Sclose(filespace);
            goto done;
        }
        memspace = H5Screate_simple(1, dims, 0);
        if (memspace < 0) {
            H5Sclose(filespace);
            goto done;
        }
        if (H5Dwrite(d_indices, H5T_NATIVE_UINT32, memspace, filespace, H5P_DEFAULT, part->minorIdx) < 0) {
            H5Sclose(memspace);
            H5Sclose(filespace);
            goto done;
        }
        H5Sclose(memspace);
        H5Sclose(filespace);
    }

    {
        hsize_t off[1];
        hsize_t dims[1];
        hid_t filespace = (hid_t) -1;
        hid_t memspace = (hid_t) -1;

        off[0] = (hsize_t) part_nnz_offsets[part_id];
        dims[0] = (hsize_t) part->nnz;
        filespace = H5Dget_space(d_values);
        if (filespace < 0) goto done;
        if (H5Sselect_hyperslab(filespace, H5S_SELECT_SET, off, 0, dims, 0) < 0) {
            H5Sclose(filespace);
            goto done;
        }
        memspace = H5Screate_simple(1, dims, 0);
        if (memspace < 0) {
            H5Sclose(filespace);
            goto done;
        }
        if (H5Dwrite(d_values, H5T_NATIVE_UINT16, memspace, filespace, H5P_DEFAULT, part->val) < 0) {
            H5Sclose(memspace);
            H5Sclose(filespace);
            goto done;
        }
        H5Sclose(memspace);
        H5Sclose(filespace);
    }

    ok = 1;

done:
    std::free(part_indptr_offsets);
    std::free(part_nnz_offsets);
    if (d_values >= 0) H5Dclose(d_values);
    if (d_indices >= 0) H5Dclose(d_indices);
    if (d_indptr >= 0) H5Dclose(d_indptr);
    if (payload >= 0) H5Gclose(payload);
    if (file >= 0) H5Fclose(file);
    return ok;
}

int bind_series_h5(shard_storage *s, const char *path) {
    std::size_t len = 0;
    char *copy = 0;
    series_h5_state *state = 0;

    if (s == 0) return 0;
    if (s->close_backend != 0) s->close_backend(s);
    close_packfile(s);
    std::free(s->packfile_path);
    std::free(s->locators);
    s->packfile_path = 0;
    s->locators = 0;
    s->capacity = 0;
    if (path == 0) return 1;

    len = std::strlen(path);
    copy = (char *) std::malloc(len + 1u);
    state = (series_h5_state *) std::calloc(1u, sizeof(series_h5_state));
    if (copy == 0 || state == 0) {
        std::free(copy);
        std::free(state);
        return 0;
    }
    std::memcpy(copy, path, len + 1u);
    series_h5_state_init(state);
    s->packfile_path = copy;
    s->backend = shard_storage_backend_series_h5;
    s->backend_state = state;
    s->open_backend = open_series_h5_backend;
    s->close_backend = close_series_h5_backend;
    return 1;
}

int bind_series_h5_part_cache(shard_storage *s, const char *cache_dir) {
    series_h5_state *state = 0;
    char *copy = 0;
    std::size_t len = 0;

    if (s == 0 || s->backend != shard_storage_backend_series_h5 || s->backend_state == 0) return 0;
    state = (series_h5_state *) s->backend_state;
    std::free(state->cache_dir);
    state->cache_dir = 0;
    if (cache_dir == 0 || *cache_dir == 0) return 1;
    if (!ensure_directory_exists(cache_dir)) return 0;
    len = std::strlen(cache_dir);
    copy = (char *) std::malloc(len + 1u);
    if (copy == 0) return 0;
    std::memcpy(copy, cache_dir, len + 1u);
    state->cache_dir = copy;
    return 1;
}

int load_series_compressed_h5_header(const char *filename,
                                     sharded<sparse::compressed> *m,
                                     shard_storage *s) {
    hid_t file = (hid_t) -1;
    hid_t matrix = (hid_t) -1;
    hid_t codecs = (hid_t) -1;
    std::uint64_t rows = 0;
    std::uint64_t cols = 0;
    std::uint64_t nnz = 0;
    std::uint64_t num_parts = 0;
    std::uint64_t num_shards = 0;
    std::uint64_t num_codecs = 0;
    std::uint64_t *part_rows = 0;
    std::uint64_t *part_nnz = 0;
    std::uint32_t *part_axes = 0;
    std::uint64_t *part_row_offsets = 0;
    std::uint64_t *shard_offsets = 0;
    unsigned long *part_rows_ul = 0;
    unsigned long *part_nnz_ul = 0;
    unsigned long *part_axes_ul = 0;
    unsigned long *shard_offsets_ul = 0;
    unsigned long i = 0;
    int ok = 0;

    if (filename == 0 || m == 0) return 0;
    file = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file < 0) return 0;
    if (!ensure_magic(file)) goto done;
    if (!read_attr_u64(file, "rows", &rows)) goto done;
    if (!read_attr_u64(file, "cols", &cols)) goto done;
    if (!read_attr_u64(file, "nnz", &nnz)) goto done;
    if (!read_attr_u64(file, "num_parts", &num_parts)) goto done;
    if (!read_attr_u64(file, "num_shards", &num_shards)) goto done;
    if (!read_attr_u64(file, "num_codecs", &num_codecs)) goto done;

    matrix = H5Gopen2(file, matrix_group, H5P_DEFAULT);
    codecs = H5Gopen2(file, codecs_group, H5P_DEFAULT);
    if (matrix < 0 || codecs < 0) goto done;

    part_rows = (std::uint64_t *) std::calloc((std::size_t) num_parts, sizeof(std::uint64_t));
    part_nnz = (std::uint64_t *) std::calloc((std::size_t) num_parts, sizeof(std::uint64_t));
    part_axes = (std::uint32_t *) std::calloc((std::size_t) num_parts, sizeof(std::uint32_t));
    part_row_offsets = (std::uint64_t *) std::calloc((std::size_t) num_parts + 1u, sizeof(std::uint64_t));
    shard_offsets = (std::uint64_t *) std::calloc((std::size_t) num_shards + 1u, sizeof(std::uint64_t));
    part_rows_ul = (unsigned long *) std::calloc((std::size_t) num_parts, sizeof(unsigned long));
    part_nnz_ul = (unsigned long *) std::calloc((std::size_t) num_parts, sizeof(unsigned long));
    part_axes_ul = (unsigned long *) std::calloc((std::size_t) num_parts, sizeof(unsigned long));
    shard_offsets_ul = (unsigned long *) std::calloc((std::size_t) num_shards + 1u, sizeof(unsigned long));
    if ((num_parts != 0) && (part_rows == 0 || part_nnz == 0 || part_axes == 0 || part_row_offsets == 0 || part_rows_ul == 0 || part_nnz_ul == 0 || part_axes_ul == 0)) goto done;
    if ((num_shards + 1u) != 0u && (shard_offsets == 0 || shard_offsets_ul == 0)) goto done;

    if (!read_dataset_1d(matrix, "part_rows", H5T_NATIVE_UINT64, part_rows)) goto done;
    if (!read_dataset_1d(matrix, "part_nnz", H5T_NATIVE_UINT64, part_nnz)) goto done;
    if (!read_dataset_1d(matrix, "part_axes", H5T_NATIVE_UINT32, part_axes)) goto done;
    if (!read_dataset_1d(matrix, "part_row_offsets", H5T_NATIVE_UINT64, part_row_offsets)) goto done;
    if (!read_dataset_1d(matrix, "shard_offsets", H5T_NATIVE_UINT64, shard_offsets)) goto done;

    clear(m);
    init(m);
    for (i = 0; i < (unsigned long) num_parts; ++i) {
        if (!sharded_from_u64(part_rows[i], part_rows_ul + i, "part_rows", filename)) goto done;
        if (!sharded_from_u64(part_nnz[i], part_nnz_ul + i, "part_nnz", filename)) goto done;
        part_axes_ul[i] = (unsigned long) part_axes[i];
    }
    for (i = 0; i <= (unsigned long) num_shards; ++i) {
        if (!sharded_from_u64(shard_offsets[i], shard_offsets_ul + i, "shard_offsets", filename)) goto done;
    }
    if (!define_parts(m, (unsigned long) cols, (unsigned long) num_parts, part_rows_ul, part_nnz_ul, part_axes_ul)) goto done;
    if (!reshard(m, (unsigned long) num_shards, shard_offsets_ul)) goto done;
    m->rows = (unsigned long) rows;
    m->nnz = (unsigned long) nnz;

    if (s != 0) {
        series_h5_state *state = 0;
        if (!bind_series_h5(s, filename)) goto done;
        s->capacity = (unsigned int) num_parts;
        state = (series_h5_state *) s->backend_state;
        state->num_parts = num_parts;
        state->num_codecs = (std::uint32_t) num_codecs;
        if (num_parts != 0) {
            state->part_indptr_offsets = (std::uint64_t *) std::calloc((std::size_t) num_parts, sizeof(std::uint64_t));
            state->part_nnz_offsets = (std::uint64_t *) std::calloc((std::size_t) num_parts, sizeof(std::uint64_t));
            state->part_codec_ids = (std::uint32_t *) std::calloc((std::size_t) num_parts, sizeof(std::uint32_t));
            if (state->part_indptr_offsets == 0 || state->part_nnz_offsets == 0 || state->part_codec_ids == 0) goto done;
        }
        if (num_codecs != 0) {
            state->codecs = (series_codec_descriptor *) std::calloc((std::size_t) num_codecs, sizeof(series_codec_descriptor));
            if (state->codecs == 0) goto done;
        }
        {
            hid_t payload = H5Gopen2(file, payload_standard_group, H5P_DEFAULT);
            if (payload < 0) goto done;
            if (!read_dataset_1d(payload, "part_indptr_offsets", H5T_NATIVE_UINT64, state->part_indptr_offsets)) {
                H5Gclose(payload);
                goto done;
            }
            if (!read_dataset_1d(payload, "part_nnz_offsets", H5T_NATIVE_UINT64, state->part_nnz_offsets)) {
                H5Gclose(payload);
                goto done;
            }
            H5Gclose(payload);
        }
        if (!read_dataset_1d(matrix, "part_codec_ids", H5T_NATIVE_UINT32, state->part_codec_ids)) goto done;
        if (!load_codec_table(codecs, state->codecs, (std::uint32_t) num_codecs)) goto done;
    }

    ok = 1;

done:
    if (!ok && s != 0) clear(s);
    std::free(part_rows);
    std::free(part_nnz);
    std::free(part_axes);
    std::free(part_row_offsets);
    std::free(shard_offsets);
    std::free(part_rows_ul);
    std::free(part_nnz_ul);
    std::free(part_axes_ul);
    std::free(shard_offsets_ul);
    if (codecs >= 0) H5Gclose(codecs);
    if (matrix >= 0) H5Gclose(matrix);
    if (file >= 0) H5Fclose(file);
    return ok;
}

int fetch_series_compressed_h5_part(sharded<sparse::compressed> *m,
                                    const shard_storage *s,
                                    unsigned long part_id) {
    shard_storage *storage = const_cast<shard_storage *>(s);
    series_h5_state *state = 0;
    const series_codec_descriptor *codec = 0;
    sparse::compressed *part = 0;
    hid_t payload = (hid_t) -1;
    hid_t d_indptr = (hid_t) -1;
    hid_t d_indices = (hid_t) -1;
    hid_t d_values = (hid_t) -1;
    int ok = 0;

    if (m == 0 || storage == 0 || storage->backend != shard_storage_backend_series_h5 || part_id >= m->num_parts || storage->backend_state == 0) return 0;
    state = (series_h5_state *) storage->backend_state;
    if (state->cache_dir != 0 && load_standard_csr_part_from_cache(m, state, part_id)) return 1;
    if (storage->open_backend == 0 || !storage->open_backend(storage)) return 0;
    codec = find_codec(state, state->part_codec_ids[part_id]);
    if (codec == 0 || codec->family != series_codec_family_standard_csr || codec->value_code != (std::uint32_t) real::code_of<real::storage_t>::code) return 0;

    if (m->parts[part_id] != 0) destroy(m->parts[part_id]);
    m->parts[part_id] = 0;

    payload = H5Gopen2(state->file, payload_standard_group, H5P_DEFAULT);
    if (payload < 0) return 0;
    d_indptr = H5Dopen2(payload, "indptr", H5P_DEFAULT);
    d_indices = H5Dopen2(payload, "indices", H5P_DEFAULT);
    d_values = H5Dopen2(payload, "values", H5P_DEFAULT);
    if (d_indptr < 0 || d_indices < 0 || d_values < 0) goto done;

    part = new sparse::compressed;
    sparse::init(part,
                 (types::dim_t) m->part_rows[part_id],
                 (types::dim_t) m->cols,
                 (types::nnz_t) m->part_nnz[part_id],
                 (types::u32) m->part_aux[part_id]);
    if (!sparse::allocate(part)) goto done;
    if (!read_hyperslab_1d(d_indptr, H5T_NATIVE_UINT32, state->part_indptr_offsets[part_id], (std::uint64_t) part->rows + 1u, part->majorPtr)) goto done;
    if (!read_hyperslab_1d(d_indices, H5T_NATIVE_UINT32, state->part_nnz_offsets[part_id], (std::uint64_t) part->nnz, part->minorIdx)) goto done;
    if (!read_hyperslab_1d(d_values, H5T_NATIVE_UINT16, state->part_nnz_offsets[part_id], (std::uint64_t) part->nnz, part->val)) goto done;
    if (state->cache_dir != 0) {
        if (!store_standard_csr_part_to_cache(state, part_id, part)) goto done;
    }
    m->parts[part_id] = part;
    part = 0;
    ok = 1;

done:
    if (part != 0) {
        sparse::clear(part);
        delete part;
    }
    if (d_values >= 0) H5Dclose(d_values);
    if (d_indices >= 0) H5Dclose(d_indices);
    if (d_indptr >= 0) H5Dclose(d_indptr);
    if (payload >= 0) H5Gclose(payload);
    return ok;
}

int fetch_series_compressed_h5_shard(sharded<sparse::compressed> *m,
                                     const shard_storage *s,
                                     unsigned long shard_id) {
    unsigned long begin = 0;
    unsigned long end = 0;
    unsigned long i = 0;

    if (m == 0 || s == 0 || shard_id >= m->num_shards) return 0;
    begin = first_part_in_shard(m, shard_id);
    end = last_part_in_shard(m, shard_id);
    for (i = begin; i < end; ++i) {
        if (!fetch_series_compressed_h5_part(m, s, i)) return 0;
    }
    return 1;
}

int prefetch_series_compressed_h5_part_to_cache(const sharded<sparse::compressed> *m,
                                                const shard_storage *s,
                                                unsigned long part_id) {
    shard_storage *storage = const_cast<shard_storage *>(s);
    series_h5_state *state = 0;
    sparse::compressed part;
    const series_codec_descriptor *codec = 0;
    hid_t payload = (hid_t) -1;
    hid_t d_indptr = (hid_t) -1;
    hid_t d_indices = (hid_t) -1;
    hid_t d_values = (hid_t) -1;
    int ok = 0;

    if (m == 0 || storage == 0 || storage->backend != shard_storage_backend_series_h5 || part_id >= m->num_parts || storage->backend_state == 0) return 0;
    if (storage->open_backend == 0 || !storage->open_backend(storage)) return 0;
    state = (series_h5_state *) storage->backend_state;
    if (state->cache_dir == 0) return 0;
    if (load_standard_csr_part_from_cache(const_cast<sharded<sparse::compressed> *>(m), state, part_id)) {
        drop_part(const_cast<sharded<sparse::compressed> *>(m), part_id);
        return 1;
    }
    codec = find_codec(state, state->part_codec_ids[part_id]);
    if (codec == 0 || codec->family != series_codec_family_standard_csr || codec->value_code != (std::uint32_t) real::code_of<real::storage_t>::code) return 0;

    sparse::init(&part,
                 (types::dim_t) m->part_rows[part_id],
                 (types::dim_t) m->cols,
                 (types::nnz_t) m->part_nnz[part_id],
                 (types::u32) m->part_aux[part_id]);
    if (!sparse::allocate(&part)) goto done;

    payload = H5Gopen2(state->file, payload_standard_group, H5P_DEFAULT);
    if (payload < 0) goto done;
    d_indptr = H5Dopen2(payload, "indptr", H5P_DEFAULT);
    d_indices = H5Dopen2(payload, "indices", H5P_DEFAULT);
    d_values = H5Dopen2(payload, "values", H5P_DEFAULT);
    if (d_indptr < 0 || d_indices < 0 || d_values < 0) goto done;
    if (!read_hyperslab_1d(d_indptr, H5T_NATIVE_UINT32, state->part_indptr_offsets[part_id], (std::uint64_t) part.rows + 1u, part.majorPtr)) goto done;
    if (!read_hyperslab_1d(d_indices, H5T_NATIVE_UINT32, state->part_nnz_offsets[part_id], (std::uint64_t) part.nnz, part.minorIdx)) goto done;
    if (!read_hyperslab_1d(d_values, H5T_NATIVE_UINT16, state->part_nnz_offsets[part_id], (std::uint64_t) part.nnz, part.val)) goto done;
    if (!store_standard_csr_part_to_cache(state, part_id, &part)) goto done;
    ok = 1;

done:
    if (d_values >= 0) H5Dclose(d_values);
    if (d_indices >= 0) H5Dclose(d_indices);
    if (d_indptr >= 0) H5Dclose(d_indptr);
    if (payload >= 0) H5Gclose(payload);
    sparse::clear(&part);
    return ok;
}

int prefetch_series_compressed_h5_shard_to_cache(const sharded<sparse::compressed> *m,
                                                 const shard_storage *s,
                                                 unsigned long shard_id) {
    unsigned long begin = 0;
    unsigned long end = 0;
    unsigned long i = 0;

    if (m == 0 || s == 0 || shard_id >= m->num_shards) return 0;
    begin = first_part_in_shard(m, shard_id);
    end = last_part_in_shard(m, shard_id);
    for (i = begin; i < end; ++i) {
        if (!prefetch_series_compressed_h5_part_to_cache(m, s, i)) return 0;
    }
    return 1;
}

} // namespace cellshard
