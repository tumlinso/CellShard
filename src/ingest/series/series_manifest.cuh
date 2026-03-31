#pragma once

#include <cstddef>
#include <cstdlib>
#include <cstring>

#include "../common/text_column.cuh"
#include "../scan.cuh"

namespace cellshard {
namespace ingest {
namespace series {

enum {
    source_unknown   = 0,
    source_mtx       = 1,
    source_tenx_mtx  = 2,
    source_tenx_h5   = 3,
    source_h5ad      = 4,
    source_loom      = 5,
    source_binary    = 6
};

struct manifest {
    unsigned int count;
    unsigned int capacity;

    common::text_column dataset_ids;
    common::text_column matrix_paths;
    common::text_column feature_paths;
    common::text_column barcode_paths;
    common::text_column metadata_paths;

    unsigned int *formats;
    unsigned long *rows;
    unsigned long *cols;
    unsigned long *nnz;
};

static inline void init(manifest *m) {
    m->count = 0;
    m->capacity = 0;
    common::init(&m->dataset_ids);
    common::init(&m->matrix_paths);
    common::init(&m->feature_paths);
    common::init(&m->barcode_paths);
    common::init(&m->metadata_paths);
    m->formats = 0;
    m->rows = 0;
    m->cols = 0;
    m->nnz = 0;
}

static inline void clear(manifest *m) {
    common::clear(&m->dataset_ids);
    common::clear(&m->matrix_paths);
    common::clear(&m->feature_paths);
    common::clear(&m->barcode_paths);
    common::clear(&m->metadata_paths);
    std::free(m->formats);
    std::free(m->rows);
    std::free(m->cols);
    std::free(m->nnz);
    init(m);
}

static inline int reserve(manifest *m, unsigned int capacity) {
    unsigned int *next_formats = 0;
    unsigned long *next_rows = 0;
    unsigned long *next_cols = 0;
    unsigned long *next_nnz = 0;

    if (capacity <= m->capacity) return 1;
    next_formats = (unsigned int *) std::calloc((std::size_t) capacity, sizeof(unsigned int));
    next_rows = (unsigned long *) std::calloc((std::size_t) capacity, sizeof(unsigned long));
    next_cols = (unsigned long *) std::calloc((std::size_t) capacity, sizeof(unsigned long));
    next_nnz = (unsigned long *) std::calloc((std::size_t) capacity, sizeof(unsigned long));
    if (next_formats == 0 || next_rows == 0 || next_cols == 0 || next_nnz == 0) {
        std::free(next_formats);
        std::free(next_rows);
        std::free(next_cols);
        std::free(next_nnz);
        return 0;
    }
    if (m->count != 0) {
        std::memcpy(next_formats, m->formats, (std::size_t) m->count * sizeof(unsigned int));
        std::memcpy(next_rows, m->rows, (std::size_t) m->count * sizeof(unsigned long));
        std::memcpy(next_cols, m->cols, (std::size_t) m->count * sizeof(unsigned long));
        std::memcpy(next_nnz, m->nnz, (std::size_t) m->count * sizeof(unsigned long));
    }
    std::free(m->formats);
    std::free(m->rows);
    std::free(m->cols);
    std::free(m->nnz);
    m->formats = next_formats;
    m->rows = next_rows;
    m->cols = next_cols;
    m->nnz = next_nnz;
    m->capacity = capacity;
    return 1;
}

static inline int parse_u64_field(const char *s, unsigned long *out) {
    char *end = 0;
    unsigned long long v = 0;

    if (s == 0 || *s == 0) {
        *out = 0;
        return 1;
    }
    v = std::strtoull(s, &end, 10);
    if (end == s || *end != 0) return 0;
    *out = (unsigned long) v;
    if ((unsigned long long) *out != v) return 0;
    return 1;
}

static inline unsigned int parse_format(const char *s) {
    if (s == 0 || *s == 0) return source_unknown;
    if (std::strcmp(s, "mtx") == 0) return source_mtx;
    if (std::strcmp(s, "tenx_mtx") == 0) return source_tenx_mtx;
    if (std::strcmp(s, "tenx_h5") == 0) return source_tenx_h5;
    if (std::strcmp(s, "h5ad") == 0) return source_h5ad;
    if (std::strcmp(s, "loom") == 0) return source_loom;
    if (std::strcmp(s, "binary") == 0) return source_binary;
    return source_unknown;
}

static inline int append(manifest *m,
                         const char *dataset_id,
                         const char *matrix_path,
                         unsigned int format,
                         const char *feature_path,
                         const char *barcode_path,
                         const char *metadata_path,
                         unsigned long rows,
                         unsigned long cols,
                         unsigned long nnz) {
    unsigned int idx = 0;

    if (m->count == m->capacity) {
        if (!reserve(m, m->capacity == 0 ? 16u : m->capacity << 1u)) return 0;
    }

    idx = m->count;
    if (!common::append(&m->dataset_ids, dataset_id != 0 ? dataset_id : "", std::strlen(dataset_id != 0 ? dataset_id : ""))) return 0;
    if (!common::append(&m->matrix_paths, matrix_path != 0 ? matrix_path : "", std::strlen(matrix_path != 0 ? matrix_path : ""))) return 0;
    if (!common::append(&m->feature_paths, feature_path != 0 ? feature_path : "", std::strlen(feature_path != 0 ? feature_path : ""))) return 0;
    if (!common::append(&m->barcode_paths, barcode_path != 0 ? barcode_path : "", std::strlen(barcode_path != 0 ? barcode_path : ""))) return 0;
    if (!common::append(&m->metadata_paths, metadata_path != 0 ? metadata_path : "", std::strlen(metadata_path != 0 ? metadata_path : ""))) return 0;
    m->formats[idx] = format;
    m->rows[idx] = rows;
    m->cols[idx] = cols;
    m->nnz[idx] = nnz;
    ++m->count;
    return 1;
}

static inline int header_index(char **fields, unsigned int count, const char *name) {
    unsigned int i = 0;
    for (i = 0; i < count; ++i) {
        if (std::strcmp(fields[i], name) == 0) return (int) i;
    }
    return -1;
}

static inline int load_tsv(const char *path, manifest *m, int has_header = 1) {
    scan::buffered_file_reader reader;
    int rc = 0;
    char *line = 0;
    std::size_t line_len = 0;
    char *fields[64];
    unsigned int nfields = 0;
    int idx_dataset = 0;
    int idx_path = 1;
    int idx_format = 2;
    int idx_features = 3;
    int idx_barcodes = 4;
    int idx_metadata = 5;
    int idx_rows = 6;
    int idx_cols = 7;
    int idx_nnz = 8;
    unsigned long rows = 0;
    unsigned long cols = 0;
    unsigned long nnz = 0;

    scan::init(&reader);
    clear(m);
    init(m);

    if (!scan::open(&reader, path)) goto fail;

    for (;;) {
        rc = scan::next_line(&reader, &line, &line_len);
        if (rc < 0) goto fail;
        if (rc == 0) break;
        if (reader.line_number == 1u) scan::strip_utf8_bom(line, &line_len);
        if (line_len == 0) continue;
        nfields = scan::split_tabs(line, fields, 64u);
        if (nfields == 0) continue;

        if (has_header && reader.line_number == 1u) {
            idx_dataset = header_index(fields, nfields, "dataset");
            if (idx_dataset < 0) idx_dataset = header_index(fields, nfields, "sample");
            if (idx_dataset < 0) idx_dataset = 0;
            idx_path = header_index(fields, nfields, "path");
            if (idx_path < 0) idx_path = header_index(fields, nfields, "matrix");
            if (idx_path < 0) idx_path = 1;
            idx_format = header_index(fields, nfields, "format");
            if (idx_format < 0) idx_format = 2;
            idx_features = header_index(fields, nfields, "features");
            idx_barcodes = header_index(fields, nfields, "barcodes");
            idx_metadata = header_index(fields, nfields, "metadata");
            idx_rows = header_index(fields, nfields, "rows");
            idx_cols = header_index(fields, nfields, "cols");
            idx_nnz = header_index(fields, nfields, "nnz");
            continue;
        }

        if (!parse_u64_field(scan::field_or_empty(fields, nfields, (unsigned int) (idx_rows >= 0 ? idx_rows : 63)), &rows)) goto fail;
        if (!parse_u64_field(scan::field_or_empty(fields, nfields, (unsigned int) (idx_cols >= 0 ? idx_cols : 63)), &cols)) goto fail;
        if (!parse_u64_field(scan::field_or_empty(fields, nfields, (unsigned int) (idx_nnz >= 0 ? idx_nnz : 63)), &nnz)) goto fail;
        if (!append(m,
                    scan::field_or_empty(fields, nfields, (unsigned int) idx_dataset),
                    scan::field_or_empty(fields, nfields, (unsigned int) idx_path),
                    parse_format(scan::field_or_empty(fields, nfields, (unsigned int) idx_format)),
                    scan::field_or_empty(fields, nfields, (unsigned int) (idx_features >= 0 ? idx_features : 63)),
                    scan::field_or_empty(fields, nfields, (unsigned int) (idx_barcodes >= 0 ? idx_barcodes : 63)),
                    scan::field_or_empty(fields, nfields, (unsigned int) (idx_metadata >= 0 ? idx_metadata : 63)),
                    rows,
                    cols,
                    nnz)) goto fail;
    }

    scan::clear(&reader);
    return 1;

fail:
    scan::clear(&reader);
    clear(m);
    return 0;
}

} // namespace series
} // namespace ingest
} // namespace cellshard
