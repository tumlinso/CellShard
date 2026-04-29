#pragma once

#include "../scan.cuh"
#include "text_column.cuh"

namespace cellshard {
namespace ingest {
namespace common {

// Feature metadata columns loaded from a TSV-like source.
struct feature_table {
    text_column ids;
    text_column names;
    text_column types;
};

// Metadata-only init / release.
static inline void init(feature_table *t) {
    init(&t->ids);
    init(&t->names);
    init(&t->types);
}

static inline void clear(feature_table *t) {
    clear(&t->ids);
    clear(&t->names);
    clear(&t->types);
}

// Cheap column accessors over packed string storage.
static inline unsigned int count(const feature_table *t) {
    return t->ids.count;
}

static inline const char *id(const feature_table *t, unsigned int idx) {
    return common::get(&t->ids, idx);
}

static inline const char *name(const feature_table *t, unsigned int idx) {
    return common::get(&t->names, idx);
}

static inline const char *type(const feature_table *t, unsigned int idx) {
    return common::get(&t->types, idx);
}

// Append one feature row by copying three strings.
static inline int append(feature_table *t,
                         const char *id_ptr,
                         std::size_t id_len,
                         const char *name_ptr,
                         std::size_t name_len,
                         const char *type_ptr,
                         std::size_t type_len) {
    if (!common::append(&t->ids, id_ptr, id_len)) return 0;
    if (!common::append(&t->names, name_ptr, name_len)) return 0;
    if (!common::append(&t->types, type_ptr, type_len)) return 0;
    return 1;
}

// Full synchronous feature TSV ingest into host-owned packed columns.
static inline int load_tsv(const char *path, feature_table *t, int skip_header = 0) {
    scan::buffered_file_reader reader;
    int rc = 0;
    char *line = 0;
    std::size_t line_len = 0;
    char *fields[4];
    unsigned int nfields = 0;

    scan::init(&reader);
    clear(t);
    init(t);

    if (!scan::open(&reader, path)) goto fail;

    for (;;) {
        rc = scan::next_line(&reader, &line, &line_len);
        if (rc < 0) goto fail;
        if (rc == 0) break;
        if (reader.line_number == 1u) scan::strip_utf8_bom(line, &line_len);
        if (line_len == 0) continue;
        if (skip_header && reader.line_number == 1u) continue;
        nfields = scan::split_tabs(line, fields, 4u);
        if (nfields == 0) continue;
        if (!append(t,
                    scan::field_or_empty(fields, nfields, 0u), std::strlen(scan::field_or_empty(fields, nfields, 0u)),
                    scan::field_or_empty(fields, nfields, 1u), std::strlen(scan::field_or_empty(fields, nfields, 1u)),
                    scan::field_or_empty(fields, nfields, 2u), std::strlen(scan::field_or_empty(fields, nfields, 2u)))) goto fail;
    }

    scan::clear(&reader);
    return 1;

fail:
    scan::clear(&reader);
    clear(t);
    return 0;
}

} // namespace common
} // namespace ingest
} // namespace cellshard
