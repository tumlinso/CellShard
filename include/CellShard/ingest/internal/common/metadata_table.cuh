#pragma once

#include "../scan.cuh"
#include "text_column.cuh"

namespace cellshard {
namespace ingest {
namespace common {

// Host-side TSV metadata table:
// - column names stored in one packed string column
// - all field values stored in one packed string column
// - row_offsets maps rows to the flat field-value array
//
// This is intentionally ingest-oriented and copy-heavy.
struct metadata_table {
    unsigned int num_rows;
    unsigned int num_cols;
    unsigned int row_capacity;
    unsigned int *row_offsets;
    text_column column_names;
    text_column field_values;
};

// Metadata-only init.
static inline void init(metadata_table *t) {
    t->num_rows = 0;
    t->num_cols = 0;
    t->row_capacity = 0;
    t->row_offsets = 0;
    init(&t->column_names);
    init(&t->field_values);
}

// Release all host-side metadata storage.
static inline void clear(metadata_table *t) {
    std::free(t->row_offsets);
    t->row_offsets = 0;
    t->row_capacity = 0;
    clear(&t->column_names);
    clear(&t->field_values);
    t->num_rows = 0;
    t->num_cols = 0;
}

// Grow the row-offset table and copy existing offsets.
static inline int reserve_rows(metadata_table *t, unsigned int capacity) {
    unsigned int *next = 0;

    if (capacity <= t->row_capacity) return 1;
    next = (unsigned int *) std::calloc((std::size_t) capacity + 1u, sizeof(unsigned int));
    if (next == 0) return 0;
    if (t->row_offsets != 0 && t->num_rows != 0) {
        std::memcpy(next, t->row_offsets, ((std::size_t) t->num_rows + 1u) * sizeof(unsigned int));
    } else {
        next[0] = 0;
    }
    std::free(t->row_offsets);
    t->row_offsets = next;
    t->row_capacity = capacity;
    return 1;
}

// Replace the current header with a fresh set of column names.
static inline int append_header(metadata_table *t, char **fields, unsigned int count) {
    unsigned int i = 0;

    clear(&t->column_names);
    init(&t->column_names);
    for (i = 0; i < count; ++i) {
        if (!common::append(&t->column_names, fields[i], std::strlen(fields[i]))) return 0;
    }
    t->num_cols = count;
    return 1;
}

// Append one row by copying every field string into field_values.
static inline int append_row(metadata_table *t, char **fields, unsigned int count) {
    unsigned int i = 0;

    if (t->num_cols == 0) t->num_cols = count;
    if (count != t->num_cols) return 0;
    if (t->num_rows + 1u > t->row_capacity) {
        if (!reserve_rows(t, t->row_capacity == 0 ? 256u : t->row_capacity << 1u)) return 0;
    }
    for (i = 0; i < count; ++i) {
        if (!common::append(&t->field_values, fields[i], std::strlen(fields[i]))) return 0;
    }
    ++t->num_rows;
    t->row_offsets[t->num_rows] = t->field_values.count;
    return 1;
}

// Column-name and field lookup helpers.
static inline const char *column_name(const metadata_table *t, unsigned int col) {
    return common::get(&t->column_names, col);
}

static inline const char *field(const metadata_table *t, unsigned int row, unsigned int col) {
    unsigned int field_idx = 0;

    if (row >= t->num_rows || col >= t->num_cols) return 0;
    field_idx = t->row_offsets[row] + col;
    return common::get(&t->field_values, field_idx);
}

// Full synchronous TSV ingest into host-owned packed string columns.
static inline int load_tsv(const char *path, metadata_table *t, int has_header = 1) {
    scan::buffered_file_reader reader;
    int rc = 0;
    char *line = 0;
    std::size_t line_len = 0;
    char *fields[1024];
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
        nfields = scan::split_tabs(line, fields, 1024u);
        if (nfields == 0) continue;

        if (has_header && reader.line_number == 1u) {
            if (!append_header(t, fields, nfields)) goto fail;
            continue;
        }
        if (!append_row(t, fields, nfields)) goto fail;
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
