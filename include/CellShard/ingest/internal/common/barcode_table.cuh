#pragma once

#include "../scan.cuh"
#include "text_column.cuh"

namespace cellshard {
namespace ingest {
namespace common {

// One-column barcode table stored in packed-string form.
struct barcode_table {
    text_column values;
};

// Metadata-only init / release.
static inline void init(barcode_table *t) {
    init(&t->values);
}

static inline void clear(barcode_table *t) {
    clear(&t->values);
}

// Cheap accessors over packed-string storage.
static inline unsigned int count(const barcode_table *t) {
    return t->values.count;
}

static inline const char *get(const barcode_table *t, unsigned int idx) {
    return common::get(&t->values, idx);
}

// Append one barcode by copying it into the packed byte blob.
static inline int append(barcode_table *t, const char *barcode, std::size_t len) {
    return common::append(&t->values, barcode, len);
}

// Full synchronous barcode ingest. Cost scales with file bytes and packed
// string-column growth.
static inline int load_lines(const char *path, barcode_table *t) {
    scan::buffered_file_reader reader;
    int rc = 0;
    char *line = 0;
    std::size_t line_len = 0;

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
        if (!append(t, line, line_len)) goto fail;
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
