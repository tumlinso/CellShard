#pragma once

#include "../../io/source/file_reader.cuh"
#include "text_column.cuh"

namespace cellshard {
namespace ingest {
namespace common {

struct barcode_table {
    text_column values;
};

static inline void init(barcode_table *t) {
    init(&t->values);
}

static inline void clear(barcode_table *t) {
    clear(&t->values);
}

static inline unsigned int count(const barcode_table *t) {
    return t->values.count;
}

static inline const char *get(const barcode_table *t, unsigned int idx) {
    return common::get(&t->values, idx);
}

static inline int append(barcode_table *t, const char *barcode, std::size_t len) {
    return common::append(&t->values, barcode, len);
}

static inline int load_lines(const char *path, barcode_table *t) {
    io::source::buffered_file_reader reader;
    int rc = 0;
    char *line = 0;
    std::size_t line_len = 0;

    io::source::init(&reader);
    clear(t);
    init(t);

    if (!io::source::open(&reader, path)) goto fail;

    for (;;) {
        rc = io::source::next_line(&reader, &line, &line_len);
        if (rc < 0) goto fail;
        if (rc == 0) break;
        if (reader.line_number == 1u) io::source::strip_utf8_bom(line, &line_len);
        if (line_len == 0) continue;
        if (!append(t, line, line_len)) goto fail;
    }

    io::source::clear(&reader);
    return 1;

fail:
    io::source::clear(&reader);
    clear(t);
    return 0;
}

} // namespace common
} // namespace ingest
} // namespace cellshard
