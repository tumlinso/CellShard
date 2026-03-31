#pragma once

#include <climits>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "../../matrix.cuh"
#include "../../io/source/file_reader.cuh"

namespace cellshard {
namespace ingest {
namespace mtx {

struct header {
    unsigned long rows;
    unsigned long cols;
    unsigned long nnz_file;
    unsigned long nnz_loaded;
    int symmetric;
    int pattern;
    int integer_values;
    int real_values;
};

static inline void init(header *h) {
    h->rows = 0;
    h->cols = 0;
    h->nnz_file = 0;
    h->nnz_loaded = 0;
    h->symmetric = 0;
    h->pattern = 0;
    h->integer_values = 0;
    h->real_values = 0;
}

static inline int parse_u64_token(const char *s, unsigned long *out) {
    char *end = 0;
    unsigned long long v = 0;

    v = std::strtoull(s, &end, 10);
    if (end == s || *end != 0) return 0;
    *out = (unsigned long) v;
    if ((unsigned long long) *out != v) return 0;
    return 1;
}

static inline int parse_f32_token(const char *s, float *out) {
    char *end = 0;
    float v = 0.0f;

    v = std::strtof(s, &end);
    if (end == s || *end != 0) return 0;
    *out = v;
    return 1;
}

static inline int parse_banner(char *line, header *h) {
    char *fields[8];
    unsigned int nfields = 0;

    init(h);
    nfields = io::source::split_ws(line, fields, 8u);
    if (nfields < 5u) return 0;
    if (std::strcmp(fields[0], "%%MatrixMarket") != 0) return 0;
    if (std::strcmp(fields[1], "matrix") != 0) return 0;
    if (std::strcmp(fields[2], "coordinate") != 0) return 0;

    if (std::strcmp(fields[3], "pattern") == 0) h->pattern = 1;
    else if (std::strcmp(fields[3], "integer") == 0) h->integer_values = 1;
    else if (std::strcmp(fields[3], "real") == 0) h->real_values = 1;
    else return 0;

    if (std::strcmp(fields[4], "general") == 0) h->symmetric = 0;
    else if (std::strcmp(fields[4], "symmetric") == 0) h->symmetric = 1;
    else return 0;

    return 1;
}

static inline int read_header(io::source::buffered_file_reader *reader, header *h) {
    int rc = 0;
    char *line = 0;
    std::size_t line_len = 0;
    char *fields[4];
    unsigned int nfields = 0;

    init(h);

    rc = io::source::next_line(reader, &line, &line_len);
    if (rc <= 0) return 0;
    if (reader->line_number == 1u) io::source::strip_utf8_bom(line, &line_len);
    if (!parse_banner(line, h)) return 0;

    rc = io::source::skip_empty_and_comment_lines(reader, &line, &line_len, '%');
    if (rc <= 0) return 0;

    nfields = io::source::split_ws(line, fields, 4u);
    if (nfields < 3u) return 0;
    if (!parse_u64_token(fields[0], &h->rows)) return 0;
    if (!parse_u64_token(fields[1], &h->cols)) return 0;
    if (!parse_u64_token(fields[2], &h->nnz_file)) return 0;
    h->nnz_loaded = h->nnz_file;
    return 1;
}

static inline int read_header(const char *path, header *h) {
    io::source::buffered_file_reader reader;
    int ok = 0;

    io::source::init(&reader);
    if (!io::source::open(&reader, path)) return 0;
    ok = read_header(&reader, h);
    io::source::clear(&reader);
    return ok;
}

static inline int read_triplet(char *line,
                               const header *h,
                               unsigned long *row,
                               unsigned long *col,
                               float *value) {
    char *fields[4];
    unsigned int nfields = 0;

    nfields = io::source::split_ws(line, fields, 4u);
    if (nfields < 2u) return 0;
    if (!parse_u64_token(fields[0], row)) return 0;
    if (!parse_u64_token(fields[1], col)) return 0;
    if (*row == 0 || *col == 0) return 0;
    --(*row);
    --(*col);
    if (*row >= h->rows || *col >= h->cols) return 0;

    if (h->pattern) {
        *value = 1.0f;
        return 1;
    }
    if (nfields < 3u) return 0;
    return parse_f32_token(fields[2], value);
}

static inline int scan_row_nnz(io::source::buffered_file_reader *reader, const header *h, unsigned long *row_nnz) {
    int rc = 0;
    char *line = 0;
    std::size_t line_len = 0;
    unsigned long row = 0;
    unsigned long col = 0;
    float value = 0.0f;

    while ((rc = io::source::next_line(reader, &line, &line_len)) > 0) {
        if (line_len == 0 || line[0] == '%') continue;
        if (!read_triplet(line, h, &row, &col, &value)) return 0;
        ++row_nnz[row];
        if (h->symmetric && row != col) ++row_nnz[col];
    }
    return rc == 0;
}

static inline int scan_row_nnz(const char *path, header *h, unsigned long **row_nnz_out) {
    io::source::buffered_file_reader reader;
    unsigned long *row_nnz = 0;
    int ok = 0;

    *row_nnz_out = 0;
    io::source::init(&reader);
    if (!io::source::open(&reader, path)) return 0;
    if (!read_header(&reader, h)) goto done;
    row_nnz = (unsigned long *) std::calloc((std::size_t) h->rows, sizeof(unsigned long));
    if (h->rows != 0 && row_nnz == 0) goto done;
    if (!scan_row_nnz(&reader, h, row_nnz)) goto done;
    ok = 1;

done:
    io::source::clear(&reader);
    if (!ok) {
        std::free(row_nnz);
        return 0;
    }
    *row_nnz_out = row_nnz;
    return 1;
}

static inline int plan_row_partitions_by_nnz(const unsigned long *row_nnz,
                                             unsigned long rows,
                                             unsigned long max_nnz,
                                             unsigned long **row_offsets_out,
                                             unsigned long *num_parts_out) {
    unsigned long *offsets = 0;
    unsigned long part_count = 0;
    unsigned long row = 0;
    unsigned long shard_nnz = 0;

    *row_offsets_out = 0;
    *num_parts_out = 0;

    offsets = (unsigned long *) std::malloc(((std::size_t) rows + 1u) * sizeof(unsigned long));
    if (offsets == 0) return 0;

    offsets[0] = 0;
    for (row = 0; row < rows; ++row) {
        if (max_nnz != 0 && shard_nnz != 0 && shard_nnz + row_nnz[row] > max_nnz) {
            ++part_count;
            offsets[part_count] = row;
            shard_nnz = 0;
        }
        shard_nnz += row_nnz[row];
    }
    ++part_count;
    offsets[part_count] = rows;
    *row_offsets_out = offsets;
    *num_parts_out = part_count;
    return 1;
}

static inline int validate_row_offsets(const header *h,
                                       const unsigned long *row_offsets,
                                       unsigned long num_parts) {
    unsigned long i = 0;

    if (row_offsets == 0 || num_parts == 0) return 0;
    if (row_offsets[0] != 0) return 0;
    if (row_offsets[num_parts] != h->rows) return 0;
    for (i = 0; i < num_parts; ++i) {
        if (row_offsets[i] > row_offsets[i + 1]) return 0;
    }
    return 1;
}

static inline int count_part_nnz(io::source::buffered_file_reader *reader,
                                 const header *h,
                                 const unsigned long *row_offsets,
                                 unsigned long num_parts,
                                 unsigned long *part_nnz) {
    int rc = 0;
    char *line = 0;
    std::size_t line_len = 0;
    unsigned long row = 0;
    unsigned long col = 0;
    unsigned long part = 0;
    float value = 0.0f;

    while ((rc = io::source::next_line(reader, &line, &line_len)) > 0) {
        if (line_len == 0 || line[0] == '%') continue;
        if (!read_triplet(line, h, &row, &col, &value)) return 0;
        part = find_offset_span(row, row_offsets, num_parts);
        if (part >= num_parts) return 0;
        ++part_nnz[part];
        if (h->symmetric && row != col) {
            part = find_offset_span(col, row_offsets, num_parts);
            if (part >= num_parts) return 0;
            ++part_nnz[part];
        }
    }
    return rc == 0;
}

static inline int cast_dim(unsigned long v, unsigned int *out) {
    if (v > (unsigned long) UINT_MAX) return 0;
    *out = (unsigned int) v;
    return 1;
}

static inline std::size_t estimate_coo_part_bytes(unsigned long rows, unsigned long nnz) {
    return sizeof(sparse::coo)
         + (std::size_t) nnz * sizeof(unsigned int)
         + (std::size_t) nnz * sizeof(unsigned int)
         + (std::size_t) nnz * sizeof(__half);
}

static inline int count_all_part_nnz(const char *path,
                                     const header *h,
                                     const unsigned long *row_offsets,
                                     unsigned long num_parts,
                                     unsigned long **part_nnz_out) {
    io::source::buffered_file_reader reader;
    unsigned long *part_nnz = 0;
    header local;
    int ok = 0;

    *part_nnz_out = 0;
    init(&local);
    io::source::init(&reader);
    if (!io::source::open(&reader, path)) return 0;
    if (!read_header(&reader, &local)) goto done;
    if (h != 0) {
        if (local.rows != h->rows || local.cols != h->cols || local.nnz_file != h->nnz_file) goto done;
    }
    part_nnz = (unsigned long *) std::calloc((std::size_t) num_parts, sizeof(unsigned long));
    if (num_parts != 0 && part_nnz == 0) goto done;
    if (!count_part_nnz(&reader, &local, row_offsets, num_parts, part_nnz)) goto done;
    ok = 1;

done:
    io::source::clear(&reader);
    if (!ok) {
        std::free(part_nnz);
        return 0;
    }
    *part_nnz_out = part_nnz;
    return 1;
}

static inline void build_part_rows(const unsigned long *row_offsets,
                                   unsigned long num_parts,
                                   unsigned long *part_rows) {
    unsigned long i = 0;
    for (i = 0; i < num_parts; ++i) part_rows[i] = row_offsets[i + 1] - row_offsets[i];
}

static inline void build_part_bytes_from_nnz(const unsigned long *row_offsets,
                                             const unsigned long *part_nnz,
                                             unsigned long num_parts,
                                             unsigned long *part_bytes) {
    unsigned long i = 0;
    for (i = 0; i < num_parts; ++i) {
        part_bytes[i] = (unsigned long) estimate_coo_part_bytes(row_offsets[i + 1] - row_offsets[i], part_nnz[i]);
    }
}

static inline unsigned long sum_part_nnz(const unsigned long *part_nnz,
                                         unsigned long part_begin,
                                         unsigned long part_end) {
    unsigned long i = 0;
    unsigned long total = 0;

    for (i = part_begin; i < part_end; ++i) total += part_nnz[i];
    return total;
}

static inline unsigned long sum_part_rows(const unsigned long *row_offsets,
                                          unsigned long part_begin,
                                          unsigned long part_end) {
    if (part_end <= part_begin) return 0;
    return row_offsets[part_end] - row_offsets[part_begin];
}

static inline int allocate_sharded_coo(const header *h,
                                       const unsigned long *row_offsets,
                                       unsigned long num_parts,
                                       const unsigned long *part_nnz,
                                       sharded<sparse::coo> *out) {
    unsigned long i = 0;
    unsigned int rows_u32 = 0;
    unsigned int cols_u32 = 0;
    unsigned int nnz_u32 = 0;
    sparse::coo *part = 0;

    clear(out);
    init(out);
    if (!reserve_parts(out, num_parts)) return 0;

    if (!cast_dim(h->cols, &cols_u32)) return 0;

    out->num_parts = num_parts;
    out->cols = h->cols;
    for (i = 0; i < num_parts; ++i) {
        if (!cast_dim(row_offsets[i + 1] - row_offsets[i], &rows_u32)) return 0;
        if (!cast_dim(part_nnz[i], &nnz_u32)) return 0;
        part = new sparse::coo;
        sparse::init(part, rows_u32, cols_u32, nnz_u32);
        if (!sparse::allocate(part)) {
            delete part;
            clear(out);
            return 0;
        }
        out->parts[i] = part;
        out->part_rows[i] = row_offsets[i + 1] - row_offsets[i];
        out->part_nnz[i] = part_nnz[i];
        out->part_aux[i] = 0;
    }
    rebuild_part_offsets(out);
    return set_shards_to_parts(out);
}

static inline int fill_sharded_coo(io::source::buffered_file_reader *reader,
                                   const header *h,
                                   const unsigned long *row_offsets,
                                   unsigned long num_parts,
                                   sharded<sparse::coo> *out) {
    int rc = 0;
    char *line = 0;
    std::size_t line_len = 0;
    unsigned long row = 0;
    unsigned long col = 0;
    unsigned long part = 0;
    unsigned long idx = 0;
    unsigned long *write_ptr = 0;
    float value = 0.0f;

    write_ptr = (unsigned long *) std::calloc((std::size_t) num_parts, sizeof(unsigned long));
    if (num_parts != 0 && write_ptr == 0) return 0;

    while ((rc = io::source::next_line(reader, &line, &line_len)) > 0) {
        if (line_len == 0 || line[0] == '%') continue;
        if (!read_triplet(line, h, &row, &col, &value)) goto fail;

        part = find_offset_span(row, row_offsets, num_parts);
        if (part >= num_parts) goto fail;
        idx = write_ptr[part]++;
        out->parts[part]->rowIdx[idx] = (unsigned int) (row - row_offsets[part]);
        out->parts[part]->colIdx[idx] = (unsigned int) col;
        out->parts[part]->val[idx] = __float2half(value);

        if (h->symmetric && row != col) {
            part = find_offset_span(col, row_offsets, num_parts);
            if (part >= num_parts) goto fail;
            idx = write_ptr[part]++;
            out->parts[part]->rowIdx[idx] = (unsigned int) (col - row_offsets[part]);
            out->parts[part]->colIdx[idx] = (unsigned int) row;
            out->parts[part]->val[idx] = __float2half(value);
        }
    }
    if (rc < 0) goto fail;

    std::free(write_ptr);
    return 1;

fail:
    std::free(write_ptr);
    return 0;
}

static inline int load_row_sharded_coo(const char *path,
                                       const unsigned long *row_offsets,
                                       unsigned long num_parts,
                                       sharded<sparse::coo> *out) {
    io::source::buffered_file_reader reader;
    header h;
    unsigned long *part_nnz = 0;
    int ok = 0;

    init(&h);
    io::source::init(&reader);

    if (!io::source::open(&reader, path)) return 0;
    if (!read_header(&reader, &h)) goto done;
    if (!validate_row_offsets(&h, row_offsets, num_parts)) goto done;

    part_nnz = (unsigned long *) std::calloc((std::size_t) num_parts, sizeof(unsigned long));
    if (num_parts != 0 && part_nnz == 0) goto done;
    if (!count_part_nnz(&reader, &h, row_offsets, num_parts, part_nnz)) goto done;
    io::source::clear(&reader);
    io::source::init(&reader);

    if (!allocate_sharded_coo(&h, row_offsets, num_parts, part_nnz, out)) goto done;

    if (!io::source::open(&reader, path)) goto done;
    if (!read_header(&reader, &h)) goto done;
    if (!fill_sharded_coo(&reader, &h, row_offsets, num_parts, out)) goto done;
    ok = 1;

done:
    io::source::clear(&reader);
    std::free(part_nnz);
    if (!ok) clear(out);
    return ok;
}

static inline int load_coo(const char *path, sparse::coo *out) {
    header h;
    unsigned long row_offsets[2];
    sharded<sparse::coo> tmp;
    sparse::coo *part = 0;
    int ok = 0;

    init(&h);
    init(&tmp);
    if (!read_header(path, &h)) return 0;
    row_offsets[0] = 0;
    row_offsets[1] = h.rows;
    if (!load_row_sharded_coo(path, row_offsets, 1, &tmp)) goto done;
    if (tmp.num_parts != 1 || tmp.parts[0] == 0) goto done;

    sparse::clear(out);
    sparse::init(out);
    part = tmp.parts[0];
    *out = *part;
    delete part;
    tmp.parts[0] = 0;
    ok = 1;

done:
    clear(&tmp);
    return ok;
}

static inline int allocate_part_window_coo(const header *h,
                                           const unsigned long *row_offsets,
                                           const unsigned long *part_nnz,
                                           unsigned long num_parts,
                                           unsigned long part_begin,
                                           unsigned long part_end,
                                           sharded<sparse::coo> *out) {
    unsigned long i = 0;
    unsigned long local = 0;
    unsigned int rows_u32 = 0;
    unsigned int cols_u32 = 0;
    unsigned int nnz_u32 = 0;
    sparse::coo *part = 0;

    clear(out);
    init(out);
    if (part_begin >= part_end || part_end > num_parts) return 0;
    if (!cast_dim(h->cols, &cols_u32)) return 0;
    if (!reserve_parts(out, part_end - part_begin)) return 0;

    out->num_parts = part_end - part_begin;
    out->cols = h->cols;
    for (i = part_begin; i < part_end; ++i) {
        local = i - part_begin;
        if (!cast_dim(row_offsets[i + 1] - row_offsets[i], &rows_u32)) return 0;
        if (!cast_dim(part_nnz[i], &nnz_u32)) return 0;
        part = new sparse::coo;
        sparse::init(part, rows_u32, cols_u32, nnz_u32);
        if (!sparse::allocate(part)) {
            delete part;
            clear(out);
            return 0;
        }
        out->parts[local] = part;
        out->part_rows[local] = row_offsets[i + 1] - row_offsets[i];
        out->part_nnz[local] = part_nnz[i];
        out->part_aux[local] = 0;
    }
    rebuild_part_offsets(out);
    return set_shards_to_parts(out);
}

static inline int fill_part_window_coo(io::source::buffered_file_reader *reader,
                                       const header *h,
                                       const unsigned long *row_offsets,
                                       unsigned long num_parts,
                                       unsigned long part_begin,
                                       unsigned long part_end,
                                       sharded<sparse::coo> *out) {
    int rc = 0;
    char *line = 0;
    std::size_t line_len = 0;
    unsigned long row = 0;
    unsigned long col = 0;
    unsigned long global_part = 0;
    unsigned long local_part = 0;
    unsigned long idx = 0;
    unsigned long *write_ptr = 0;
    float value = 0.0f;

    write_ptr = (unsigned long *) std::calloc((std::size_t) out->num_parts, sizeof(unsigned long));
    if (out->num_parts != 0 && write_ptr == 0) return 0;

    while ((rc = io::source::next_line(reader, &line, &line_len)) > 0) {
        if (line_len == 0 || line[0] == '%') continue;
        if (!read_triplet(line, h, &row, &col, &value)) goto fail;

        global_part = find_offset_span(row, row_offsets, num_parts);
        if (global_part >= part_begin && global_part < part_end) {
            local_part = global_part - part_begin;
            idx = write_ptr[local_part]++;
            out->parts[local_part]->rowIdx[idx] = (unsigned int) (row - row_offsets[global_part]);
            out->parts[local_part]->colIdx[idx] = (unsigned int) col;
            out->parts[local_part]->val[idx] = __float2half(value);
        }

        if (h->symmetric && row != col) {
            global_part = find_offset_span(col, row_offsets, num_parts);
            if (global_part >= part_begin && global_part < part_end) {
                local_part = global_part - part_begin;
                idx = write_ptr[local_part]++;
                out->parts[local_part]->rowIdx[idx] = (unsigned int) (col - row_offsets[global_part]);
                out->parts[local_part]->colIdx[idx] = (unsigned int) row;
                out->parts[local_part]->val[idx] = __float2half(value);
            }
        }
    }
    if (rc < 0) goto fail;
    std::free(write_ptr);
    return 1;

fail:
    std::free(write_ptr);
    return 0;
}

static inline int load_part_window_coo(const char *path,
                                       const header *h,
                                       const unsigned long *row_offsets,
                                       const unsigned long *part_nnz,
                                       unsigned long num_parts,
                                       unsigned long part_begin,
                                       unsigned long part_end,
                                       sharded<sparse::coo> *out) {
    io::source::buffered_file_reader reader;
    header local;
    int ok = 0;

    init(&local);
    io::source::init(&reader);

    if (!allocate_part_window_coo(h, row_offsets, part_nnz, num_parts, part_begin, part_end, out)) goto done;
    if (!io::source::open(&reader, path)) goto done;
    if (!read_header(&reader, &local)) goto done;
    if (local.rows != h->rows || local.cols != h->cols || local.nnz_file != h->nnz_file) goto done;
    if (!fill_part_window_coo(&reader, &local, row_offsets, num_parts, part_begin, part_end, out)) goto done;
    ok = 1;

done:
    io::source::clear(&reader);
    if (!ok) clear(out);
    return ok;
}

} // namespace mtx
} // namespace ingest
} // namespace cellshard
