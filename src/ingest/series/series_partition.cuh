#pragma once

#include <cstddef>
#include <cstdlib>
#include <cstring>

namespace cellshard {
namespace ingest {
namespace series {

struct shard_range {
    unsigned long part_begin;
    unsigned long part_end;
    unsigned long row_begin;
    unsigned long row_end;
    unsigned long nnz;
    unsigned long bytes;
};

struct partition {
    unsigned long count;
    unsigned long capacity;
    shard_range *ranges;
};

static inline void init(partition *p) {
    p->count = 0;
    p->capacity = 0;
    p->ranges = 0;
}

static inline void clear(partition *p) {
    std::free(p->ranges);
    init(p);
}

static inline int reserve(partition *p, unsigned long capacity) {
    shard_range *next = 0;

    if (capacity <= p->capacity) return 1;
    next = (shard_range *) std::calloc((std::size_t) capacity, sizeof(shard_range));
    if (next == 0) return 0;
    if (p->count != 0) std::memcpy(next, p->ranges, (std::size_t) p->count * sizeof(shard_range));
    std::free(p->ranges);
    p->ranges = next;
    p->capacity = capacity;
    return 1;
}

static inline int append(partition *p,
                         unsigned long part_begin,
                         unsigned long part_end,
                         unsigned long row_begin,
                         unsigned long row_end,
                         unsigned long nnz,
                         unsigned long bytes) {
    if (p->count == p->capacity) {
        if (!reserve(p, p->capacity == 0 ? 16ul : p->capacity << 1ul)) return 0;
    }
    p->ranges[p->count].part_begin = part_begin;
    p->ranges[p->count].part_end = part_end;
    p->ranges[p->count].row_begin = row_begin;
    p->ranges[p->count].row_end = row_end;
    p->ranges[p->count].nnz = nnz;
    p->ranges[p->count].bytes = bytes;
    ++p->count;
    return 1;
}

static inline int build_by_limit(partition *p,
                                 const unsigned long *part_rows,
                                 const unsigned long *part_nnz,
                                 const unsigned long *part_bytes,
                                 unsigned long num_parts,
                                 unsigned long max_nnz,
                                 unsigned long max_bytes) {
    unsigned long i = 0;
    unsigned long row_cursor = 0;
    unsigned long shard_part_begin = 0;
    unsigned long shard_row_begin = 0;
    unsigned long shard_nnz = 0;
    unsigned long shard_bytes = 0;

    clear(p);
    init(p);

    for (i = 0; i < num_parts; ++i) {
        const unsigned long rows = part_rows != 0 ? part_rows[i] : 0;
        const unsigned long nnz = part_nnz != 0 ? part_nnz[i] : 0;
        const unsigned long bytes = part_bytes != 0 ? part_bytes[i] : 0;
        const int break_nnz = (max_nnz != 0 && shard_nnz != 0 && shard_nnz + nnz > max_nnz);
        const int break_bytes = (max_bytes != 0 && shard_bytes != 0 && shard_bytes + bytes > max_bytes);

        if (break_nnz || break_bytes) {
            if (!append(p, shard_part_begin, i, shard_row_begin, row_cursor, shard_nnz, shard_bytes)) return 0;
            shard_part_begin = i;
            shard_row_begin = row_cursor;
            shard_nnz = 0;
            shard_bytes = 0;
        }

        shard_nnz += nnz;
        shard_bytes += bytes;
        row_cursor += rows;
    }

    if (num_parts != 0) {
        if (!append(p, shard_part_begin, num_parts, shard_row_begin, row_cursor, shard_nnz, shard_bytes)) return 0;
    }
    return 1;
}

static inline int build_by_nnz(partition *p,
                               const unsigned long *part_rows,
                               const unsigned long *part_nnz,
                               unsigned long num_parts,
                               unsigned long max_nnz) {
    return build_by_limit(p, part_rows, part_nnz, 0, num_parts, max_nnz, 0);
}

static inline int build_by_bytes(partition *p,
                                 const unsigned long *part_rows,
                                 const unsigned long *part_bytes,
                                 unsigned long num_parts,
                                 unsigned long max_bytes) {
    return build_by_limit(p, part_rows, 0, part_bytes, num_parts, 0, max_bytes);
}

} // namespace series
} // namespace ingest
} // namespace cellshard
