#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>

#include <CellShard/formats/blocked_ell.cuh>
#include <CellShard/formats/sliced_ell.cuh>

namespace cellshard {
namespace ingest {
namespace dataset {

// One shard/window description over a contiguous part range.
struct shard_range {
    unsigned long part_begin;
    unsigned long part_end;
    unsigned long row_begin;
    unsigned long row_end;
    unsigned long nnz;
    unsigned long bytes;
};

// Dynamic array of shard/window ranges.
struct partition {
    unsigned long count;
    unsigned long capacity;
    shard_range *ranges;
};

// Metadata-only init / release.
static inline void init(partition *p) {
    p->count = 0;
    p->capacity = 0;
    p->ranges = 0;
}

static inline void clear(partition *p) {
    std::free(p->ranges);
    init(p);
}

// Grow the range table and copy existing ranges.
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

// Append one contiguous range.
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

// Partition parts by nnz and/or byte limits.
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

    // Rebuild the plan from scratch each time. That keeps the data structure
    // simple and the cost linear in part count.
    clear(p);
    init(p);

    // Greedy host-only packing is enough here because the goal is just to cap
    // later fetch/convert/store windows.
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

static inline int build_blocked_ell_shards(partition *p,
                                           const unsigned long *part_rows,
                                           const unsigned long *part_nnz,
                                           const unsigned long *part_aux,
                                           const unsigned long *part_bytes,
                                           unsigned long num_parts,
                                           unsigned long target_bytes) {
    const std::uint64_t max_u32 = (std::uint64_t) std::numeric_limits<std::uint32_t>::max();
    unsigned long i = 0ul;
    unsigned long row_cursor = 0ul;
    unsigned long shard_part_begin = 0ul;
    unsigned long shard_row_begin = 0ul;
    std::uint64_t shard_rows = 0u;
    std::uint64_t shard_nnz = 0u;
    std::uint64_t shard_bytes = 0u;
    std::uint64_t shard_block_idx = 0u;
    std::uint64_t shard_values = 0u;

    clear(p);
    init(p);

    for (i = 0ul; i < num_parts; ++i) {
        const std::uint64_t rows = part_rows != 0 ? (std::uint64_t) part_rows[i] : 0u;
        const std::uint64_t nnz = part_nnz != 0 ? (std::uint64_t) part_nnz[i] : 0u;
        const std::uint64_t bytes = part_bytes != 0 ? (std::uint64_t) part_bytes[i] : 0u;
        const unsigned long aux = part_aux != 0 ? part_aux[i] : 0ul;
        const std::uint64_t block_size = (std::uint64_t) ::cellshard::sparse::unpack_blocked_ell_block_size(aux);
        const std::uint64_t ell_width = (std::uint64_t) ::cellshard::sparse::unpack_blocked_ell_ell_width(aux);
        const std::uint64_t row_blocks = block_size != 0u ? (rows + block_size - 1u) / block_size : 0u;
        const std::uint64_t block_idx = row_blocks * ell_width;
        const std::uint64_t values = rows * ell_width * block_size;
        const int break_bytes = target_bytes != 0ul && shard_bytes != 0u && shard_bytes + bytes > (std::uint64_t) target_bytes;
        const int break_rows = shard_rows != 0u && shard_rows + rows > max_u32;
        const int break_nnz = shard_nnz != 0u && shard_nnz + nnz > max_u32;
        const int break_block_idx = shard_block_idx != 0u && shard_block_idx + block_idx > max_u32;
        const int break_values = shard_values != 0u && shard_values + values > max_u32;

        if (rows > max_u32 || nnz > max_u32 || block_idx > max_u32 || values > max_u32) return 0;

        if (break_bytes || break_rows || break_nnz || break_block_idx || break_values) {
            if (!append(p,
                        shard_part_begin,
                        i,
                        shard_row_begin,
                        row_cursor,
                        (unsigned long) shard_nnz,
                        (unsigned long) shard_bytes)) return 0;
            shard_part_begin = i;
            shard_row_begin = row_cursor;
            shard_rows = 0u;
            shard_nnz = 0u;
            shard_bytes = 0u;
            shard_block_idx = 0u;
            shard_values = 0u;
        }

        shard_rows += rows;
        shard_nnz += nnz;
        shard_bytes += bytes;
        shard_block_idx += block_idx;
        shard_values += values;
        row_cursor += (unsigned long) rows;
    }

    if (num_parts != 0ul) {
        if (!append(p,
                    shard_part_begin,
                    num_parts,
                    shard_row_begin,
                    row_cursor,
                    (unsigned long) shard_nnz,
                    (unsigned long) shard_bytes)) return 0;
    }
    return 1;
}

static inline int build_sliced_ell_shards(partition *p,
                                          const unsigned long *part_rows,
                                          const unsigned long *part_nnz,
                                          const unsigned long *part_aux,
                                          const unsigned long *part_bytes,
                                          unsigned long num_parts,
                                          unsigned long target_bytes) {
    const std::uint64_t max_u32 = (std::uint64_t) std::numeric_limits<std::uint32_t>::max();
    unsigned long i = 0ul;
    unsigned long row_cursor = 0ul;
    unsigned long shard_part_begin = 0ul;
    unsigned long shard_row_begin = 0ul;
    std::uint64_t shard_rows = 0u;
    std::uint64_t shard_nnz = 0u;
    std::uint64_t shard_bytes = 0u;
    std::uint64_t shard_slots = 0u;

    clear(p);
    init(p);

    for (i = 0ul; i < num_parts; ++i) {
        const std::uint64_t rows = part_rows != 0 ? (std::uint64_t) part_rows[i] : 0u;
        const std::uint64_t nnz = part_nnz != 0 ? (std::uint64_t) part_nnz[i] : 0u;
        const std::uint64_t bytes = part_bytes != 0 ? (std::uint64_t) part_bytes[i] : 0u;
        const std::uint64_t slots =
            part_aux != 0 ? (std::uint64_t) ::cellshard::sparse::unpack_sliced_ell_total_slots(part_aux[i]) : 0u;
        const int break_bytes = target_bytes != 0ul && shard_bytes != 0u && shard_bytes + bytes > (std::uint64_t) target_bytes;
        const int break_rows = shard_rows != 0u && shard_rows + rows > max_u32;
        const int break_nnz = shard_nnz != 0u && shard_nnz + nnz > max_u32;
        const int break_slots = shard_slots != 0u && shard_slots + slots > max_u32;

        if (rows > max_u32 || nnz > max_u32 || slots > max_u32) return 0;

        if (break_bytes || break_rows || break_nnz || break_slots) {
            if (!append(p,
                        shard_part_begin,
                        i,
                        shard_row_begin,
                        row_cursor,
                        (unsigned long) shard_nnz,
                        (unsigned long) shard_bytes)) return 0;
            shard_part_begin = i;
            shard_row_begin = row_cursor;
            shard_rows = 0u;
            shard_nnz = 0u;
            shard_bytes = 0u;
            shard_slots = 0u;
        }

        shard_rows += rows;
        shard_nnz += nnz;
        shard_bytes += bytes;
        shard_slots += slots;
        row_cursor += (unsigned long) rows;
    }

    if (num_parts != 0ul) {
        if (!append(p,
                    shard_part_begin,
                    num_parts,
                    shard_row_begin,
                    row_cursor,
                    (unsigned long) shard_nnz,
                    (unsigned long) shard_bytes)) return 0;
    }
    return 1;
}

} // namespace dataset
} // namespace ingest
} // namespace cellshard
