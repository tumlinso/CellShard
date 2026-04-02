#pragma once

#include <cstddef>
#include <cstdlib>
#include <cstring>

namespace cellshard {
namespace ingest {
namespace common {

// Packed string column:
// - offsets[] points into one contiguous byte blob
// - data[] stores zero-terminated strings back to back
//
// Appends can trigger host realloc+copy for offsets or data.
struct text_column {
    unsigned int count;
    unsigned int capacity;
    unsigned int bytes;
    unsigned int bytes_capacity;
    unsigned int *offsets;
    char *data;
};

// Metadata-only init.
static inline void init(text_column *c) {
    c->count = 0;
    c->capacity = 0;
    c->bytes = 0;
    c->bytes_capacity = 0;
    c->offsets = 0;
    c->data = 0;
}

// Release both host arrays.
static inline void clear(text_column *c) {
    std::free(c->offsets);
    std::free(c->data);
    init(c);
}

// Grow the offsets table and copy existing offsets.
static inline int reserve_entries(text_column *c, unsigned int capacity) {
    unsigned int *next = 0;

    if (capacity <= c->capacity) return 1;
    next = (unsigned int *) std::calloc((std::size_t) capacity + 1u, sizeof(unsigned int));
    if (next == 0) return 0;
    if (c->offsets != 0 && c->count != 0) {
        std::memcpy(next, c->offsets, ((std::size_t) c->count + 1u) * sizeof(unsigned int));
    } else {
        next[0] = 0;
    }
    std::free(c->offsets);
    c->offsets = next;
    c->capacity = capacity;
    return 1;
}

// Grow the string-data blob and copy existing bytes.
static inline int reserve_bytes(text_column *c, unsigned int bytes) {
    char *next = 0;

    if (bytes <= c->bytes_capacity) return 1;
    next = (char *) std::malloc(bytes);
    if (next == 0) return 0;
    if (c->bytes != 0) std::memcpy(next, c->data, c->bytes);
    std::free(c->data);
    c->data = next;
    c->bytes_capacity = bytes;
    return 1;
}

// Append one string by copying it into the packed byte blob.
static inline int append(text_column *c, const char *src, std::size_t len) {
    unsigned int next_count = 0;
    unsigned int next_bytes = 0;

    next_count = c->count + 1u;
    if (next_count > c->capacity) {
        if (!reserve_entries(c, c->capacity == 0 ? 16u : c->capacity << 1u)) return 0;
    }

    if (len > (std::size_t) (0xFFFFFFFFu - c->bytes - 1u)) return 0;
    next_bytes = c->bytes + (unsigned int) len + 1u;
    if (next_bytes > c->bytes_capacity) {
        unsigned int target = c->bytes_capacity == 0 ? 4096u : c->bytes_capacity;
        while (target < next_bytes) {
            if (target > 0x7FFFFFFFu) {
                target = next_bytes;
                break;
            }
            target <<= 1u;
        }
        if (!reserve_bytes(c, target)) return 0;
    }

    c->offsets[c->count] = c->bytes;
    if (len != 0) std::memcpy(c->data + c->bytes, src, len);
    c->data[c->bytes + len] = 0;
    c->bytes = next_bytes;
    c->count = next_count;
    c->offsets[c->count] = c->bytes;
    return 1;
}

// Random access through the offsets table.
static inline const char *get(const text_column *c, unsigned int idx) {
    if (idx >= c->count) return 0;
    return c->data + c->offsets[idx];
}

static inline unsigned int length(const text_column *c, unsigned int idx) {
    if (idx >= c->count) return 0;
    return c->offsets[idx + 1u] - c->offsets[idx] - 1u;
}

} // namespace common
} // namespace ingest
} // namespace cellshard
