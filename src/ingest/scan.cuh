#pragma once

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include <fcntl.h>
#include <unistd.h>

namespace cellshard {
namespace ingest {
namespace scan {

// Buffered sequential file reader for text ingest.
//
// The design is intentionally simple:
// - one file descriptor
// - one reusable read buffer
// - one scratch buffer for lines that span refill boundaries
//
// This is a host-side ingest utility and therefore copy-heavy by nature.
struct buffered_file_reader {
    int fd;
    char *buf;
    std::size_t cap;
    std::size_t begin;
    std::size_t end;
    int eof;
    int err;

    char *scratch;
    std::size_t scratch_cap;
    std::size_t scratch_size;

    unsigned long long line_number;
};

// Metadata-only init.
static inline void init(buffered_file_reader *r) {
    r->fd = -1;
    r->buf = 0;
    r->cap = 0;
    r->begin = 0;
    r->end = 0;
    r->eof = 0;
    r->err = 0;
    r->scratch = 0;
    r->scratch_cap = 0;
    r->scratch_size = 0;
    r->line_number = 0;
}

// Close the fd and release both heap buffers.
static inline void clear(buffered_file_reader *r) {
    if (r->fd >= 0) ::close(r->fd);
    std::free(r->buf);
    std::free(r->scratch);
    init(r);
}

// Grow the read buffer. This copies any unread bytes to the new allocation.
static inline int reserve_buffer(buffered_file_reader *r, std::size_t cap) {
    char *next = 0;

    if (cap <= r->cap) return 1;
    next = (char *) std::malloc(cap);
    if (next == 0) return 0;
    if (r->end > r->begin) {
        std::memcpy(next, r->buf + r->begin, r->end - r->begin);
        r->end -= r->begin;
        r->begin = 0;
    } else {
        r->begin = 0;
        r->end = 0;
    }
    std::free(r->buf);
    r->buf = next;
    r->cap = cap;
    return 1;
}

// Grow the scratch buffer used for cross-chunk lines. This copies any existing
// partial line bytes.
static inline int reserve_scratch(buffered_file_reader *r, std::size_t cap) {
    char *next = 0;

    if (cap <= r->scratch_cap) return 1;
    next = (char *) std::malloc(cap);
    if (next == 0) return 0;
    if (r->scratch_size != 0) std::memcpy(next, r->scratch, r->scratch_size);
    std::free(r->scratch);
    r->scratch = next;
    r->scratch_cap = cap;
    return 1;
}

// Open one source file for sequential buffered scanning.
static inline int open(buffered_file_reader *r, const char *path, std::size_t cap = (std::size_t) 8u << 20u) {
    clear(r);
    if (!reserve_buffer(r, cap)) return 0;
    r->fd = ::open(path, O_RDONLY);
    if (r->fd < 0) {
        r->err = errno;
        return 0;
    }
#if defined(POSIX_FADV_SEQUENTIAL)
    (void) ::posix_fadvise(r->fd, 0, 0, POSIX_FADV_SEQUENTIAL);
#endif
    return 1;
}

// Refill the read buffer from the file descriptor. This is the actual read()
// path for text ingest.
static inline int refill(buffered_file_reader *r) {
    ssize_t got = 0;

    if (r->fd < 0 || r->eof || r->err != 0) return 0;

    if (r->begin != 0 && r->begin < r->end) {
        std::memmove(r->buf, r->buf + r->begin, r->end - r->begin);
        r->end -= r->begin;
        r->begin = 0;
    } else if (r->begin >= r->end) {
        r->begin = 0;
        r->end = 0;
    }

    got = ::read(r->fd, r->buf + r->end, r->cap - r->end);
    if (got < 0) {
        r->err = errno;
        return 0;
    }
    if (got == 0) {
        r->eof = 1;
        return 0;
    }
    r->end += (std::size_t) got;
    return 1;
}

// Return the next line as a mutable zero-terminated buffer.
// If a line crosses chunk boundaries it is copied through scratch.
static inline int next_line(buffered_file_reader *r, char **out, std::size_t *len) {
    std::size_t i = 0;
    std::size_t chunk = 0;

    if (out == 0 || len == 0) return -1;
    *out = 0;
    *len = 0;
    r->scratch_size = 0;

    for (;;) {
        if (r->begin >= r->end) {
            if (!refill(r)) {
                if (r->err != 0) return -1;
                return 0;
            }
        }

        i = r->begin;
        while (i < r->end && r->buf[i] != '\n') ++i;

        if (i < r->end) {
            chunk = i - r->begin;
            if (r->scratch_size == 0) {
                if (chunk != 0 && r->buf[r->begin + chunk - 1] == '\r') --chunk;
                r->buf[r->begin + chunk] = 0;
                *out = r->buf + r->begin;
                *len = chunk;
                r->begin = i + 1;
                ++r->line_number;
                return 1;
            }

            if (!reserve_scratch(r, r->scratch_size + chunk + 1)) {
                r->err = ENOMEM;
                return -1;
            }
            if (chunk != 0) {
                std::memcpy(r->scratch + r->scratch_size, r->buf + r->begin, chunk);
                r->scratch_size += chunk;
            }
            if (r->scratch_size != 0 && r->scratch[r->scratch_size - 1] == '\r') --r->scratch_size;
            r->scratch[r->scratch_size] = 0;
            *out = r->scratch;
            *len = r->scratch_size;
            r->begin = i + 1;
            ++r->line_number;
            return 1;
        }

        chunk = r->end - r->begin;
        if (!reserve_scratch(r, r->scratch_size + chunk + 1)) {
            r->err = ENOMEM;
            return -1;
        }
        if (chunk != 0) {
            std::memcpy(r->scratch + r->scratch_size, r->buf + r->begin, chunk);
            r->scratch_size += chunk;
        }
        r->begin = r->end;

        if (!refill(r)) {
            if (r->err != 0) return -1;
            if (r->scratch_size == 0) return 0;
            if (r->scratch[r->scratch_size - 1] == '\r') --r->scratch_size;
            r->scratch[r->scratch_size] = 0;
            *out = r->scratch;
            *len = r->scratch_size;
            ++r->line_number;
            return 1;
        }
    }
}

// Skip blank and comment lines without allocating new strings.
static inline int skip_empty_and_comment_lines(buffered_file_reader *r, char **out, std::size_t *len, char comment) {
    int rc = 0;
    char *line = 0;
    std::size_t line_len = 0;

    for (;;) {
        rc = next_line(r, &line, &line_len);
        if (rc <= 0) return rc;
        if (line_len == 0) continue;
        if (comment != 0 && line[0] == comment) continue;
        *out = line;
        *len = line_len;
        return 1;
    }
}

// In-place tab splitter.
static inline unsigned int split_tabs(char *line, char **fields, unsigned int max_fields) {
    unsigned int count = 0;
    char *p = line;

    if (max_fields == 0) return 0;
    for (;;) {
        fields[count++] = p;
        while (*p != 0 && *p != '\t') ++p;
        if (*p == 0 || count == max_fields) return count;
        *p = 0;
        ++p;
    }
}

// In-place whitespace splitter.
static inline unsigned int split_ws(char *line, char **fields, unsigned int max_fields) {
    unsigned int count = 0;
    char *p = line;

    while (*p == ' ' || *p == '\t') ++p;
    while (*p != 0 && count < max_fields) {
        fields[count++] = p;
        while (*p != 0 && *p != ' ' && *p != '\t') ++p;
        if (*p == 0) break;
        *p = 0;
        ++p;
        while (*p == ' ' || *p == '\t') ++p;
    }
    return count;
}

static inline void strip_utf8_bom(char *line, std::size_t *len) {
    if (line == 0 || len == 0) return;
    if (*len < 3) return;
    if ((unsigned char) line[0] != 0xEFu) return;
    if ((unsigned char) line[1] != 0xBBu) return;
    if ((unsigned char) line[2] != 0xBFu) return;
    std::memmove(line, line + 3, *len - 2);
    *len -= 3;
}

static inline const char *field_or_empty(char **fields, unsigned int count, unsigned int idx) {
    if (idx >= count) return "";
    return fields[idx];
}

} // namespace scan
} // namespace ingest
} // namespace cellshard
