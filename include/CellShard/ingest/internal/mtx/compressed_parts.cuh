#pragma once

#include <cerrno>
#include <climits>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <fcntl.h>
#include <unistd.h>

#include "../../../../../src/convert/compressed_from_coo_raw.cuh"
#include <CellShard/io/pack/packfile.cuh>
#include <CellShard/runtime/storage/disk.cuh>
#include "mtx_reader.cuh"

namespace cellshard {
namespace ingest {
namespace mtx {

namespace sparse = ::cellshard::sparse;

// Chunk sizes for byte-range MTX scanning during transposed compressed ingest.
enum {
    mtx_part_source_bytes = 128u << 20u,
    mtx_part_source_tail_bytes = 1u << 20u
};

// Host/device staging workspace for building compressed parts from MTX triplets.
//
// Important behavior:
// - h_* buffers are pinned host buffers
// - d_* buffers are device buffers
// - reserve() may free and rebuild the whole workspace
// - build_pinned_triplet_to_compressed() performs H2D and D2H copies
struct compressed_workspace {
    int device;
    cudaStream_t stream;

    unsigned int cdim_cap;
    unsigned int udim_cap;
    unsigned int nnz_cap;

    void *h_block;
    unsigned int *h_row_idx;
    unsigned int *h_col_idx;
    __half *h_in_val;
    unsigned int *h_major_ptr;
    unsigned int *h_minor_idx;
    __half *h_out_val;

    void *d_block;
    unsigned int *d_row_idx;
    unsigned int *d_col_idx;
    __half *d_in_val;
    unsigned int *d_major_ptr;
    unsigned int *d_heads;
    unsigned int *d_minor_idx;
    __half *d_out_val;
    unsigned int *d_sort_cax;
    unsigned int *d_permutation;

    void *d_tmp;
    std::size_t d_tmp_bytes;
    std::size_t d_sort_bytes;
    std::size_t d_scan_bytes;
};

// Metadata-only init.
static inline void init(compressed_workspace *ws) {
    ws->device = -1;
    ws->stream = (cudaStream_t) 0;
    ws->cdim_cap = 0;
    ws->udim_cap = 0;
    ws->nnz_cap = 0;
    ws->h_block = 0;
    ws->h_row_idx = 0;
    ws->h_col_idx = 0;
    ws->h_in_val = 0;
    ws->h_major_ptr = 0;
    ws->h_minor_idx = 0;
    ws->h_out_val = 0;
    ws->d_block = 0;
    ws->d_row_idx = 0;
    ws->d_col_idx = 0;
    ws->d_in_val = 0;
    ws->d_major_ptr = 0;
    ws->d_heads = 0;
    ws->d_minor_idx = 0;
    ws->d_out_val = 0;
    ws->d_sort_cax = 0;
    ws->d_permutation = 0;
    ws->d_tmp = 0;
    ws->d_tmp_bytes = 0;
    ws->d_sort_bytes = 0;
    ws->d_scan_bytes = 0;
}

// Release all pinned host buffers, device buffers, and scan scratch.
static inline void clear(compressed_workspace *ws) {
    if (ws->d_tmp != 0) cudaFree(ws->d_tmp);
    if (ws->d_block != 0) cudaFree(ws->d_block);
    if (ws->h_block != 0) cudaFreeHost(ws->h_block);
    init(ws);
}

// Bind the workspace to one device and stream.
static inline int setup(compressed_workspace *ws, int device, cudaStream_t stream = (cudaStream_t) 0) {
    if (device >= 0 && cudaSetDevice(device) != cudaSuccess) return 0;
    ws->device = device;
    ws->stream = stream;
    return 1;
}

static inline std::size_t ingest_align_up_bytes(std::size_t value, std::size_t alignment) {
    return (value + alignment - 1u) & ~(alignment - 1u);
}

// Ensure the workspace can hold one compressed-build job of the requested size.
// This now allocates one pinned host slab, one device slab, and one temporary
// device workspace. The sorted COO path uses the temporary block for cuSPARSE
// sorting; the compatibility atomic path reuses the same block for CUB scan.
static inline int reserve(compressed_workspace *ws, unsigned int cdim, unsigned int udim, unsigned int nnz) {
    const int bound_device = ws->device;
    const cudaStream_t bound_stream = ws->stream;
    std::size_t scan_bytes = 0;
    std::size_t sort_bytes = 0;
    std::size_t tmp_bytes = 0;
    std::size_t h_bytes = 0;
    std::size_t d_bytes = 0;
    char *h_base = 0;
    char *d_base = 0;

    if (cub::DeviceScan::ExclusiveSum(0, scan_bytes, (unsigned int *) 0, (unsigned int *) 0, cdim + 1, ws->stream) != cudaSuccess) return 0;

    h_bytes = ingest_align_up_bytes(h_bytes, alignof(unsigned int));
    h_bytes += (std::size_t) nnz * sizeof(unsigned int);
    h_bytes = ingest_align_up_bytes(h_bytes, alignof(unsigned int));
    h_bytes += (std::size_t) nnz * sizeof(unsigned int);
    h_bytes = ingest_align_up_bytes(h_bytes, alignof(__half));
    h_bytes += (std::size_t) nnz * sizeof(__half);
    h_bytes = ingest_align_up_bytes(h_bytes, alignof(unsigned int));
    h_bytes += (std::size_t) (cdim + 1u) * sizeof(unsigned int);
    h_bytes = ingest_align_up_bytes(h_bytes, alignof(unsigned int));
    h_bytes += (std::size_t) nnz * sizeof(unsigned int);
    h_bytes = ingest_align_up_bytes(h_bytes, alignof(__half));
    h_bytes += (std::size_t) nnz * sizeof(__half);

    d_bytes = ingest_align_up_bytes(d_bytes, alignof(unsigned int));
    d_bytes += (std::size_t) nnz * sizeof(unsigned int);
    d_bytes = ingest_align_up_bytes(d_bytes, alignof(unsigned int));
    d_bytes += (std::size_t) nnz * sizeof(unsigned int);
    d_bytes = ingest_align_up_bytes(d_bytes, alignof(__half));
    d_bytes += (std::size_t) nnz * sizeof(__half);
    d_bytes = ingest_align_up_bytes(d_bytes, alignof(unsigned int));
    d_bytes += (std::size_t) (cdim + 1u) * sizeof(unsigned int);
    d_bytes = ingest_align_up_bytes(d_bytes, alignof(unsigned int));
    d_bytes += (std::size_t) cdim * sizeof(unsigned int);
    d_bytes = ingest_align_up_bytes(d_bytes, alignof(unsigned int));
    d_bytes += (std::size_t) nnz * sizeof(unsigned int);
    d_bytes = ingest_align_up_bytes(d_bytes, alignof(__half));
    d_bytes += (std::size_t) nnz * sizeof(__half);
    d_bytes = ingest_align_up_bytes(d_bytes, alignof(unsigned int));
    d_bytes += (std::size_t) nnz * sizeof(unsigned int);
    d_bytes = ingest_align_up_bytes(d_bytes, alignof(unsigned int));
    d_bytes += (std::size_t) nnz * sizeof(unsigned int);

    if (cdim <= ws->cdim_cap &&
        udim <= ws->udim_cap &&
        nnz <= ws->nnz_cap &&
        !::cellshard::convert::compressed_from_coo_sorted_workspace_bytes(
            cdim,
            udim,
            nnz,
            ws->d_sort_cax,
            ws->d_minor_idx,
            ws->stream,
            &sort_bytes)) return 0;

    tmp_bytes = scan_bytes > sort_bytes ? scan_bytes : sort_bytes;
    if (cdim <= ws->cdim_cap &&
        udim <= ws->udim_cap &&
        nnz <= ws->nnz_cap &&
        tmp_bytes <= ws->d_tmp_bytes) {
        ws->d_sort_bytes = sort_bytes;
        ws->d_scan_bytes = scan_bytes;
        return 1;
    }

    clear(ws);
    ws->device = bound_device;
    ws->stream = bound_stream;

    if (h_bytes != 0 && cudaMallocHost(&ws->h_block, h_bytes) != cudaSuccess) goto fail;
    if (d_bytes != 0 && cudaMalloc(&ws->d_block, d_bytes) != cudaSuccess) goto fail;

    h_base = (char *) ws->h_block;
    d_base = (char *) ws->d_block;

    h_bytes = 0;
    h_bytes = ingest_align_up_bytes(h_bytes, alignof(unsigned int));
    ws->h_row_idx = (unsigned int *) (h_base + h_bytes);
    h_bytes += (std::size_t) nnz * sizeof(unsigned int);
    h_bytes = ingest_align_up_bytes(h_bytes, alignof(unsigned int));
    ws->h_col_idx = (unsigned int *) (h_base + h_bytes);
    h_bytes += (std::size_t) nnz * sizeof(unsigned int);
    h_bytes = ingest_align_up_bytes(h_bytes, alignof(__half));
    ws->h_in_val = (__half *) (h_base + h_bytes);
    h_bytes += (std::size_t) nnz * sizeof(__half);
    h_bytes = ingest_align_up_bytes(h_bytes, alignof(unsigned int));
    ws->h_major_ptr = (unsigned int *) (h_base + h_bytes);
    h_bytes += (std::size_t) (cdim + 1u) * sizeof(unsigned int);
    h_bytes = ingest_align_up_bytes(h_bytes, alignof(unsigned int));
    ws->h_minor_idx = (unsigned int *) (h_base + h_bytes);
    h_bytes += (std::size_t) nnz * sizeof(unsigned int);
    h_bytes = ingest_align_up_bytes(h_bytes, alignof(__half));
    ws->h_out_val = (__half *) (h_base + h_bytes);

    d_bytes = 0;
    d_bytes = ingest_align_up_bytes(d_bytes, alignof(unsigned int));
    ws->d_row_idx = (unsigned int *) (d_base + d_bytes);
    d_bytes += (std::size_t) nnz * sizeof(unsigned int);
    d_bytes = ingest_align_up_bytes(d_bytes, alignof(unsigned int));
    ws->d_col_idx = (unsigned int *) (d_base + d_bytes);
    d_bytes += (std::size_t) nnz * sizeof(unsigned int);
    d_bytes = ingest_align_up_bytes(d_bytes, alignof(__half));
    ws->d_in_val = (__half *) (d_base + d_bytes);
    d_bytes += (std::size_t) nnz * sizeof(__half);
    d_bytes = ingest_align_up_bytes(d_bytes, alignof(unsigned int));
    ws->d_major_ptr = (unsigned int *) (d_base + d_bytes);
    d_bytes += (std::size_t) (cdim + 1u) * sizeof(unsigned int);
    d_bytes = ingest_align_up_bytes(d_bytes, alignof(unsigned int));
    ws->d_heads = (unsigned int *) (d_base + d_bytes);
    d_bytes += (std::size_t) cdim * sizeof(unsigned int);
    d_bytes = ingest_align_up_bytes(d_bytes, alignof(unsigned int));
    ws->d_minor_idx = (unsigned int *) (d_base + d_bytes);
    d_bytes += (std::size_t) nnz * sizeof(unsigned int);
    d_bytes = ingest_align_up_bytes(d_bytes, alignof(__half));
    ws->d_out_val = (__half *) (d_base + d_bytes);
    d_bytes += (std::size_t) nnz * sizeof(__half);
    d_bytes = ingest_align_up_bytes(d_bytes, alignof(unsigned int));
    ws->d_sort_cax = (unsigned int *) (d_base + d_bytes);
    d_bytes += (std::size_t) nnz * sizeof(unsigned int);
    d_bytes = ingest_align_up_bytes(d_bytes, alignof(unsigned int));
    ws->d_permutation = (unsigned int *) (d_base + d_bytes);

    if (!::cellshard::convert::compressed_from_coo_sorted_workspace_bytes(
            cdim,
            udim,
            nnz,
            ws->d_sort_cax,
            ws->d_minor_idx,
            ws->stream,
            &sort_bytes)) goto fail;

    tmp_bytes = scan_bytes > sort_bytes ? scan_bytes : sort_bytes;
    if (tmp_bytes != 0 && cudaMalloc(&ws->d_tmp, tmp_bytes) != cudaSuccess) goto fail;
    ws->d_tmp_bytes = tmp_bytes;
    ws->d_sort_bytes = sort_bytes;
    ws->d_scan_bytes = scan_bytes;
    ws->cdim_cap = cdim;
    ws->udim_cap = udim;
    ws->nnz_cap = nnz;
    return 1;

fail:
    clear(ws);
    ws->device = bound_device;
    ws->stream = bound_stream;
    return 0;
}

// Open the MTX source for byte-range reads.
static inline int open_part_source(const char *path) {
    int fd = ::open(path, O_RDONLY);
    if (fd < 0) {
        std::fprintf(stderr, "Error: open failed for %s: %s\n", path, std::strerror(errno));
        return -1;
    }
#if defined(POSIX_FADV_SEQUENTIAL)
    (void) ::posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL);
#endif
    return fd;
}

// Load one byte range from the source MTX file into pinned COO-style triplet
// buffers, transposing row/column roles on the way in.
//
// This is a host-side parsing path with:
// - pread()
// - line splitting
// - triplet parsing
// - copies into pinned host buffers
static inline int load_transposed_byte_range_pinned_triplet(const char *path,
                                                            unsigned int cols_out,
                                                            unsigned long row_begin,
                                                            unsigned long scan_begin,
                                                            unsigned long scan_end,
                                                            unsigned int rows_out,
                                                            unsigned int nnz_out,
                                                            compressed_workspace *ws) {
    header h;
    int fd = -1;
    char *buf = 0;
    std::size_t carry = 0;
    unsigned long next_read_offset = scan_begin;
    unsigned int write_pos = 0;

    if (!read_header(path, &h)) return 0;
    if (h.symmetric) return 0;
    if (h.cols < row_begin + rows_out) return 0;
    if (h.rows != cols_out) return 0;
    if (!reserve(ws, rows_out, cols_out, nnz_out)) return 0;
    if (nnz_out == 0) return 1;

    fd = open_part_source(path);
    if (fd < 0) return 0;
#if defined(POSIX_FADV_SEQUENTIAL)
    (void) ::posix_fadvise(fd, (off_t) scan_begin, (off_t) (scan_end - scan_begin), POSIX_FADV_SEQUENTIAL);
#endif
    buf = (char *) std::malloc((std::size_t) mtx_part_source_bytes + (std::size_t) mtx_part_source_tail_bytes + 1u);
    if (buf == 0) goto fail;

    while (write_pos < nnz_out) {
        const unsigned long chunk_bytes_u64 = next_read_offset < scan_end
            ? ((scan_end - next_read_offset) > (unsigned long) mtx_part_source_bytes
                ? (unsigned long) mtx_part_source_bytes
                : (scan_end - next_read_offset))
            : 0;
        const std::size_t chunk_bytes = (std::size_t) chunk_bytes_u64;
        const ssize_t got = chunk_bytes != 0
            ? ::pread(fd, buf + carry, chunk_bytes, (off_t) next_read_offset)
            : 0;
        const std::size_t total = carry + (got > 0 ? (std::size_t) got : 0u);
        std::size_t parse_bytes = total;
        char *p = buf;
        char *end = 0;

        if (got < 0) goto fail;
        next_read_offset += (got > 0 ? (unsigned long) got : 0ul);
        buf[total] = 0;
        if (next_read_offset < scan_end) {
            while (parse_bytes != 0 && buf[parse_bytes - 1] != '\n') --parse_bytes;
            if (parse_bytes == 0) goto fail;
        }
        end = buf + parse_bytes;

        while (p < end && write_pos < nnz_out) {
            char *line = p;
            unsigned long file_row = 0;
            unsigned long file_col = 0;
            unsigned long out_row = 0;
            float value = 0.0f;

            while (p < end && *p != '\n') ++p;
            if (p < end) {
                *p = 0;
                ++p;
            }
            if (line[0] == 0 || line[0] == '%') continue;
            if (!read_triplet(line, &h, &file_row, &file_col, &value)) goto fail;
            if (file_col < row_begin) goto fail;
            out_row = file_col - row_begin;
            if (out_row >= rows_out) goto fail;
            ws->h_row_idx[write_pos] = (unsigned int) out_row;
            ws->h_col_idx[write_pos] = (unsigned int) file_row;
            ws->h_in_val[write_pos] = __float2half(value);
            ++write_pos;
        }

        carry = total - parse_bytes;
        if (carry != 0) std::memmove(buf, buf + parse_bytes, carry);
        if (got == 0 && carry == 0) break;
    }

    ::close(fd);
    std::free(buf);
    return write_pos == nnz_out;

fail:
    if (fd >= 0) ::close(fd);
    std::free(buf);
    return 0;
}

// Convert the pinned triplet buffers into compressed form on device, then copy
// the result back into pinned host output buffers.
//
// This is one of the heaviest ingest operations in the library:
// - H2D copies
// - device scan/scatter work
// - D2H copies
// - stream synchronize
static inline int build_pinned_triplet_to_compressed(compressed_workspace *ws,
                                                     unsigned int rows,
                                                     unsigned int cols,
                                                     unsigned int nnz,
                                                     unsigned int axis) {
    const unsigned int cdim = axis == sparse::compressed_by_col ? cols : rows;
    const unsigned int udim = axis == sparse::compressed_by_col ? rows : cols;
    const unsigned int *d_cax = axis == sparse::compressed_by_col ? ws->d_col_idx : ws->d_row_idx;
    const unsigned int *d_uax = axis == sparse::compressed_by_col ? ws->d_row_idx : ws->d_col_idx;

    if (!reserve(ws, cdim, udim, nnz)) return 0;
    if (nnz != 0) {
        if (cudaMemcpyAsync(ws->d_row_idx, ws->h_row_idx, (std::size_t) nnz * sizeof(unsigned int), cudaMemcpyHostToDevice, ws->stream) != cudaSuccess) return 0;
        if (cudaMemcpyAsync(ws->d_col_idx, ws->h_col_idx, (std::size_t) nnz * sizeof(unsigned int), cudaMemcpyHostToDevice, ws->stream) != cudaSuccess) return 0;
        if (cudaMemcpyAsync(ws->d_in_val, ws->h_in_val, (std::size_t) nnz * sizeof(__half), cudaMemcpyHostToDevice, ws->stream) != cudaSuccess) return 0;
    }
    if (!::cellshard::convert::build_compressed_from_coo_sorted_raw(cdim,
                                                                    udim,
                                                                    nnz,
                                                                    d_cax,
                                                                    d_uax,
                                                                    ws->d_in_val,
                                                                    ws->d_major_ptr,
                                                                    ws->d_sort_cax,
                                                                    ws->d_minor_idx,
                                                                    ws->d_out_val,
                                                                    ws->d_permutation,
                                                                    ws->d_tmp,
                                                                    ws->d_sort_bytes,
                                                                    ws->stream)) {
        if (!::cellshard::convert::build_compressed_from_coo_raw(cdim,
                                                                 nnz,
                                                                 d_cax,
                                                                 d_uax,
                                                                 ws->d_in_val,
                                                                 ws->d_major_ptr,
                                                                 ws->d_heads,
                                                                 ws->d_minor_idx,
                                                                 ws->d_out_val,
                                                                 ws->d_tmp,
                                                                 ws->d_scan_bytes,
                                                                 ws->stream)) return 0;
    }
    if (cudaMemcpyAsync(ws->h_major_ptr, ws->d_major_ptr, (std::size_t) (cdim + 1) * sizeof(unsigned int), cudaMemcpyDeviceToHost, ws->stream) != cudaSuccess) return 0;
    if (nnz != 0) {
        if (cudaMemcpyAsync(ws->h_minor_idx, ws->d_minor_idx, (std::size_t) nnz * sizeof(unsigned int), cudaMemcpyDeviceToHost, ws->stream) != cudaSuccess) return 0;
        if (cudaMemcpyAsync(ws->h_out_val, ws->d_out_val, (std::size_t) nnz * sizeof(__half), cudaMemcpyDeviceToHost, ws->stream) != cudaSuccess) return 0;
    }
    return cudaStreamSynchronize(ws->stream) == cudaSuccess;
}

// Split one large compressed window into per-part compressed files.
static inline int store_row_compressed_part_window(const char *part_prefix,
                                                   unsigned long global_part_begin,
                                                   unsigned int cols_out,
                                                   const unsigned long *part_rows,
                                                   const unsigned long *part_nnz,
                                                   unsigned long part_count,
                                                   const unsigned int *row_ptr,
                                                   const unsigned int *col_idx,
                                                   const __half *val) {
    unsigned long part_i = 0;
    unsigned long local_row_begin = 0;

    for (part_i = 0; part_i < part_count; ++part_i) {
        const unsigned int rows = (unsigned int) part_rows[part_i];
        const unsigned int nnz = (unsigned int) part_nnz[part_i];
        const unsigned int nnz_begin = row_ptr[local_row_begin];
        char path[4096];
        unsigned int *part_row_ptr = 0;
        unsigned int r = 0;

        if (std::snprintf(path, sizeof(path), "%s.%lu", part_prefix, global_part_begin + part_i) <= 0) return 0;
        part_row_ptr = (unsigned int *) std::malloc((std::size_t) (rows + 1) * sizeof(unsigned int));
        if (part_row_ptr == 0) return 0;
        for (r = 0; r <= rows; ++r) part_row_ptr[r] = row_ptr[local_row_begin + r] - nnz_begin;
        if (!::cellshard::store_compressed_raw(path,
                                               rows,
                                               cols_out,
                                               nnz,
                                               sparse::compressed_by_row,
                                               rows,
                                               part_row_ptr,
                                               col_idx + nnz_begin,
                                               val + nnz_begin,
                                               sizeof(__half))) {
            std::free(part_row_ptr);
            return 0;
        }
        std::free(part_row_ptr);
        local_row_begin += rows;
    }
    return 1;
}

// End-to-end byte-range conversion:
// - parse a source range into pinned triplets
// - build compressed form on device
// - copy compressed results back to host
// - split/store one compressed file per part
static inline int convert_transposed_byte_range_to_row_compressed_parts(const char *path,
                                                                        unsigned int cols_out,
                                                                        const char *part_prefix,
                                                                        unsigned long global_part_begin,
                                                                        unsigned long row_begin,
                                                                        const unsigned long *part_rows,
                                                                        const unsigned long *part_nnz,
                                                                        unsigned long part_count,
                                                                        unsigned long scan_begin,
                                                                        unsigned long scan_end,
                                                                        compressed_workspace *ws) {
    unsigned long part_i = 0;
    unsigned long rows_total_ul = 0;
    unsigned long nnz_total_ul = 0;
    unsigned int rows_total = 0;
    unsigned int nnz_total = 0;

    for (part_i = 0; part_i < part_count; ++part_i) {
        rows_total_ul += part_rows[part_i];
        nnz_total_ul += part_nnz[part_i];
    }
    if (rows_total_ul > (unsigned long) UINT_MAX || nnz_total_ul > (unsigned long) UINT_MAX) return 0;
    rows_total = (unsigned int) rows_total_ul;
    nnz_total = (unsigned int) nnz_total_ul;

    if (!load_transposed_byte_range_pinned_triplet(path,
                                                   cols_out,
                                                   row_begin,
                                                   scan_begin,
                                                   scan_end,
                                                   rows_total,
                                                   nnz_total,
                                                   ws)) return 0;
    if (!build_pinned_triplet_to_compressed(ws,
                                            rows_total,
                                            cols_out,
                                            nnz_total,
                                            sparse::compressed_by_row)) return 0;
    return store_row_compressed_part_window(part_prefix,
                                            global_part_begin,
                                            cols_out,
                                            part_rows,
                                            part_nnz,
                                            part_count,
                                            ws->h_major_ptr,
                                            ws->h_minor_idx,
                                            ws->h_out_val);
}

} // namespace mtx
} // namespace ingest
} // namespace cellshard
