#pragma once

#include <cstddef>
#include <cstdlib>
#include <cstring>

#include "series_manifest.cuh"
#include "series_partition.cuh"
#include "../mtx/mtx_reader.cuh"
#include "../mtx/mtx_shard_writer.cuh"

namespace cellshard {
namespace ingest {
namespace series {

// Conversion knobs for MTX-series ingest.
struct mtx_convert_options {
    unsigned long max_part_nnz;
    unsigned long max_window_bytes;
    std::size_t reader_bytes;
};

// Metadata-only init with throughput-oriented defaults.
static inline void init(mtx_convert_options *opts) {
    opts->max_part_nnz = 1ul << 26ul;
    opts->max_window_bytes = 1ul << 30ul;
    opts->reader_bytes = (std::size_t) 8u << 20u;
}

// Build trivial auxiliary metadata and shard boundaries for freshly generated
// COO parts.
static inline int build_default_part_aux(unsigned long num_parts, unsigned long **part_aux_out) {
    unsigned long *part_aux = 0;

    *part_aux_out = 0;
    if (num_parts == 0) return 1;
    part_aux = (unsigned long *) std::calloc((std::size_t) num_parts, sizeof(unsigned long));
    if (part_aux == 0) return 0;
    *part_aux_out = part_aux;
    return 1;
}

static inline int build_identity_shards(unsigned long num_parts,
                                        const unsigned long *row_offsets,
                                        unsigned long **shard_offsets_out) {
    unsigned long *shard_offsets = 0;
    unsigned long i = 0;

    *shard_offsets_out = 0;
    shard_offsets = (unsigned long *) std::malloc((std::size_t) (num_parts + 1ul) * sizeof(unsigned long));
    if (shard_offsets == 0) return 0;
    for (i = 0; i <= num_parts; ++i) shard_offsets[i] = row_offsets[i];
    *shard_offsets_out = shard_offsets;
    return 1;
}

// Allocate one in-memory part window plus one write pointer per local part.
static inline int prepare_part_window(const mtx::header *h,
                                      const unsigned long *row_offsets,
                                      const unsigned long *part_nnz,
                                      unsigned long num_parts,
                                      unsigned long part_begin,
                                      unsigned long part_end,
                                      sharded<sparse::coo> *window_view,
                                      unsigned long **write_ptr_out) {
    unsigned long *write_ptr = 0;

    *write_ptr_out = 0;
    if (!mtx::allocate_part_window_coo(h, row_offsets, part_nnz, num_parts, part_begin, part_end, window_view)) return 0;
    if (window_view->num_parts != 0) {
        write_ptr = (unsigned long *) std::calloc((std::size_t) window_view->num_parts, sizeof(unsigned long));
        if (write_ptr == 0) {
            clear(window_view);
            init(window_view);
            return 0;
        }
    }
    *write_ptr_out = write_ptr;
    return 1;
}

// Flush one in-memory window to disk after validating all local write counts.
static inline int flush_part_window(const char *part_prefix,
                                    unsigned long global_part_begin,
                                    sharded<sparse::coo> *window_view,
                                    const unsigned long *write_ptr) {
    unsigned long i = 0;

    for (i = 0; i < window_view->num_parts; ++i) {
        if (write_ptr != 0 && write_ptr[i] != window_view->part_nnz[i]) return 0;
    }
    return mtx::store_part_window_coo(part_prefix, global_part_begin, window_view);
}

// Fast path for general, row-sorted MTX:
// - stream once through the source
// - fill only one part window at a time
// - flush each completed window to disk
//
// This avoids materializing the whole source as one giant COO object.
static inline int stream_sorted_general_mtx_to_sharded_coo(const char *mtx_path,
                                                           const char *part_prefix,
                                                           const mtx::header *h,
                                                           const partition *windows,
                                                           const unsigned long *row_offsets,
                                                           const unsigned long *part_nnz,
                                                           unsigned long num_parts,
                                                           std::size_t reader_bytes) {
    scan::buffered_file_reader reader;
    mtx::header local;
    sharded<sparse::coo> window_view;
    unsigned long *write_ptr = 0;
    unsigned long w = 0;
    int rc = 0;
    int ok = 0;
    char *line = 0;
    std::size_t line_len = 0;
    unsigned long row = 0;
    unsigned long col = 0;
    unsigned long global_part = 0;
    unsigned long local_part = 0;
    unsigned long idx = 0;
    float value = 0.0f;

    if (windows->count == 0) return 1;

    mtx::init(&local);
    scan::init(&reader);
    init(&window_view);

    if (!scan::open(&reader, mtx_path, reader_bytes)) goto done;
    if (!mtx::read_header(&reader, &local)) goto done;
    if (local.rows != h->rows || local.cols != h->cols || local.nnz_file != h->nnz_file) goto done;
    if (local.symmetric || !local.row_sorted) goto done;

    if (!prepare_part_window(h,
                             row_offsets,
                             part_nnz,
                             num_parts,
                             windows->ranges[0].part_begin,
                             windows->ranges[0].part_end,
                             &window_view,
                             &write_ptr)) goto done;

    while ((rc = scan::next_line(&reader, &line, &line_len)) > 0) {
        if (line_len == 0 || line[0] == '%') continue;
        if (!mtx::read_triplet(line, &local, &row, &col, &value)) goto done;

        global_part = find_offset_span(row, row_offsets, num_parts);
        if (global_part >= num_parts) goto done;

        while (w < windows->count && global_part >= windows->ranges[w].part_end) {
            if (!flush_part_window(part_prefix, windows->ranges[w].part_begin, &window_view, write_ptr)) goto done;
            std::free(write_ptr);
            write_ptr = 0;
            clear(&window_view);
            init(&window_view);
            ++w;
            if (w < windows->count) {
                if (!prepare_part_window(h,
                                         row_offsets,
                                         part_nnz,
                                         num_parts,
                                         windows->ranges[w].part_begin,
                                         windows->ranges[w].part_end,
                                         &window_view,
                                         &write_ptr)) goto done;
            }
        }

        if (w >= windows->count) goto done;
        if (global_part < windows->ranges[w].part_begin) goto done;

        local_part = global_part - windows->ranges[w].part_begin;
        idx = write_ptr[local_part]++;
        if (idx >= window_view.part_nnz[local_part]) goto done;
        window_view.parts[local_part]->rowIdx[idx] = (unsigned int) (row - row_offsets[global_part]);
        window_view.parts[local_part]->colIdx[idx] = (unsigned int) col;
        window_view.parts[local_part]->val[idx] = __float2half(value);
    }
    if (rc < 0) goto done;

    while (w < windows->count) {
        if (!flush_part_window(part_prefix, windows->ranges[w].part_begin, &window_view, write_ptr)) goto done;
        std::free(write_ptr);
        write_ptr = 0;
        clear(&window_view);
        init(&window_view);
        ++w;
        if (w < windows->count) {
            if (!prepare_part_window(h,
                                     row_offsets,
                                     part_nnz,
                                     num_parts,
                                     windows->ranges[w].part_begin,
                                     windows->ranges[w].part_end,
                                     &window_view,
                                     &write_ptr)) goto done;
        }
    }

    ok = 1;

done:
    std::free(write_ptr);
    scan::clear(&reader);
    clear(&window_view);
    return ok;
}

// End-to-end single-dataset MTX -> sharded COO conversion.
// This is a host-heavy multi-pass ingest path by design because MTX is text.
static inline int convert_single_mtx_to_sharded_coo(const char *mtx_path,
                                                    const char *header_path,
                                                    const char *part_prefix,
                                                    const mtx_convert_options *opts) {
    mtx::header h;
    partition windows;
    unsigned long *row_nnz = 0;
    unsigned long *row_offsets = 0;
    unsigned long *part_nnz = 0;
    unsigned long *part_rows = 0;
    unsigned long *part_bytes = 0;
    unsigned long *part_aux = 0;
    unsigned long *shard_offsets = 0;
    unsigned long num_parts = 0;
    unsigned long w = 0;
    sharded<sparse::coo> window_view;
    int ok = 0;

    mtx::init(&h);
    init(&windows);
    init(&window_view);

    if (!mtx::scan_row_nnz(mtx_path, &h, &row_nnz, opts->reader_bytes)) goto done;
    if (!mtx::plan_row_partitions_by_nnz(row_nnz, h.rows, opts->max_part_nnz, &row_offsets, &num_parts)) goto done;
    if (!mtx::build_part_nnz_from_row_nnz(row_nnz, row_offsets, num_parts, &part_nnz)) goto done;

    part_rows = (unsigned long *) std::malloc((std::size_t) num_parts * sizeof(unsigned long));
    part_bytes = (unsigned long *) std::malloc((std::size_t) num_parts * sizeof(unsigned long));
    if ((num_parts != 0) && (part_rows == 0 || part_bytes == 0)) goto done;
    mtx::build_part_rows(row_offsets, num_parts, part_rows);
    mtx::build_part_bytes_from_nnz(row_offsets, part_nnz, num_parts, part_bytes);

    if (!build_default_part_aux(num_parts, &part_aux)) goto done;
    if (!build_identity_shards(num_parts, row_offsets, &shard_offsets)) goto done;
    if (!build_by_bytes(&windows, part_rows, part_bytes, num_parts, opts->max_window_bytes)) goto done;

    if (!mtx::store_coo_header(header_path,
                               h.rows,
                               h.cols,
                               mtx::sum_part_nnz(part_nnz, 0, num_parts),
                               num_parts,
                               num_parts,
                               part_rows,
                               part_nnz,
                               part_aux,
                               shard_offsets)) goto done;

    if (!h.symmetric && h.row_sorted) {
        if (!stream_sorted_general_mtx_to_sharded_coo(mtx_path,
                                                      part_prefix,
                                                      &h,
                                                      &windows,
                                                      row_offsets,
                                                      part_nnz,
                                                      num_parts,
                                                      opts->reader_bytes)) goto done;
        ok = 1;
        goto done;
    }

    for (w = 0; w < windows.count; ++w) {
        if (!mtx::load_part_window_coo(mtx_path,
                                       &h,
                                       row_offsets,
                                       part_nnz,
                                       num_parts,
                                       windows.ranges[w].part_begin,
                                       windows.ranges[w].part_end,
                                       &window_view,
                                       opts->reader_bytes)) goto done;
        if (!mtx::store_part_window_coo(part_prefix, windows.ranges[w].part_begin, &window_view)) goto done;
        clear(&window_view);
        init(&window_view);
    }

    ok = 1;

done:
    clear(&window_view);
    clear(&windows);
    std::free(row_nnz);
    std::free(row_offsets);
    std::free(part_nnz);
    std::free(part_rows);
    std::free(part_bytes);
    std::free(part_aux);
    std::free(shard_offsets);
    return ok;
}

// Small manifest access helpers.
static inline const char *dataset_id_at(const manifest *m, unsigned int idx) {
    return common::get(&m->dataset_ids, idx);
}

static inline const char *matrix_path_at(const manifest *m, unsigned int idx) {
    return common::get(&m->matrix_paths, idx);
}

static inline unsigned int format_at(const manifest *m, unsigned int idx) {
    return idx < m->count ? m->formats[idx] : source_unknown;
}

// Build output file names for one dataset.
static inline int build_output_paths(const char *out_dir,
                                     const char *dataset_id,
                                     char *header_path,
                                     std::size_t header_cap,
                                     char *part_prefix,
                                     std::size_t part_cap) {
    if (std::snprintf(header_path, header_cap, "%s/%s.cshdr", out_dir, dataset_id) <= 0) return 0;
    if (std::snprintf(part_prefix, part_cap, "%s/%s.part", out_dir, dataset_id) <= 0) return 0;
    return 1;
}

// Convert every manifest row that points at an MTX-like source.
static inline int convert_manifest_mtx_series(const manifest *m,
                                              const char *out_dir,
                                              const mtx_convert_options *opts) {
    unsigned int i = 0;
    char header_path[4096];
    char part_prefix[4096];

    for (i = 0; i < m->count; ++i) {
        if (format_at(m, i) != source_mtx && format_at(m, i) != source_tenx_mtx) continue;
        if (!build_output_paths(out_dir, dataset_id_at(m, i), header_path, sizeof(header_path), part_prefix, sizeof(part_prefix))) return 0;
        if (!convert_single_mtx_to_sharded_coo(matrix_path_at(m, i), header_path, part_prefix, opts)) return 0;
    }
    return 1;
}

} // namespace series
} // namespace ingest
} // namespace cellshard
