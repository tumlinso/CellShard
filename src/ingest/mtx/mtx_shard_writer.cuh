#pragma once

#include "../../sharded/disk.cuh"

namespace cellshard {
namespace ingest {
namespace mtx {

static inline int store_sharded_coo(const char *header_path,
                                    const char *part_prefix,
                                    const sharded<sparse::coo> *view) {
    shard_storage files;
    int ok = 0;

    init(&files);
    if (!bind_sequential(&files, (unsigned int) view->num_parts, part_prefix)) goto done;
    ok = store(header_path, view, &files);

done:
    clear(&files);
    return ok;
}

static inline int store_part_window_coo(const char *part_prefix,
                                        unsigned long global_part_begin,
                                        const sharded<sparse::coo> *view) {
    char path[4096];
    unsigned long i = 0;

    for (i = 0; i < view->num_parts; ++i) {
        if (std::snprintf(path, sizeof(path), "%s.%lu", part_prefix, global_part_begin + i) <= 0) return 0;
        if (view->parts[i] == 0) return 0;
        if (!store(path, view->parts[i])) return 0;
    }
    return 1;
}

static inline int store_coo_header(const char *header_path,
                                   unsigned long rows,
                                   unsigned long cols,
                                   unsigned long total_nnz,
                                   unsigned long num_parts,
                                   unsigned long num_shards,
                                   const unsigned long *part_rows,
                                   const unsigned long *part_nnz,
                                   const unsigned long *part_aux,
                                   const unsigned long *shard_offsets) {
    unsigned int *part_rows_u32 = 0;
    unsigned int *part_nnz_u32 = 0;
    unsigned int *part_aux_u32 = 0;
    unsigned int *shard_offsets_u32 = 0;
    unsigned long i = 0;
    int ok = 0;

    if (rows > (unsigned long) UINT_MAX || cols > (unsigned long) UINT_MAX || total_nnz > (unsigned long) UINT_MAX) return 0;
    if (num_parts > (unsigned long) UINT_MAX || num_shards > (unsigned long) UINT_MAX) return 0;

    if (num_parts != 0) {
        part_rows_u32 = (unsigned int *) std::malloc((std::size_t) num_parts * sizeof(unsigned int));
        part_nnz_u32 = (unsigned int *) std::malloc((std::size_t) num_parts * sizeof(unsigned int));
        part_aux_u32 = (unsigned int *) std::malloc((std::size_t) num_parts * sizeof(unsigned int));
        if (part_rows_u32 == 0 || part_nnz_u32 == 0 || part_aux_u32 == 0) goto done;
        for (i = 0; i < num_parts; ++i) {
            if (part_rows[i] > (unsigned long) UINT_MAX) goto done;
            if (part_nnz[i] > (unsigned long) UINT_MAX) goto done;
            if (part_aux[i] > (unsigned long) UINT_MAX) goto done;
            part_rows_u32[i] = (unsigned int) part_rows[i];
            part_nnz_u32[i] = (unsigned int) part_nnz[i];
            part_aux_u32[i] = (unsigned int) part_aux[i];
        }
    }

    if (num_shards != 0) {
        shard_offsets_u32 = (unsigned int *) std::malloc((std::size_t) (num_shards + 1ul) * sizeof(unsigned int));
        if (shard_offsets_u32 == 0) goto done;
        for (i = 0; i <= num_shards; ++i) {
            if (shard_offsets[i] > (unsigned long) UINT_MAX) goto done;
            shard_offsets_u32[i] = (unsigned int) shard_offsets[i];
        }
    }

    ok = store_sharded_header_raw(header_path,
                                  disk_format_coo,
                                  (unsigned int) rows,
                                  (unsigned int) cols,
                                  (unsigned int) total_nnz,
                                  (unsigned int) num_parts,
                                  (unsigned int) num_shards,
                                  part_rows_u32,
                                  part_nnz_u32,
                                  part_aux_u32,
                                  shard_offsets_u32);

done:
    std::free(part_rows_u32);
    std::free(part_nnz_u32);
    std::free(part_aux_u32);
    std::free(shard_offsets_u32);
    return ok;
}

} // namespace mtx
} // namespace ingest
} // namespace cellshard
