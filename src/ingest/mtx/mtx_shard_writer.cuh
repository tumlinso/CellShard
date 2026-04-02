#pragma once

#include "../../sharded/disk.cuh"

namespace cellshard {
namespace ingest {
namespace mtx {

// Thin wrappers kept so ingest code can talk in MTX/COO terms while the real
// storage implementation lives in sharded/disk.cuh.
static inline int store_sharded_coo(const char *header_path,
                                    const sharded<sparse::coo> *view) {
    return store(header_path, view, 0);
}

static inline int store_part_window_coo(const char *packfile_path,
                                        unsigned long,
                                        const sharded<sparse::coo> *view) {
    return store(packfile_path, view, 0);
}

// Header-only metadata store for sharded COO outputs.
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
    std::uint64_t *part_rows_u64 = 0;
    std::uint64_t *part_nnz_u64 = 0;
    std::uint64_t *part_aux_u64 = 0;
    std::uint64_t *shard_offsets_u64 = 0;
    std::uint64_t *part_offsets_u64 = 0;
    std::uint64_t *part_bytes_u64 = 0;
    unsigned long i = 0;
    int ok = 0;

    if (num_parts != 0) {
        part_rows_u64 = (std::uint64_t *) std::malloc((std::size_t) num_parts * sizeof(std::uint64_t));
        part_nnz_u64 = (std::uint64_t *) std::malloc((std::size_t) num_parts * sizeof(std::uint64_t));
        part_aux_u64 = (std::uint64_t *) std::malloc((std::size_t) num_parts * sizeof(std::uint64_t));
        part_offsets_u64 = (std::uint64_t *) std::calloc((std::size_t) num_parts, sizeof(std::uint64_t));
        part_bytes_u64 = (std::uint64_t *) std::calloc((std::size_t) num_parts, sizeof(std::uint64_t));
        if (part_rows_u64 == 0 || part_nnz_u64 == 0 || part_aux_u64 == 0 || part_offsets_u64 == 0 || part_bytes_u64 == 0) goto done;
        for (i = 0; i < num_parts; ++i) {
            part_rows_u64[i] = (std::uint64_t) part_rows[i];
            part_nnz_u64[i] = (std::uint64_t) part_nnz[i];
            part_aux_u64[i] = (std::uint64_t) part_aux[i];
        }
    }

    shard_offsets_u64 = (std::uint64_t *) std::malloc((std::size_t) (num_shards + 1ul) * sizeof(std::uint64_t));
    if (shard_offsets_u64 == 0) goto done;
    for (i = 0; i <= num_shards; ++i) {
        shard_offsets_u64[i] = (std::uint64_t) shard_offsets[i];
    }

    ok = store_sharded_header_raw(header_path,
                                  disk_format_coo,
                                  (std::uint64_t) rows,
                                  (std::uint64_t) cols,
                                  (std::uint64_t) total_nnz,
                                  (std::uint64_t) num_parts,
                                  (std::uint64_t) num_shards,
                                  4096,
                                  0,
                                  part_rows_u64,
                                  part_nnz_u64,
                                  part_aux_u64,
                                  shard_offsets_u64,
                                  part_offsets_u64,
                                  part_bytes_u64);

done:
    std::free(part_rows_u64);
    std::free(part_nnz_u64);
    std::free(part_aux_u64);
    std::free(shard_offsets_u64);
    std::free(part_offsets_u64);
    std::free(part_bytes_u64);
    return ok;
}

} // namespace mtx
} // namespace ingest
} // namespace cellshard
