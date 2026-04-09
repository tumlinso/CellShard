#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "../formats/dense.cuh"
#include "../formats/compressed.cuh"
#include "../formats/triplet.cuh"
#include "../formats/diagonal.cuh"
#include "sharded.cuh"
#include "sharded_host.cuh"
#include "../disk/matrix.cuh"
#include "shard_paths.cuh"
#include "series_h5.cuh"

namespace cellshard {

// Map a sharded matrix type to its per-part disk tag.
template<typename MatrixT>
inline int check_sharded_disk_format(unsigned char actual) {
    return check_disk_format((unsigned char) disk_format_code<MatrixT>::value, actual, disk_format_code<MatrixT>::name());
}

// Temporary owned result for a packfile metadata load.
struct sharded_header_load_result {
    disk_header h;
    std::uint64_t num_parts;
    std::uint64_t num_shards;
    std::uint64_t payload_alignment;
    std::uint64_t payload_offset;
    std::uint64_t *part_rows;
    std::uint64_t *part_nnz;
    std::uint64_t *part_aux;
    std::uint64_t *shard_offsets;
    std::uint64_t *part_offsets;
    std::uint64_t *part_bytes;
};

// Raw metadata-only header store/load. These do not move part payload bytes.
int store_sharded_header_raw(const char *filename,
                             unsigned char format,
                             std::uint64_t rows,
                             std::uint64_t cols,
                             std::uint64_t nnz,
                             std::uint64_t num_parts,
                             std::uint64_t num_shards,
                             std::uint64_t payload_alignment,
                             std::uint64_t payload_offset,
                             const std::uint64_t *part_rows,
                             const std::uint64_t *part_nnz,
                             const std::uint64_t *part_aux,
                             const std::uint64_t *shard_offsets,
                             const std::uint64_t *part_offsets,
                             const std::uint64_t *part_bytes);

int load_sharded_header_raw(const char *filename, sharded_header_load_result *out);

// Metadata block I/O wrappers.
inline int write_sharded_block(std::FILE *fp, const void *ptr, std::size_t elem_size, std::size_t count) {
    if (count == 0) return 1;
    return std::fwrite(ptr, elem_size, count, fp) == count;
}

inline int read_sharded_block(std::FILE *fp, void *ptr, std::size_t elem_size, std::size_t count) {
    if (count == 0) return 1;
    return std::fread(ptr, elem_size, count, fp) == count;
}

// Host-sized unsigned long metadata is normalized to fixed u64 on disk.
inline int sharded_to_u64(unsigned long value, std::uint64_t *out, const char *label, const char *filename) {
    *out = (std::uint64_t) value;
    if ((unsigned long) *out != value) {
        std::fprintf(stderr, "Error: %s out of disk u64 range in %s\n", label, filename);
        return 0;
    }
    return 1;
}

inline int sharded_from_u64(std::uint64_t value, unsigned long *out, const char *label, const char *filename) {
    *out = (unsigned long) value;
    if ((std::uint64_t) *out != value) {
        std::fprintf(stderr, "Error: %s does not fit target sharded index type in %s\n", label, filename);
        return 0;
    }
    return 1;
}

inline int store_sharded_index_array(std::FILE *fp, const unsigned long *src, std::size_t count, const char *label, const char *filename) {
    std::uint64_t value = 0;
    std::size_t i = 0;

    for (i = 0; i < count; ++i) {
        if (!sharded_to_u64(src[i], &value, label, filename)) return 0;
        if (!write_sharded_block(fp, &value, sizeof(value), 1)) return 0;
    }
    return 1;
}

inline int load_sharded_index_array(std::FILE *fp, unsigned long *dst, std::size_t count, const char *label, const char *filename) {
    std::uint64_t value = 0;
    std::size_t i = 0;

    for (i = 0; i < count; ++i) {
        if (!read_sharded_block(fp, &value, sizeof(value), 1)) return 0;
        if (!sharded_from_u64(value, dst + i, label, filename)) return 0;
    }
    return 1;
}

// Header-only store path. This allocates temporary u64 metadata arrays and
// writes only the sharded metadata block.
template<typename MatrixT>
inline int store_header(const char *filename, const sharded<MatrixT> *m) {
    std::uint64_t *part_rows = 0;
    std::uint64_t *part_nnz = 0;
    std::uint64_t *part_aux = 0;
    std::uint64_t *part_offsets = 0;
    std::uint64_t *part_bytes = 0;
    std::uint64_t *shard_offsets = 0;
    unsigned long i = 0;
    int ok = 0;

    if (m == 0) return 0;
    if (m->num_parts != 0) {
        part_rows = (std::uint64_t *) std::calloc((std::size_t) m->num_parts, sizeof(std::uint64_t));
        part_nnz = (std::uint64_t *) std::calloc((std::size_t) m->num_parts, sizeof(std::uint64_t));
        part_aux = (std::uint64_t *) std::calloc((std::size_t) m->num_parts, sizeof(std::uint64_t));
        part_offsets = (std::uint64_t *) std::calloc((std::size_t) m->num_parts, sizeof(std::uint64_t));
        part_bytes = (std::uint64_t *) std::calloc((std::size_t) m->num_parts, sizeof(std::uint64_t));
        if (part_rows == 0 || part_nnz == 0 || part_aux == 0 || part_offsets == 0 || part_bytes == 0) goto done;
        for (i = 0; i < m->num_parts; ++i) {
            part_rows[i] = (std::uint64_t) m->part_rows[i];
            part_nnz[i] = (std::uint64_t) m->part_nnz[i];
            part_aux[i] = (std::uint64_t) m->part_aux[i];
        }
    }

    shard_offsets = (std::uint64_t *) std::calloc((std::size_t) (m->num_shards + 1), sizeof(std::uint64_t));
    if (shard_offsets == 0 && m->num_shards != 0) goto done;
    for (i = 0; i <= m->num_shards; ++i) {
        shard_offsets[i] = m->shard_offsets != 0 ? (std::uint64_t) m->shard_offsets[i] : 0;
    }

    ok = store_sharded_header_raw(filename,
                                  (unsigned char) disk_format_code<MatrixT>::value,
                                  (std::uint64_t) m->rows,
                                  (std::uint64_t) m->cols,
                                  (std::uint64_t) m->nnz,
                                  (std::uint64_t) m->num_parts,
                                  (std::uint64_t) m->num_shards,
                                  4096,
                                  0,
                                  part_rows,
                                  part_nnz,
                                  part_aux,
                                  shard_offsets,
                                  part_offsets,
                                  part_bytes);

done:
    std::free(part_rows);
    std::free(part_nnz);
    std::free(part_aux);
    std::free(part_offsets);
    std::free(part_bytes);
    std::free(shard_offsets);
    return ok;
}

// Header-only load path. This allocates host metadata tables and optionally
// binds packfile locators, but does not fetch any part payload.
template<typename MatrixT>
inline int load_packfile_header(const char *filename, sharded<MatrixT> *m, shard_storage *s) {
    sharded_header_load_result tmp;
    unsigned long i = 0;
    int ok = 0;

    tmp.num_parts = 0;
    tmp.num_shards = 0;
    tmp.payload_alignment = 0;
    tmp.payload_offset = 0;
    tmp.part_rows = 0;
    tmp.part_nnz = 0;
    tmp.part_aux = 0;
    tmp.shard_offsets = 0;
    tmp.part_offsets = 0;
    tmp.part_bytes = 0;
    if (!load_sharded_header_raw(filename, &tmp)) return 0;
    clear(m);
    init(m);
    if (!check_sharded_disk_format<MatrixT>(tmp.h.format)) goto done;
    if (!sharded_from_u64((std::uint64_t) tmp.h.rows, &m->rows, "rows", filename)) goto done;
    if (!sharded_from_u64((std::uint64_t) tmp.h.cols, &m->cols, "cols", filename)) goto done;
    if (!sharded_from_u64((std::uint64_t) tmp.h.nnz, &m->nnz, "nnz", filename)) goto done;
    if (!sharded_from_u64(tmp.num_parts, &m->num_parts, "num_parts", filename)) goto done;
    if (!sharded_from_u64(tmp.num_shards, &m->num_shards, "num_shards", filename)) goto done;
    m->part_capacity = m->num_parts;
    m->shard_capacity = m->num_shards;

    if (m->part_capacity != 0) {
        m->parts = (MatrixT **) std::calloc((std::size_t) m->part_capacity, sizeof(MatrixT *));
        m->part_offsets = (unsigned long *) std::calloc((std::size_t) (m->part_capacity + 1), sizeof(unsigned long));
        m->part_rows = (unsigned long *) std::calloc((std::size_t) m->part_capacity, sizeof(unsigned long));
        m->part_nnz = (unsigned long *) std::calloc((std::size_t) m->part_capacity, sizeof(unsigned long));
        m->part_aux = (unsigned long *) std::calloc((std::size_t) m->part_capacity, sizeof(unsigned long));
        if (m->parts == 0 || m->part_offsets == 0 || m->part_rows == 0 || m->part_nnz == 0 || m->part_aux == 0) {
            clear(m);
            goto done;
        }
    }
    if (m->shard_capacity != 0) {
        m->shard_offsets = (unsigned long *) std::calloc((std::size_t) (m->shard_capacity + 1), sizeof(unsigned long));
        m->shard_parts = (unsigned long *) std::calloc((std::size_t) (m->shard_capacity + 1), sizeof(unsigned long));
        if (m->shard_offsets == 0 || m->shard_parts == 0) {
            clear(m);
            goto done;
        }
    }

    for (i = 0; i < m->num_parts; ++i) {
        if (!sharded_from_u64(tmp.part_rows[i], &m->part_rows[i], "part_rows", filename)) goto done;
        if (!sharded_from_u64(tmp.part_nnz[i], &m->part_nnz[i], "part_nnz", filename)) goto done;
        if (!sharded_from_u64(tmp.part_aux[i], &m->part_aux[i], "part_aux", filename)) goto done;
    }
    for (i = 0; i < m->num_shards; ++i) {
        if (!sharded_from_u64(tmp.shard_offsets[i], &m->shard_offsets[i], "shard_offsets", filename)) goto done;
    }
    if (m->num_shards != 0 && !sharded_from_u64(tmp.shard_offsets[m->num_shards], &m->shard_offsets[m->num_shards], "shard_offsets", filename)) goto done;
    rebuild_part_offsets(m);
    rebuild_shard_parts(m);
    if (s != 0) {
        init(s);
        if (!reserve(s, (unsigned int) m->num_parts)) goto done;
        if (!bind_packfile(s, filename)) goto done;
        for (i = 0; i < m->num_parts; ++i) {
            if (!bind_part(s, (unsigned int) i, tmp.part_offsets[i], tmp.part_bytes[i])) goto done;
        }
    }
    ok = 1;

done:
    if (!ok) clear(m);
    std::free(tmp.part_rows);
    std::free(tmp.part_nnz);
    std::free(tmp.part_aux);
    std::free(tmp.shard_offsets);
    std::free(tmp.part_offsets);
    std::free(tmp.part_bytes);
    return ok;
}

template<typename MatrixT>
inline int load_header(const char *filename, sharded<MatrixT> *m) {
    return load_packfile_header(filename, m, 0);
}

template<typename MatrixT>
inline int load_header(const char *filename, sharded<MatrixT> *m, shard_storage *s) {
    return load_packfile_header(filename, m, s);
}

inline int load_header(const char *filename, sharded<sparse::compressed> *m, shard_storage *s) {
    const char *ext = std::strrchr(filename != 0 ? filename : "", '.');
    if (ext != 0) {
        if (std::strcmp(ext, ".csh5") == 0 || std::strcmp(ext, ".h5") == 0 || std::strcmp(ext, ".hdf5") == 0) {
            return load_series_compressed_h5_header(filename, m, s);
        }
    }
    return load_packfile_header(filename, m, s);
}

inline int load_header(const char *filename, sharded<sparse::compressed> *m) {
    return load_header(filename, m, 0);
}

// Full sharded store path:
// - requires every part to be materialized on host
// - computes aligned payload offsets
// - writes one header block
// - seeks and stores each part payload into the same packfile
template<typename MatrixT>
inline int store(const char *filename, const sharded<MatrixT> *m, shard_storage *s) {
    static const unsigned char magic[8] = { 'C', 'S', 'P', 'A', 'C', 'K', '0', '1' };
    static const std::uint64_t payload_alignment = 4096;
    std::uint64_t *part_offsets = 0;
    std::uint64_t *part_sizes = 0;
    std::uint64_t payload_offset = 0;
    std::uint64_t cursor = 0;
    std::FILE *fp = 0;
    unsigned long i = 0;
    int ok = 0;

    if (m == 0) return 0;
    for (i = 0; i < m->num_parts; ++i) {
        if (m->parts[i] == 0) return 0;
    }

    if (m->num_parts != 0) {
        part_offsets = (std::uint64_t *) std::calloc((std::size_t) m->num_parts, sizeof(std::uint64_t));
        part_sizes = (std::uint64_t *) std::calloc((std::size_t) m->num_parts, sizeof(std::uint64_t));
        if (part_offsets == 0 || part_sizes == 0) goto done;
    }

    payload_offset = 8
        + sizeof(unsigned char)
        + 7
        + sizeof(std::uint64_t) * 7
        + sizeof(std::uint64_t) * (std::size_t) m->num_parts * 3
        + sizeof(std::uint64_t) * (std::size_t) (m->num_shards + 1)
        + sizeof(std::uint64_t) * (std::size_t) m->num_parts * 2;
    payload_offset = (payload_offset + payload_alignment - 1) & ~(payload_alignment - 1);
    cursor = payload_offset;

    for (i = 0; i < m->num_parts; ++i) {
        const std::size_t part_size = packed_bytes((const MatrixT *) 0,
                                                   (types::dim_t) m->part_rows[i],
                                                   (types::dim_t) m->cols,
                                                   (types::nnz_t) m->part_nnz[i],
                                                   m->part_aux[i],
                                                   sizeof(real::storage_t));
        part_offsets[i] = cursor;
        part_sizes[i] = (std::uint64_t) part_size;
        cursor += (std::uint64_t) part_size;
        cursor = (cursor + payload_alignment - 1) & ~(payload_alignment - 1);
    }

    fp = std::fopen(filename, "wb");
    if (fp == 0) goto done;
    if (!write_sharded_block(fp, magic, sizeof(magic), 1)) goto done;
    {
        const unsigned char format = (unsigned char) disk_format_code<MatrixT>::value;
        const unsigned char reserved[7] = { 0, 0, 0, 0, 0, 0, 0 };
        std::uint64_t rows = (std::uint64_t) m->rows;
        std::uint64_t cols = (std::uint64_t) m->cols;
        std::uint64_t nnz = (std::uint64_t) m->nnz;
        std::uint64_t num_parts = (std::uint64_t) m->num_parts;
        std::uint64_t num_shards = (std::uint64_t) m->num_shards;

        if (!write_sharded_block(fp, &format, sizeof(format), 1)) goto done;
        if (!write_sharded_block(fp, reserved, sizeof(reserved), 1)) goto done;
        if (!write_sharded_block(fp, &rows, sizeof(rows), 1)) goto done;
        if (!write_sharded_block(fp, &cols, sizeof(cols), 1)) goto done;
        if (!write_sharded_block(fp, &nnz, sizeof(nnz), 1)) goto done;
        if (!write_sharded_block(fp, &num_parts, sizeof(num_parts), 1)) goto done;
        if (!write_sharded_block(fp, &num_shards, sizeof(num_shards), 1)) goto done;
        if (!write_sharded_block(fp, &payload_alignment, sizeof(payload_alignment), 1)) goto done;
        if (!write_sharded_block(fp, &payload_offset, sizeof(payload_offset), 1)) goto done;
        if (!store_sharded_index_array(fp, m->part_rows, (std::size_t) m->num_parts, "part_rows", filename)) goto done;
        if (!store_sharded_index_array(fp, m->part_nnz, (std::size_t) m->num_parts, "part_nnz", filename)) goto done;
        if (!store_sharded_index_array(fp, m->part_aux, (std::size_t) m->num_parts, "part_aux", filename)) goto done;
        if (m->num_shards != 0) {
            if (!store_sharded_index_array(fp, m->shard_offsets, (std::size_t) (m->num_shards + 1), "shard_offsets", filename)) goto done;
        } else {
            const std::uint64_t zero = 0;
            if (!write_sharded_block(fp, &zero, sizeof(zero), 1)) goto done;
        }
        if (!write_sharded_block(fp, part_offsets, sizeof(std::uint64_t), (std::size_t) m->num_parts)) goto done;
        if (!write_sharded_block(fp, part_sizes, sizeof(std::uint64_t), (std::size_t) m->num_parts)) goto done;
    }

    if (std::fflush(fp) != 0) goto done;
    for (i = 0; i < m->num_parts; ++i) {
        if (std::fseek(fp, (long) part_offsets[i], SEEK_SET) != 0) goto done;
        if (!::cellshard::store(fp, m->parts[i])) goto done;
    }
    if (s != 0) {
        init(s);
        if (!reserve(s, (unsigned int) m->num_parts)) goto done;
        if (!bind_packfile(s, filename)) goto done;
        for (i = 0; i < m->num_parts; ++i) {
            if (!bind_part(s, (unsigned int) i, part_offsets[i], part_sizes[i])) goto done;
        }
    }
    ok = 1;

done:
    if (fp != 0) std::fclose(fp);
    std::free(part_offsets);
    std::free(part_sizes);
    return ok;
}

} // namespace cellshard
