#include "disk.cuh"

#include <cstdio>
#include <cstdlib>

namespace cellshard {

namespace {

inline int write_block(std::FILE *fp, const void *ptr, std::size_t elem_size, std::size_t count) {
    if (count == 0) return 1;
    return std::fwrite(ptr, elem_size, count, fp) == count;
}

inline int read_block(std::FILE *fp, void *ptr, std::size_t elem_size, std::size_t count) {
    if (count == 0) return 1;
    return std::fread(ptr, elem_size, count, fp) == count;
}

inline void *alloc_bytes(std::size_t bytes) {
    if (bytes == 0) return 0;
    return std::malloc(bytes);
}

inline void free_sharded_header_result(sharded_header_load_result *out) {
    std::free(out->part_rows);
    std::free(out->part_nnz);
    std::free(out->part_aux);
    std::free(out->shard_offsets);
    std::free(out->part_offsets);
    std::free(out->part_bytes);
    out->part_rows = 0;
    out->part_nnz = 0;
    out->part_aux = 0;
    out->shard_offsets = 0;
    out->part_offsets = 0;
    out->part_bytes = 0;
    out->num_parts = 0;
    out->num_shards = 0;
}

} // namespace

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
                             const std::uint64_t *part_bytes) {
    static const unsigned char magic[8] = { 'C', 'S', 'P', 'A', 'C', 'K', '0', '1' };
    const unsigned char reserved[7] = { 0, 0, 0, 0, 0, 0, 0 };
    std::FILE *fp = 0;
    int ok = 0;

    fp = std::fopen(filename, "wb");
    if (fp == 0) return 0;
    if (!write_block(fp, magic, sizeof(magic), 1)) goto done;
    if (!write_block(fp, &format, sizeof(format), 1)) goto done;
    if (!write_block(fp, reserved, sizeof(reserved), 1)) goto done;
    if (!write_block(fp, &rows, sizeof(rows), 1)) goto done;
    if (!write_block(fp, &cols, sizeof(cols), 1)) goto done;
    if (!write_block(fp, &nnz, sizeof(nnz), 1)) goto done;
    if (!write_block(fp, &num_parts, sizeof(num_parts), 1)) goto done;
    if (!write_block(fp, &num_shards, sizeof(num_shards), 1)) goto done;
    if (!write_block(fp, &payload_alignment, sizeof(payload_alignment), 1)) goto done;
    if (!write_block(fp, &payload_offset, sizeof(payload_offset), 1)) goto done;
    if (!write_block(fp, part_rows, sizeof(std::uint64_t), (std::size_t) num_parts)) goto done;
    if (!write_block(fp, part_nnz, sizeof(std::uint64_t), (std::size_t) num_parts)) goto done;
    if (!write_block(fp, part_aux, sizeof(std::uint64_t), (std::size_t) num_parts)) goto done;
    if (!write_block(fp, shard_offsets, sizeof(std::uint64_t), (std::size_t) num_shards + 1)) goto done;
    if (!write_block(fp, part_offsets, sizeof(std::uint64_t), (std::size_t) num_parts)) goto done;
    if (!write_block(fp, part_bytes, sizeof(std::uint64_t), (std::size_t) num_parts)) goto done;
    ok = 1;

done:
    std::fclose(fp);
    return ok;
}

int load_sharded_header_raw(const char *filename, sharded_header_load_result *out) {
    static const unsigned char magic[8] = { 'C', 'S', 'P', 'A', 'C', 'K', '0', '1' };
    unsigned char got_magic[8];
    unsigned char reserved[7];
    std::uint64_t rows = 0;
    std::uint64_t cols = 0;
    std::uint64_t nnz = 0;
    std::FILE *fp = 0;
    int ok = 0;

    out->num_parts = 0;
    out->num_shards = 0;
    out->part_rows = 0;
    out->part_nnz = 0;
    out->part_aux = 0;
    out->shard_offsets = 0;
    out->part_offsets = 0;
    out->part_bytes = 0;
    fp = std::fopen(filename, "rb");
    if (fp == 0) return 0;
    if (!read_block(fp, got_magic, sizeof(got_magic), 1)) goto done;
    if (std::memcmp(got_magic, magic, sizeof(magic)) != 0) goto done;
    if (!read_block(fp, &out->h.format, sizeof(out->h.format), 1)) goto done;
    if (!read_block(fp, reserved, sizeof(reserved), 1)) goto done;
    if (!read_block(fp, &rows, sizeof(rows), 1)) goto done;
    if (!read_block(fp, &cols, sizeof(cols), 1)) goto done;
    if (!read_block(fp, &nnz, sizeof(nnz), 1)) goto done;
    out->h.rows = (types::dim_t) rows;
    out->h.cols = (types::dim_t) cols;
    out->h.nnz = (types::nnz_t) nnz;
    if (!read_block(fp, &out->num_parts, sizeof(std::uint64_t), 1)) goto done;
    if (!read_block(fp, &out->num_shards, sizeof(std::uint64_t), 1)) goto done;
    if (!read_block(fp, &out->payload_alignment, sizeof(std::uint64_t), 1)) goto done;
    if (!read_block(fp, &out->payload_offset, sizeof(std::uint64_t), 1)) goto done;

    out->part_rows = (std::uint64_t *) alloc_bytes((std::size_t) out->num_parts * sizeof(std::uint64_t));
    out->part_nnz = (std::uint64_t *) alloc_bytes((std::size_t) out->num_parts * sizeof(std::uint64_t));
    out->part_aux = (std::uint64_t *) alloc_bytes((std::size_t) out->num_parts * sizeof(std::uint64_t));
    out->part_offsets = (std::uint64_t *) alloc_bytes((std::size_t) out->num_parts * sizeof(std::uint64_t));
    out->part_bytes = (std::uint64_t *) alloc_bytes((std::size_t) out->num_parts * sizeof(std::uint64_t));
    if (out->num_shards != 0) out->shard_offsets = (std::uint64_t *) alloc_bytes((std::size_t) (out->num_shards + 1) * sizeof(std::uint64_t));
    if (out->num_parts != 0 && (out->part_rows == 0 || out->part_nnz == 0 || out->part_aux == 0)) goto done;
    if (out->num_parts != 0 && (out->part_offsets == 0 || out->part_bytes == 0)) goto done;
    if (out->num_shards != 0 && out->shard_offsets == 0) goto done;

    if (!read_block(fp, out->part_rows, sizeof(std::uint64_t), (std::size_t) out->num_parts)) goto done;
    if (!read_block(fp, out->part_nnz, sizeof(std::uint64_t), (std::size_t) out->num_parts)) goto done;
    if (!read_block(fp, out->part_aux, sizeof(std::uint64_t), (std::size_t) out->num_parts)) goto done;
    if (!read_block(fp, out->shard_offsets, sizeof(std::uint64_t), (std::size_t) out->num_shards + 1)) goto done;
    if (!read_block(fp, out->part_offsets, sizeof(std::uint64_t), (std::size_t) out->num_parts)) goto done;
    if (!read_block(fp, out->part_bytes, sizeof(std::uint64_t), (std::size_t) out->num_parts)) goto done;
    ok = 1;

done:
    if (!ok) free_sharded_header_result(out);
    std::fclose(fp);
    return ok;
}

} // namespace cellshard
