#include "sharded_file.cuh"

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

inline int write_header(std::FILE *fp, unsigned char format, unsigned int rows, unsigned int cols, unsigned int nnz) {
    if (!write_block(fp, &format, sizeof(format), 1)) return 0;
    if (!write_block(fp, &rows, sizeof(rows), 1)) return 0;
    if (!write_block(fp, &cols, sizeof(cols), 1)) return 0;
    if (!write_block(fp, &nnz, sizeof(nnz), 1)) return 0;
    return 1;
}

inline int read_header(std::FILE *fp, disk_header *out) {
    if (!read_block(fp, &out->format, sizeof(out->format), 1)) return 0;
    if (!read_block(fp, &out->rows, sizeof(out->rows), 1)) return 0;
    if (!read_block(fp, &out->cols, sizeof(out->cols), 1)) return 0;
    if (!read_block(fp, &out->nnz, sizeof(out->nnz), 1)) return 0;
    return 1;
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
    out->part_rows = 0;
    out->part_nnz = 0;
    out->part_aux = 0;
    out->shard_offsets = 0;
    out->num_parts = 0;
    out->num_shards = 0;
}

} // namespace

int store_sharded_header_raw(const char *filename,
                             unsigned char format,
                             unsigned int rows,
                             unsigned int cols,
                             unsigned int nnz,
                             unsigned int num_parts,
                             unsigned int num_shards,
                             const unsigned int *part_rows,
                             const unsigned int *part_nnz,
                             const unsigned int *part_aux,
                             const unsigned int *shard_offsets) {
    std::FILE *fp = 0;
    int ok = 0;

    fp = std::fopen(filename, "wb");
    if (fp == 0) return 0;
    if (!write_header(fp, format, rows, cols, nnz)) goto done;
    if (!write_block(fp, &num_parts, sizeof(unsigned int), 1)) goto done;
    if (!write_block(fp, &num_shards, sizeof(unsigned int), 1)) goto done;
    if (!write_block(fp, part_rows, sizeof(unsigned int), num_parts)) goto done;
    if (!write_block(fp, part_nnz, sizeof(unsigned int), num_parts)) goto done;
    if (!write_block(fp, part_aux, sizeof(unsigned int), num_parts)) goto done;
    if (num_shards != 0 && !write_block(fp, shard_offsets, sizeof(unsigned int), (std::size_t) num_shards + 1)) goto done;
    ok = 1;

done:
    std::fclose(fp);
    return ok;
}

int load_sharded_header_raw(const char *filename, sharded_header_load_result *out) {
    std::FILE *fp = 0;
    int ok = 0;

    out->num_parts = 0;
    out->num_shards = 0;
    out->part_rows = 0;
    out->part_nnz = 0;
    out->part_aux = 0;
    out->shard_offsets = 0;
    fp = std::fopen(filename, "rb");
    if (fp == 0) return 0;
    if (!read_header(fp, &out->h)) goto done;
    if (!read_block(fp, &out->num_parts, sizeof(unsigned int), 1)) goto done;
    if (!read_block(fp, &out->num_shards, sizeof(unsigned int), 1)) goto done;

    out->part_rows = (unsigned int *) alloc_bytes((std::size_t) out->num_parts * sizeof(unsigned int));
    out->part_nnz = (unsigned int *) alloc_bytes((std::size_t) out->num_parts * sizeof(unsigned int));
    out->part_aux = (unsigned int *) alloc_bytes((std::size_t) out->num_parts * sizeof(unsigned int));
    if (out->num_shards != 0) out->shard_offsets = (unsigned int *) alloc_bytes((std::size_t) (out->num_shards + 1) * sizeof(unsigned int));
    if (out->num_parts != 0 && (out->part_rows == 0 || out->part_nnz == 0 || out->part_aux == 0)) goto done;
    if (out->num_shards != 0 && out->shard_offsets == 0) goto done;

    if (!read_block(fp, out->part_rows, sizeof(unsigned int), out->num_parts)) goto done;
    if (!read_block(fp, out->part_nnz, sizeof(unsigned int), out->num_parts)) goto done;
    if (!read_block(fp, out->part_aux, sizeof(unsigned int), out->num_parts)) goto done;
    if (out->num_shards != 0 && !read_block(fp, out->shard_offsets, sizeof(unsigned int), (std::size_t) out->num_shards + 1)) goto done;
    ok = 1;

done:
    if (!ok) free_sharded_header_result(out);
    std::fclose(fp);
    return ok;
}

} // namespace cellshard
