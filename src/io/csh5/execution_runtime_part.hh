#pragma once

inline int load_sliced_bucketed_partition_from_exec_pack(const dataset_h5_state *state,
                                                         unsigned long shard_id,
                                                         unsigned long partition_id,
                                                         bucketed_sliced_ell_partition *out);

inline int load_optimized_partition_blob(std::FILE *fp, bucketed_blocked_ell_partition *part) {
    std::uint32_t segment = 0u;
    if (fp == 0 || part == 0) return 0;
    clear(part);
    if (!read_sharded_block(fp, &part->rows, sizeof(part->rows), 1u)) return 0;
    if (!read_sharded_block(fp, &part->cols, sizeof(part->cols), 1u)) return 0;
    if (!read_sharded_block(fp, &part->nnz, sizeof(part->nnz), 1u)) return 0;
    if (!read_sharded_block(fp, &part->segment_count, sizeof(part->segment_count), 1u)) return 0;
    part->segments = part->segment_count != 0u ? (sparse::blocked_ell *) std::calloc((std::size_t) part->segment_count, sizeof(sparse::blocked_ell)) : 0;
    part->segment_row_offsets = (std::uint32_t *) std::calloc((std::size_t) part->segment_count + 1u, sizeof(std::uint32_t));
    part->exec_to_canonical_rows = part->rows != 0u ? (std::uint32_t *) std::calloc((std::size_t) part->rows, sizeof(std::uint32_t)) : 0;
    part->canonical_to_exec_rows = part->rows != 0u ? (std::uint32_t *) std::calloc((std::size_t) part->rows, sizeof(std::uint32_t)) : 0;
    if ((part->segment_count != 0u && (part->segments == 0 || part->segment_row_offsets == 0))
        || (part->rows != 0u && (part->exec_to_canonical_rows == 0 || part->canonical_to_exec_rows == 0))) {
        clear(part);
        return 0;
    }
    for (segment = 0u; segment < part->segment_count; ++segment) sparse::init(part->segments + segment);
    if (!read_packed_u32_array(fp, part->segment_row_offsets, (std::size_t) part->segment_count + 1u, 0)) {
        clear(part);
        return 0;
    }
    if (!read_packed_u32_array(fp, part->exec_to_canonical_rows, part->rows, 1)
        || !invert_u32_permutation(part->canonical_to_exec_rows, part->exec_to_canonical_rows, part->rows)) {
        clear(part);
        return 0;
    }
    for (segment = 0u; segment < part->segment_count; ++segment) {
        if (!::cellshard::load(fp, part->segments + segment)) {
            clear(part);
            return 0;
        }
    }
    return 1;
}

inline int serialize_optimized_shard(const bucketed_blocked_ell_shard *shard,
                                     unsigned char **data_out,
                                     std::size_t *bytes_out) {
    char *buffer = 0;
    std::size_t bytes = 0u;
    std::FILE *fp = 0;
    std::uint32_t partition = 0u;
    if (shard == 0 || data_out == 0 || bytes_out == 0) return 0;
    *data_out = 0;
    *bytes_out = 0u;
    fp = open_memstream(&buffer, &bytes);
    if (fp == 0) return 0;
    if (!write_sharded_block(fp, optimized_blocked_shard_magic, sizeof(optimized_blocked_shard_magic), 1u)
        || !write_sharded_block(fp, &shard->rows, sizeof(shard->rows), 1u)
        || !write_sharded_block(fp, &shard->cols, sizeof(shard->cols), 1u)
        || !write_sharded_block(fp, &shard->nnz, sizeof(shard->nnz), 1u)
        || !write_sharded_block(fp, &shard->partition_count, sizeof(shard->partition_count), 1u)
        || !write_packed_u32_array(fp, shard->partition_row_offsets, (std::size_t) shard->partition_count + 1u, 0)
        || !write_packed_u32_array(fp, shard->exec_to_canonical_cols, shard->cols, 1)) {
        std::fclose(fp);
        std::free(buffer);
        return 0;
    }
    for (partition = 0u; partition < shard->partition_count; ++partition) {
        if (!write_optimized_partition_blob(fp, shard->partitions + partition)) {
            std::fclose(fp);
            std::free(buffer);
            return 0;
        }
    }
    if (std::fclose(fp) != 0) {
        std::free(buffer);
        return 0;
    }
    *data_out = (unsigned char *) buffer;
    *bytes_out = bytes;
    return 1;
}

inline int deserialize_optimized_shard(const unsigned char *data,
                                       std::size_t bytes,
                                       bucketed_blocked_ell_shard *shard) {
    std::FILE *fp = 0;
    std::uint32_t partition = 0u;
    unsigned char magic[sizeof(optimized_blocked_shard_magic)] = {};
    if (data == 0 || shard == 0) return 0;
    clear(shard);
    init(shard);
    fp = fmemopen((void *) data, bytes, "rb");
    if (fp == 0) return 0;
    if (!read_sharded_block(fp, magic, sizeof(magic), 1u)) {
        std::fclose(fp);
        clear(shard);
        return 0;
    }
    if (std::memcmp(magic, optimized_blocked_shard_magic, sizeof(magic)) != 0) {
        std::rewind(fp);
        if (!read_sharded_block(fp, &shard->rows, sizeof(shard->rows), 1u)
            || !read_sharded_block(fp, &shard->cols, sizeof(shard->cols), 1u)
            || !read_sharded_block(fp, &shard->nnz, sizeof(shard->nnz), 1u)
            || !read_sharded_block(fp, &shard->partition_count, sizeof(shard->partition_count), 1u)) {
            std::fclose(fp);
            clear(shard);
            return 0;
        }
    } else {
        if (!read_sharded_block(fp, &shard->rows, sizeof(shard->rows), 1u)
            || !read_sharded_block(fp, &shard->cols, sizeof(shard->cols), 1u)
            || !read_sharded_block(fp, &shard->nnz, sizeof(shard->nnz), 1u)
            || !read_sharded_block(fp, &shard->partition_count, sizeof(shard->partition_count), 1u)) {
            std::fclose(fp);
            clear(shard);
            return 0;
        }
    }
    shard->partition_row_offsets = (std::uint32_t *) std::calloc((std::size_t) shard->partition_count + 1u, sizeof(std::uint32_t));
    shard->partitions = shard->partition_count != 0u
        ? (bucketed_blocked_ell_partition *) std::calloc((std::size_t) shard->partition_count, sizeof(bucketed_blocked_ell_partition))
        : 0;
    shard->exec_to_canonical_cols = shard->cols != 0u ? (std::uint32_t *) std::calloc((std::size_t) shard->cols, sizeof(std::uint32_t)) : 0;
    shard->canonical_to_exec_cols = shard->cols != 0u ? (std::uint32_t *) std::calloc((std::size_t) shard->cols, sizeof(std::uint32_t)) : 0;
    if ((shard->partition_count != 0u && (shard->partition_row_offsets == 0 || shard->partitions == 0))
        || (shard->cols != 0u && (shard->exec_to_canonical_cols == 0 || shard->canonical_to_exec_cols == 0))) {
        std::fclose(fp);
        clear(shard);
        return 0;
    }
    for (partition = 0u; partition < shard->partition_count; ++partition) init(shard->partitions + partition);
    if (std::memcmp(magic, optimized_blocked_shard_magic, sizeof(magic)) == 0) {
        if (!read_packed_u32_array(fp, shard->partition_row_offsets, (std::size_t) shard->partition_count + 1u, 0)
            || !read_packed_u32_array(fp, shard->exec_to_canonical_cols, shard->cols, 1)
            || !invert_u32_permutation(shard->canonical_to_exec_cols, shard->exec_to_canonical_cols, shard->cols)) {
            std::fclose(fp);
            clear(shard);
            return 0;
        }
    } else if (!read_sharded_block(fp, shard->partition_row_offsets, sizeof(std::uint32_t), (std::size_t) shard->partition_count + 1u)
               || !read_sharded_block(fp, shard->exec_to_canonical_cols, sizeof(std::uint32_t), shard->cols)
               || !read_sharded_block(fp, shard->canonical_to_exec_cols, sizeof(std::uint32_t), shard->cols)) {
        std::fclose(fp);
        clear(shard);
        return 0;
    }
    for (partition = 0u; partition < shard->partition_count; ++partition) {
        if (!(std::memcmp(magic, optimized_blocked_shard_magic, sizeof(magic)) == 0
                  ? load_optimized_partition_blob(fp, shard->partitions + partition)
                  : load_execution_partition_blob(fp, shard->partitions + partition))
            || !assign_partition_col_maps(shard->partitions + partition,
                                          shard->exec_to_canonical_cols,
                                          shard->canonical_to_exec_cols,
                                          shard->cols)) {
            std::fclose(fp);
            clear(shard);
            return 0;
        }
    }
    std::fclose(fp);
    return 1;
}

inline int write_sliced_execution_partition_blob(std::FILE *fp, const bucketed_sliced_ell_partition *part) {
    std::uint32_t segment = 0u;
    if (fp == 0 || part == 0) return 0;
    if (!write_sharded_block(fp, &part->rows, sizeof(part->rows), 1u)) return 0;
    if (!write_sharded_block(fp, &part->cols, sizeof(part->cols), 1u)) return 0;
    if (!write_sharded_block(fp, &part->nnz, sizeof(part->nnz), 1u)) return 0;
    if (!write_sharded_block(fp, &part->segment_count, sizeof(part->segment_count), 1u)) return 0;
    if (!write_sharded_block(fp, &part->canonical_slice_count, sizeof(part->canonical_slice_count), 1u)) return 0;
    if (!write_sharded_block(fp, part->segment_row_offsets, sizeof(std::uint32_t), (std::size_t) part->segment_count + 1u)) return 0;
    if (!write_sharded_block(fp, part->exec_to_canonical_rows, sizeof(std::uint32_t), part->rows)) return 0;
    if (!write_sharded_block(fp, part->canonical_to_exec_rows, sizeof(std::uint32_t), part->rows)) return 0;
    if (!write_sharded_block(fp,
                             part->canonical_slice_row_offsets,
                             sizeof(std::uint32_t),
                             (std::size_t) part->canonical_slice_count + 1u)) {
        return 0;
    }
    if (!write_sharded_block(fp,
                             part->canonical_slice_widths,
                             sizeof(std::uint32_t),
                             (std::size_t) part->canonical_slice_count)) {
        return 0;
    }
    for (segment = 0u; segment < part->segment_count; ++segment) {
        if (!::cellshard::store(fp, part->segments + segment)) return 0;
    }
    return 1;
}

inline int load_sliced_execution_partition_blob(std::FILE *fp, bucketed_sliced_ell_partition *part) {
    std::uint32_t segment = 0u;
    if (fp == 0 || part == 0) return 0;
    clear(part);
    if (!read_sharded_block(fp, &part->rows, sizeof(part->rows), 1u)) return 0;
    if (!read_sharded_block(fp, &part->cols, sizeof(part->cols), 1u)) return 0;
    if (!read_sharded_block(fp, &part->nnz, sizeof(part->nnz), 1u)) return 0;
    if (!read_sharded_block(fp, &part->segment_count, sizeof(part->segment_count), 1u)) return 0;
    if (!read_sharded_block(fp, &part->canonical_slice_count, sizeof(part->canonical_slice_count), 1u)) return 0;
    part->segments = part->segment_count != 0u ? (sparse::sliced_ell *) std::calloc((std::size_t) part->segment_count, sizeof(sparse::sliced_ell)) : 0;
    part->segment_row_offsets = (std::uint32_t *) std::calloc((std::size_t) part->segment_count + 1u, sizeof(std::uint32_t));
    part->exec_to_canonical_rows = part->rows != 0u ? (std::uint32_t *) std::calloc((std::size_t) part->rows, sizeof(std::uint32_t)) : 0;
    part->canonical_to_exec_rows = part->rows != 0u ? (std::uint32_t *) std::calloc((std::size_t) part->rows, sizeof(std::uint32_t)) : 0;
    part->canonical_slice_row_offsets =
        (std::uint32_t *) std::calloc((std::size_t) part->canonical_slice_count + 1u, sizeof(std::uint32_t));
    part->canonical_slice_widths = part->canonical_slice_count != 0u
        ? (std::uint32_t *) std::calloc((std::size_t) part->canonical_slice_count, sizeof(std::uint32_t))
        : 0;
    if ((part->segment_count != 0u && (part->segments == 0 || part->segment_row_offsets == 0))
        || (part->rows != 0u && (part->exec_to_canonical_rows == 0 || part->canonical_to_exec_rows == 0))
        || part->canonical_slice_row_offsets == 0
        || (part->canonical_slice_count != 0u && part->canonical_slice_widths == 0)) {
        clear(part);
        return 0;
    }
    for (segment = 0u; segment < part->segment_count; ++segment) sparse::init(part->segments + segment);
    if (!read_sharded_block(fp, part->segment_row_offsets, sizeof(std::uint32_t), (std::size_t) part->segment_count + 1u)) {
        clear(part);
        return 0;
    }
    if (!read_sharded_block(fp, part->exec_to_canonical_rows, sizeof(std::uint32_t), part->rows)) {
        clear(part);
        return 0;
    }
    if (!read_sharded_block(fp, part->canonical_to_exec_rows, sizeof(std::uint32_t), part->rows)) {
        clear(part);
        return 0;
    }
    if (!read_sharded_block(fp,
                            part->canonical_slice_row_offsets,
                            sizeof(std::uint32_t),
                            (std::size_t) part->canonical_slice_count + 1u)) {
        clear(part);
        return 0;
    }
    if (!read_sharded_block(fp,
                            part->canonical_slice_widths,
                            sizeof(std::uint32_t),
                            (std::size_t) part->canonical_slice_count)) {
        clear(part);
        return 0;
    }
    for (segment = 0u; segment < part->segment_count; ++segment) {
        if (!::cellshard::load(fp, part->segments + segment)) {
            clear(part);
            return 0;
        }
    }
    return 1;
}

inline int find_bucketed_sliced_segment(const bucketed_sliced_ell_partition *part,
                                        std::uint32_t exec_row,
                                        std::uint32_t *segment_out,
                                        std::uint32_t *segment_row_out) {
    if (part == 0 || segment_out == 0 || segment_row_out == 0 || exec_row >= part->rows) return 0;
    for (std::uint32_t segment = 0u; segment < part->segment_count; ++segment) {
        const std::uint32_t row_begin = part->segment_row_offsets[segment];
        const std::uint32_t row_end = part->segment_row_offsets[segment + 1u];
        if (exec_row >= row_begin && exec_row < row_end) {
            *segment_out = segment;
            *segment_row_out = exec_row - row_begin;
            return 1;
        }
    }
    return 0;
}

inline int rebuild_canonical_sliced_partition(const bucketed_sliced_ell_partition *src,
                                              sparse::sliced_ell *out) {
    if (src == 0 || out == 0) return 0;
    sparse::clear(out);
    sparse::init(out, src->rows, src->cols, src->nnz);
    if (!sparse::allocate(out,
                          src->canonical_slice_count,
                          src->canonical_slice_row_offsets,
                          src->canonical_slice_widths)) {
        sparse::clear(out);
        return 0;
    }
    for (std::uint32_t exec_row = 0u; exec_row < src->rows; ++exec_row) {
        std::uint32_t segment = 0u, segment_row = 0u;
        const std::uint32_t canonical_row = src->exec_to_canonical_rows != 0 ? src->exec_to_canonical_rows[exec_row] : exec_row;
        const std::uint32_t dst_slice = sparse::find_slice(out, canonical_row);
        const std::uint32_t dst_row_begin = dst_slice < out->slice_count ? out->slice_row_offsets[dst_slice] : 0u;
        const std::uint32_t dst_width = dst_slice < out->slice_count ? out->slice_widths[dst_slice] : 0u;
        const std::size_t dst_base =
            sparse::slice_slot_base(out, dst_slice) + (std::size_t) (canonical_row - dst_row_begin) * (std::size_t) dst_width;
        std::size_t dst_slot = 0u;
        if (!find_bucketed_sliced_segment(src, exec_row, &segment, &segment_row)) {
            sparse::clear(out);
            return 0;
        }
        {
            const sparse::sliced_ell *seg = src->segments + segment;
            const std::uint32_t seg_width = seg->slice_count != 0u ? seg->slice_widths[0] : 0u;
            const std::size_t seg_base = (std::size_t) segment_row * (std::size_t) seg_width;
            for (std::uint32_t slot = 0u; slot < seg_width; ++slot) {
                const types::idx_t col = seg->col_idx[seg_base + slot];
                if (col == sparse::sliced_ell_invalid_col) continue;
                if (dst_slot >= dst_width) {
                    sparse::clear(out);
                    return 0;
                }
                out->col_idx[dst_base + dst_slot] = col;
                out->val[dst_base + dst_slot] = seg->val[seg_base + slot];
                ++dst_slot;
            }
        }
    }
    return 1;
}

inline int serialize_optimized_sliced_shard(const bucketed_sliced_ell_shard *shard,
                                            unsigned char **data_out,
                                            std::size_t *bytes_out) {
    char *buffer = 0;
    std::size_t bytes = 0u;
    std::FILE *fp = 0;
    std::uint32_t partition = 0u;
    if (shard == 0 || data_out == 0 || bytes_out == 0) return 0;
    *data_out = 0;
    *bytes_out = 0u;
    fp = open_memstream(&buffer, &bytes);
    if (fp == 0) return 0;
    if (!write_sharded_block(fp, &shard->rows, sizeof(shard->rows), 1u)
        || !write_sharded_block(fp, &shard->cols, sizeof(shard->cols), 1u)
        || !write_sharded_block(fp, &shard->nnz, sizeof(shard->nnz), 1u)
        || !write_sharded_block(fp, &shard->partition_count, sizeof(shard->partition_count), 1u)
        || !write_sharded_block(fp, shard->partition_row_offsets, sizeof(std::uint32_t), (std::size_t) shard->partition_count + 1u)) {
        std::fclose(fp);
        std::free(buffer);
        return 0;
    }
    for (partition = 0u; partition < shard->partition_count; ++partition) {
        if (!write_sliced_execution_partition_blob(fp, shard->partitions + partition)) {
            std::fclose(fp);
            std::free(buffer);
            return 0;
        }
    }
    if (std::fclose(fp) != 0) {
        std::free(buffer);
        return 0;
    }
    *data_out = (unsigned char *) buffer;
    *bytes_out = bytes;
    return 1;
}

inline int deserialize_optimized_sliced_shard(const unsigned char *data,
                                              std::size_t bytes,
                                              bucketed_sliced_ell_shard *shard) {
    std::FILE *fp = 0;
    std::uint32_t partition = 0u;
    if (data == 0 || shard == 0) return 0;
    clear(shard);
    init(shard);
    fp = fmemopen((void *) data, bytes, "rb");
    if (fp == 0) return 0;
    if (!read_sharded_block(fp, &shard->rows, sizeof(shard->rows), 1u)
        || !read_sharded_block(fp, &shard->cols, sizeof(shard->cols), 1u)
        || !read_sharded_block(fp, &shard->nnz, sizeof(shard->nnz), 1u)
        || !read_sharded_block(fp, &shard->partition_count, sizeof(shard->partition_count), 1u)) {
        std::fclose(fp);
        clear(shard);
        return 0;
    }
    shard->partition_row_offsets = (std::uint32_t *) std::calloc((std::size_t) shard->partition_count + 1u, sizeof(std::uint32_t));
    shard->partitions = shard->partition_count != 0u
        ? (bucketed_sliced_ell_partition *) std::calloc((std::size_t) shard->partition_count, sizeof(bucketed_sliced_ell_partition))
        : 0;
    if (shard->partition_count != 0u && (shard->partition_row_offsets == 0 || shard->partitions == 0)) {
        std::fclose(fp);
        clear(shard);
        return 0;
    }
    for (partition = 0u; partition < shard->partition_count; ++partition) init(shard->partitions + partition);
    if (!read_sharded_block(fp, shard->partition_row_offsets, sizeof(std::uint32_t), (std::size_t) shard->partition_count + 1u)) {
        std::fclose(fp);
        clear(shard);
        return 0;
    }
    for (partition = 0u; partition < shard->partition_count; ++partition) {
        if (!load_sliced_execution_partition_blob(fp, shard->partitions + partition)) {
            std::fclose(fp);
            clear(shard);
            return 0;
        }
    }
    std::fclose(fp);
    return 1;
}

inline void close_cached_shard_file(dataset_h5_state *state, unsigned long shard_id) {
    dataset_h5_cache_runtime *runtime = cache_runtime(state);
    if (state == 0 || runtime == 0 || shard_id >= state->num_shards || state->shard_cache_files == 0) return;
    std::lock_guard<std::mutex> file_lock(runtime->shard_file_mutexes[shard_id]);
    if (state->shard_cache_files[shard_id] != 0) {
        std::fclose(state->shard_cache_files[shard_id]);
        state->shard_cache_files[shard_id] = 0;
    }
}

inline int ensure_cached_shard_file_open(dataset_h5_state *state, unsigned long shard_id) {
    dataset_h5_cache_runtime *runtime = cache_runtime(state);
    if (state == 0 || runtime == 0 || shard_id >= state->num_shards || state->shard_cache_paths == 0) return 0;
    std::lock_guard<std::mutex> file_lock(runtime->shard_file_mutexes[shard_id]);
    if (state->shard_cache_files[shard_id] != 0) return 1;
    if (state->shard_cache_paths[shard_id] == 0) return 0;
    state->shard_cache_files[shard_id] = std::fopen(state->shard_cache_paths[shard_id], "rb");
    if (state->shard_cache_files[shard_id] == 0) return 0;
    std::setvbuf(state->shard_cache_files[shard_id], 0, _IOFBF, (std::size_t) 8u << 20u);
    return 1;
}

template<typename MatrixT>
inline void compute_cached_part_locator(const dataset_h5_state *state,
                                        unsigned long partition_id,
                                        std::uint64_t *offset,
                                        std::uint64_t *bytes) {
    const std::uint64_t shard_id = state != 0 && state->partition_shard_ids != 0 ? state->partition_shard_ids[partition_id] : 0u;
    const std::uint64_t begin = state != 0 && state->shard_part_begin != 0 ? state->shard_part_begin[shard_id] : 0u;
    const std::uint64_t end = state != 0 && state->shard_part_end != 0 ? state->shard_part_end[shard_id] : 0u;
    std::uint64_t cursor = sharded_pack_payload_offset(end - begin, 1u, shard_pack_payload_alignment);
    std::uint64_t i = 0u;

    if (offset != 0) *offset = cursor;
    if (bytes != 0) *bytes = 0u;
    if (state == 0 || partition_id >= state->num_partitions || begin > partition_id || end <= partition_id) return;
    for (i = begin; i < partition_id; ++i) {
        const std::size_t partition_bytes = packed_bytes((const MatrixT *) 0,
                                                    (types::dim_t) state->partition_rows[i],
                                                    (types::dim_t) state->cols,
                                                    (types::nnz_t) state->partition_nnz[i],
                                                    (unsigned long) state->partition_aux[i],
                                                    sizeof(real::storage_t));
        cursor += (std::uint64_t) partition_bytes;
        cursor = (cursor + shard_pack_payload_alignment - 1u) & ~(shard_pack_payload_alignment - 1u);
    }
    if (offset != 0) *offset = cursor;
    if (bytes != 0) {
        *bytes = (std::uint64_t) packed_bytes((const MatrixT *) 0,
                                              (types::dim_t) state->partition_rows[partition_id],
                                              (types::dim_t) state->cols,
                                              (types::nnz_t) state->partition_nnz[partition_id],
                                              (unsigned long) state->partition_aux[partition_id],
                                              sizeof(real::storage_t));
    }
}

inline int load_blocked_ell_part_from_cached_pack(sharded<sparse::blocked_ell> *m,
                                                  dataset_h5_state *state,
                                                  unsigned long partition_id) {
    const unsigned long shard_id = state != 0 && state->partition_shard_ids != 0 ? (unsigned long) state->partition_shard_ids[partition_id] : 0ul;
    dataset_h5_cache_runtime *runtime = cache_runtime(state);
    sparse::blocked_ell *part = 0;
    std::uint64_t offset = 0u;
    int ok = 0;

    if (m == 0 || state == 0 || runtime == 0 || partition_id >= m->num_partitions) return 0;
    if (!ensure_cached_shard_file_open(state, shard_id)) {
        std::fprintf(stderr, "cellshard: failed to open cached blocked shard file %lu for partition %lu\n", shard_id, partition_id);
        return 0;
    }
    compute_cached_part_locator<sparse::blocked_ell>(state, partition_id, &offset, 0);
    std::lock_guard<std::mutex> file_lock(runtime->shard_file_mutexes[shard_id]);
    if (state->shard_cache_files[shard_id] == 0) return 0;
    part = new sparse::blocked_ell;
    sparse::init(part);
    if (fseeko(state->shard_cache_files[shard_id], (off_t) offset, SEEK_SET) != 0) {
        std::fprintf(stderr, "cellshard: failed to seek cached blocked shard %lu to offset %llu\n", shard_id, (unsigned long long) offset);
        goto done;
    }
    if (!::cellshard::load(state->shard_cache_files[shard_id], part)) {
        std::fprintf(stderr, "cellshard: failed to load blocked partition %lu from cached shard %lu\n", partition_id, shard_id);
        goto done;
    }
    if (part->rows != m->partition_rows[partition_id]) {
        std::fprintf(stderr, "cellshard: cached blocked part rows mismatch for partition %lu: got=%u expected=%llu\n", partition_id, part->rows, (unsigned long long) m->partition_rows[partition_id]);
        goto done;
    }
    if (part->cols != m->cols) {
        std::fprintf(stderr, "cellshard: cached blocked part cols mismatch for partition %lu: got=%u expected=%lu\n", partition_id, part->cols, m->cols);
        goto done;
    }
    if (part->nnz != m->partition_nnz[partition_id]) {
        std::fprintf(stderr,
                     "cellshard: cached blocked part nnz mismatch for partition %lu: got=%llu expected=%llu\n",
                     partition_id,
                     (unsigned long long) part->nnz,
                     (unsigned long long) m->partition_nnz[partition_id]);
        goto done;
    }
    if (::cellshard::partition_aux(part) != m->partition_aux[partition_id]) {
        std::fprintf(stderr,
                     "cellshard: cached blocked part aux mismatch for partition %lu: got=%lu expected=%llu\n",
                     partition_id,
                     (unsigned long) ::cellshard::partition_aux(part),
                     (unsigned long long) m->partition_aux[partition_id]);
        goto done;
    }
    if (m->parts[partition_id] != 0) destroy(m->parts[partition_id]);
    m->parts[partition_id] = part;
    part = 0;
    ok = 1;

done:
    if (part != 0) {
        sparse::clear(part);
        delete part;
    }
    return ok;
}

inline int load_quantized_blocked_ell_part_from_cached_pack(sharded<sparse::quantized_blocked_ell> *m,
                                                            dataset_h5_state *state,
                                                            unsigned long partition_id) {
    const unsigned long shard_id = state != 0 && state->partition_shard_ids != 0 ? (unsigned long) state->partition_shard_ids[partition_id] : 0ul;
    dataset_h5_cache_runtime *runtime = cache_runtime(state);
    sparse::quantized_blocked_ell *part = 0;
    std::uint64_t offset = 0u;
    int ok = 0;

    if (m == 0 || state == 0 || runtime == 0 || partition_id >= m->num_partitions) return 0;
    if (!ensure_cached_shard_file_open(state, shard_id)) return 0;
    compute_cached_part_locator<sparse::quantized_blocked_ell>(state, partition_id, &offset, 0);
    std::lock_guard<std::mutex> file_lock(runtime->shard_file_mutexes[shard_id]);
    if (state->shard_cache_files[shard_id] == 0) return 0;
    part = new sparse::quantized_blocked_ell;
    sparse::init(part);
    if (fseeko(state->shard_cache_files[shard_id], (off_t) offset, SEEK_SET) != 0) goto done;
    if (!::cellshard::load(state->shard_cache_files[shard_id], part)) goto done;
    if (part->rows != m->partition_rows[partition_id]) goto done;
    if (part->cols != m->cols) goto done;
    if (part->nnz != m->partition_nnz[partition_id]) goto done;
    if (::cellshard::partition_aux(part) != m->partition_aux[partition_id]) goto done;
    if (m->parts[partition_id] != 0) destroy(m->parts[partition_id]);
    m->parts[partition_id] = part;
    part = 0;
    ok = 1;

done:
    if (part != 0) {
        sparse::clear(part);
        delete part;
    }
    return ok;
}

inline int load_sliced_ell_part_from_cached_pack(sharded<sparse::sliced_ell> *m,
                                                 dataset_h5_state *state,
                                                 unsigned long partition_id) {
    const unsigned long shard_id = state != 0 && state->partition_shard_ids != 0 ? (unsigned long) state->partition_shard_ids[partition_id] : 0ul;
    sparse::sliced_ell *part = 0;
    bucketed_sliced_ell_partition stored;
    int ok = 0;

    init(&stored);
    if (m == 0 || state == 0 || partition_id >= m->num_partitions) return 0;
    part = new sparse::sliced_ell;
    sparse::init(part);
    if (!load_sliced_bucketed_partition_from_exec_pack(state, shard_id, partition_id, &stored)) goto done;
    if (!rebuild_canonical_sliced_partition(&stored, part)) goto done;
    if (part->rows != m->partition_rows[partition_id]) goto done;
    if (part->cols != m->cols) goto done;
    if (part->nnz != m->partition_nnz[partition_id]) goto done;
    if (::cellshard::partition_aux(part) != m->partition_aux[partition_id]) goto done;
    if (m->parts[partition_id] != 0) destroy(m->parts[partition_id]);
    m->parts[partition_id] = part;
    part = 0;
    ok = 1;

done:
    clear(&stored);
    if (part != 0) {
        sparse::clear(part);
        delete part;
    }
    return ok;
}

inline int load_execution_partition_from_pack(const dataset_h5_state *state,
                                              unsigned long shard_id,
                                              unsigned long partition_id,
                                              bucketed_blocked_ell_partition *out) {
    char path[4096];
    std::FILE *fp = 0;
    unsigned char magic[8];
    std::uint64_t file_shard_id = 0u;
    std::uint64_t partition_count = 0u;
    std::uint64_t *partition_offsets = 0;
    std::uint64_t local_partition_id = 0u;
    int ok = 0;

    if (state == 0 || out == 0 || shard_id >= state->num_shards || partition_id >= state->num_partitions) return 0;
    if (!build_execution_pack_path(state, shard_id, path, sizeof(path))) return 0;
    fp = std::fopen(path, "rb");
    if (fp == 0) return 0;
    if (!read_sharded_block(fp, magic, sizeof(magic), 1u)) goto done;
    if (std::memcmp(magic, execution_pack_magic, sizeof(magic)) != 0) goto done;
    if (!read_sharded_block(fp, &file_shard_id, sizeof(file_shard_id), 1u)) goto done;
    if (!read_sharded_block(fp, &partition_count, sizeof(partition_count), 1u)) goto done;
    if (file_shard_id != shard_id) goto done;
    local_partition_id = partition_id - state->shard_part_begin[shard_id];
    if (local_partition_id >= partition_count) goto done;
    partition_offsets = (std::uint64_t *) std::calloc((std::size_t) partition_count, sizeof(std::uint64_t));
    if (partition_count != 0u && partition_offsets == 0) goto done;
    if (!read_sharded_block(fp, partition_offsets, sizeof(std::uint64_t), (std::size_t) partition_count)) goto done;
    if (fseeko(fp, (off_t) partition_offsets[local_partition_id], SEEK_SET) != 0) goto done;
    if (!load_execution_partition_blob(fp, out)) goto done;
    ok = 1;

done:
    if (fp != 0) std::fclose(fp);
    std::free(partition_offsets);
    return ok;
}

inline int load_sliced_execution_partition_from_pack(const dataset_h5_state *state,
                                                     unsigned long shard_id,
                                                     unsigned long partition_id,
                                                     bucketed_sliced_ell_partition *out) {
    char path[4096];
    std::FILE *fp = 0;
    unsigned char magic[8];
    std::uint64_t file_shard_id = 0u;
    std::uint64_t partition_count = 0u;
    std::uint64_t *partition_offsets = 0;
    std::uint64_t local_partition_id = 0u;
    int ok = 0;

    if (state == 0 || out == 0 || shard_id >= state->num_shards || partition_id >= state->num_partitions) return 0;
    if (!build_execution_pack_path(state, shard_id, path, sizeof(path))) return 0;
    fp = std::fopen(path, "rb");
    if (fp == 0) return 0;
    if (!read_sharded_block(fp, magic, sizeof(magic), 1u)) goto done;
    if (std::memcmp(magic, execution_pack_magic, sizeof(magic)) != 0) goto done;
    if (!read_sharded_block(fp, &file_shard_id, sizeof(file_shard_id), 1u)) goto done;
    if (!read_sharded_block(fp, &partition_count, sizeof(partition_count), 1u)) goto done;
    if (file_shard_id != shard_id) goto done;
    local_partition_id = partition_id - state->shard_part_begin[shard_id];
    if (local_partition_id >= partition_count) goto done;
    partition_offsets = (std::uint64_t *) std::calloc((std::size_t) partition_count, sizeof(std::uint64_t));
    if (partition_count != 0u && partition_offsets == 0) goto done;
    if (!read_sharded_block(fp, partition_offsets, sizeof(std::uint64_t), (std::size_t) partition_count)) goto done;
    if (fseeko(fp, (off_t) partition_offsets[local_partition_id], SEEK_SET) != 0) goto done;
    if (!load_sliced_execution_partition_blob(fp, out)) goto done;
    ok = 1;

done:
    if (fp != 0) std::fclose(fp);
    std::free(partition_offsets);
    return ok;
}

inline int load_sliced_bucketed_partition_from_exec_pack(const dataset_h5_state *state,
                                                         unsigned long shard_id,
                                                         unsigned long partition_id,
                                                         bucketed_sliced_ell_partition *out) {
    char path[4096];
    std::FILE *fp = 0;
    unsigned char magic[8];
    std::uint64_t file_shard_id = 0u;
    std::uint64_t partition_count = 0u;
    std::uint64_t *partition_offsets = 0;
    std::uint64_t local_partition_id = 0u;
    int ok = 0;

    if (state == 0 || out == 0 || shard_id >= state->num_shards || partition_id >= state->num_partitions) return 0;
    if (!build_execution_pack_path(state, shard_id, path, sizeof(path))) return 0;
    fp = std::fopen(path, "rb");
    if (fp == 0) return 0;
    if (!read_sharded_block(fp, magic, sizeof(magic), 1u)) goto done;
    if (std::memcmp(magic, execution_pack_magic, sizeof(magic)) != 0) goto done;
    if (!read_sharded_block(fp, &file_shard_id, sizeof(file_shard_id), 1u)) goto done;
    if (!read_sharded_block(fp, &partition_count, sizeof(partition_count), 1u)) goto done;
    if (file_shard_id != shard_id) goto done;
    local_partition_id = partition_id - state->shard_part_begin[shard_id];
    if (local_partition_id >= partition_count) goto done;
    partition_offsets = (std::uint64_t *) std::calloc((std::size_t) partition_count, sizeof(std::uint64_t));
    if (partition_count != 0u && partition_offsets == 0) goto done;
    if (!read_sharded_block(fp, partition_offsets, sizeof(std::uint64_t), (std::size_t) partition_count)) goto done;
    if (fseeko(fp, (off_t) partition_offsets[local_partition_id], SEEK_SET) != 0) goto done;
    if (!load_sliced_execution_partition_blob(fp, out)) goto done;
    ok = 1;

done:
    if (fp != 0) std::fclose(fp);
    std::free(partition_offsets);
    return ok;
}

inline int materialize_blocked_ell_execution_pack(shard_storage *s, dataset_h5_state *state, unsigned long shard_id) {
    const std::uint64_t begin = state != 0 && state->shard_part_begin != 0 ? state->shard_part_begin[shard_id] : 0u;
    const std::uint64_t end = state != 0 && state->shard_part_end != 0 ? state->shard_part_end[shard_id] : 0u;
    const std::uint64_t partition_count = end >= begin ? (end - begin) : 0u;
    sparse::blocked_ell **parts = 0;
    bucketed_blocked_ell_partition *exec_parts = 0;
    const bucketed_blocked_ell_shard *optimized_shard = 0;
    std::uint64_t *partition_offsets = 0;
    char tmp_path[4096];
    char final_path[4096];
    std::FILE *fp = 0;
    std::uint64_t local = 0u;
    std::uint64_t pack_bytes = 0u;
    int ok = 0;

    if (s == 0 || state == 0 || shard_id >= state->num_shards) return 0;
    if (!open_dataset_h5_backend(s)) return 0;
    if (partition_count != 0u) {
        partition_offsets = (std::uint64_t *) std::calloc((std::size_t) partition_count, sizeof(std::uint64_t));
        if (partition_offsets == 0) goto done;
    }
    if (state->matrix_family == dataset_matrix_family_optimized_blocked_ell) {
        if (!load_optimized_blocked_ell_shard_payload(state, shard_id)) goto done;
        optimized_shard = &state->loaded_optimized_shard;
        if (optimized_shard->partition_count != partition_count) goto done;
    } else {
        if (!load_blocked_ell_shard_payload(state, shard_id)) goto done;
        if (partition_count != 0u) {
            parts = (sparse::blocked_ell **) std::calloc((std::size_t) partition_count, sizeof(sparse::blocked_ell *));
            exec_parts = (bucketed_blocked_ell_partition *) std::calloc((std::size_t) partition_count, sizeof(bucketed_blocked_ell_partition));
            if (parts == 0 || exec_parts == 0) goto done;
            for (local = 0u; local < partition_count; ++local) init(exec_parts + local);
        }
        if (!prepare_blocked_ell_parts_from_state(state, (unsigned long) begin, (unsigned long) end, parts)) goto done;
        if (!fill_blocked_ell_parts_from_loaded_shard(state, shard_id, (unsigned long) begin, (unsigned long) end, parts)) goto done;
        for (local = 0u; local < partition_count; ++local) {
            const std::uint64_t partition_id = begin + local;
            const std::uint32_t requested_bucket_count =
                state->partition_execution_formats != 0
                && state->partition_execution_formats[partition_id] == dataset_execution_format_bucketed_blocked_ell
                && state->partition_blocked_ell_bucket_counts != 0
                    ? std::max<std::uint32_t>(1u, state->partition_blocked_ell_bucket_counts[partition_id])
                    : 1u;
            std::uint64_t bucketed_bytes = 0u;
            if (!build_bucketed_execution_partition(exec_parts + local, parts[local], requested_bucket_count, &bucketed_bytes)) goto done;
            if (state->partition_bucketed_blocked_ell_bytes != 0) state->partition_bucketed_blocked_ell_bytes[partition_id] = bucketed_bytes;
        }
    }
    if (!build_execution_pack_temp_path(state, shard_id, tmp_path, sizeof(tmp_path))) goto done;
    if (!build_execution_pack_path(state, shard_id, final_path, sizeof(final_path))) goto done;
    fp = std::fopen(tmp_path, "wb");
    if (fp == 0) goto done;
    std::setvbuf(fp, 0, _IOFBF, (std::size_t) 8u << 20u);
    if (!write_sharded_block(fp, execution_pack_magic, sizeof(execution_pack_magic), 1u)) goto done;
    if (!write_sharded_block(fp, &shard_id, sizeof(shard_id), 1u)) goto done;
    if (!write_sharded_block(fp, &partition_count, sizeof(partition_count), 1u)) goto done;
    for (local = 0u; local < partition_count; ++local) {
        const std::uint64_t zero = 0u;
        if (!write_sharded_block(fp, &zero, sizeof(zero), 1u)) goto done;
    }
    for (local = 0u; local < partition_count; ++local) {
        partition_offsets[local] = (std::uint64_t) ftello(fp);
        if (optimized_shard != 0) {
            if (!write_execution_partition_blob(fp, optimized_shard->partitions + local)) goto done;
        } else {
            if (!write_execution_partition_blob(fp, exec_parts + local)) goto done;
        }
    }
    if (fseeko(fp, (off_t) (sizeof(execution_pack_magic) + sizeof(std::uint64_t) * 2u), SEEK_SET) != 0) goto done;
    if (!write_sharded_block(fp, partition_offsets, sizeof(std::uint64_t), (std::size_t) partition_count)) goto done;
    if (std::fflush(fp) != 0) goto done;
    std::fclose(fp);
    fp = 0;
    if (::rename(tmp_path, final_path) != 0) {
        std::remove(tmp_path);
        goto done;
    }
    ok = 1;

done:
    if (fp != 0) std::fclose(fp);
    if (!ok && build_execution_pack_temp_path(state, shard_id, tmp_path, sizeof(tmp_path))) std::remove(tmp_path);
    if (exec_parts != 0) {
        for (local = 0u; local < partition_count; ++local) clear(exec_parts + local);
    }
    clear_blocked_ell_parts(parts, (unsigned long) partition_count);
    std::free(partition_offsets);
    std::free(exec_parts);
    std::free(parts);
    return ok;
}

inline int materialize_sliced_ell_execution_pack(shard_storage *s, dataset_h5_state *state, unsigned long shard_id) {
    const std::uint64_t begin = state != 0 && state->shard_part_begin != 0 ? state->shard_part_begin[shard_id] : 0u;
    const std::uint64_t end = state != 0 && state->shard_part_end != 0 ? state->shard_part_end[shard_id] : 0u;
    const std::uint64_t partition_count = end >= begin ? (end - begin) : 0u;
    bucketed_sliced_ell_partition *parts = 0;
    std::uint64_t *partition_offsets = 0;
    char tmp_path[4096];
    char final_path[4096];
    std::uint64_t local = 0u;
    std::FILE *fp = 0;
    int ok = 0;

    if (s == 0 || state == 0 || shard_id >= state->num_shards) return 0;
    if (!open_dataset_h5_backend(s)) return 0;
    if (partition_count != 0u) {
        parts = (bucketed_sliced_ell_partition *) std::calloc((std::size_t) partition_count, sizeof(bucketed_sliced_ell_partition));
        partition_offsets = (std::uint64_t *) std::calloc((std::size_t) partition_count, sizeof(std::uint64_t));
        if (parts == 0 || partition_offsets == 0) return 0;
        for (local = 0u; local < partition_count; ++local) init(parts + local);
    }
    for (local = 0u; local < partition_count; ++local) {
        const std::uint64_t partition_id = begin + local;
        if (!load_bucketed_sliced_ell_partition_payload(state, (unsigned long) partition_id, parts + local)) goto done;
    }
    if (!build_execution_pack_temp_path(state, shard_id, tmp_path, sizeof(tmp_path))) goto done;
    if (!build_execution_pack_path(state, shard_id, final_path, sizeof(final_path))) goto done;
    fp = std::fopen(tmp_path, "wb");
    if (fp == 0) goto done;
    std::setvbuf(fp, 0, _IOFBF, (std::size_t) 8u << 20u);
    if (!write_sharded_block(fp, execution_pack_magic, sizeof(execution_pack_magic), 1u)) goto done;
    if (!write_sharded_block(fp, &shard_id, sizeof(shard_id), 1u)) goto done;
    if (!write_sharded_block(fp, &partition_count, sizeof(partition_count), 1u)) goto done;
    for (local = 0u; local < partition_count; ++local) {
        const std::uint64_t zero = 0u;
        if (!write_sharded_block(fp, &zero, sizeof(zero), 1u)) goto done;
    }
    for (local = 0u; local < partition_count; ++local) {
        partition_offsets[local] = (std::uint64_t) ftello(fp);
        if (!write_sliced_execution_partition_blob(fp, parts + local)) goto done;
    }
    if (fseeko(fp, (off_t) (sizeof(execution_pack_magic) + sizeof(std::uint64_t) * 2u), SEEK_SET) != 0) goto done;
    if (!write_sharded_block(fp, partition_offsets, sizeof(std::uint64_t), (std::size_t) partition_count)) goto done;
    if (std::fflush(fp) != 0) goto done;
    std::fclose(fp);
    fp = 0;
    if (::rename(tmp_path, final_path) != 0) {
        std::remove(tmp_path);
        goto done;
    }
    ok = 1;

done:
    if (fp != 0) std::fclose(fp);
    if (!ok && build_execution_pack_temp_path(state, shard_id, tmp_path, sizeof(tmp_path))) std::remove(tmp_path);
    if (parts != 0) {
        for (local = 0u; local < partition_count; ++local) clear(parts + local);
    }
    std::free(partition_offsets);
    std::free(parts);
    return ok;
}

inline int ensure_execution_pack_ready(shard_storage *s, dataset_h5_state *state, unsigned long shard_id) {
    char path[4096];
    if (s == 0 || state == 0 || shard_id >= state->num_shards) return 0;
    if (!ensure_dataset_cache_layout(s)) return 0;
    if (!build_execution_pack_path(state, shard_id, path, sizeof(path))) return 0;
    if (::access(path, R_OK) == 0) return 1;
    if (!require_storage_capability(s,
                                    shard_storage_cap_materialize_execution_pack,
                                    "materialize execution pack")) {
        return 0;
    }
    if (state->matrix_family == dataset_matrix_family_sliced_ell) return materialize_sliced_ell_execution_pack(s, state, shard_id);
    return materialize_blocked_ell_execution_pack(s, state, shard_id);
}

inline int materialize_blocked_ell_shard_pack(shard_storage *s, dataset_h5_state *state, unsigned long shard_id) {
    const std::uint64_t begin = state != 0 && state->shard_part_begin != 0 ? state->shard_part_begin[shard_id] : 0u;
    const std::uint64_t end = state != 0 && state->shard_part_end != 0 ? state->shard_part_end[shard_id] : 0u;
    const std::uint64_t partition_count = end >= begin ? (end - begin) : 0u;
    sparse::blocked_ell **parts = 0;
    char tmp_path[4096];
    char final_path[4096];
    int ok = 0;

    if (s == 0 || state == 0 || shard_id >= state->num_shards) return 0;
    if (!open_dataset_h5_backend(s)) return 0;
    if (partition_count != 0u) {
        parts = (sparse::blocked_ell **) std::calloc((std::size_t) partition_count, sizeof(sparse::blocked_ell *));
        if (parts == 0) return 0;
    }
    if (state->matrix_family == dataset_matrix_family_optimized_blocked_ell) {
        if (!load_optimized_blocked_ell_shard_payload(state, shard_id)) {
            std::fprintf(stderr, "cellshard: failed to load optimized blocked shard payload %lu while materializing canonical pack\n", shard_id);
            goto done;
        }
        if (state->loaded_optimized_shard.partition_count != partition_count) {
            std::fprintf(stderr,
                         "cellshard: optimized blocked shard %lu partition count mismatch: loaded=%u expected=%llu\n",
                         shard_id,
                         state->loaded_optimized_shard.partition_count,
                         (unsigned long long) partition_count);
            goto done;
        }
        for (std::uint64_t local = 0u; local < partition_count; ++local) {
            const std::uint64_t partition_id = begin + local;
            const std::uint32_t block_size =
                sparse::unpack_blocked_ell_block_size((unsigned long) state->partition_aux[partition_id]);
            sparse::blocked_ell *part = new sparse::blocked_ell;
            sparse::init(part);
            if (!reconstruct_canonical_blocked_ell_part(state->loaded_optimized_shard.partitions + local,
                                                        state->loaded_optimized_shard.exec_to_canonical_cols,
                                                        block_size,
                                                        part)) {
                std::fprintf(stderr,
                             "cellshard: failed to reconstruct canonical blocked partition %llu from optimized shard %lu (block_size=%u)\n",
                             (unsigned long long) partition_id,
                             shard_id,
                             block_size);
                sparse::clear(part);
                delete part;
                goto done;
            }
            parts[local] = part;
        }
    } else {
        if (!load_blocked_ell_shard_payload(state, shard_id)) return 0;
        if (!prepare_blocked_ell_parts_from_state(state, (unsigned long) begin, (unsigned long) end, parts)) goto done;
        if (!fill_blocked_ell_parts_from_loaded_shard(state, shard_id, (unsigned long) begin, (unsigned long) end, parts)) goto done;
    }
    if (!build_shard_pack_temp_path(state, shard_id, tmp_path, sizeof(tmp_path))) goto done;
    if (!build_shard_pack_path(state, shard_id, final_path, sizeof(final_path))) goto done;
    if (!write_shard_pack_file<sparse::blocked_ell>(tmp_path,
                                                    state->cols,
                                                    state->partition_rows + begin,
                                                    state->partition_nnz + begin,
                                                    state->partition_aux + begin,
                                                    partition_count,
                                                    parts)) {
        std::fprintf(stderr, "cellshard: failed to write canonical blocked shard pack for shard %lu\n", shard_id);
        goto done;
    }
    if (::rename(tmp_path, final_path) != 0) {
        std::remove(tmp_path);
        goto done;
    }
    ok = 1;

done:
    if (!ok && build_shard_pack_temp_path(state, shard_id, tmp_path, sizeof(tmp_path))) std::remove(tmp_path);
    clear_blocked_ell_parts(parts, (unsigned long) partition_count);
    std::free(parts);
    return ok;
}

inline int materialize_quantized_blocked_ell_shard_pack(shard_storage *s, dataset_h5_state *state, unsigned long shard_id) {
    const std::uint64_t begin = state != 0 && state->shard_part_begin != 0 ? state->shard_part_begin[shard_id] : 0u;
    const std::uint64_t end = state != 0 && state->shard_part_end != 0 ? state->shard_part_end[shard_id] : 0u;
    const std::uint64_t partition_count = end >= begin ? (end - begin) : 0u;
    sparse::quantized_blocked_ell **parts = 0;
    char tmp_path[4096];
    char final_path[4096];
    std::uint64_t local = 0u;
    int ok = 0;

    if (s == 0 || state == 0 || shard_id >= state->num_shards) return 0;
    if (!open_dataset_h5_backend(s)) return 0;
    if (partition_count != 0u) {
        parts = (sparse::quantized_blocked_ell **) std::calloc((std::size_t) partition_count, sizeof(sparse::quantized_blocked_ell *));
        if (parts == 0) return 0;
    }
    for (local = 0u; local < partition_count; ++local) {
        const std::uint64_t partition_id = begin + local;
        sparse::quantized_blocked_ell *part = new sparse::quantized_blocked_ell;
        sparse::init(part);
        if (!load_quantized_blocked_ell_partition_payload(state, (unsigned long) partition_id, part)) {
            sparse::clear(part);
            delete part;
            goto done;
        }
        parts[local] = part;
    }
    if (!build_shard_pack_temp_path(state, shard_id, tmp_path, sizeof(tmp_path))) goto done;
    if (!build_shard_pack_path(state, shard_id, final_path, sizeof(final_path))) goto done;
    if (!write_shard_pack_file<sparse::quantized_blocked_ell>(tmp_path,
                                                              state->cols,
                                                              state->partition_rows + begin,
                                                              state->partition_nnz + begin,
                                                              state->partition_aux + begin,
                                                              partition_count,
                                                              parts)) {
        goto done;
    }
    if (::rename(tmp_path, final_path) != 0) {
        std::remove(tmp_path);
        goto done;
    }
    ok = 1;

done:
    if (!ok && build_shard_pack_temp_path(state, shard_id, tmp_path, sizeof(tmp_path))) std::remove(tmp_path);
    if (parts != 0) {
        for (local = 0u; local < partition_count; ++local) {
            if (parts[local] != 0) {
                sparse::clear(parts[local]);
                delete parts[local];
            }
        }
    }
    std::free(parts);
    return ok;
}

inline int materialize_sliced_ell_shard_pack(shard_storage *s, dataset_h5_state *state, unsigned long shard_id) {
    const std::uint64_t begin = state != 0 && state->shard_part_begin != 0 ? state->shard_part_begin[shard_id] : 0u;
    const std::uint64_t end = state != 0 && state->shard_part_end != 0 ? state->shard_part_end[shard_id] : 0u;
    const std::uint64_t partition_count = end >= begin ? (end - begin) : 0u;
    bucketed_sliced_ell_partition *parts = 0;
    std::uint64_t *partition_offsets = 0;
    char tmp_path[4096];
    char final_path[4096];
    std::uint64_t local = 0u;
    std::FILE *fp = 0;
    int ok = 0;

    if (s == 0 || state == 0 || shard_id >= state->num_shards) return 0;
    if (!open_dataset_h5_backend(s)) return 0;
    if (partition_count != 0u) {
        parts = (bucketed_sliced_ell_partition *) std::calloc((std::size_t) partition_count, sizeof(bucketed_sliced_ell_partition));
        partition_offsets = (std::uint64_t *) std::calloc((std::size_t) partition_count, sizeof(std::uint64_t));
        if (parts == 0 || partition_offsets == 0) return 0;
        for (local = 0u; local < partition_count; ++local) init(parts + local);
    }
    for (local = 0u; local < partition_count; ++local) {
        const std::uint64_t partition_id = begin + local;
        if (!load_bucketed_sliced_ell_partition_payload(state, (unsigned long) partition_id, parts + local)) {
            goto done;
        }
    }
    if (!build_shard_pack_temp_path(state, shard_id, tmp_path, sizeof(tmp_path))) goto done;
    if (!build_shard_pack_path(state, shard_id, final_path, sizeof(final_path))) goto done;
    fp = std::fopen(tmp_path, "wb");
    if (fp == 0) goto done;
    std::setvbuf(fp, 0, _IOFBF, (std::size_t) 8u << 20u);
    if (!write_sharded_block(fp, execution_pack_magic, sizeof(execution_pack_magic), 1u)) goto done;
    if (!write_sharded_block(fp, &shard_id, sizeof(shard_id), 1u)) goto done;
    if (!write_sharded_block(fp, &partition_count, sizeof(partition_count), 1u)) goto done;
    for (local = 0u; local < partition_count; ++local) {
        const std::uint64_t zero = 0u;
        if (!write_sharded_block(fp, &zero, sizeof(zero), 1u)) goto done;
    }
    for (local = 0u; local < partition_count; ++local) {
        partition_offsets[local] = (std::uint64_t) ftello(fp);
        if (!write_sliced_execution_partition_blob(fp, parts + local)) goto done;
    }
    if (fseeko(fp, (off_t) (sizeof(execution_pack_magic) + sizeof(std::uint64_t) * 2u), SEEK_SET) != 0) goto done;
    if (!write_sharded_block(fp, partition_offsets, sizeof(std::uint64_t), (std::size_t) partition_count)) goto done;
    if (std::fflush(fp) != 0) goto done;
    std::fclose(fp);
    fp = 0;
    if (::rename(tmp_path, final_path) != 0) {
        std::remove(tmp_path);
        goto done;
    }
    ok = 1;

done:
    if (fp != 0) std::fclose(fp);
    if (!ok && build_shard_pack_temp_path(state, shard_id, tmp_path, sizeof(tmp_path))) std::remove(tmp_path);
    if (parts != 0) {
        for (local = 0u; local < partition_count; ++local) {
            clear(parts + local);
        }
    }
    std::free(partition_offsets);
    std::free(parts);
    return ok;
}

inline int materialize_shard_pack(shard_storage *s, dataset_h5_state *state, unsigned long shard_id) {
    if (state == 0) return 0;
    if (state->matrix_family == dataset_matrix_family_blocked_ell) return materialize_blocked_ell_shard_pack(s, state, shard_id);
    if (state->matrix_family == dataset_matrix_family_quantized_blocked_ell) return materialize_quantized_blocked_ell_shard_pack(s, state, shard_id);
    if (state->matrix_family == dataset_matrix_family_optimized_blocked_ell) return materialize_blocked_ell_shard_pack(s, state, shard_id);
    if (state->matrix_family == dataset_matrix_family_sliced_ell) return materialize_sliced_ell_shard_pack(s, state, shard_id);
    return 0;
}

inline void touch_shard_locked(dataset_h5_state *state, unsigned long shard_id) {
    if (state == 0 || shard_id >= state->num_shards) return;
    state->shard_access_count[shard_id] += 1u;
    state->shard_last_access_tick[shard_id] = ++state->access_clock;
    state->last_requested_shard = shard_id;
}

inline std::uint64_t shard_eviction_score(const dataset_h5_state *state,
                                          unsigned long shard_id,
                                          std::uint64_t now_tick) {
    const std::uint64_t age = now_tick >= state->shard_last_access_tick[shard_id] ? (now_tick - state->shard_last_access_tick[shard_id]) : 0u;
    const std::uint64_t accesses = state->shard_access_count[shard_id];
    return (age + 1u) * (state->shard_cache_bytes[shard_id] + 1u) / (accesses + 1u);
}

inline void evict_cached_shard_locked(dataset_h5_state *state, unsigned long shard_id) {
    if (state == 0 || shard_id >= state->num_shards) return;
    close_cached_shard_file(state, shard_id);
    if (state->shard_cache_paths != 0 && state->shard_cache_paths[shard_id] != 0) ::unlink(state->shard_cache_paths[shard_id]);
    if (state->shard_cache_state[shard_id] == dataset_cache_shard_ready) {
        if (state->cache_resident_bytes >= state->shard_cache_bytes[shard_id]) state->cache_resident_bytes -= state->shard_cache_bytes[shard_id];
        else state->cache_resident_bytes = 0u;
    }
    state->shard_cache_state[shard_id] = dataset_cache_shard_missing;
    state->shard_cache_bytes[shard_id] = 0u;
    state->shard_access_count[shard_id] = 0u;
    state->shard_last_access_tick[shard_id] = 0u;
    state->shard_pin_count[shard_id] = 0u;
}

inline void maybe_evict_cached_shards_locked(dataset_h5_state *state, unsigned long keep_shard_id) {
    while (state != 0
           && state->predictor_enabled
           && state->cache_budget_bytes != 0u
           && state->cache_resident_bytes > state->cache_budget_bytes) {
        unsigned long victim = (unsigned long) state->num_shards;
        std::uint64_t best_score = 0u;
        unsigned long shard_id = 0ul;
        for (shard_id = 0ul; shard_id < (unsigned long) state->num_shards; ++shard_id) {
            const std::uint64_t score = shard_eviction_score(state, shard_id, state->access_clock + 1u);
            if (shard_id == keep_shard_id) continue;
            if (state->shard_cache_state[shard_id] != dataset_cache_shard_ready) continue;
            if (state->shard_pin_count[shard_id] != 0u) continue;
            if (victim >= state->num_shards || score > best_score) {
                victim = shard_id;
                best_score = score;
            }
        }
        if (victim >= state->num_shards) break;
        evict_cached_shard_locked(state, victim);
    }
}

inline void reader_materialize_loop(shard_storage *s) {
    dataset_h5_state *state = s != 0 ? (dataset_h5_state *) s->backend_state : 0;
    dataset_h5_cache_runtime *runtime = cache_runtime(state);
    if (state == 0 || runtime == 0) return;
    for (;;) {
        unsigned long shard_id = 0ul;
        {
            std::unique_lock<std::mutex> lock(runtime->state_mutex);
            runtime->state_cv.wait(lock, [&]() { return runtime->stop_requested || !runtime->shard_queue.empty(); });
            if (runtime->stop_requested) break;
            shard_id = runtime->shard_queue.front();
            runtime->shard_queue.pop_front();
            state->shard_cache_state[shard_id] = dataset_cache_shard_building;
        }

        const int ok = materialize_shard_pack(s, state, shard_id);

        {
            std::lock_guard<std::mutex> lock(runtime->state_mutex);
            if (ok) {
                if (state->shard_cache_state[shard_id] != dataset_cache_shard_ready) {
                    state->shard_cache_bytes[shard_id] = estimate_shard_pack_bytes(state, shard_id);
                    state->cache_resident_bytes += state->shard_cache_bytes[shard_id];
                }
                state->shard_cache_state[shard_id] = dataset_cache_shard_ready;
                touch_shard_locked(state, shard_id);
                maybe_evict_cached_shards_locked(state, shard_id);
            } else {
                std::fprintf(stderr, "cellshard: shard cache materialization failed for shard %lu\n", shard_id);
                state->shard_cache_state[shard_id] = dataset_cache_shard_failed;
            }
            runtime->state_cv.notify_all();
        }
    }
}

inline int ensure_cache_reader_started(shard_storage *s) {
    dataset_h5_state *state = 0;
    dataset_h5_cache_runtime *runtime = 0;
    if (s == 0 || s->backend_state == 0) return 0;
    state = (dataset_h5_state *) s->backend_state;
    runtime = cache_runtime(state);
    if (runtime == 0) return 0;
    {
        std::lock_guard<std::mutex> lock(runtime->state_mutex);
        if (runtime->reader_started) return 1;
        runtime->stop_requested = false;
        runtime->reader_thread = std::thread(reader_materialize_loop, s);
        runtime->reader_started = true;
    }
    return 1;
}

inline int ensure_cached_shard_ready(shard_storage *s, unsigned long shard_id) {
    dataset_h5_state *state = 0;
    dataset_h5_cache_runtime *runtime = 0;
    if (s == 0 || s->backend_state == 0) return 0;
    state = (dataset_h5_state *) s->backend_state;
    if (!ensure_dataset_cache_layout(s)) return 0;
    runtime = cache_runtime(state);
    if (runtime == 0 || shard_id >= state->num_shards) return 0;
    if (!ensure_cache_reader_started(s)) return 0;

    {
        std::unique_lock<std::mutex> lock(runtime->state_mutex);
        if (state->shard_cache_state[shard_id] == dataset_cache_shard_ready) {
            touch_shard_locked(state, shard_id);
            return 1;
        }
        if (!shard_storage_has_capability(s, shard_storage_cap_materialize_canonical_pack)) {
            return 0;
        }
        if (state->shard_cache_state[shard_id] == dataset_cache_shard_missing
            || state->shard_cache_state[shard_id] == dataset_cache_shard_failed) {
            state->shard_cache_state[shard_id] = dataset_cache_shard_queued;
            runtime->shard_queue.push_back(shard_id);
            runtime->state_cv.notify_all();
        }
        runtime->state_cv.wait(lock, [&]() {
            return state->shard_cache_state[shard_id] == dataset_cache_shard_ready
                || state->shard_cache_state[shard_id] == dataset_cache_shard_failed;
        });
        if (state->shard_cache_state[shard_id] != dataset_cache_shard_ready) return 0;
        touch_shard_locked(state, shard_id);
    }
    return 1;
}

inline void close_dataset_h5_open_handles(dataset_h5_state *state) {
    if (state == 0) return;
    if (state->d_blocked_ell_values >= 0) H5Dclose(state->d_blocked_ell_values);
    if (state->d_blocked_ell_block_idx >= 0) H5Dclose(state->d_blocked_ell_block_idx);
    if (state->payload_blocked_ell >= 0) H5Gclose(state->payload_blocked_ell);
    if (state->payload_quantized_blocked_ell >= 0) H5Gclose(state->payload_quantized_blocked_ell);
    if (state->payload_sliced_ell >= 0) H5Gclose(state->payload_sliced_ell);
    if (state->payload_optimized_blocked_ell >= 0) H5Gclose(state->payload_optimized_blocked_ell);
    if (state->payload_optimized_sliced_ell >= 0) H5Gclose(state->payload_optimized_sliced_ell);
    if (state->file >= 0) H5Fclose(state->file);
    state->d_blocked_ell_values = (hid_t) -1;
    state->d_blocked_ell_block_idx = (hid_t) -1;
    state->payload_blocked_ell = (hid_t) -1;
    state->payload_quantized_blocked_ell = (hid_t) -1;
    state->payload_sliced_ell = (hid_t) -1;
    state->payload_optimized_blocked_ell = (hid_t) -1;
    state->payload_optimized_sliced_ell = (hid_t) -1;
    state->file = (hid_t) -1;
    state->loaded_blocked_ell_shard_id = std::numeric_limits<std::uint64_t>::max();
    state->loaded_optimized_shard_id = std::numeric_limits<std::uint64_t>::max();
    clear(&state->loaded_optimized_shard);
    init(&state->loaded_optimized_shard);
    state->loaded_optimized_sliced_shard_id = std::numeric_limits<std::uint64_t>::max();
    clear(&state->loaded_optimized_sliced_shard);
    init(&state->loaded_optimized_sliced_shard);
}

int open_dataset_h5_backend(shard_storage *s) {
    dataset_h5_state *state = 0;
    if (s == 0 || s->source_path == 0 || s->backend_state == 0) return 0;
    if (!require_storage_capability(s, shard_storage_cap_canonical_read, "open dataset h5 backend")) return 0;
    state = (dataset_h5_state *) s->backend_state;
    if (state->file >= 0) return 1;
    state->file = H5Fopen(s->source_path, H5F_ACC_RDONLY, H5P_DEFAULT);
    return state->file >= 0;
}

void close_dataset_h5_backend(shard_storage *s) {
    dataset_h5_state *state = 0;
    if (s == 0 || s->backend_state == 0) return;
    state = (dataset_h5_state *) s->backend_state;
    dataset_h5_state_clear(state);
    std::free(state);
    s->backend_state = 0;
    s->open_backend = 0;
    s->close_backend = 0;
    s->backend = shard_storage_backend_none;
}
