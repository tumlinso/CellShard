#pragma once

inline int choose_bucket_count_for_blocked_part_host_exact(const sparse::blocked_ell *part,
                                                           std::uint32_t *bucket_count_out) {
    const std::uint32_t row_blocks = part != nullptr ? sparse::row_block_count(part) : 0u;
    const std::uint32_t max_buckets = std::min<std::uint32_t>(8u, std::max<std::uint32_t>(1u, row_blocks));
    bucketed_blocked_ell_partition trial;
    std::uint32_t best_buckets = 1u;
    std::uint64_t best_bytes = std::numeric_limits<std::uint64_t>::max();

    if (part == nullptr || bucket_count_out == nullptr) return 0;
    init(&trial);
    for (std::uint32_t buckets = 1u; buckets <= max_buckets; ++buckets) {
        std::uint64_t bytes = 0u;
        clear(&trial);
        init(&trial);
        if (!build_bucketed_blocked_ell_partition(&trial, part, buckets, &bytes)) {
            clear(&trial);
            return 0;
        }
        if (bytes < best_bytes || (bytes == best_bytes && buckets < best_buckets)) {
            best_bytes = bytes;
            best_buckets = buckets;
        }
    }
    clear(&trial);
    *bucket_count_out = best_buckets;
    return 1;
}

inline int build_filtered_blocked_ell_part_from_sliced(const sparse::sliced_ell *src,
                                                       const std::uint8_t *keep_rows,
                                                       std::uint64_t global_row_begin,
                                                       const std::uint32_t *col_remap,
                                                       std::uint32_t out_cols,
                                                       sparse::blocked_ell *dst,
                                                       std::uint32_t *rows_out,
                                                       std::uint32_t *nnz_out) {
    sparse::coo coo;
    std::uint32_t live_rows = 0u;
    std::uint32_t live_nnz = 0u;
    static constexpr unsigned int candidates[] = {4u, 8u, 16u, 32u};

    if (src == nullptr || dst == nullptr || rows_out == nullptr || nnz_out == nullptr) return 0;
    *rows_out = 0u;
    *nnz_out = 0u;
    sparse::clear(dst);
    sparse::init(dst);

    for (std::uint32_t row = 0u; row < src->rows; ++row) {
        const std::uint64_t global_row = global_row_begin + row;
        const std::uint32_t slice = sparse::find_slice(src, row);
        const std::uint32_t row_begin = slice < src->slice_count ? src->slice_row_offsets[slice] : 0u;
        const std::uint32_t width = slice < src->slice_count ? src->slice_widths[slice] : 0u;
        const std::size_t row_base = sparse::slice_slot_base(src, slice) + (std::size_t) (row - row_begin) * (std::size_t) width;
        if (keep_rows == nullptr || keep_rows[global_row] == 0u) continue;
        ++live_rows;
        for (std::uint32_t slot = 0u; slot < width; ++slot) {
            const std::uint32_t col = src->col_idx[row_base + slot];
            if (col == sparse::sliced_ell_invalid_col || col >= src->cols) continue;
            if (col_remap[col] == std::numeric_limits<std::uint32_t>::max()) continue;
            ++live_nnz;
        }
    }

    *rows_out = live_rows;
    *nnz_out = live_nnz;
    if (live_rows == 0u) return 1;
    if (live_nnz == 0u || out_cols == 0u) {
        sparse::init(dst, live_rows, out_cols, 0u, 8u, 0u);
        return sparse::allocate(dst);
    }

    sparse::init(&coo, live_rows, out_cols, live_nnz);
    if (!sparse::allocate(&coo)) return 0;

    {
        std::uint32_t out_row = 0u;
        std::uint32_t emitted = 0u;
        for (std::uint32_t row = 0u; row < src->rows; ++row) {
            const std::uint64_t global_row = global_row_begin + row;
            const std::uint32_t slice = sparse::find_slice(src, row);
            const std::uint32_t row_begin = slice < src->slice_count ? src->slice_row_offsets[slice] : 0u;
            const std::uint32_t width = slice < src->slice_count ? src->slice_widths[slice] : 0u;
            const std::size_t row_base = sparse::slice_slot_base(src, slice) + (std::size_t) (row - row_begin) * (std::size_t) width;
            if (keep_rows == nullptr || keep_rows[global_row] == 0u) continue;
            for (std::uint32_t slot = 0u; slot < width; ++slot) {
                const std::uint32_t col = src->col_idx[row_base + slot];
                const std::uint32_t mapped =
                    col < src->cols && col_remap != nullptr ? col_remap[col] : std::numeric_limits<std::uint32_t>::max();
                if (col == sparse::sliced_ell_invalid_col || col >= src->cols || mapped == std::numeric_limits<std::uint32_t>::max()) continue;
                coo.rowIdx[emitted] = out_row;
                coo.colIdx[emitted] = mapped;
                coo.val[emitted] = src->val[row_base + slot];
                ++emitted;
            }
            ++out_row;
        }
    }

    if (!convert::blocked_ell_from_coo_auto(&coo,
                                            out_cols,
                                            nullptr,
                                            candidates,
                                            (unsigned int) (sizeof(candidates) / sizeof(candidates[0])),
                                            dst,
                                            nullptr)) {
        sparse::clear(&coo);
        return 0;
    }
    sparse::clear(&coo);
    return 1;
}

inline int build_filtered_blocked_ell_part_from_blocked(const sparse::blocked_ell *src,
                                                        const std::uint8_t *keep_rows,
                                                        std::uint64_t global_row_begin,
                                                        const std::uint32_t *col_remap,
                                                        std::uint32_t out_cols,
                                                        sparse::blocked_ell *dst,
                                                        std::uint32_t *rows_out,
                                                        std::uint32_t *nnz_out) {
    sparse::coo canonical;
    sparse::coo filtered;
    std::uint32_t live_rows = 0u;
    std::uint32_t live_nnz = 0u;
    static constexpr unsigned int candidates[] = {4u, 8u, 16u, 32u};

    if (src == nullptr || dst == nullptr || rows_out == nullptr || nnz_out == nullptr) return 0;
    *rows_out = 0u;
    *nnz_out = 0u;
    sparse::clear(dst);
    sparse::init(dst);
    sparse::init(&canonical);
    sparse::init(&filtered);

    if (!blocked_ell_to_canonical_coo(src, &canonical)) goto done;
    for (std::uint32_t row = 0u; row < src->rows; ++row) {
        const std::uint64_t global_row = global_row_begin + row;
        if (keep_rows == nullptr || keep_rows[global_row] == 0u) continue;
        ++live_rows;
    }
    for (types::nnz_t idx = 0u; idx < canonical.nnz; ++idx) {
        const std::uint64_t global_row = global_row_begin + canonical.rowIdx[idx];
        const std::uint32_t col = canonical.colIdx[idx];
        if (keep_rows == nullptr || keep_rows[global_row] == 0u) continue;
        if (col_remap == nullptr || col >= src->cols || col_remap[col] == std::numeric_limits<std::uint32_t>::max()) continue;
        ++live_nnz;
    }

    *rows_out = live_rows;
    *nnz_out = live_nnz;
    if (live_rows == 0u) {
        sparse::clear(&canonical);
        return 1;
    }
    if (live_nnz == 0u || out_cols == 0u) {
        sparse::init(dst, live_rows, out_cols, 0u, src->block_size != 0u ? src->block_size : 8u, 0u);
        sparse::clear(&canonical);
        return sparse::allocate(dst);
    }

    sparse::init(&filtered, live_rows, out_cols, live_nnz);
    if (!sparse::allocate(&filtered)) goto done;

    {
        std::vector<std::uint32_t> row_remap((std::size_t) src->rows, std::numeric_limits<std::uint32_t>::max());
        std::uint32_t next_row = 0u;
        types::nnz_t emitted = 0u;
        for (std::uint32_t row = 0u; row < src->rows; ++row) {
            const std::uint64_t global_row = global_row_begin + row;
            if (keep_rows == nullptr || keep_rows[global_row] == 0u) continue;
            row_remap[(std::size_t) row] = next_row++;
        }
        for (types::nnz_t idx = 0u; idx < canonical.nnz; ++idx) {
            const std::uint32_t row = canonical.rowIdx[idx];
            const std::uint32_t col = canonical.colIdx[idx];
            const std::uint64_t global_row = global_row_begin + row;
            const std::uint32_t mapped =
                (col_remap != nullptr && col < src->cols) ? col_remap[col] : std::numeric_limits<std::uint32_t>::max();
            if (keep_rows == nullptr || keep_rows[global_row] == 0u || mapped == std::numeric_limits<std::uint32_t>::max()) continue;
            filtered.rowIdx[emitted] = row_remap[(std::size_t) row];
            filtered.colIdx[emitted] = mapped;
            filtered.val[emitted] = canonical.val[idx];
            ++emitted;
        }
        if (emitted != filtered.nnz) goto done;
    }

    if (!convert::blocked_ell_from_coo_auto(&filtered,
                                            out_cols,
                                            nullptr,
                                            candidates,
                                            (unsigned int) (sizeof(candidates) / sizeof(candidates[0])),
                                            dst,
                                            nullptr)) {
        goto done;
    }
    sparse::clear(&filtered);
    sparse::clear(&canonical);
    return 1;

done:
    sparse::clear(&filtered);
    sparse::clear(&canonical);
    sparse::clear(dst);
    sparse::init(dst);
    return 0;
}

inline int build_filtered_sliced_ell_part_from_sliced(const sparse::sliced_ell *src,
                                                      const std::uint8_t *keep_rows,
                                                      std::uint64_t global_row_begin,
                                                      const std::uint32_t *col_remap,
                                                      std::uint32_t out_cols,
                                                      sparse::sliced_ell *dst,
                                                      std::uint32_t *rows_out,
                                                      std::uint32_t *nnz_out) {
    std::vector<std::uint32_t> row_widths;
    std::vector<std::uint32_t> slice_row_offsets;
    std::vector<std::uint32_t> slice_widths;
    std::uint32_t live_rows = 0u;
    std::uint32_t live_nnz = 0u;
    std::uint32_t slice_rows = 0u;

    if (src == nullptr || dst == nullptr || rows_out == nullptr || nnz_out == nullptr) return 0;
    *rows_out = 0u;
    *nnz_out = 0u;
    sparse::clear(dst);
    sparse::init(dst);

    row_widths.reserve((std::size_t) src->rows);
    for (std::uint32_t row = 0u; row < src->rows; ++row) {
        const std::uint64_t global_row = global_row_begin + row;
        const std::uint32_t slice = sparse::find_slice(src, row);
        const std::uint32_t row_begin = slice < src->slice_count ? src->slice_row_offsets[slice] : 0u;
        const std::uint32_t width = slice < src->slice_count ? src->slice_widths[slice] : 0u;
        const std::size_t row_base = sparse::slice_slot_base(src, slice) + (std::size_t) (row - row_begin) * (std::size_t) width;
        std::uint32_t filtered_row_nnz = 0u;
        if (keep_rows == nullptr || keep_rows[global_row] == 0u) continue;
        for (std::uint32_t slot = 0u; slot < width; ++slot) {
            const std::uint32_t col = src->col_idx[row_base + slot];
            if (col == sparse::sliced_ell_invalid_col || col >= src->cols) continue;
            if (col_remap == nullptr || col_remap[col] == std::numeric_limits<std::uint32_t>::max()) continue;
            ++filtered_row_nnz;
        }
        row_widths.push_back(filtered_row_nnz);
        ++live_rows;
        live_nnz += filtered_row_nnz;
    }

    *rows_out = live_rows;
    *nnz_out = live_nnz;
    if (live_rows == 0u) return 1;

    slice_rows = sparse::uniform_slice_rows(src);
    if (slice_rows == 0u) {
        if (src->slice_count != 0u && src->slice_row_offsets != nullptr) {
            for (std::uint32_t slice = 0u; slice < src->slice_count; ++slice) {
                slice_rows = std::max<std::uint32_t>(slice_rows, src->slice_row_offsets[slice + 1u] - src->slice_row_offsets[slice]);
            }
        }
        if (slice_rows == 0u) slice_rows = live_rows;
    }

    slice_row_offsets.push_back(0u);
    for (std::uint32_t row_begin = 0u; row_begin < live_rows; row_begin += slice_rows) {
        const std::uint32_t row_end = std::min<std::uint32_t>(live_rows, row_begin + slice_rows);
        std::uint32_t width = 0u;
        for (std::uint32_t row = row_begin; row < row_end; ++row) width = std::max<std::uint32_t>(width, row_widths[(std::size_t) row]);
        slice_widths.push_back(width);
        slice_row_offsets.push_back(row_end);
    }

    sparse::init(dst, live_rows, out_cols, live_nnz);
    if (!sparse::allocate(dst, (std::uint32_t) slice_widths.size(), slice_row_offsets.data(), slice_widths.data())) {
        sparse::clear(dst);
        sparse::init(dst);
        return 0;
    }

    {
        std::uint32_t out_row = 0u;
        for (std::uint32_t row = 0u; row < src->rows; ++row) {
            const std::uint64_t global_row = global_row_begin + row;
            const std::uint32_t src_slice = sparse::find_slice(src, row);
            const std::uint32_t src_row_begin = src_slice < src->slice_count ? src->slice_row_offsets[src_slice] : 0u;
            const std::uint32_t src_width = src_slice < src->slice_count ? src->slice_widths[src_slice] : 0u;
            const std::size_t src_row_base =
                sparse::slice_slot_base(src, src_slice) + (std::size_t) (row - src_row_begin) * (std::size_t) src_width;
            std::uint32_t emitted = 0u;
            if (keep_rows == nullptr || keep_rows[global_row] == 0u) continue;
            if (out_row >= live_rows) break;
            {
                const std::uint32_t dst_slice = sparse::find_slice(dst, out_row);
                const std::uint32_t dst_row_begin =
                    dst_slice < dst->slice_count ? dst->slice_row_offsets[dst_slice] : 0u;
                const std::uint32_t dst_width =
                    dst_slice < dst->slice_count ? dst->slice_widths[dst_slice] : 0u;
                const std::size_t dst_row_base =
                    sparse::slice_slot_base(dst, dst_slice) + (std::size_t) (out_row - dst_row_begin) * (std::size_t) dst_width;
                for (std::uint32_t slot = 0u; slot < src_width; ++slot) {
                    const std::uint32_t col = src->col_idx[src_row_base + slot];
                    const std::uint32_t mapped =
                        (col_remap != nullptr && col < src->cols) ? col_remap[col] : std::numeric_limits<std::uint32_t>::max();
                    if (col == sparse::sliced_ell_invalid_col || col >= src->cols || mapped == std::numeric_limits<std::uint32_t>::max()) continue;
                    if (emitted >= dst_width) {
                        sparse::clear(dst);
                        sparse::init(dst);
                        return 0;
                    }
                    dst->col_idx[dst_row_base + emitted] = mapped;
                    dst->val[dst_row_base + emitted] = src->val[src_row_base + slot];
                    ++emitted;
                }
            }
            ++out_row;
        }
    }
    return 1;
}

inline std::size_t blocked_ell_part_block_index_count(std::uint64_t rows, std::uint64_t aux) {
    const types::u32 block_size = sparse::unpack_blocked_ell_block_size((unsigned long) aux);
    const std::uint64_t ell_width = sparse::unpack_blocked_ell_ell_width((unsigned long) aux);
    return block_size == 0u ? 0u : ((rows + block_size - 1u) / block_size) * ell_width;
}

inline std::size_t blocked_ell_part_value_count(std::uint64_t rows, std::uint64_t aux) {
    return (std::size_t) rows * (std::size_t) sparse::unpack_blocked_ell_cols((unsigned long) aux);
}

inline std::size_t sliced_ell_part_total_slots(std::uint64_t aux) {
    return (std::size_t) sparse::unpack_sliced_ell_total_slots((unsigned long) aux);
}

inline std::uint64_t local_dim_limit() {
    return (std::uint64_t) std::numeric_limits<types::dim_t>::max();
}

inline std::uint64_t local_nnz_limit() {
    return (std::uint64_t) std::numeric_limits<types::nnz_t>::max();
}

inline std::uint64_t local_index_limit() {
    return (std::uint64_t) std::numeric_limits<types::idx_t>::max();
}

inline int fail_dataset_u32_limit(const char *filename,
                                 const char *scope,
                                 std::uint64_t id,
                                 const char *field,
                                 std::uint64_t value,
                                 std::uint64_t limit) {
    std::fprintf(stderr,
                 "cellshard: %s exceeds the current u32 execution limit while writing %s (%s=%llu, %s=%llu, limit=%llu)\n",
                 scope != 0 ? scope : "dataset payload",
                 filename != 0 ? filename : "<memory>",
                 scope != 0 && std::strcmp(scope, "part") == 0 ? "partition_id" : "id",
                 (unsigned long long) id,
                 field != 0 ? field : "value",
                 (unsigned long long) value,
                 (unsigned long long) limit);
    return 0;
}

inline void warn_dataset_u32_limit(const char *filename,
                                  const char *scope,
                                  std::uint64_t id,
                                  const char *field,
                                  std::uint64_t value,
                                  std::uint64_t limit) {
    std::fprintf(stderr,
                 "cellshard: warning: %s exceeds the current u32 execution limit for %s (%s=%llu, %s=%llu, limit=%llu)\n",
                 scope != 0 ? scope : "dataset payload",
                 filename != 0 ? filename : "<memory>",
                 scope != 0 && std::strcmp(scope, "shard") == 0 ? "shard_id" : "id",
                 (unsigned long long) id,
                 field != 0 ? field : "value",
                 (unsigned long long) value,
                 (unsigned long long) limit);
}

inline int reserve_blocked_ell_shard_scratch(dataset_h5_state *state,
                                             std::size_t block_idx_count,
                                             std::size_t value_count) {
    types::idx_t *block_idx = state != 0 ? state->blocked_ell_block_idx_scratch : 0;
    real::storage_t *values = state != 0 ? state->blocked_ell_value_scratch : 0;

    if (state == 0) return 0;
    if (block_idx_count > state->blocked_ell_block_idx_capacity) {
        block_idx = (types::idx_t *) std::realloc(state->blocked_ell_block_idx_scratch,
                                                  block_idx_count * sizeof(types::idx_t));
        if (block_idx == 0) return 0;
        state->blocked_ell_block_idx_scratch = block_idx;
        state->blocked_ell_block_idx_capacity = block_idx_count;
    }
    if (value_count > state->blocked_ell_value_capacity) {
        values = (real::storage_t *) std::realloc(state->blocked_ell_value_scratch,
                                                  value_count * sizeof(real::storage_t));
        if (values == 0) return 0;
        state->blocked_ell_value_scratch = values;
        state->blocked_ell_value_capacity = value_count;
    }
    return 1;
}

int open_dataset_h5_backend(shard_storage *s);
void close_dataset_h5_backend(shard_storage *s);

inline int ensure_directory_exists(const char *path) {
    struct stat st;
    if (path == 0 || *path == 0) return 0;
    if (::stat(path, &st) == 0) return S_ISDIR(st.st_mode) ? 1 : 0;
    if (::mkdir(path, 0775) == 0) return 1;
    if (errno == EEXIST) return 1;
    return 0;
}

inline int directory_exists(const char *path) {
    struct stat st;
    if (path == 0 || *path == 0) return 0;
    if (::stat(path, &st) != 0) return 0;
    return S_ISDIR(st.st_mode) ? 1 : 0;
}

inline int require_storage_capability(const shard_storage *s,
                                      std::uint32_t capability,
                                      const char *operation) {
    if (shard_storage_has_capability(s, capability)) return 1;
    std::fprintf(stderr,
                 "cellshard: %s requires capability 0x%x for storage role %s\n",
                 operation != 0 ? operation : "operation",
                 (unsigned int) capability,
                 shard_storage_role_name(s != 0 ? s->role : shard_storage_role_unknown));
    return 0;
}

inline int assign_owned_string(char **dst, const char *src) {
    char *copy = 0;
    std::size_t len = 0u;

    if (dst == 0) return 0;
    std::free(*dst);
    *dst = 0;
    if (src == 0) return 1;
    len = std::strlen(src);
    copy = (char *) std::malloc(len + 1u);
    if (copy == 0) return 0;
    std::memcpy(copy, src, len + 1u);
    *dst = copy;
    return 1;
}

inline int build_default_cache_root(const char *source_path, char *path, std::size_t cap) {
    const char *slash = 0;
    int written = 0;
    if (source_path == 0 || path == 0 || cap == 0u) return 0;
    slash = std::strrchr(source_path, '/');
    if (slash == 0) return std::snprintf(path, cap, ".cellshard_cache") > 0;
    written = std::snprintf(path, cap, "%.*s/.cellshard_cache", (int) (slash - source_path), source_path);
    return written > 0 && (std::size_t) written < cap;
}

inline std::uint64_t fnv1a_mix(std::uint64_t h, const void *data, std::size_t bytes) {
    const unsigned char *ptr = (const unsigned char *) data;
    std::size_t i = 0u;
    for (i = 0u; i < bytes; ++i) {
        h ^= (std::uint64_t) ptr[i];
        h *= 1099511628211ull;
    }
    return h;
}

inline std::uint64_t build_source_fingerprint_u64(const char *source_path,
                                                  std::uint64_t size_bytes,
                                                  std::uint64_t mtime_ns,
                                                  std::uint32_t matrix_family,
                                                  std::uint64_t num_partitions,
                                                  std::uint64_t num_shards) {
    const std::uint32_t schema_version = dataset_h5_schema_version;
    std::uint64_t h = 1469598103934665603ull;
    if (source_path != 0) h = fnv1a_mix(h, source_path, std::strlen(source_path));
    h = fnv1a_mix(h, &size_bytes, sizeof(size_bytes));
    h = fnv1a_mix(h, &mtime_ns, sizeof(mtime_ns));
    h = fnv1a_mix(h, &matrix_family, sizeof(matrix_family));
    h = fnv1a_mix(h, &num_partitions, sizeof(num_partitions));
    h = fnv1a_mix(h, &num_shards, sizeof(num_shards));
    h = fnv1a_mix(h, &schema_version, sizeof(schema_version));
    h = fnv1a_mix(h, dataset_magic, std::strlen(dataset_magic));
    return h;
}

inline int build_cache_instance_path(const char *cache_root,
                                     std::uint64_t fingerprint,
                                     char *path,
                                     std::size_t cap) {
    if (cache_root == 0 || path == 0 || cap == 0u) return 0;
    return std::snprintf(path, cap, "%s/instances/%016llx", cache_root, (unsigned long long) fingerprint) > 0;
}

inline int build_cache_instances_root_path(const char *cache_root,
                                           char *path,
                                           std::size_t cap) {
    if (cache_root == 0 || path == 0 || cap == 0u) return 0;
    return std::snprintf(path, cap, "%s/instances", cache_root) > 0;
}

inline int build_cache_metadata_dir_path(const char *cache_instance_dir,
                                         char *path,
                                         std::size_t cap) {
    if (cache_instance_dir == 0 || path == 0 || cap == 0u) return 0;
    return std::snprintf(path, cap, "%s/metadata", cache_instance_dir) > 0;
}

inline int build_cache_pack_root_path(const char *cache_instance_dir,
                                      char *path,
                                      std::size_t cap) {
    if (cache_instance_dir == 0 || path == 0 || cap == 0u) return 0;
    return std::snprintf(path, cap, "%s/packs", cache_instance_dir) > 0;
}

inline int build_cache_canonical_pack_dir_path(const char *cache_instance_dir,
                                               char *path,
                                               std::size_t cap) {
    if (cache_instance_dir == 0 || path == 0 || cap == 0u) return 0;
    return std::snprintf(path, cap, "%s/packs/canonical", cache_instance_dir) > 0;
}

inline int build_cache_execution_pack_dir_path(const char *cache_instance_dir,
                                               char *path,
                                               std::size_t cap) {
    if (cache_instance_dir == 0 || path == 0 || cap == 0u) return 0;
    return std::snprintf(path, cap, "%s/packs/execution", cache_instance_dir) > 0;
}

inline int build_active_execution_pack_dir_path(const dataset_h5_state *state,
                                                char *path,
                                                std::size_t cap) {
    if (state == 0 || state->cache_instance_dir == 0 || path == 0 || cap == 0u) return 0;
    return std::snprintf(path,
                         cap,
                         "%s/packs/execution/plan.%llu-pack.%llu-epoch.%llu",
                         state->cache_instance_dir,
                         (unsigned long long) state->runtime_service.execution_plan_generation,
                         (unsigned long long) state->runtime_service.pack_generation,
                         (unsigned long long) state->runtime_service.service_epoch) > 0;
}

inline int build_cache_manifest_path(const char *cache_instance_dir,
                                     char *path,
                                     std::size_t cap) {
    if (cache_instance_dir == 0 || path == 0 || cap == 0u) return 0;
    return std::snprintf(path, cap, "%s/metadata/manifest.txt", cache_instance_dir) > 0;
}

inline int build_shard_pack_path(const dataset_h5_state *state,
                                 unsigned long shard_id,
                                 char *path,
                                 std::size_t cap) {
    if (state == 0 || state->cache_instance_dir == 0 || path == 0 || cap == 0u) return 0;
    return std::snprintf(path, cap, "%s/packs/canonical/shard.%lu.pack", state->cache_instance_dir, shard_id) > 0;
}

inline int build_shard_pack_temp_path(const dataset_h5_state *state,
                                      unsigned long shard_id,
                                      char *path,
                                      std::size_t cap) {
    if (state == 0 || state->cache_instance_dir == 0 || path == 0 || cap == 0u) return 0;
    return std::snprintf(path, cap, "%s/packs/canonical/shard.%lu.pack.tmp", state->cache_instance_dir, shard_id) > 0;
}

inline int build_execution_pack_path(const dataset_h5_state *state,
                                     unsigned long shard_id,
                                     char *path,
                                     std::size_t cap) {
    char dir[4096];
    if (!build_active_execution_pack_dir_path(state, dir, sizeof(dir)) || path == 0 || cap == 0u) return 0;
    return std::snprintf(path, cap, "%s/shard.%lu.exec.pack", dir, shard_id) > 0;
}

inline int build_execution_pack_temp_path(const dataset_h5_state *state,
                                          unsigned long shard_id,
                                          char *path,
                                          std::size_t cap) {
    char dir[4096];
    if (!build_active_execution_pack_dir_path(state, dir, sizeof(dir)) || path == 0 || cap == 0u) return 0;
    return std::snprintf(path, cap, "%s/shard.%lu.exec.pack.tmp", dir, shard_id) > 0;
}

inline std::uint64_t sharded_pack_payload_offset(std::uint64_t partition_count,
                                                 std::uint64_t shard_count,
                                                 std::uint64_t payload_alignment) {
    std::uint64_t offset = 8u
        + sizeof(unsigned char)
        + 7u
        + sizeof(std::uint64_t) * 7u
        + sizeof(std::uint64_t) * partition_count * 3u
        + sizeof(std::uint64_t) * (shard_count + 1u)
        + sizeof(std::uint64_t) * partition_count * 2u;
    offset = (offset + payload_alignment - 1u) & ~(payload_alignment - 1u);
    return offset;
}

template<typename MatrixT>
inline int compute_shard_pack_locators(const std::uint64_t *partition_rows,
                                       const std::uint64_t *partition_nnz,
                                       const std::uint64_t *partition_aux,
                                       std::uint64_t cols,
                                       std::uint64_t partition_count,
                                       std::uint64_t *partition_offsets,
                                       std::uint64_t *part_sizes) {
    std::uint64_t cursor = sharded_pack_payload_offset(partition_count, 1u, shard_pack_payload_alignment);
    std::uint64_t i = 0u;
    if ((partition_count != 0u) && (partition_rows == 0 || partition_nnz == 0 || partition_offsets == 0 || part_sizes == 0)) return 0;
    for (i = 0u; i < partition_count; ++i) {
        const std::size_t bytes = packed_bytes((const MatrixT *) 0,
                                               (types::dim_t) partition_rows[i],
                                               (types::dim_t) cols,
                                               (types::nnz_t) partition_nnz[i],
                                               partition_aux != 0 ? (unsigned long) partition_aux[i] : 0ul,
                                               sizeof(real::storage_t));
        partition_offsets[i] = cursor;
        part_sizes[i] = (std::uint64_t) bytes;
        cursor += (std::uint64_t) bytes;
        cursor = (cursor + shard_pack_payload_alignment - 1u) & ~(shard_pack_payload_alignment - 1u);
    }
    return 1;
}

inline dataset_h5_cache_runtime *cache_runtime(dataset_h5_state *state) {
    return state != 0 ? (dataset_h5_cache_runtime *) state->cache_runtime : 0;
}

inline int refresh_dataset_source_stat(const char *source_path,
                                      dataset_h5_state *state) {
    struct stat st;
    if (source_path == 0 || state == 0) return 0;
    if (::stat(source_path, &st) != 0) return 0;
    state->source_size_bytes = (std::uint64_t) st.st_size;
#if defined(__linux__)
    state->source_mtime_ns = (std::uint64_t) st.st_mtim.tv_sec * 1000000000ull + (std::uint64_t) st.st_mtim.tv_nsec;
#else
    state->source_mtime_ns = (std::uint64_t) st.st_mtime * 1000000000ull;
#endif
    return 1;
}

inline int ensure_cache_tracking_allocated(dataset_h5_state *state) {
    std::size_t count = 0u;
    if (state == 0) return 0;
    if (state->cache_runtime == 0) {
        state->cache_runtime = new dataset_h5_cache_runtime((std::size_t) state->num_shards);
        if (state->cache_runtime == 0) return 0;
    }
    count = (std::size_t) state->num_shards;
    if (count == 0u) return 1;
    if (state->shard_cache_paths == 0) state->shard_cache_paths = (char **) std::calloc(count, sizeof(char *));
    if (state->shard_cache_files == 0) state->shard_cache_files = (std::FILE **) std::calloc(count, sizeof(std::FILE *));
    if (state->shard_cache_state == 0) state->shard_cache_state = (std::uint8_t *) std::calloc(count, sizeof(std::uint8_t));
    if (state->shard_pin_count == 0) state->shard_pin_count = (std::uint32_t *) std::calloc(count, sizeof(std::uint32_t));
    if (state->shard_cache_bytes == 0) state->shard_cache_bytes = (std::uint64_t *) std::calloc(count, sizeof(std::uint64_t));
    if (state->shard_access_count == 0) state->shard_access_count = (std::uint64_t *) std::calloc(count, sizeof(std::uint64_t));
    if (state->shard_last_access_tick == 0) state->shard_last_access_tick = (std::uint64_t *) std::calloc(count, sizeof(std::uint64_t));
    return state->shard_cache_paths != 0
        && state->shard_cache_files != 0
        && state->shard_cache_state != 0
        && state->shard_pin_count != 0
        && state->shard_cache_bytes != 0
        && state->shard_access_count != 0
        && state->shard_last_access_tick != 0;
}

inline int ensure_execution_metadata_allocated(dataset_h5_state *state) {
    if (state == 0) return 0;
    if (state->num_partitions != 0u) {
        if (state->partition_execution_formats == 0) {
            state->partition_execution_formats = (std::uint32_t *) std::calloc((std::size_t) state->num_partitions, sizeof(std::uint32_t));
        }
        if (state->partition_blocked_ell_block_sizes == 0) {
            state->partition_blocked_ell_block_sizes = (std::uint32_t *) std::calloc((std::size_t) state->num_partitions, sizeof(std::uint32_t));
        }
        if (state->partition_blocked_ell_bucket_counts == 0) {
            state->partition_blocked_ell_bucket_counts = (std::uint32_t *) std::calloc((std::size_t) state->num_partitions, sizeof(std::uint32_t));
        }
        if (state->partition_blocked_ell_fill_ratios == 0) {
            state->partition_blocked_ell_fill_ratios = (float *) std::calloc((std::size_t) state->num_partitions, sizeof(float));
        }
        if (state->partition_execution_bytes == 0) {
            state->partition_execution_bytes = (std::uint64_t *) std::calloc((std::size_t) state->num_partitions, sizeof(std::uint64_t));
        }
        if (state->partition_blocked_ell_bytes == 0) {
            state->partition_blocked_ell_bytes = (std::uint64_t *) std::calloc((std::size_t) state->num_partitions, sizeof(std::uint64_t));
        }
        if (state->partition_bucketed_blocked_ell_bytes == 0) {
            state->partition_bucketed_blocked_ell_bytes = (std::uint64_t *) std::calloc((std::size_t) state->num_partitions, sizeof(std::uint64_t));
        }
        if (state->partition_sliced_ell_slice_counts == 0) {
            state->partition_sliced_ell_slice_counts = (std::uint32_t *) std::calloc((std::size_t) state->num_partitions, sizeof(std::uint32_t));
        }
        if (state->partition_sliced_ell_slice_rows == 0) {
            state->partition_sliced_ell_slice_rows = (std::uint32_t *) std::calloc((std::size_t) state->num_partitions, sizeof(std::uint32_t));
        }
        if (state->partition_sliced_ell_bytes == 0) {
            state->partition_sliced_ell_bytes = (std::uint64_t *) std::calloc((std::size_t) state->num_partitions, sizeof(std::uint64_t));
        }
        if (state->partition_bucketed_sliced_ell_bytes == 0) {
            state->partition_bucketed_sliced_ell_bytes = (std::uint64_t *) std::calloc((std::size_t) state->num_partitions, sizeof(std::uint64_t));
        }
    }
    if (state->num_shards != 0u) {
        if (state->shard_execution_formats == 0) {
            state->shard_execution_formats = (std::uint32_t *) std::calloc((std::size_t) state->num_shards, sizeof(std::uint32_t));
        }
        if (state->shard_blocked_ell_block_sizes == 0) {
            state->shard_blocked_ell_block_sizes = (std::uint32_t *) std::calloc((std::size_t) state->num_shards, sizeof(std::uint32_t));
        }
        if (state->shard_bucketed_partition_counts == 0) {
            state->shard_bucketed_partition_counts = (std::uint32_t *) std::calloc((std::size_t) state->num_shards, sizeof(std::uint32_t));
        }
        if (state->shard_bucketed_segment_counts == 0) {
            state->shard_bucketed_segment_counts = (std::uint32_t *) std::calloc((std::size_t) state->num_shards, sizeof(std::uint32_t));
        }
        if (state->shard_blocked_ell_fill_ratios == 0) {
            state->shard_blocked_ell_fill_ratios = (float *) std::calloc((std::size_t) state->num_shards, sizeof(float));
        }
        if (state->shard_execution_bytes == 0) {
            state->shard_execution_bytes = (std::uint64_t *) std::calloc((std::size_t) state->num_shards, sizeof(std::uint64_t));
        }
        if (state->shard_bucketed_blocked_ell_bytes == 0) {
            state->shard_bucketed_blocked_ell_bytes = (std::uint64_t *) std::calloc((std::size_t) state->num_shards, sizeof(std::uint64_t));
        }
        if (state->shard_sliced_ell_slice_counts == 0) {
            state->shard_sliced_ell_slice_counts = (std::uint32_t *) std::calloc((std::size_t) state->num_shards, sizeof(std::uint32_t));
        }
        if (state->shard_sliced_ell_slice_rows == 0) {
            state->shard_sliced_ell_slice_rows = (std::uint32_t *) std::calloc((std::size_t) state->num_shards, sizeof(std::uint32_t));
        }
        if (state->shard_bucketed_sliced_ell_bytes == 0) {
            state->shard_bucketed_sliced_ell_bytes = (std::uint64_t *) std::calloc((std::size_t) state->num_shards, sizeof(std::uint64_t));
        }
        if (state->shard_preferred_pair_ids == 0) {
            state->shard_preferred_pair_ids = (std::uint32_t *) std::calloc((std::size_t) state->num_shards, sizeof(std::uint32_t));
        }
        if (state->shard_owner_node_ids == 0) {
            state->shard_owner_node_ids = (std::uint32_t *) std::calloc((std::size_t) state->num_shards, sizeof(std::uint32_t));
        }
        if (state->shard_owner_rank_ids == 0) {
            state->shard_owner_rank_ids = (std::uint32_t *) std::calloc((std::size_t) state->num_shards, sizeof(std::uint32_t));
        }
    }
    return (state->num_partitions == 0u
            || (state->partition_execution_formats != 0
                && state->partition_blocked_ell_block_sizes != 0
                && state->partition_blocked_ell_bucket_counts != 0
                && state->partition_blocked_ell_fill_ratios != 0
                && state->partition_execution_bytes != 0
                && state->partition_blocked_ell_bytes != 0
                && state->partition_bucketed_blocked_ell_bytes != 0
                && state->partition_sliced_ell_slice_counts != 0
                && state->partition_sliced_ell_slice_rows != 0
                && state->partition_sliced_ell_bytes != 0
                && state->partition_bucketed_sliced_ell_bytes != 0))
        && (state->num_shards == 0u
            || (state->shard_execution_formats != 0
                && state->shard_blocked_ell_block_sizes != 0
                && state->shard_bucketed_partition_counts != 0
                && state->shard_bucketed_segment_counts != 0
                && state->shard_blocked_ell_fill_ratios != 0
                && state->shard_execution_bytes != 0
                && state->shard_bucketed_blocked_ell_bytes != 0
                && state->shard_sliced_ell_slice_counts != 0
                && state->shard_sliced_ell_slice_rows != 0
                && state->shard_bucketed_sliced_ell_bytes != 0
                && state->shard_preferred_pair_ids != 0
                && state->shard_owner_node_ids != 0
                && state->shard_owner_rank_ids != 0));
}

inline void default_execution_metadata(dataset_h5_state *state) {
    std::uint64_t shard_id = 0u;
    std::uint64_t partition_id = 0u;
    const std::uint32_t default_format =
        blocked_ell_uses_execution_payload(state)
            ? dataset_execution_format_bucketed_blocked_ell
            : (state != 0 && state->matrix_family == dataset_matrix_family_quantized_blocked_ell
                   ? dataset_execution_format_quantized_blocked_ell
                   : (state != 0 && state->matrix_family == dataset_matrix_family_sliced_ell
                   ? dataset_execution_format_bucketed_sliced_ell
                   : dataset_execution_format_blocked_ell));
    if (state == 0) return;
    state->preferred_base_format = default_format;
    for (partition_id = 0u; partition_id < state->num_partitions; ++partition_id) {
        state->partition_execution_formats[partition_id] = default_format;
        state->partition_blocked_ell_block_sizes[partition_id] = 0u;
        state->partition_blocked_ell_bucket_counts[partition_id] = 1u;
        state->partition_blocked_ell_fill_ratios[partition_id] = 0.0f;
        state->partition_execution_bytes[partition_id] = 0u;
        state->partition_blocked_ell_bytes[partition_id] = 0u;
        state->partition_bucketed_blocked_ell_bytes[partition_id] = 0u;
        state->partition_sliced_ell_slice_counts[partition_id] = 0u;
        state->partition_sliced_ell_slice_rows[partition_id] = 0u;
        state->partition_sliced_ell_bytes[partition_id] = 0u;
        state->partition_bucketed_sliced_ell_bytes[partition_id] = 0u;
    }
    for (shard_id = 0u; shard_id < state->num_shards; ++shard_id) {
        state->shard_execution_formats[shard_id] = default_format;
        state->shard_blocked_ell_block_sizes[shard_id] = 0u;
        state->shard_bucketed_partition_counts[shard_id] =
            state->shard_part_end != 0 && state->shard_part_begin != 0
                ? (std::uint32_t) (state->shard_part_end[shard_id] - state->shard_part_begin[shard_id])
                : 0u;
        state->shard_bucketed_segment_counts[shard_id] = state->shard_bucketed_partition_counts[shard_id];
        state->shard_blocked_ell_fill_ratios[shard_id] = 0.0f;
        state->shard_execution_bytes[shard_id] = 0u;
        state->shard_bucketed_blocked_ell_bytes[shard_id] = 0u;
        state->shard_sliced_ell_slice_counts[shard_id] = 0u;
        state->shard_sliced_ell_slice_rows[shard_id] = 0u;
        state->shard_bucketed_sliced_ell_bytes[shard_id] = 0u;
        state->shard_preferred_pair_ids[shard_id] = 0u;
        state->shard_owner_node_ids[shard_id] = 0u;
        state->shard_owner_rank_ids[shard_id] = 0u;
    }
}

inline int load_dataset_execution_metadata(hid_t file, dataset_h5_state *state) {
    hid_t execution = (hid_t) -1;
    if (state == 0) return 0;
    if (!ensure_execution_metadata_allocated(state)) return 0;
    default_execution_metadata(state);
    execution = open_optional_group(file, execution_group);
    if (execution < 0) return 1;
    if (!read_optional_attr_u32(execution, "preferred_base_format", &state->preferred_base_format)) goto done;
    if (state->num_partitions != 0u) {
        if (dataset_exists(execution, "partition_execution_formats")
            && !read_dataset_1d(execution, "partition_execution_formats", H5T_NATIVE_UINT32, state->num_partitions, state->partition_execution_formats)) goto done;
        if (dataset_exists(execution, "partition_blocked_ell_block_sizes")
            && !read_dataset_1d(execution, "partition_blocked_ell_block_sizes", H5T_NATIVE_UINT32, state->num_partitions, state->partition_blocked_ell_block_sizes)) goto done;
        if (dataset_exists(execution, "partition_blocked_ell_bucket_counts")
            && !read_dataset_1d(execution, "partition_blocked_ell_bucket_counts", H5T_NATIVE_UINT32, state->num_partitions, state->partition_blocked_ell_bucket_counts)) goto done;
        if (dataset_exists(execution, "partition_blocked_ell_fill_ratios")
            && !read_dataset_1d(execution, "partition_blocked_ell_fill_ratios", H5T_NATIVE_FLOAT, state->num_partitions, state->partition_blocked_ell_fill_ratios)) goto done;
        if (dataset_exists(execution, "partition_execution_bytes")
            && !read_dataset_1d(execution, "partition_execution_bytes", H5T_NATIVE_UINT64, state->num_partitions, state->partition_execution_bytes)) goto done;
        if (dataset_exists(execution, "partition_blocked_ell_bytes")
            && !read_dataset_1d(execution, "partition_blocked_ell_bytes", H5T_NATIVE_UINT64, state->num_partitions, state->partition_blocked_ell_bytes)) goto done;
        if (dataset_exists(execution, "partition_bucketed_blocked_ell_bytes")
            && !read_dataset_1d(execution, "partition_bucketed_blocked_ell_bytes", H5T_NATIVE_UINT64, state->num_partitions, state->partition_bucketed_blocked_ell_bytes)) goto done;
        if (dataset_exists(execution, "partition_sliced_ell_slice_counts")
            && !read_dataset_1d(execution, "partition_sliced_ell_slice_counts", H5T_NATIVE_UINT32, state->num_partitions, state->partition_sliced_ell_slice_counts)) goto done;
        if (dataset_exists(execution, "partition_sliced_ell_slice_rows")
            && !read_dataset_1d(execution, "partition_sliced_ell_slice_rows", H5T_NATIVE_UINT32, state->num_partitions, state->partition_sliced_ell_slice_rows)) goto done;
        if (dataset_exists(execution, "partition_sliced_ell_bytes")
            && !read_dataset_1d(execution, "partition_sliced_ell_bytes", H5T_NATIVE_UINT64, state->num_partitions, state->partition_sliced_ell_bytes)) goto done;
        if (dataset_exists(execution, "partition_bucketed_sliced_ell_bytes")
            && !read_dataset_1d(execution, "partition_bucketed_sliced_ell_bytes", H5T_NATIVE_UINT64, state->num_partitions, state->partition_bucketed_sliced_ell_bytes)) goto done;
    }
    if (state->num_shards != 0u) {
        if (dataset_exists(execution, "shard_execution_formats")
            && !read_dataset_1d(execution, "shard_execution_formats", H5T_NATIVE_UINT32, state->num_shards, state->shard_execution_formats)) goto done;
        if (dataset_exists(execution, "shard_blocked_ell_block_sizes")
            && !read_dataset_1d(execution, "shard_blocked_ell_block_sizes", H5T_NATIVE_UINT32, state->num_shards, state->shard_blocked_ell_block_sizes)) goto done;
        if (dataset_exists(execution, "shard_bucketed_partition_counts")
            && !read_dataset_1d(execution, "shard_bucketed_partition_counts", H5T_NATIVE_UINT32, state->num_shards, state->shard_bucketed_partition_counts)) goto done;
        if (dataset_exists(execution, "shard_bucketed_segment_counts")
            && !read_dataset_1d(execution, "shard_bucketed_segment_counts", H5T_NATIVE_UINT32, state->num_shards, state->shard_bucketed_segment_counts)) goto done;
        if (dataset_exists(execution, "shard_blocked_ell_fill_ratios")
            && !read_dataset_1d(execution, "shard_blocked_ell_fill_ratios", H5T_NATIVE_FLOAT, state->num_shards, state->shard_blocked_ell_fill_ratios)) goto done;
        if (dataset_exists(execution, "shard_execution_bytes")
            && !read_dataset_1d(execution, "shard_execution_bytes", H5T_NATIVE_UINT64, state->num_shards, state->shard_execution_bytes)) goto done;
        if (dataset_exists(execution, "shard_bucketed_blocked_ell_bytes")
            && !read_dataset_1d(execution, "shard_bucketed_blocked_ell_bytes", H5T_NATIVE_UINT64, state->num_shards, state->shard_bucketed_blocked_ell_bytes)) goto done;
        if (dataset_exists(execution, "shard_sliced_ell_slice_counts")
            && !read_dataset_1d(execution, "shard_sliced_ell_slice_counts", H5T_NATIVE_UINT32, state->num_shards, state->shard_sliced_ell_slice_counts)) goto done;
        if (dataset_exists(execution, "shard_sliced_ell_slice_rows")
            && !read_dataset_1d(execution, "shard_sliced_ell_slice_rows", H5T_NATIVE_UINT32, state->num_shards, state->shard_sliced_ell_slice_rows)) goto done;
        if (dataset_exists(execution, "shard_bucketed_sliced_ell_bytes")
            && !read_dataset_1d(execution, "shard_bucketed_sliced_ell_bytes", H5T_NATIVE_UINT64, state->num_shards, state->shard_bucketed_sliced_ell_bytes)) goto done;
        if (dataset_exists(execution, "shard_preferred_pair_ids")
            && !read_dataset_1d(execution, "shard_preferred_pair_ids", H5T_NATIVE_UINT32, state->num_shards, state->shard_preferred_pair_ids)) goto done;
        if (dataset_exists(execution, "shard_owner_node_ids")
            && !read_dataset_1d(execution, "shard_owner_node_ids", H5T_NATIVE_UINT32, state->num_shards, state->shard_owner_node_ids)) goto done;
        if (dataset_exists(execution, "shard_owner_rank_ids")
            && !read_dataset_1d(execution, "shard_owner_rank_ids", H5T_NATIVE_UINT32, state->num_shards, state->shard_owner_rank_ids)) goto done;
    }
    H5Gclose(execution);
    return 1;

done:
    if (execution >= 0) H5Gclose(execution);
    return 0;
}

inline void default_runtime_service_metadata(dataset_h5_state *state) {
    if (state == 0) return;
    init(&state->runtime_service);
    state->runtime_service.service_mode = dataset_runtime_service_mode_local_cache;
    state->runtime_service.live_write_mode = dataset_live_write_mode_read_only;
    state->runtime_service.prefer_pack_delivery = 1u;
    state->runtime_service.remote_pack_delivery = 0u;
    state->runtime_service.single_reader_coordinator = 0u;
    state->runtime_service.maintenance_lock_blocks_overwrite = 1u;
    state->runtime_service.canonical_generation = 1u;
    state->runtime_service.execution_plan_generation = 1u;
    state->runtime_service.pack_generation = 1u;
    state->runtime_service.service_epoch = 1u;
    state->runtime_service.active_read_generation = 1u;
    state->runtime_service.staged_write_generation = 1u;
}

inline int load_dataset_runtime_service_metadata(hid_t file, dataset_h5_state *state) {
    hid_t runtime = (hid_t) -1;
    if (state == 0) return 0;
    default_runtime_service_metadata(state);
    runtime = open_optional_group(file, runtime_service_group);
    if (runtime < 0) return 1;
    if (!read_optional_attr_u32(runtime, "service_mode", &state->runtime_service.service_mode)
        || !read_optional_attr_u32(runtime, "live_write_mode", &state->runtime_service.live_write_mode)
        || !read_optional_attr_u32(runtime, "prefer_pack_delivery", &state->runtime_service.prefer_pack_delivery)
        || !read_optional_attr_u32(runtime, "remote_pack_delivery", &state->runtime_service.remote_pack_delivery)
        || !read_optional_attr_u32(runtime, "single_reader_coordinator", &state->runtime_service.single_reader_coordinator)
        || !read_optional_attr_u32(runtime, "maintenance_lock_blocks_overwrite", &state->runtime_service.maintenance_lock_blocks_overwrite)
        || !read_optional_attr_u64(runtime, "canonical_generation", &state->runtime_service.canonical_generation)
        || !read_optional_attr_u64(runtime, "execution_plan_generation", &state->runtime_service.execution_plan_generation)
        || !read_optional_attr_u64(runtime, "pack_generation", &state->runtime_service.pack_generation)
        || !read_optional_attr_u64(runtime, "service_epoch", &state->runtime_service.service_epoch)
        || !read_optional_attr_u64(runtime, "active_read_generation", &state->runtime_service.active_read_generation)
        || !read_optional_attr_u64(runtime, "staged_write_generation", &state->runtime_service.staged_write_generation)) {
        goto done;
    }
    H5Gclose(runtime);
    return 1;

done:
    if (runtime >= 0) H5Gclose(runtime);
    return 0;
}

inline std::uint64_t estimate_shard_pack_bytes(const dataset_h5_state *state, unsigned long shard_id) {
    std::uint64_t begin = 0u;
    std::uint64_t end = 0u;
    std::uint64_t local_count = 0u;
    std::uint64_t rows = 0u;
    std::uint64_t nnz = 0u;
    std::uint64_t *local_offsets = 0;
    std::uint64_t *local_sizes = 0;
    std::uint64_t bytes = 0u;
    std::uint64_t i = 0u;

    if (state == 0 || shard_id >= state->num_shards || state->shard_part_begin == 0 || state->shard_part_end == 0) return 0u;
    begin = state->shard_part_begin[shard_id];
    end = state->shard_part_end[shard_id];
    local_count = end >= begin ? (end - begin) : 0u;
    if (local_count == 0u) return sharded_pack_payload_offset(0u, 1u, shard_pack_payload_alignment);
    local_offsets = (std::uint64_t *) std::calloc((std::size_t) local_count, sizeof(std::uint64_t));
    local_sizes = (std::uint64_t *) std::calloc((std::size_t) local_count, sizeof(std::uint64_t));
    if (local_offsets == 0 || local_sizes == 0) goto done;
    for (i = begin; i < end; ++i) {
        rows += state->partition_rows[i];
        nnz += state->partition_nnz[i];
    }
    if (state->matrix_family == dataset_matrix_family_blocked_ell) {
        if (!compute_shard_pack_locators<sparse::blocked_ell>(state->partition_rows + begin,
                                                              state->partition_nnz + begin,
                                                              state->partition_aux + begin,
                                                              state->cols,
                                                              local_count,
                                                              local_offsets,
                                                              local_sizes)) {
            goto done;
        }
    } else if (state->matrix_family == dataset_matrix_family_quantized_blocked_ell) {
        if (!compute_shard_pack_locators<sparse::quantized_blocked_ell>(state->partition_rows + begin,
                                                                        state->partition_nnz + begin,
                                                                        state->partition_aux + begin,
                                                                        state->cols,
                                                                        local_count,
                                                                        local_offsets,
                                                                        local_sizes)) {
            goto done;
        }
    } else if (state->matrix_family == dataset_matrix_family_sliced_ell) {
        if (!compute_shard_pack_locators<sparse::sliced_ell>(state->partition_rows + begin,
                                                             state->partition_nnz + begin,
                                                             state->partition_aux + begin,
                                                             state->cols,
                                                             local_count,
                                                             local_offsets,
                                                             local_sizes)) {
            goto done;
        }
    } else {
        goto done;
    }
    bytes = sharded_pack_payload_offset(local_count, 1u, shard_pack_payload_alignment);
    if (local_count != 0u) bytes = local_offsets[local_count - 1u] + local_sizes[local_count - 1u];
    (void) rows;
    (void) nnz;

done:
    std::free(local_offsets);
    std::free(local_sizes);
    return bytes;
}

inline int write_dataset_cache_manifest(const char *source_path,
                                       const dataset_h5_state *state) {
    std::FILE *fp = 0;
    unsigned long shard_id = 0ul;
    char canonical_pack_dir[4096];
    char execution_pack_root_dir[4096];
    char execution_pack_dir[4096];
    if (source_path == 0 || state == 0 || state->cache_manifest_path == 0) return 0;
    fp = std::fopen(state->cache_manifest_path, "wb");
    if (fp == 0) return 0;
    canonical_pack_dir[0] = '\0';
    execution_pack_root_dir[0] = '\0';
    execution_pack_dir[0] = '\0';
    if (state->cache_instance_dir != 0) {
        (void) build_cache_canonical_pack_dir_path(state->cache_instance_dir, canonical_pack_dir, sizeof(canonical_pack_dir));
        (void) build_cache_execution_pack_dir_path(state->cache_instance_dir, execution_pack_root_dir, sizeof(execution_pack_root_dir));
        (void) build_active_execution_pack_dir_path(state, execution_pack_dir, sizeof(execution_pack_dir));
    }
    std::fprintf(fp, "cache_schema_version=%u\n", (unsigned int) dataset_cache_schema_version);
    std::fprintf(fp, "source_path=%s\n", source_path);
    std::fprintf(fp, "cache_root=%s\n", state->cache_root != 0 ? state->cache_root : "");
    std::fprintf(fp, "cache_instance_dir=%s\n", state->cache_instance_dir != 0 ? state->cache_instance_dir : "");
    std::fprintf(fp, "canonical_pack_dir=%s\n", canonical_pack_dir);
    std::fprintf(fp, "execution_pack_root_dir=%s\n", execution_pack_root_dir);
    std::fprintf(fp, "execution_pack_dir=%s\n", execution_pack_dir);
    std::fprintf(fp, "source_size_bytes=%llu\n", (unsigned long long) state->source_size_bytes);
    std::fprintf(fp, "source_mtime_ns=%llu\n", (unsigned long long) state->source_mtime_ns);
    std::fprintf(fp, "matrix_family=%u\n", (unsigned int) state->matrix_family);
    std::fprintf(fp, "service_mode=%u\n", (unsigned int) state->runtime_service.service_mode);
    std::fprintf(fp, "live_write_mode=%u\n", (unsigned int) state->runtime_service.live_write_mode);
    std::fprintf(fp, "canonical_generation=%llu\n", (unsigned long long) state->runtime_service.canonical_generation);
    std::fprintf(fp, "execution_plan_generation=%llu\n", (unsigned long long) state->runtime_service.execution_plan_generation);
    std::fprintf(fp, "pack_generation=%llu\n", (unsigned long long) state->runtime_service.pack_generation);
    std::fprintf(fp, "service_epoch=%llu\n", (unsigned long long) state->runtime_service.service_epoch);
    std::fprintf(fp, "num_partitions=%llu\n", (unsigned long long) state->num_partitions);
    std::fprintf(fp, "num_shards=%llu\n", (unsigned long long) state->num_shards);
    for (shard_id = 0ul; shard_id < (unsigned long) state->num_shards; ++shard_id) {
        std::fprintf(fp,
                     "shard.%lu=%llu,%llu,%llu\n",
                     shard_id,
                     (unsigned long long) state->shard_part_begin[shard_id],
                     (unsigned long long) state->shard_part_end[shard_id],
                     (unsigned long long) estimate_shard_pack_bytes(state, shard_id));
    }
    std::fclose(fp);
    return 1;
}

inline int ensure_dataset_cache_layout(shard_storage *s) {
    dataset_h5_state *state = 0;
    char path[4096];
    char instances_root[4096];
    char metadata_dir[4096];
    char pack_root_dir[4096];
    char canonical_pack_dir[4096];
    char execution_pack_dir[4096];
    std::uint64_t fingerprint = 0u;
    unsigned long shard_id = 0ul;
    struct statvfs vfs;

    if (s == 0 || s->backend != shard_storage_backend_dataset_h5 || s->backend_state == 0 || s->source_path == 0) return 0;
    state = (dataset_h5_state *) s->backend_state;
    if (!refresh_dataset_source_stat(s->source_path, state)) return 0;
    if (state->cache_root == 0) {
        if (!build_default_cache_root(s->source_path, path, sizeof(path))) return 0;
        if (!assign_owned_string(&state->cache_root, path)) return 0;
    }
    if (shard_storage_has_capability(s, shard_storage_cap_materialize_canonical_pack | shard_storage_cap_materialize_execution_pack)) {
        if (!ensure_directory_exists(state->cache_root)) return 0;
    } else if (!directory_exists(state->cache_root)) {
        return 0;
    }
    if (!build_cache_instances_root_path(state->cache_root, instances_root, sizeof(instances_root))) return 0;
    if (shard_storage_has_capability(s, shard_storage_cap_materialize_canonical_pack | shard_storage_cap_materialize_execution_pack)) {
        if (!ensure_directory_exists(instances_root)) return 0;
    } else if (!directory_exists(instances_root)) {
        return 0;
    }
    fingerprint = build_source_fingerprint_u64(s->source_path,
                                               state->source_size_bytes,
                                               state->source_mtime_ns,
                                               state->matrix_family,
                                               state->num_partitions,
                                               state->num_shards);
    if (!build_cache_instance_path(state->cache_root, fingerprint, path, sizeof(path))) return 0;
    if (state->cache_instance_dir == 0 || std::strcmp(state->cache_instance_dir, path) != 0) {
        if (!assign_owned_string(&state->cache_instance_dir, path)) return 0;
    }
    if (shard_storage_has_capability(s, shard_storage_cap_materialize_canonical_pack | shard_storage_cap_materialize_execution_pack)) {
        if (!ensure_directory_exists(state->cache_instance_dir)) return 0;
    } else if (!directory_exists(state->cache_instance_dir)) {
        return 0;
    }
    if (!build_cache_metadata_dir_path(state->cache_instance_dir, metadata_dir, sizeof(metadata_dir))) return 0;
    if (shard_storage_has_capability(s, shard_storage_cap_materialize_canonical_pack | shard_storage_cap_materialize_execution_pack)
        && !ensure_directory_exists(metadata_dir)) {
        return 0;
    }
    if (!build_cache_pack_root_path(state->cache_instance_dir, pack_root_dir, sizeof(pack_root_dir))) return 0;
    if (shard_storage_has_capability(s, shard_storage_cap_materialize_canonical_pack | shard_storage_cap_materialize_execution_pack)) {
        if (!ensure_directory_exists(pack_root_dir)) return 0;
    } else if (!directory_exists(pack_root_dir)) {
        return 0;
    }
    if (!build_cache_canonical_pack_dir_path(state->cache_instance_dir, canonical_pack_dir, sizeof(canonical_pack_dir))) return 0;
    if (shard_storage_has_capability(s, shard_storage_cap_materialize_canonical_pack)) {
        if (!ensure_directory_exists(canonical_pack_dir)) return 0;
    } else if (!directory_exists(canonical_pack_dir)) {
        return 0;
    }
    if (!build_cache_execution_pack_dir_path(state->cache_instance_dir, execution_pack_dir, sizeof(execution_pack_dir))) return 0;
    if (shard_storage_has_capability(s, shard_storage_cap_materialize_execution_pack)) {
        if (!ensure_directory_exists(execution_pack_dir)) return 0;
    } else if (!directory_exists(execution_pack_dir)) {
        return 0;
    }
    if (!build_active_execution_pack_dir_path(state, execution_pack_dir, sizeof(execution_pack_dir))) return 0;
    if (shard_storage_has_capability(s, shard_storage_cap_materialize_execution_pack)) {
        if (!ensure_directory_exists(execution_pack_dir)) return 0;
    } else if (!directory_exists(execution_pack_dir)) {
        return 0;
    }
    if (!build_cache_manifest_path(state->cache_instance_dir, path, sizeof(path))) return 0;
    if (!assign_owned_string(&state->cache_manifest_path, path)) return 0;
    if (!ensure_cache_tracking_allocated(state)) return 0;
    for (shard_id = 0ul; shard_id < (unsigned long) state->num_shards; ++shard_id) {
        if (state->shard_cache_paths[shard_id] == 0) {
            if (!build_shard_pack_path(state, shard_id, path, sizeof(path))) return 0;
            if (!assign_owned_string(state->shard_cache_paths + shard_id, path)) return 0;
        }
        if (::access(state->shard_cache_paths[shard_id], R_OK) == 0) {
            if (state->shard_cache_state[shard_id] != dataset_cache_shard_ready) {
                state->shard_cache_state[shard_id] = dataset_cache_shard_ready;
                state->shard_cache_bytes[shard_id] = estimate_shard_pack_bytes(state, shard_id);
                state->cache_resident_bytes += state->shard_cache_bytes[shard_id];
            }
        }
    }
    if (shard_storage_has_capability(s, shard_storage_cap_materialize_canonical_pack | shard_storage_cap_materialize_execution_pack)
        && !write_dataset_cache_manifest(s->source_path, state)) {
        return 0;
    }
    if (!state->cache_budget_explicit) {
        std::uint64_t free_half = 0u;
        std::uint64_t estimated_total = 0u;
        if (::statvfs(state->cache_root, &vfs) == 0) {
            free_half = ((std::uint64_t) vfs.f_bavail * (std::uint64_t) vfs.f_frsize) / 2u;
        }
        for (shard_id = 0ul; shard_id < (unsigned long) state->num_shards; ++shard_id) {
            estimated_total += estimate_shard_pack_bytes(state, shard_id);
        }
        state->cache_budget_bytes = free_half == 0u ? estimated_total : (estimated_total < free_half ? estimated_total : free_half);
    }
    return 1;
}

inline int build_shard_partition_spans(dataset_h5_state *state) {
    std::uint64_t shard_id = 0u;
    std::uint64_t partition_id = 0u;
    if (state == 0) return 0;
    if (state->num_shards == 0u) return 1;
    if (state->shard_part_begin == 0) state->shard_part_begin = (std::uint64_t *) std::calloc((std::size_t) state->num_shards, sizeof(std::uint64_t));
    if (state->shard_part_end == 0) state->shard_part_end = (std::uint64_t *) std::calloc((std::size_t) state->num_shards, sizeof(std::uint64_t));
    if (state->partition_shard_ids == 0 && state->num_partitions != 0u) state->partition_shard_ids = (std::uint64_t *) std::calloc((std::size_t) state->num_partitions, sizeof(std::uint64_t));
    if (state->shard_part_begin == 0 || state->shard_part_end == 0 || (state->num_partitions != 0u && state->partition_shard_ids == 0)) return 0;
    for (shard_id = 0u; shard_id < state->num_shards; ++shard_id) {
        const std::uint64_t row_begin = state->shard_offsets[shard_id];
        const std::uint64_t row_end = state->shard_offsets[shard_id + 1u];
        while (partition_id < state->num_partitions && state->partition_row_offsets[partition_id] < row_begin) ++partition_id;
        state->shard_part_begin[shard_id] = partition_id;
        while (partition_id < state->num_partitions && state->partition_row_offsets[partition_id + 1u] <= row_end) {
            state->partition_shard_ids[partition_id] = shard_id;
            ++partition_id;
        }
        state->shard_part_end[shard_id] = partition_id;
    }
    return 1;
}

template<typename MatrixT>
inline int write_shard_pack_file(const char *filename,
                                 std::uint64_t cols,
                                 const std::uint64_t *partition_rows,
                                 const std::uint64_t *partition_nnz,
                                 const std::uint64_t *partition_aux,
                                 std::uint64_t partition_count,
                                 MatrixT *const *parts) {
    static const unsigned char magic[8] = { 'C', 'S', 'P', 'A', 'C', 'K', '0', '1' };
    std::uint64_t *partition_offsets = 0;
    std::uint64_t *part_sizes = 0;
    std::uint64_t *shard_offsets = 0;
    std::FILE *fp = 0;
    std::uint64_t rows = 0u;
    std::uint64_t nnz = 0u;
    std::uint64_t payload_offset = 0u;
    std::uint64_t i = 0u;
    int ok = 0;

    if (filename == 0 || ((partition_count != 0u) && (partition_rows == 0 || partition_nnz == 0 || parts == 0))) return 0;
    if (partition_count != 0u) {
        partition_offsets = (std::uint64_t *) std::calloc((std::size_t) partition_count, sizeof(std::uint64_t));
        part_sizes = (std::uint64_t *) std::calloc((std::size_t) partition_count, sizeof(std::uint64_t));
        if (partition_offsets == 0 || part_sizes == 0) goto done;
    }
    shard_offsets = (std::uint64_t *) std::calloc(2u, sizeof(std::uint64_t));
    if (shard_offsets == 0) goto done;
    if (!compute_shard_pack_locators<MatrixT>(partition_rows, partition_nnz, partition_aux, cols, partition_count, partition_offsets, part_sizes)) goto done;
    payload_offset = sharded_pack_payload_offset(partition_count, 1u, shard_pack_payload_alignment);
    for (i = 0u; i < partition_count; ++i) {
        rows += partition_rows[i];
        nnz += partition_nnz[i];
    }
    shard_offsets[0] = 0u;
    shard_offsets[1] = rows;

    fp = std::fopen(filename, "wb");
    if (fp == 0) goto done;
    std::setvbuf(fp, 0, _IOFBF, (std::size_t) 8u << 20u);
    if (!write_sharded_block(fp, magic, sizeof(magic), 1u)) goto done;
    {
        const unsigned char format = (unsigned char) disk_format_code<MatrixT>::value;
        const unsigned char reserved[7] = { 0, 0, 0, 0, 0, 0, 0 };
        const std::uint64_t num_partitions = partition_count;
        const std::uint64_t num_shards = 1u;
        if (!write_sharded_block(fp, &format, sizeof(format), 1u)) goto done;
        if (!write_sharded_block(fp, reserved, sizeof(reserved), 1u)) goto done;
        if (!write_sharded_block(fp, &rows, sizeof(rows), 1u)) goto done;
        if (!write_sharded_block(fp, &cols, sizeof(cols), 1u)) goto done;
        if (!write_sharded_block(fp, &nnz, sizeof(nnz), 1u)) goto done;
        if (!write_sharded_block(fp, &num_partitions, sizeof(num_partitions), 1u)) goto done;
        if (!write_sharded_block(fp, &num_shards, sizeof(num_shards), 1u)) goto done;
        if (!write_sharded_block(fp, &shard_pack_payload_alignment, sizeof(shard_pack_payload_alignment), 1u)) goto done;
        if (!write_sharded_block(fp, &payload_offset, sizeof(payload_offset), 1u)) goto done;
        if (!write_sharded_block(fp, partition_rows, sizeof(std::uint64_t), (std::size_t) partition_count)) goto done;
        if (!write_sharded_block(fp, partition_nnz, sizeof(std::uint64_t), (std::size_t) partition_count)) goto done;
        if (partition_aux != 0) {
            if (!write_sharded_block(fp, partition_aux, sizeof(std::uint64_t), (std::size_t) partition_count)) goto done;
        } else {
            const std::uint64_t zero = 0u;
            for (i = 0u; i < partition_count; ++i) {
                if (!write_sharded_block(fp, &zero, sizeof(zero), 1u)) goto done;
            }
        }
        if (!write_sharded_block(fp, shard_offsets, sizeof(std::uint64_t), 2u)) goto done;
        if (!write_sharded_block(fp, partition_offsets, sizeof(std::uint64_t), (std::size_t) partition_count)) goto done;
        if (!write_sharded_block(fp, part_sizes, sizeof(std::uint64_t), (std::size_t) partition_count)) goto done;
    }
    if (std::fflush(fp) != 0) goto done;
    for (i = 0u; i < partition_count; ++i) {
        if (parts[i] == 0) goto done;
        if (std::fseek(fp, (long) partition_offsets[i], SEEK_SET) != 0) goto done;
        if (!::cellshard::store(fp, parts[i])) goto done;
    }
    ok = 1;

done:
    if (fp != 0) std::fclose(fp);
    std::free(partition_offsets);
    std::free(part_sizes);
    std::free(shard_offsets);
    return ok;
}

inline int load_codec_table(hid_t codecs, dataset_codec_descriptor *descs, std::uint32_t count) {
    std::uint32_t i = 0;
    std::uint32_t *codec_ids = 0;
    std::uint32_t *families = 0;
    std::uint32_t *value_codes = 0;
    std::uint32_t *scale_value_codes = 0;
    std::uint32_t *bits = 0;
    std::uint32_t *flags = 0;
    int ok = 0;

    if (count == 0) return 1;
    codec_ids = (std::uint32_t *) std::calloc((std::size_t) count, sizeof(std::uint32_t));
    families = (std::uint32_t *) std::calloc((std::size_t) count, sizeof(std::uint32_t));
    value_codes = (std::uint32_t *) std::calloc((std::size_t) count, sizeof(std::uint32_t));
    scale_value_codes = (std::uint32_t *) std::calloc((std::size_t) count, sizeof(std::uint32_t));
    bits = (std::uint32_t *) std::calloc((std::size_t) count, sizeof(std::uint32_t));
    flags = (std::uint32_t *) std::calloc((std::size_t) count, sizeof(std::uint32_t));
    if (codec_ids == 0 || families == 0 || value_codes == 0 || scale_value_codes == 0 || bits == 0 || flags == 0) goto done;
    if (!read_dataset_1d(codecs, "codec_id", H5T_NATIVE_UINT32, count, codec_ids)) goto done;
    if (!read_dataset_1d(codecs, "family", H5T_NATIVE_UINT32, count, families)) goto done;
    if (!read_dataset_1d(codecs, "value_code", H5T_NATIVE_UINT32, count, value_codes)) goto done;
    if (!read_dataset_1d(codecs, "scale_value_code", H5T_NATIVE_UINT32, count, scale_value_codes)) goto done;
    if (!read_dataset_1d(codecs, "bits", H5T_NATIVE_UINT32, count, bits)) goto done;
    if (!read_dataset_1d(codecs, "flags", H5T_NATIVE_UINT32, count, flags)) goto done;
    for (i = 0; i < count; ++i) {
        descs[i].codec_id = codec_ids[i];
        descs[i].family = families[i];
        descs[i].value_code = value_codes[i];
        descs[i].scale_value_code = scale_value_codes[i];
        descs[i].bits = bits[i];
        descs[i].flags = flags[i];
    }
    ok = 1;

done:
    std::free(codec_ids);
    std::free(families);
    std::free(value_codes);
    std::free(scale_value_codes);
    std::free(bits);
    std::free(flags);
    return ok;
}

inline const dataset_codec_descriptor *find_codec(const dataset_h5_state *state, std::uint32_t codec_id) {
    std::uint32_t i = 0;
    if (state == 0) return 0;
    for (i = 0; i < state->num_codecs; ++i) {
        if (state->codecs[i].codec_id == codec_id) return state->codecs + i;
    }
    return 0;
}
