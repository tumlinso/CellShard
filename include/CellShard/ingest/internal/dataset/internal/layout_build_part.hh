#pragma once

static inline int build_bucketed_optimized_shard(const std::vector<sparse::blocked_ell> &parts,
                                                 std::uint32_t cols,
                                                 int device,
                                                 cellshard::bucketed_blocked_ell_shard *out) {
    owned_buffer<std::uint32_t> exec_to_canonical_cols;
    owned_buffer<std::uint32_t> canonical_to_exec_cols;
    std::uint32_t local_rows = 0u;
    std::uint32_t local_nnz = 0u;
    sparse::coo canonical_coo;
    sparse::blocked_ell permuted;

    if (out == nullptr) return 0;
    cellshard::clear(out);
    cellshard::init(out);
    if (!build_shard_column_maps(parts, cols, &exec_to_canonical_cols, &canonical_to_exec_cols)) return 0;

    out->rows = 0u;
    out->cols = cols;
    out->nnz = 0u;
    out->partition_count = (std::uint32_t) parts.size();
    out->partitions = parts.empty()
        ? nullptr
        : (cellshard::bucketed_blocked_ell_partition *) std::calloc(parts.size(), sizeof(cellshard::bucketed_blocked_ell_partition));
    out->partition_row_offsets = (std::uint32_t *) std::calloc(parts.size() + 1u, sizeof(std::uint32_t));
    out->exec_to_canonical_cols = cols != 0u ? (std::uint32_t *) std::calloc(cols, sizeof(std::uint32_t)) : nullptr;
    out->canonical_to_exec_cols = cols != 0u ? (std::uint32_t *) std::calloc(cols, sizeof(std::uint32_t)) : nullptr;
    if ((out->partition_count != 0u && (out->partitions == nullptr || out->partition_row_offsets == nullptr))
        || (cols != 0u && (out->exec_to_canonical_cols == nullptr || out->canonical_to_exec_cols == nullptr))) {
        cellshard::clear(out);
        return 0;
    }
    if (cols != 0u) {
        std::memcpy(out->exec_to_canonical_cols, exec_to_canonical_cols.data(), (std::size_t) cols * sizeof(std::uint32_t));
        std::memcpy(out->canonical_to_exec_cols, canonical_to_exec_cols.data(), (std::size_t) cols * sizeof(std::uint32_t));
    }
    sparse::init(&canonical_coo);
    sparse::init(&permuted);
    for (std::size_t part_i = 0; part_i < parts.size(); ++part_i) {
        std::uint32_t bucket_count = 1u;
        std::uint64_t bucketed_bytes = 0u;
        cellshard::bucketed_blocked_ell_partition *bucketed = out->partitions + part_i;
        cellshard::init(bucketed);
        out->partition_row_offsets[part_i] = local_rows;
        if (!blocked_ell_to_canonical_coo(&parts[part_i], &canonical_coo)
            || !cellshard::convert::blocked_ell_from_coo(&canonical_coo,
                                                         cols,
                                                         canonical_to_exec_cols.data(),
                                                         parts[part_i].block_size,
                                                         &permuted)
            || !choose_bucket_count_for_part(&permuted, device, &bucket_count)
            || !cellshard::build_bucketed_blocked_ell_partition(bucketed, &permuted, bucket_count, &bucketed_bytes)) {
            sparse::clear(&canonical_coo);
            sparse::clear(&permuted);
            cellshard::clear(out);
            return 0;
        }
        if (cols != 0u) {
            bucketed->exec_to_canonical_cols = (std::uint32_t *) std::calloc(cols, sizeof(std::uint32_t));
            bucketed->canonical_to_exec_cols = (std::uint32_t *) std::calloc(cols, sizeof(std::uint32_t));
            if (bucketed->exec_to_canonical_cols == nullptr || bucketed->canonical_to_exec_cols == nullptr) {
                sparse::clear(&canonical_coo);
                sparse::clear(&permuted);
                cellshard::clear(out);
                return 0;
            }
            std::memcpy(bucketed->exec_to_canonical_cols, out->exec_to_canonical_cols, (std::size_t) cols * sizeof(std::uint32_t));
            std::memcpy(bucketed->canonical_to_exec_cols, out->canonical_to_exec_cols, (std::size_t) cols * sizeof(std::uint32_t));
        }
        local_rows += bucketed->rows;
        local_nnz += bucketed->nnz;
        out->nnz += bucketed->nnz;
        sparse::clear(&canonical_coo);
        sparse::init(&canonical_coo);
        sparse::clear(&permuted);
        sparse::init(&permuted);
    }
    out->partition_row_offsets[parts.size()] = local_rows;
    out->rows = local_rows;
    (void) local_nnz;
    sparse::clear(&canonical_coo);
    sparse::clear(&permuted);
    return 1;
}

static inline int clone_bucketed_sliced_partition(cellshard::bucketed_sliced_ell_partition *dst,
                                                  const cellshard::bucketed_sliced_ell_partition *src) {
    if (dst == nullptr || src == nullptr) return 0;
    cellshard::clear(dst);
    cellshard::init(dst);
    dst->rows = src->rows;
    dst->cols = src->cols;
    dst->nnz = src->nnz;
    dst->segment_count = src->segment_count;
    dst->canonical_slice_count = src->canonical_slice_count;
    dst->segments = dst->segment_count != 0u
        ? (sparse::sliced_ell *) std::calloc((std::size_t) dst->segment_count, sizeof(sparse::sliced_ell))
        : nullptr;
    dst->segment_row_offsets = (std::uint32_t *) std::calloc((std::size_t) dst->segment_count + 1u, sizeof(std::uint32_t));
    dst->canonical_slice_row_offsets =
        (std::uint32_t *) std::calloc((std::size_t) dst->canonical_slice_count + 1u, sizeof(std::uint32_t));
    dst->canonical_slice_widths = dst->canonical_slice_count != 0u
        ? (std::uint32_t *) std::calloc((std::size_t) dst->canonical_slice_count, sizeof(std::uint32_t))
        : nullptr;
    dst->exec_to_canonical_rows = dst->rows != 0u ? (std::uint32_t *) std::malloc((std::size_t) dst->rows * sizeof(std::uint32_t)) : nullptr;
    dst->canonical_to_exec_rows = dst->rows != 0u ? (std::uint32_t *) std::malloc((std::size_t) dst->rows * sizeof(std::uint32_t)) : nullptr;
    if ((dst->segment_count != 0u && (dst->segments == nullptr || dst->segment_row_offsets == nullptr))
        || (dst->canonical_slice_row_offsets == nullptr)
        || (dst->canonical_slice_count != 0u && dst->canonical_slice_widths == nullptr)
        || (dst->rows != 0u && (dst->exec_to_canonical_rows == nullptr || dst->canonical_to_exec_rows == nullptr))) {
        cellshard::clear(dst);
        return 0;
    }
    if (dst->segment_count != 0u) {
        std::memcpy(dst->segment_row_offsets,
                    src->segment_row_offsets,
                    ((std::size_t) dst->segment_count + 1u) * sizeof(std::uint32_t));
    }
    if (dst->rows != 0u) {
        std::memcpy(dst->exec_to_canonical_rows, src->exec_to_canonical_rows, (std::size_t) dst->rows * sizeof(std::uint32_t));
        std::memcpy(dst->canonical_to_exec_rows, src->canonical_to_exec_rows, (std::size_t) dst->rows * sizeof(std::uint32_t));
    }
    std::memcpy(dst->canonical_slice_row_offsets,
                src->canonical_slice_row_offsets,
                ((std::size_t) dst->canonical_slice_count + 1u) * sizeof(std::uint32_t));
    if (dst->canonical_slice_widths != nullptr) {
        std::memcpy(dst->canonical_slice_widths,
                    src->canonical_slice_widths,
                    (std::size_t) dst->canonical_slice_count * sizeof(std::uint32_t));
    }
    for (std::uint32_t segment = 0u; segment < dst->segment_count; ++segment) {
        const sparse::sliced_ell *src_segment = src->segments + segment;
        sparse::init(dst->segments + segment, src_segment->rows, src_segment->cols, src_segment->nnz);
        if (!sparse::allocate(dst->segments + segment,
                              src_segment->slice_count,
                              src_segment->slice_row_offsets,
                              src_segment->slice_widths)) {
            cellshard::clear(dst);
            return 0;
        }
        {
            const std::size_t slots = (std::size_t) sparse::total_slots(src_segment);
            if (slots != 0u) {
                std::memcpy(dst->segments[segment].col_idx, src_segment->col_idx, slots * sizeof(cellshard::types::idx_t));
                std::memcpy(dst->segments[segment].val, src_segment->val, slots * sizeof(real::storage_t));
            }
        }
    }
    return 1;
}

static inline int choose_bucket_count_for_sliced_part_exact(const sparse::sliced_ell *part,
                                                            std::uint32_t *bucket_count_out,
                                                            std::uint64_t *bucketed_bytes_out = nullptr) {
    const std::uint32_t max_buckets = std::min<std::uint32_t>(8u, part != nullptr ? part->rows : 0u);
    cellshard::bucketed_sliced_ell_partition trial;
    std::uint32_t best_buckets = 1u;
    std::uint64_t best_bytes = std::numeric_limits<std::uint64_t>::max();
    if (part == nullptr || bucket_count_out == nullptr) return 0;
    cellshard::init(&trial);
    for (std::uint32_t buckets = 1u; buckets <= std::max<std::uint32_t>(1u, max_buckets); ++buckets) {
        std::uint64_t bytes = 0u;
        cellshard::clear(&trial);
        cellshard::init(&trial);
        if (!cellshard::build_bucketed_sliced_ell_partition(&trial, part, buckets, &bytes)) {
            cellshard::clear(&trial);
            return 0;
        }
        if (bytes < best_bytes || (bytes == best_bytes && buckets < best_buckets)) {
            best_bytes = bytes;
            best_buckets = buckets;
        }
    }
    cellshard::clear(&trial);
    *bucket_count_out = best_buckets;
    if (bucketed_bytes_out != nullptr) *bucketed_bytes_out = best_bytes == std::numeric_limits<std::uint64_t>::max() ? 0u : best_bytes;
    return 1;
}

static inline int build_bucketed_optimized_sliced_shard(const std::vector<sparse::sliced_ell> &parts,
                                                        std::uint32_t cols,
                                                        cellshard::bucketed_sliced_ell_shard *out,
                                                        std::vector<std::uint32_t> *bucket_counts = nullptr,
                                                        std::vector<std::uint64_t> *bucketed_bytes = nullptr) {
    std::uint32_t local_rows = 0u;
    if (out == nullptr) return 0;
    cellshard::clear(out);
    cellshard::init(out);
    out->rows = 0u;
    out->cols = cols;
    out->nnz = 0u;
    out->partition_count = (std::uint32_t) parts.size();
    out->partitions = parts.empty()
        ? nullptr
        : (cellshard::bucketed_sliced_ell_partition *) std::calloc(parts.size(), sizeof(cellshard::bucketed_sliced_ell_partition));
    out->partition_row_offsets = (std::uint32_t *) std::calloc(parts.size() + 1u, sizeof(std::uint32_t));
    if ((out->partition_count != 0u && (out->partitions == nullptr || out->partition_row_offsets == nullptr))) {
        cellshard::clear(out);
        return 0;
    }
    if (bucket_counts != nullptr) bucket_counts->assign(parts.size(), 0u);
    if (bucketed_bytes != nullptr) bucketed_bytes->assign(parts.size(), 0u);
    for (std::size_t part_i = 0u; part_i < parts.size(); ++part_i) {
        std::uint32_t bucket_count = 1u;
        std::uint64_t execution_bytes = 0u;
        cellshard::bucketed_sliced_ell_partition built;
        cellshard::init(&built);
        out->partition_row_offsets[part_i] = local_rows;
        if (!choose_bucket_count_for_sliced_part_exact(&parts[part_i], &bucket_count)
            || !cellshard::build_bucketed_sliced_ell_partition(&built, &parts[part_i], bucket_count, &execution_bytes)
            || !clone_bucketed_sliced_partition(out->partitions + part_i, &built)) {
            cellshard::clear(&built);
            cellshard::clear(out);
            return 0;
        }
        if (bucket_counts != nullptr) (*bucket_counts)[part_i] = bucket_count;
        if (bucketed_bytes != nullptr) (*bucketed_bytes)[part_i] = execution_bytes;
        local_rows += built.rows;
        out->nnz += built.nnz;
        cellshard::clear(&built);
    }
    out->partition_row_offsets[parts.size()] = local_rows;
    out->rows = local_rows;
    return 1;
}

static inline const std::string *find_spool_path_for_part(const std::vector<dataset_dataset_plan> &plans,
                                                          unsigned long global_part_id) {
    for (const dataset_dataset_plan &plan : plans) {
        if (global_part_id < plan.global_part_begin) continue;
        const unsigned long local_part = global_part_id - plan.global_part_begin;
        if (local_part < (unsigned long) plan.spool_paths.size()) return &plan.spool_paths[(std::size_t) local_part];
    }
    return nullptr;
}

static inline std::string build_ingest_spool_root(const char *out_path,
                                                  const std::string &working_root,
                                                  const std::string &cache_root) {
    namespace fs = std::filesystem;
    const fs::path out = out_path != 0 ? fs::path(out_path) : fs::path("dataset.csh5");
    const std::string stem = out.filename().string();
    if (!working_root.empty()) {
        return (fs::path(working_root) / (stem + ".ingest_spool")).string();
    }
    if (!cache_root.empty()) {
        return (fs::path(cache_root) / (stem + ".ingest_spool")).string();
    }
    if (!out.parent_path().empty()) {
        return (out.parent_path() / (stem + ".ingest_spool")).string();
    }
    return stem + ".ingest_spool";
}

static inline int prepare_ingest_spool_root(const std::string &root) {
    namespace fs = std::filesystem;
    std::error_code ec;
    if (root.empty()) return 0;
    fs::remove_all(root, ec);
    ec.clear();
    return fs::create_directories(root, ec) || (!ec && fs::exists(root, ec));
}

static inline std::string build_ingest_spool_part_path(const std::string &root,
                                                       unsigned long global_part_id) {
    char filename[64];
    std::snprintf(filename, sizeof(filename), "part.%08lu.cspool", global_part_id);
    return (std::filesystem::path(root) / filename).string();
}
