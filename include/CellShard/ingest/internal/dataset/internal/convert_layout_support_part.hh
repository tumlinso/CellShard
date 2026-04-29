#pragma once

struct shard_column_signature {
    std::uint32_t canonical_col;
    std::uint32_t support;
    std::uint64_t hash_a;
    std::uint64_t hash_b;
    std::uint32_t min_row_block;
};

static inline std::uint64_t mix_signature(std::uint64_t seed, std::uint64_t value) {
    const unsigned char *ptr = reinterpret_cast<const unsigned char *>(&value);
    for (std::size_t i = 0; i < sizeof(value); ++i) {
        seed ^= (std::uint64_t) ptr[i];
        seed *= 1099511628211ull;
    }
    return seed;
}

static inline int build_shard_column_maps(const std::vector<sparse::blocked_ell> &parts,
                                          std::uint32_t cols,
                                          owned_buffer<std::uint32_t> *exec_to_canonical,
                                          owned_buffer<std::uint32_t> *canonical_to_exec) {
    owned_buffer<shard_column_signature> signatures;
    std::uint32_t global_row_block = 0u;
    if (exec_to_canonical == nullptr || canonical_to_exec == nullptr) return 0;
    exec_to_canonical->assign_fill(cols, 0u);
    canonical_to_exec->assign_fill(cols, 0u);
    signatures.resize((std::size_t) cols);
    for (std::uint32_t col = 0u; col < cols; ++col) {
        signatures[(std::size_t) col].canonical_col = col;
        signatures[(std::size_t) col].support = 0u;
        signatures[(std::size_t) col].hash_a = 1469598103934665603ull;
        signatures[(std::size_t) col].hash_b = 1099511628211ull;
        signatures[(std::size_t) col].min_row_block = std::numeric_limits<std::uint32_t>::max();
    }
    for (const sparse::blocked_ell &part : parts) {
        const std::uint32_t row_block_count = cellshard::sparse::row_block_count(&part);
        const std::uint32_t width_blocks = cellshard::sparse::ell_width_blocks(&part);
        for (std::uint32_t row_block = 0u; row_block < row_block_count; ++row_block, ++global_row_block) {
            const std::uint32_t rows_in_block = std::min<std::uint32_t>(part.block_size, part.rows - row_block * part.block_size);
            for (std::uint32_t slot = 0u; slot < width_blocks; ++slot) {
                const cellshard::types::idx_t block_col = part.blockColIdx[(std::size_t) row_block * width_blocks + slot];
                if (block_col == cellshard::sparse::blocked_ell_invalid_col) continue;
                for (std::uint32_t col_in_block = 0u; col_in_block < part.block_size; ++col_in_block) {
                    const std::uint32_t col = (std::uint32_t) block_col * part.block_size + col_in_block;
                    bool seen = false;
                    if (col >= cols) continue;
                    for (std::uint32_t row_in_block = 0u; row_in_block < rows_in_block; ++row_in_block) {
                        const std::size_t offset =
                            (std::size_t) (row_block * part.block_size + row_in_block) * part.ell_cols
                            + (std::size_t) slot * part.block_size + col_in_block;
                        if (__half2float(part.val[offset]) != 0.0f) {
                            seen = true;
                            break;
                        }
                    }
                    if (!seen) continue;
                    signatures[(std::size_t) col].support += 1u;
                    signatures[(std::size_t) col].hash_a =
                        mix_signature(signatures[(std::size_t) col].hash_a, (std::uint64_t) global_row_block + 1u);
                    signatures[(std::size_t) col].hash_b =
                        mix_signature(signatures[(std::size_t) col].hash_b, ((std::uint64_t) global_row_block + 1u) * 1315423911ull);
                    signatures[(std::size_t) col].min_row_block =
                        std::min(signatures[(std::size_t) col].min_row_block, global_row_block);
                }
            }
        }
    }
    std::stable_sort(signatures.begin(),
                     signatures.end(),
                     [](const shard_column_signature &lhs, const shard_column_signature &rhs) {
                         if (lhs.support == 0u || rhs.support == 0u) {
                             if (lhs.support != rhs.support) return lhs.support > rhs.support;
                         }
                         if (lhs.min_row_block != rhs.min_row_block) return lhs.min_row_block < rhs.min_row_block;
                         if (lhs.hash_a != rhs.hash_a) return lhs.hash_a < rhs.hash_a;
                         if (lhs.hash_b != rhs.hash_b) return lhs.hash_b < rhs.hash_b;
                         if (lhs.support != rhs.support) return lhs.support > rhs.support;
                         return lhs.canonical_col < rhs.canonical_col;
                     });
    for (std::uint32_t exec_col = 0u; exec_col < cols; ++exec_col) {
        const std::uint32_t canonical_col = signatures[(std::size_t) exec_col].canonical_col;
        (*exec_to_canonical)[exec_col] = canonical_col;
        (*canonical_to_exec)[canonical_col] = exec_col;
    }
    return 1;
}

static inline int choose_bucket_count_for_part_host_exact(const sparse::blocked_ell *part,
                                                          std::uint32_t *bucket_count_out) {
    const std::uint32_t row_blocks = part != nullptr ? cellshard::sparse::row_block_count(part) : 0u;
    const std::uint32_t max_buckets = std::min<std::uint32_t>(8u, row_blocks);
    cellshard::bucketed_blocked_ell_partition trial;
    std::uint32_t best_buckets = 1u;
    std::uint64_t best_bytes = std::numeric_limits<std::uint64_t>::max();
    if (part == nullptr || bucket_count_out == nullptr) return 0;
    cellshard::init(&trial);
    for (std::uint32_t buckets = 1u; buckets <= std::max<std::uint32_t>(1u, max_buckets); ++buckets) {
        std::uint64_t bytes = 0u;
        cellshard::clear(&trial);
        cellshard::init(&trial);
        if (!cellshard::build_bucketed_blocked_ell_partition(&trial, part, buckets, &bytes)) {
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
    return 1;
}

static inline std::uint64_t estimate_bucketed_bytes_from_gpu_plan(const sparse::blocked_ell *part,
                                                                  const cellshard::bucket::blocked_ell_bucket_plan &plan,
                                                                  std::uint32_t bucket_count) {
    const std::uint32_t row_block_count = (std::uint32_t) plan.row_block_order.size();
    std::uint64_t bytes = 0u;
    if (part == nullptr || row_block_count == 0u || bucket_count == 0u) return 0u;
    for (std::uint32_t bucket = 0u; bucket < bucket_count; ++bucket) {
        const std::uint32_t rb_begin = (bucket * row_block_count) / bucket_count;
        const std::uint32_t rb_end = ((bucket + 1u) * row_block_count) / bucket_count;
        std::uint32_t seg_rows = 0u;
        std::uint32_t seg_width = 0u;
        if (rb_end <= rb_begin) continue;
        seg_width = plan.row_block_width_sorted[(std::size_t) rb_end - 1u];
        for (std::uint32_t pos = rb_begin; pos < rb_end; ++pos) {
            const std::uint32_t rb = plan.row_block_order[pos];
            const std::uint32_t row_begin = rb * part->block_size;
            seg_rows += row_begin < part->rows ? std::min<std::uint32_t>(part->block_size, part->rows - row_begin) : 0u;
        }
        bytes += (std::uint64_t) (rb_end - rb_begin) * seg_width * sizeof(cellshard::types::idx_t);
        bytes += (std::uint64_t) seg_rows * seg_width * part->block_size * sizeof(::real::storage_t);
    }
    bytes += (std::uint64_t) part->rows * sizeof(std::uint32_t) * 2u;
    bytes += (std::uint64_t) (bucket_count + 1u) * sizeof(std::uint32_t);
    return bytes;
}

static inline int choose_bucket_count_for_part(const sparse::blocked_ell *part,
                                               int device,
                                               std::uint32_t *bucket_count_out) {
    static thread_local cellshard::bucket::blocked_ell_bucket_workspace ws;
    static thread_local int ws_ready = 0;
    cellshard::bucket::blocked_ell_bucket_plan plan;
    const std::uint32_t row_blocks = part != nullptr ? cellshard::sparse::row_block_count(part) : 0u;
    const std::uint32_t max_buckets = std::min<std::uint32_t>(8u, row_blocks);
    const std::uint64_t original_bytes =
        (std::uint64_t) cellshard::packed_bytes((const sparse::blocked_ell *) nullptr,
                                                part != nullptr ? part->rows : 0u,
                                                part != nullptr ? part->cols : 0u,
                                                part != nullptr ? part->nnz : 0u,
                                                part != nullptr ? (unsigned long) cellshard::partition_aux(part) : 0ul,
                                                sizeof(::real::storage_t));
    std::uint32_t best_buckets = 1u;
    std::uint64_t best_bytes = original_bytes;

    if (part == nullptr || bucket_count_out == nullptr) return 0;
    if (row_blocks <= 1u) {
        *bucket_count_out = 1u;
        return 1;
    }
    if (!ws_ready) {
        ws_ready = cellshard::bucket::setup(&ws, device) ? 1 : 0;
    }
    if (ws_ready && cellshard::bucket::build_plan(part, &ws, &plan)) {
        for (std::uint32_t buckets = 1u; buckets <= std::max<std::uint32_t>(1u, max_buckets); ++buckets) {
            const std::uint64_t candidate = estimate_bucketed_bytes_from_gpu_plan(part, plan, buckets);
            if (candidate < best_bytes || (candidate == best_bytes && buckets < best_buckets)) {
                best_bytes = candidate;
                best_buckets = buckets;
            }
        }
        *bucket_count_out = best_buckets;
        return 1;
    }
    return choose_bucket_count_for_part_host_exact(part, bucket_count_out);
}
