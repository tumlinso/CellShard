#pragma once

#include "../execution_internal.hh"

struct dataset_header_layout_buffers {
    std::uint64_t *partition_rows = 0;
    std::uint64_t *partition_nnz = 0;
    std::uint64_t *partition_aux = 0;
    std::uint64_t *partition_row_offsets = 0;
    std::uint64_t *shard_offsets = 0;
    unsigned long *part_rows_ul = 0;
    unsigned long *part_nnz_ul = 0;
    unsigned long *part_aux_ul = 0;
    unsigned long *shard_offsets_ul = 0;

    void clear() {
        std::free(partition_rows);
        std::free(partition_nnz);
        std::free(partition_aux);
        std::free(partition_row_offsets);
        std::free(shard_offsets);
        std::free(part_rows_ul);
        std::free(part_nnz_ul);
        std::free(part_aux_ul);
        std::free(shard_offsets_ul);
        partition_rows = 0;
        partition_nnz = 0;
        partition_aux = 0;
        partition_row_offsets = 0;
        shard_offsets = 0;
        part_rows_ul = 0;
        part_nnz_ul = 0;
        part_aux_ul = 0;
        shard_offsets_ul = 0;
    }
};

inline dataset_h5_state *dataset_h5_state_from_storage(shard_storage *s) {
    if (s == 0 || s->backend != shard_storage_backend_dataset_h5 || s->backend_state == 0) return 0;
    return (dataset_h5_state *) s->backend_state;
}

inline const dataset_h5_state *dataset_h5_state_from_storage(const shard_storage *s) {
    if (s == 0 || s->backend != shard_storage_backend_dataset_h5 || s->backend_state == 0) return 0;
    return (const dataset_h5_state *) s->backend_state;
}

inline int allocate_header_layout_buffers(std::uint64_t num_partitions,
                                          std::uint64_t num_shards,
                                          dataset_header_layout_buffers *buffers) {
    if (buffers == 0) return 0;
    buffers->clear();
    buffers->partition_rows = (std::uint64_t *) std::calloc((std::size_t) num_partitions, sizeof(std::uint64_t));
    buffers->partition_nnz = (std::uint64_t *) std::calloc((std::size_t) num_partitions, sizeof(std::uint64_t));
    buffers->partition_aux = (std::uint64_t *) std::calloc((std::size_t) num_partitions, sizeof(std::uint64_t));
    buffers->partition_row_offsets =
        (std::uint64_t *) std::calloc((std::size_t) num_partitions + 1u, sizeof(std::uint64_t));
    buffers->shard_offsets = (std::uint64_t *) std::calloc((std::size_t) num_shards + 1u, sizeof(std::uint64_t));
    buffers->part_rows_ul = (unsigned long *) std::calloc((std::size_t) num_partitions, sizeof(unsigned long));
    buffers->part_nnz_ul = (unsigned long *) std::calloc((std::size_t) num_partitions, sizeof(unsigned long));
    buffers->part_aux_ul = (unsigned long *) std::calloc((std::size_t) num_partitions, sizeof(unsigned long));
    buffers->shard_offsets_ul = (unsigned long *) std::calloc((std::size_t) num_shards + 1u, sizeof(unsigned long));
    if ((num_partitions != 0)
        && (buffers->partition_rows == 0
            || buffers->partition_nnz == 0
            || buffers->partition_aux == 0
            || buffers->partition_row_offsets == 0
            || buffers->part_rows_ul == 0
            || buffers->part_nnz_ul == 0
            || buffers->part_aux_ul == 0)) {
        return 0;
    }
    if ((num_shards + 1u) != 0u && (buffers->shard_offsets == 0 || buffers->shard_offsets_ul == 0)) return 0;
    return 1;
}

inline int read_header_layout_tables(hid_t matrix,
                                     const char *filename,
                                     std::uint64_t rows,
                                     std::uint64_t nnz,
                                     std::uint64_t num_partitions,
                                     std::uint64_t num_shards,
                                     dataset_header_layout_buffers *buffers) {
    if (matrix < 0 || buffers == 0) return 0;
    if (!allocate_header_layout_buffers(num_partitions, num_shards, buffers)) return 0;
    if (!read_dataset_1d(matrix, "partition_rows", H5T_NATIVE_UINT64, num_partitions, buffers->partition_rows)) return 0;
    if (!read_dataset_1d(matrix, "partition_nnz", H5T_NATIVE_UINT64, num_partitions, buffers->partition_nnz)) return 0;
    if (!read_dataset_1d(matrix, "partition_aux", H5T_NATIVE_UINT64, num_partitions, buffers->partition_aux)) return 0;
    if (!read_dataset_1d(matrix,
                         "partition_row_offsets",
                         H5T_NATIVE_UINT64,
                         num_partitions + 1u,
                         buffers->partition_row_offsets)) {
        return 0;
    }
    if (!read_dataset_1d(matrix, "shard_offsets", H5T_NATIVE_UINT64, num_shards + 1u, buffers->shard_offsets)) return 0;
    return validate_dataset_layout_tables(filename,
                                          rows,
                                          nnz,
                                          num_partitions,
                                          num_shards,
                                          buffers->partition_rows,
                                          buffers->partition_nnz,
                                          buffers->partition_row_offsets,
                                          buffers->shard_offsets);
}

template<typename MatrixT>
inline int initialize_sharded_header_view(const char *filename,
                                          sharded<MatrixT> *m,
                                          unsigned long rows_ul,
                                          unsigned long cols_ul,
                                          unsigned long nnz_ul,
                                          std::uint64_t num_partitions,
                                          std::uint64_t num_shards,
                                          const dataset_header_layout_buffers &buffers) {
    unsigned long i = 0ul;
    if (m == 0) return 0;
    clear(m);
    init(m);
    for (i = 0; i < (unsigned long) num_partitions; ++i) {
        if (!sharded_from_u64(buffers.partition_rows[i], buffers.part_rows_ul + i, "partition_rows", filename)) return 0;
        if (!sharded_from_u64(buffers.partition_nnz[i], buffers.part_nnz_ul + i, "partition_nnz", filename)) return 0;
        if (!sharded_from_u64(buffers.partition_aux[i], buffers.part_aux_ul + i, "partition_aux", filename)) return 0;
    }
    for (i = 0; i <= (unsigned long) num_shards; ++i) {
        if (!sharded_from_u64(buffers.shard_offsets[i], buffers.shard_offsets_ul + i, "shard_offsets", filename)) return 0;
    }
    if (!define_partitions(m,
                           cols_ul,
                           (unsigned long) num_partitions,
                           buffers.part_rows_ul,
                           buffers.part_nnz_ul,
                           buffers.part_aux_ul)) {
        return 0;
    }
    if (!reshard(m, (unsigned long) num_shards, buffers.shard_offsets_ul)) return 0;
    return validate_loaded_sharded_header(filename, m, rows_ul, nnz_ul);
}

inline int populate_common_dataset_h5_state(const char *filename,
                                            hid_t matrix,
                                            hid_t codecs,
                                            dataset_h5_state *state,
                                            std::uint64_t rows,
                                            std::uint64_t cols,
                                            std::uint64_t nnz,
                                            std::uint64_t num_partitions,
                                            std::uint64_t num_shards,
                                            std::uint64_t num_codecs,
                                            const dataset_header_layout_buffers &buffers) {
    if (state == 0 || matrix < 0 || codecs < 0) return 0;
    state->rows = rows;
    state->cols = cols;
    state->nnz = nnz;
    state->num_partitions = num_partitions;
    state->num_shards = num_shards;
    state->num_codecs = (std::uint32_t) num_codecs;
    if (num_partitions != 0) {
        state->partition_rows = (std::uint64_t *) std::calloc((std::size_t) num_partitions, sizeof(std::uint64_t));
        state->partition_nnz = (std::uint64_t *) std::calloc((std::size_t) num_partitions, sizeof(std::uint64_t));
        state->partition_aux = (std::uint64_t *) std::calloc((std::size_t) num_partitions, sizeof(std::uint64_t));
        state->partition_row_offsets =
            (std::uint64_t *) std::calloc((std::size_t) num_partitions + 1u, sizeof(std::uint64_t));
        state->partition_codec_ids = (std::uint32_t *) std::calloc((std::size_t) num_partitions, sizeof(std::uint32_t));
        if (state->partition_rows == 0
            || state->partition_nnz == 0
            || state->partition_aux == 0
            || state->partition_row_offsets == 0
            || state->partition_codec_ids == 0) {
            return 0;
        }
        std::memcpy(state->partition_rows, buffers.partition_rows, (std::size_t) num_partitions * sizeof(std::uint64_t));
        std::memcpy(state->partition_nnz, buffers.partition_nnz, (std::size_t) num_partitions * sizeof(std::uint64_t));
        std::memcpy(state->partition_aux, buffers.partition_aux, (std::size_t) num_partitions * sizeof(std::uint64_t));
        std::memcpy(state->partition_row_offsets,
                    buffers.partition_row_offsets,
                    ((std::size_t) num_partitions + 1u) * sizeof(std::uint64_t));
    }
    state->shard_offsets = (std::uint64_t *) std::calloc((std::size_t) num_shards + 1u, sizeof(std::uint64_t));
    if (state->shard_offsets == 0 && (num_shards + 1u) != 0u) return 0;
    if (state->shard_offsets != 0) {
        std::memcpy(state->shard_offsets, buffers.shard_offsets, ((std::size_t) num_shards + 1u) * sizeof(std::uint64_t));
    }
    if (num_codecs != 0) {
        state->codecs = (dataset_codec_descriptor *) std::calloc((std::size_t) num_codecs, sizeof(dataset_codec_descriptor));
        if (state->codecs == 0) return 0;
    }
    if (!read_dataset_1d(matrix, "partition_codec_ids", H5T_NATIVE_UINT32, num_partitions, state->partition_codec_ids)) return 0;
    if (!load_codec_table(codecs, state->codecs, (std::uint32_t) num_codecs)) return 0;
    return validate_partition_codec_ids(filename,
                                        num_partitions,
                                        state->partition_codec_ids,
                                        (std::uint32_t) num_codecs,
                                        state->codecs);
}

template<typename MatrixT>
inline int fetch_cached_partition_common(sharded<MatrixT> *m,
                                         const shard_storage *s,
                                         unsigned long partition_id,
                                         int (*load_part)(sharded<MatrixT> *, dataset_h5_state *, unsigned long)) {
    shard_storage *storage = const_cast<shard_storage *>(s);
    dataset_h5_state *state = dataset_h5_state_from_storage(storage);
    if (m == 0 || storage == 0 || state == 0 || partition_id >= m->num_partitions || load_part == 0) return 0;
    if (!ensure_cached_shard_ready(storage, (unsigned long) state->partition_shard_ids[partition_id])) return 0;
    return load_part(m, state, partition_id);
}

template<typename MatrixT>
inline int fetch_cached_shard_common(sharded<MatrixT> *m,
                                     const shard_storage *s,
                                     unsigned long shard_id,
                                     int (*load_part)(sharded<MatrixT> *, dataset_h5_state *, unsigned long)) {
    const unsigned long begin = first_partition_in_shard(m, shard_id);
    const unsigned long end = last_partition_in_shard(m, shard_id);
    dataset_h5_state *state = dataset_h5_state_from_storage(const_cast<shard_storage *>(s));
    unsigned long i = 0ul;

    if (m == 0 || s == 0 || state == 0 || shard_id >= m->num_shards || load_part == 0) return 0;
    if (!ensure_cached_shard_ready(const_cast<shard_storage *>(s), shard_id)) return 0;
    for (i = begin; i < end; ++i) {
        if (!load_part(m, state, i)) return 0;
    }
    return 1;
}

template<typename MatrixT>
inline int prefetch_partition_cache_common(const sharded<MatrixT> *m,
                                           shard_storage *s,
                                           unsigned long partition_id) {
    dataset_h5_state *state = dataset_h5_state_from_storage(s);
    if (m == 0 || s == 0 || state == 0 || partition_id >= m->num_partitions) return 0;
    return ensure_cached_shard_ready(s, (unsigned long) state->partition_shard_ids[partition_id]);
}

template<typename MatrixT>
inline int prefetch_shard_cache_common(const sharded<MatrixT> *m,
                                       shard_storage *s,
                                       unsigned long shard_id) {
    if (m == 0 || s == 0 || shard_id >= m->num_shards) return 0;
    return ensure_cached_shard_ready(s, shard_id);
}

template<typename MatrixT>
inline int warm_cache_range_common(const char *filename,
                                   const char *cache_root,
                                   unsigned long shard_begin,
                                   unsigned long shard_end,
                                   int (*load_header)(const char *, sharded<MatrixT> *, shard_storage *),
                                   int (*prefetch_shard)(const sharded<MatrixT> *, shard_storage *, unsigned long)) {
    sharded<MatrixT> matrix;
    shard_storage storage;
    unsigned long shard_id = 0ul;
    int ok = 0;

    if (filename == 0 || cache_root == 0 || *cache_root == '\0' || load_header == 0 || prefetch_shard == 0) return 0;

    init(&matrix);
    init(&storage);
    if (!load_header(filename, &matrix, &storage)) goto done;
    if (!bind_dataset_h5_cache(&storage, cache_root)) goto done;
    if (shard_begin > matrix.num_shards) goto done;
    if (shard_end > matrix.num_shards) shard_end = matrix.num_shards;
    for (shard_id = shard_begin; shard_id < shard_end; ++shard_id) {
        if (!prefetch_shard(&matrix, &storage, shard_id)) goto done;
    }
    ok = 1;

done:
    clear(&storage);
    clear(&matrix);
    return ok;
}

template<typename MatrixT>
inline int warm_cache_common(const char *filename,
                             const char *cache_root,
                             int (*load_header)(const char *, sharded<MatrixT> *, shard_storage *),
                             int (*prefetch_shard)(const sharded<MatrixT> *, shard_storage *, unsigned long)) {
    return warm_cache_range_common(filename,
                                   cache_root,
                                   0ul,
                                   std::numeric_limits<unsigned long>::max(),
                                   load_header,
                                   prefetch_shard);
}
