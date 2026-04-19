#include "shared.hh"

inline int warm_dataset_blocked_ell_h5_execution_cache_common(const char *filename,
                                                              const char *cache_root,
                                                              unsigned long shard_begin,
                                                              unsigned long shard_end) {
    sharded<sparse::blocked_ell> matrix;
    shard_storage storage;
    dataset_h5_state *state = 0;
    unsigned long shard_id = 0ul;
    int ok = 0;

    if (filename == 0 || cache_root == 0 || *cache_root == '\0') return 0;

    init(&matrix);
    init(&storage);
    if (!load_dataset_blocked_ell_h5_header(filename, &matrix, &storage)) goto done;
    if (!bind_dataset_h5_cache(&storage, cache_root)) goto done;
    if (shard_begin > matrix.num_shards) goto done;
    if (shard_end > matrix.num_shards) shard_end = matrix.num_shards;
    state = dataset_h5_state_from_storage(&storage);
    if (state == 0) goto done;
    for (shard_id = shard_begin; shard_id < shard_end; ++shard_id) {
        if (!ensure_execution_pack_ready(&storage, state, shard_id)) goto done;
    }
    ok = 1;

done:
    clear(&storage);
    clear(&matrix);
    return ok;
}

int warm_dataset_blocked_ell_h5_cache_range(const char *filename,
                                            const char *cache_root,
                                            unsigned long shard_begin,
                                            unsigned long shard_end) {
    return warm_cache_range_common(filename,
                                   cache_root,
                                   shard_begin,
                                   shard_end,
                                   load_dataset_blocked_ell_h5_header,
                                   prefetch_dataset_blocked_ell_h5_shard_cache);
}

int warm_dataset_blocked_ell_h5_cache(const char *filename,
                                      const char *cache_root) {
    return warm_cache_common(filename,
                             cache_root,
                             load_dataset_blocked_ell_h5_header,
                             prefetch_dataset_blocked_ell_h5_shard_cache);
}

int warm_dataset_quantized_blocked_ell_h5_cache_range(const char *filename,
                                                      const char *cache_root,
                                                      unsigned long shard_begin,
                                                      unsigned long shard_end) {
    return warm_cache_range_common(filename,
                                   cache_root,
                                   shard_begin,
                                   shard_end,
                                   load_dataset_quantized_blocked_ell_h5_header,
                                   prefetch_dataset_quantized_blocked_ell_h5_shard_cache);
}

int warm_dataset_quantized_blocked_ell_h5_cache(const char *filename,
                                                const char *cache_root) {
    return warm_cache_common(filename,
                             cache_root,
                             load_dataset_quantized_blocked_ell_h5_header,
                             prefetch_dataset_quantized_blocked_ell_h5_shard_cache);
}

int warm_dataset_sliced_ell_h5_cache_range(const char *filename,
                                           const char *cache_root,
                                           unsigned long shard_begin,
                                           unsigned long shard_end) {
    return warm_cache_range_common(filename,
                                   cache_root,
                                   shard_begin,
                                   shard_end,
                                   load_dataset_sliced_ell_h5_header,
                                   prefetch_dataset_sliced_ell_h5_shard_cache);
}

int warm_dataset_sliced_ell_h5_cache(const char *filename,
                                     const char *cache_root) {
    return warm_cache_common(filename,
                             cache_root,
                             load_dataset_sliced_ell_h5_header,
                             prefetch_dataset_sliced_ell_h5_shard_cache);
}

int warm_dataset_blocked_ell_h5_execution_cache_range(const char *filename,
                                                      const char *cache_root,
                                                      unsigned long shard_begin,
                                                      unsigned long shard_end) {
    return warm_dataset_blocked_ell_h5_execution_cache_common(filename, cache_root, shard_begin, shard_end);
}

int warm_dataset_blocked_ell_h5_execution_cache(const char *filename,
                                                const char *cache_root) {
    return warm_dataset_blocked_ell_h5_execution_cache_common(filename,
                                                              cache_root,
                                                              0ul,
                                                              std::numeric_limits<unsigned long>::max());
}

} // namespace cellshard
