#include "shared.hh"

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

} // namespace cellshard
