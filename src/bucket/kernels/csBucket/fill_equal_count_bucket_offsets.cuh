__global__ static void fill_equal_count_bucket_offsets(
    types::dim_t major_dim,
    types::idx_t bucket_count,
    types::idx_t * __restrict__ bucket_offsets
) {
    types::idx_t bucket = 0;
    types::idx_t bucket_size = 0;
    types::idx_t remainder = 0;
    types::idx_t offset = 0;

    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    if (bucket_offsets == 0) return;
    if (bucket_count < 1u) bucket_count = 1u;
    if (major_dim == 0u) {
        bucket_offsets[0] = 0u;
        return;
    }
    if (bucket_count > major_dim) bucket_count = major_dim;
    bucket_size = (types::idx_t) (major_dim / bucket_count);
    remainder = (types::idx_t) (major_dim % bucket_count);
    for (bucket = 0; bucket < bucket_count; ++bucket) {
        bucket_offsets[bucket] = offset;
        offset = ::cellshard::ptx::add_u32(offset, bucket_size + (bucket < remainder ? 1u : 0u));
    }
    bucket_offsets[bucket_count] = (types::idx_t) major_dim;
}
