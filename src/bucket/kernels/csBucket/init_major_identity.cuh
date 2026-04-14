__global__ static void init_major_identity(
    types::dim_t major_dim,
    types::idx_t * __restrict__ major_index
) {
    const types::dim_t tid = (types::dim_t) ::cellshard::ptx::global_tid_1d();
    const types::dim_t stride = (types::dim_t) ::cellshard::ptx::global_stride_1d();
    types::dim_t i = tid;
    while (i < major_dim) {
        major_index[i] = (types::idx_t) i;
        i += stride;
    }
}
