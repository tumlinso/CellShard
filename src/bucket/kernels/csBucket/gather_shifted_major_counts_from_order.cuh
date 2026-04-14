__global__ static void gather_shifted_major_counts_from_order(
    types::dim_t major_dim,
    const types::idx_t * __restrict__ major_order,
    const types::ptr_t * __restrict__ src_major_ptr,
    types::ptr_t * __restrict__ dst_major_counts_shifted
) {
    const types::dim_t tid = (types::dim_t) ::cellshard::ptx::global_tid_1d();
    const types::dim_t stride = (types::dim_t) ::cellshard::ptx::global_stride_1d();
    types::dim_t i = tid;

    if (tid == 0u) dst_major_counts_shifted[0] = 0u;
    while (i < major_dim) {
        const types::idx_t src_major = major_order[i];
        dst_major_counts_shifted[i + 1u] = (types::ptr_t) ::cellshard::ptx::sub_u32(src_major_ptr[src_major + 1u], src_major_ptr[src_major]);
        i += stride;
    }
}
