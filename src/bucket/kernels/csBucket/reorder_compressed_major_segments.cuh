template<typename ValueT>
__global__ static void reorder_compressed_major_segments(
    types::dim_t major_dim,
    const types::idx_t * __restrict__ major_order,
    const types::ptr_t * __restrict__ src_major_ptr,
    const types::idx_t * __restrict__ src_minor_idx,
    const ValueT * __restrict__ src_val,
    const types::ptr_t * __restrict__ dst_major_ptr,
    types::idx_t * __restrict__ dst_minor_idx,
    ValueT * __restrict__ dst_val
) {
    const types::dim_t major_stride = (types::dim_t) gridDim.x;
    types::dim_t dst_major = (types::dim_t) blockIdx.x;

    while (dst_major < major_dim) {
        const types::idx_t src_major = major_order[dst_major];
        const types::ptr_t src_begin = src_major_ptr[src_major];
        const types::ptr_t src_end = src_major_ptr[src_major + 1u];
        const types::ptr_t dst_begin = dst_major_ptr[dst_major];
        const types::ptr_t len = src_end - src_begin;
        types::ptr_t j = (types::ptr_t) threadIdx.x;

        while (j < len) {
            dst_minor_idx[dst_begin + j] = src_minor_idx[src_begin + j];
            dst_val[dst_begin + j] = src_val[src_begin + j];
            j += (types::ptr_t) blockDim.x;
        }

        dst_major += major_stride;
    }
}
