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
        const types::ptr_t len = ::cellshard::ptx::sub_u32(src_end, src_begin);
        types::ptr_t j = (types::ptr_t) threadIdx.x;

        while (j < len) {
            const types::ptr_t src_offset = ::cellshard::ptx::add_u32(src_begin, j);
            const types::ptr_t dst_offset = ::cellshard::ptx::add_u32(dst_begin, j);
            dst_minor_idx[dst_offset] = src_minor_idx[src_offset];
            dst_val[dst_offset] = src_val[src_offset];
            j += (types::ptr_t) blockDim.x;
        }

        dst_major += major_stride;
    }
}
