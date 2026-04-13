template<typename ValueT>
__global__ static void reorder_shard_major_segments(
    types::dim_t shard_rows,
    const types::idx_t * __restrict__ major_order,
    const types::idx_t * __restrict__ part_row_offsets,
    types::idx_t part_count,
    const types::ptr_t * const * __restrict__ part_major_ptr,
    const types::idx_t * const * __restrict__ part_minor_idx,
    const ValueT * const * __restrict__ part_val,
    const types::ptr_t * __restrict__ dst_major_ptr,
    types::idx_t * __restrict__ dst_minor_idx,
    ValueT * __restrict__ dst_val
) {
    const types::dim_t major_stride = (types::dim_t) gridDim.x;
    types::dim_t dst_row = (types::dim_t) blockIdx.x;

    while (dst_row < shard_rows) {
        const types::idx_t src_row = major_order[dst_row];
        const types::idx_t part = (types::idx_t) ::cellshard::find_offset_span(src_row, part_row_offsets, part_count);
        const types::idx_t local_row = (types::idx_t) (src_row - part_row_offsets[part]);
        const types::ptr_t *src_major_ptr = part_major_ptr[part];
        const types::idx_t *src_minor = part_minor_idx[part];
        const ValueT *src_values = part_val[part];
        const types::ptr_t src_begin = src_major_ptr[local_row];
        const types::ptr_t src_end = src_major_ptr[local_row + 1u];
        const types::ptr_t dst_begin = dst_major_ptr[dst_row];
        const types::ptr_t len = src_end - src_begin;
        types::ptr_t j = (types::ptr_t) threadIdx.x;

        while (j < len) {
            dst_minor_idx[dst_begin + j] = src_minor[src_begin + j];
            dst_val[dst_begin + j] = src_values[src_begin + j];
            j += (types::ptr_t) blockDim.x;
        }

        dst_row += major_stride;
    }
}
