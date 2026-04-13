__global__ static void gather_shifted_shard_major_counts_from_order(
    types::dim_t shard_rows,
    const types::idx_t * __restrict__ major_order,
    const types::idx_t * __restrict__ part_row_offsets,
    types::idx_t part_count,
    const types::ptr_t * const * __restrict__ part_major_ptr,
    types::ptr_t * __restrict__ dst_major_counts_shifted
) {
    const types::dim_t tid = (types::dim_t) (blockIdx.x * blockDim.x + threadIdx.x);
    const types::dim_t stride = (types::dim_t) (gridDim.x * blockDim.x);
    types::dim_t row = tid;

    if (tid == 0u) dst_major_counts_shifted[0] = 0u;
    while (row < shard_rows) {
        const types::idx_t src_row = major_order[row];
        const types::idx_t part = (types::idx_t) ::cellshard::find_offset_span(src_row, part_row_offsets, part_count);
        const types::idx_t local_row = (types::idx_t) (src_row - part_row_offsets[part]);
        const types::ptr_t *major_ptr = part_major_ptr[part];
        dst_major_counts_shifted[row + 1u] = (types::ptr_t) (major_ptr[local_row + 1u] - major_ptr[local_row]);
        row += stride;
    }
}
