__global__ static void count_shard_major_nnz(
    types::dim_t shard_rows,
    const types::idx_t * __restrict__ part_row_offsets,
    types::idx_t part_count,
    const types::ptr_t * const * __restrict__ part_major_ptr,
    types::idx_t * __restrict__ major_nnz
) {
    const types::dim_t tid = (types::dim_t) (blockIdx.x * blockDim.x + threadIdx.x);
    const types::dim_t stride = (types::dim_t) (gridDim.x * blockDim.x);
    types::dim_t row = tid;

    while (row < shard_rows) {
        const types::idx_t part = (types::idx_t) ::cellshard::find_offset_span(row, part_row_offsets, part_count);
        const types::idx_t local_row = (types::idx_t) (row - part_row_offsets[part]);
        const types::ptr_t *major_ptr = part_major_ptr[part];
        major_nnz[row] = (types::idx_t) (major_ptr[local_row + 1u] - major_ptr[local_row]);
        row += stride;
    }
}
