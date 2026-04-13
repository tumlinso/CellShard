template<typename ValueT>
__global__ static void transpose_coo_entries(
    const types::nnz_t nnz,
    const types::idx_t * __restrict__ src_rowIdx,
    const types::idx_t * __restrict__ src_colIdx,
    const ValueT * __restrict__ src_val,
    types::idx_t * __restrict__ dst_rowIdx,
    types::idx_t * __restrict__ dst_colIdx,
    ValueT * __restrict__ dst_val
) {
    const types::nnz_t tid = (types::nnz_t) (blockIdx.x * blockDim.x + threadIdx.x);
    const types::nnz_t stride = (types::nnz_t) (gridDim.x * blockDim.x);
    types::nnz_t i = tid;

    while (i < nnz) {
        const types::idx_t r = src_rowIdx[i];
        const types::idx_t c = src_colIdx[i];
        dst_rowIdx[i] = c;
        dst_colIdx[i] = r;
        if (dst_val != 0 && dst_val != src_val) dst_val[i] = src_val[i];
        i += stride;
    }
}
