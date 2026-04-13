__global__ static void gather_major_nnz_by_order(
    types::dim_t major_dim,
    const types::idx_t * __restrict__ major_nnz,
    const types::idx_t * __restrict__ major_order,
    types::idx_t * __restrict__ sorted_major_nnz
) {
    const types::dim_t tid = (types::dim_t) (blockIdx.x * blockDim.x + threadIdx.x);
    const types::dim_t stride = (types::dim_t) (gridDim.x * blockDim.x);
    types::dim_t i = tid;
    while (i < major_dim) {
        sorted_major_nnz[i] = major_nnz[major_order[i]];
        i += stride;
    }
}
