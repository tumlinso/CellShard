__global__ static void count_major_nnz(
    types::dim_t major_dim,
    const types::ptr_t * __restrict__ major_ptr,
    types::idx_t * __restrict__ major_nnz
) {
    const types::dim_t tid = (types::dim_t) (blockIdx.x * blockDim.x + threadIdx.x);
    const types::dim_t stride = (types::dim_t) (gridDim.x * blockDim.x);
    types::dim_t i = tid;
    while (i < major_dim) {
        major_nnz[i] = (types::idx_t) (major_ptr[i + 1] - major_ptr[i]);
        i += stride;
    }
}
