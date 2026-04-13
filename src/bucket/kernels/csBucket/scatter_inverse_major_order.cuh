__global__ static void scatter_inverse_major_order(
    types::dim_t major_dim,
    const types::idx_t * __restrict__ major_order,
    types::idx_t * __restrict__ inverse_major_order
) {
    const types::dim_t tid = (types::dim_t) (blockIdx.x * blockDim.x + threadIdx.x);
    const types::dim_t stride = (types::dim_t) (gridDim.x * blockDim.x);
    types::dim_t i = tid;
    while (i < major_dim) {
        inverse_major_order[major_order[i]] = (types::idx_t) i;
        i += stride;
    }
}
