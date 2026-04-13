__global__ static void shift_ptr_idx_count(
    const unsigned int nnz,
    const unsigned int * __restrict__ axIdx,
    unsigned int * __restrict__ axPtr_shifted
) {
    const unsigned int tid = (unsigned int) (blockIdx.x * blockDim.x + threadIdx.x);
    const unsigned int stride = (unsigned int) (gridDim.x * blockDim.x);
    unsigned int i = tid;
    while (i < nnz) {
        atomicAdd(axPtr_shifted + axIdx[i] + 1, 1u);
        i += stride;
    }
}
