__global__ static void count_cs_transpose_targets(
    const types::dim_t cDim,
    const types::ptr_t * __restrict__ cAxPtr,
    const types::idx_t * __restrict__ uAxIdx,
    types::ptr_t * __restrict__ out_uAxPtr_shifted
) {
    types::dim_t seg = (types::dim_t) blockIdx.x * blockDim.y + (types::dim_t) threadIdx.y;
    const types::dim_t seg_stride = (types::dim_t) gridDim.x * blockDim.y;

    while (seg < cDim) {
        const types::ptr_t begin = cAxPtr[seg];
        const types::ptr_t end = cAxPtr[seg + 1];
        types::ptr_t i = begin + (types::ptr_t) threadIdx.x;
        while (i < end) {
            atomicAdd(out_uAxPtr_shifted + uAxIdx[i] + 1u, 1u);
            i += (types::ptr_t) blockDim.x;
        }
        seg += seg_stride;
    }
}
