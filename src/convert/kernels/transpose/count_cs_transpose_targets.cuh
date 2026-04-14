__global__ static void count_cs_transpose_targets(
    const types::dim_t cDim,
    const types::ptr_t * __restrict__ cAxPtr,
    const types::idx_t * __restrict__ uAxIdx,
    types::ptr_t * __restrict__ out_uAxPtr_shifted
) {
    types::dim_t seg = (types::dim_t) ::cellshard::ptx::segment_tid_2d();
    const types::dim_t seg_stride = (types::dim_t) ::cellshard::ptx::segment_stride_2d();

    while (seg < cDim) {
        const types::ptr_t begin = cAxPtr[seg];
        const types::ptr_t end = cAxPtr[seg + 1];
        types::ptr_t i = ::cellshard::ptx::add_u32(begin, (types::ptr_t) threadIdx.x);
        while (i < end) {
            ::cellshard::ptx::atomic_add_u32(out_uAxPtr_shifted + ::cellshard::ptx::add_u32(uAxIdx[i], 1u), 1u);
            i = ::cellshard::ptx::add_u32(i, (types::ptr_t) blockDim.x);
        }
        seg += seg_stride;
    }
}
