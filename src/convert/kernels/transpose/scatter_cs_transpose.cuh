__global__ static void scatter_cs_transpose(
    const types::dim_t cDim,
    const types::ptr_t * __restrict__ cAxPtr,
    const types::idx_t * __restrict__ uAxIdx,
    const real::storage_t * __restrict__ val,
    types::ptr_t * __restrict__ heads,
    types::idx_t * __restrict__ out_cAxIdx,
    real::storage_t * __restrict__ out_val
) {
    types::dim_t seg = (types::dim_t) ::cellshard::ptx::segment_tid_2d();
    const types::dim_t seg_stride = (types::dim_t) ::cellshard::ptx::segment_stride_2d();

    while (seg < cDim) {
        const types::ptr_t begin = cAxPtr[seg];
        const types::ptr_t end = cAxPtr[seg + 1];
        types::ptr_t i = ::cellshard::ptx::add_u32(begin, (types::ptr_t) threadIdx.x);
        while (i < end) {
            const types::idx_t dst_seg = uAxIdx[i];
            const types::ptr_t dst = ::cellshard::ptx::atomic_add_u32(heads + dst_seg, 1u);
            out_cAxIdx[dst] = (types::idx_t) seg;
            out_val[dst] = val[i];
            i = ::cellshard::ptx::add_u32(i, (types::ptr_t) blockDim.x);
        }
        seg += seg_stride;
    }
}
