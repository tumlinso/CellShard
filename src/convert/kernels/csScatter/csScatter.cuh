__global__ static void csScatter(
    const unsigned int nnz,
    const unsigned int * __restrict__ cAxIdx,
    const unsigned int * __restrict__ uAxIdx,
    const __half * __restrict__ val,
    unsigned int * __restrict__ heads,
    unsigned int * __restrict__ out_uAx,
    __half * __restrict__ out_val
) {
    const unsigned int tid = (unsigned int) ::cellshard::ptx::global_tid_1d();
    const unsigned int stride = (unsigned int) ::cellshard::ptx::global_stride_1d();
    unsigned int i = tid;
    while (i < nnz) {
        const unsigned int dst = ::cellshard::ptx::atomic_add_u32(heads + cAxIdx[i], 1u);
        out_uAx[dst] = uAxIdx[i];
        out_val[dst] = val[i];
        i += stride;
    }
}
