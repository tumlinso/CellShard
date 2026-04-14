__global__ static void shift_ptr_idx_count(
    const unsigned int nnz,
    const unsigned int * __restrict__ axIdx,
    unsigned int * __restrict__ axPtr_shifted
) {
    const unsigned int tid = (unsigned int) ::cellshard::ptx::global_tid_1d();
    const unsigned int stride = (unsigned int) ::cellshard::ptx::global_stride_1d();
    unsigned int i = tid;
    while (i < nnz) {
        ::cellshard::ptx::atomic_add_u32(axPtr_shifted + ::cellshard::ptx::add_u32(axIdx[i], 1u), 1u);
        i += stride;
    }
}
