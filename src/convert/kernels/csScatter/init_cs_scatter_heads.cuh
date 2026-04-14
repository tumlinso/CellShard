__global__ static void init_cs_scatter_heads(
    const unsigned int cDim,
    const unsigned int * __restrict__ cAxPtr,
    unsigned int * __restrict__ heads
) {
    const unsigned int tid = (unsigned int) ::cellshard::ptx::global_tid_1d();
    const unsigned int stride = (unsigned int) ::cellshard::ptx::global_stride_1d();
    unsigned int i = tid;
    while (i < cDim) {
        heads[i] = cAxPtr[i];
        i += stride;
    }
}
