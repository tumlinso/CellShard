__global__ static void csExpandToCoo(
    const unsigned int cDim,
    const unsigned int * __restrict__ cAxPtr,
    const unsigned int * __restrict__ uAxIdx,
    const __half * __restrict__ val,
    unsigned int * __restrict__ out_cAxIdx,
    unsigned int * __restrict__ out_uAxIdx,
    __half * __restrict__ out_val
) {
    unsigned int c = (unsigned int) blockIdx.x;
    const unsigned int cStride = (unsigned int) gridDim.x;

    while (c < cDim) {
        const unsigned int begin = cAxPtr[c];
        const unsigned int end = cAxPtr[c + 1];
        unsigned int i = begin + (unsigned int) threadIdx.x;

        while (i < end) {
            out_cAxIdx[i] = c;
            out_uAxIdx[i] = uAxIdx[i];
            out_val[i] = val[i];
            i += (unsigned int) blockDim.x;
        }
        c += cStride;
    }
}
