__global__ static void csSearchToCoo(
    const unsigned int cDim,
    const unsigned int nnz,
    const unsigned int * __restrict__ cAxPtr,
    const unsigned int * __restrict__ uAxIdx,
    const __half * __restrict__ val,
    unsigned int * __restrict__ out_cAxIdx,
    unsigned int * __restrict__ out_uAxIdx,
    __half * __restrict__ out_val
) {
    const unsigned int tid = (unsigned int) (blockIdx.x * blockDim.x + threadIdx.x);
    const unsigned int stride = (unsigned int) (gridDim.x * blockDim.x);
    unsigned int i = tid;

    while (i < nnz) {
        unsigned int lo = 0;
        unsigned int hi = cDim;

        while (lo < hi) {
            const unsigned int mid = lo + ((hi - lo) >> 1);
            if (i >= cAxPtr[mid + 1]) lo = mid + 1;
            else hi = mid;
        }

        out_cAxIdx[i] = lo;
        out_uAxIdx[i] = uAxIdx[i];
        out_val[i] = val[i];
        i += stride;
    }
}
