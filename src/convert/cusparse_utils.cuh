#pragma once

#include <cstdio>

#include <cuda_runtime.h>
#include <cusparse.h>

namespace cellshard {
namespace convert {
namespace cusparse_utils {

// cuSPARSE setup cost is noticeable when these helpers are called per part.
// Cache one handle per host thread and per active device, then just retarget
// the stream on each call. This keeps the library-first fast paths cheap
// enough to use inside CellShard's inner conversion loops.
struct thread_handle_cache {
    int device;
    cusparseHandle_t handle;

    __host__ thread_handle_cache() : device(-1), handle(0) {}

    __host__ ~thread_handle_cache() {
        if (handle != 0) (void) cusparseDestroy(handle);
    }
};

static inline int cuda_check(cudaError_t err, const char *label) {
    if (err == cudaSuccess) return 1;
    std::fprintf(stderr, "CUDA error at %s: %s\n", label, cudaGetErrorString(err));
    return 0;
}

static inline int check(cusparseStatus_t status, const char *label) {
    if (status == CUSPARSE_STATUS_SUCCESS) return 1;
    std::fprintf(stderr, "cuSPARSE error at %s: %d\n", label, (int) status);
    return 0;
}

// Borrow a handle bound to the current CUDA device and requested stream.
// The handle stays owned by thread-local cache; callers do not destroy it.
static inline int acquire(cudaStream_t stream, cusparseHandle_t *handle) {
    thread_local thread_handle_cache cache;
    int device = -1;

    if (!cuda_check(cudaGetDevice(&device), "cudaGetDevice acquire cuSPARSE handle")) return 0;

    if (cache.handle == 0 || cache.device != device) {
        if (cache.handle != 0) {
            if (!check(cusparseDestroy(cache.handle), "cusparseDestroy stale handle")) return 0;
            cache.handle = 0;
        }
        if (!check(cusparseCreate(&cache.handle), "cusparseCreate")) return 0;
        cache.device = device;
    }

    if (!check(cusparseSetStream(cache.handle, stream), "cusparseSetStream")) return 0;
    *handle = cache.handle;
    return 1;
}

} // namespace cusparse_utils
} // namespace convert
} // namespace cellshard
