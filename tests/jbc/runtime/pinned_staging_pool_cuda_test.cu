#include <CellShard/runtime/v2/pinned_staging_pool.hh>

#include <cuda_runtime_api.h>

#include <cassert>

using namespace cellshard;
using namespace cellshard::runtime_v2;

int main() {
    assert(cudaSetDevice(0) == cudaSuccess);
    pinned_staging_pool pool;
    assert(pool.initialize(0, 1 << 20, 2, 64,
                           cuda_pinned_staging_allocator())
           == status_code::success);
    pinned_staging_lease lease{};
    assert(pool.acquire(1 << 20, &lease) == status_code::success);
    cudaPointerAttributes attributes{};
    assert(cudaPointerGetAttributes(&attributes, lease.data) == cudaSuccess);
    assert(attributes.type == cudaMemoryTypeHost);
    assert(pool.release(lease) == status_code::success);
}
