#include <CellShard/runtime/v2/pinned_staging_pool.hh>

#include <cuda_runtime_api.h>

namespace cellshard::runtime_v2 {
namespace {
status_code allocate(void *, std::uint32_t, std::size_t bytes,
                     std::size_t alignment, void **out) noexcept {
    if (alignment > 4096 || out == nullptr) {
        return status_code::invalid_input;
    }
    return cudaHostAlloc(out, bytes, cudaHostAllocPortable) == cudaSuccess
        ? status_code::success
        : status_code::allocation_failure;
}

void deallocate(void *, void *allocation) noexcept {
    if (allocation != nullptr) {
        (void)cudaFreeHost(allocation);
    }
}

const pinned_staging_allocator_ops operations{allocate, deallocate};
} // namespace

pinned_staging_allocator_ref cuda_pinned_staging_allocator() noexcept {
    return {nullptr, &operations};
}

} // namespace cellshard::runtime_v2
