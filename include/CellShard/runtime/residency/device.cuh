#pragma once

#include <CellShard/runtime/residency/host.hh>
#include <cuda_runtime_api.h>

namespace cellshard {

struct device_allocator_ops {
    cudaError_t (*allocate)(void *context, int device_id, std::size_t bytes,
                            std::size_t alignment, void **out) noexcept = nullptr;
    cudaError_t (*deallocate)(void *context, int device_id,
                              void *allocation) noexcept = nullptr;
};
struct device_allocator_ref {
    void *context = nullptr;
    const device_allocator_ops *ops = nullptr;
};

// CS-FOUND-LEGACY: transitional convenience only. Production callers should
// supply the allocator that owns their device-memory policy.
[[nodiscard]] device_allocator_ref legacy_cuda_device_allocator() noexcept;

struct device_residency_view {
    image_id image{};
    const std::byte *payload = nullptr;
    std::size_t payload_bytes = 0;
    int device_id = -1;
    content_digest payload_digest{};
};

class device_residency {
public:
    device_residency() noexcept = default;
    ~device_residency() noexcept;
    device_residency(const device_residency &) = delete;
    device_residency &operator=(const device_residency &) = delete;
    device_residency(device_residency &&other) noexcept;
    device_residency &operator=(device_residency &&other) noexcept;

    [[nodiscard]] device_residency_view view() const noexcept;
    [[nodiscard]] bool valid() const noexcept { return allocation_ != nullptr; }
    [[nodiscard]] cudaError_t reset() noexcept;

private:
    image_id image_{};
    void *allocation_ = nullptr;
    std::size_t payload_bytes_ = 0;
    int device_id_ = -1;
    content_digest digest_{};
    device_allocator_ref allocator_{};

    friend cudaError_t stage_host_residency_async(
        const host_residency_view &, int, cudaStream_t, device_allocator_ref,
        device_residency *) noexcept;
};

[[nodiscard]] cudaError_t stage_host_residency_async(
    const host_residency_view &host,
    int device_id,
    cudaStream_t caller_stream,
    device_allocator_ref allocator,
    device_residency *out) noexcept;

} // namespace cellshard
