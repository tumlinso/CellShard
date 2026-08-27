#include <CellShard/runtime/residency/device.cuh>

#include <utility>

namespace cellshard {
namespace {
cudaError_t legacy_allocate(void *, int, std::size_t bytes, std::size_t,
                            void **out) noexcept {
    return cudaMalloc(out, bytes);
}
cudaError_t legacy_deallocate(void *, int, void *allocation) noexcept {
    return cudaFree(allocation);
}
const device_allocator_ops legacy_ops{&legacy_allocate, &legacy_deallocate};

cudaError_t restore_device(int original, cudaError_t current) noexcept {
    const cudaError_t restored = cudaSetDevice(original);
    return current != cudaSuccess ? current : restored;
}
}

device_allocator_ref legacy_cuda_device_allocator() noexcept {
    return {nullptr, &legacy_ops};
}

device_residency::~device_residency() noexcept { (void) reset(); }
device_residency::device_residency(device_residency &&other) noexcept {
    *this = std::move(other);
}
device_residency &device_residency::operator=(device_residency &&other) noexcept {
    if (this != &other) {
        (void) reset();
        image_ = other.image_;
        allocation_ = other.allocation_;
        payload_bytes_ = other.payload_bytes_;
        device_id_ = other.device_id_;
        digest_ = other.digest_;
        allocator_ = other.allocator_;
        other.allocation_ = nullptr;
        other.payload_bytes_ = 0;
        other.device_id_ = -1;
    }
    return *this;
}
device_residency_view device_residency::view() const noexcept {
    return {image_, static_cast<const std::byte *>(allocation_), payload_bytes_,
            device_id_, digest_};
}
cudaError_t device_residency::reset() noexcept {
    if (allocation_ == nullptr) return cudaSuccess;
    int original = -1;
    cudaError_t status = cudaGetDevice(&original);
    if (status == cudaSuccess && original != device_id_) {
        status = cudaSetDevice(device_id_);
    }
    if (status == cudaSuccess) {
        status = allocator_.ops->deallocate(
            allocator_.context, device_id_, allocation_);
    }
    if (original >= 0 && original != device_id_) {
        status = restore_device(original, status);
    }
    if (status == cudaSuccess) {
        allocation_ = nullptr;
        payload_bytes_ = 0;
        device_id_ = -1;
    }
    return status;
}

cudaError_t stage_host_residency_async(
    const host_residency_view &host, int device_id, cudaStream_t caller_stream,
    device_allocator_ref allocator, device_residency *out) noexcept {
    if (host.payload == nullptr || host.payload_bytes == 0 || !host.image.valid()
        || !valid_content_digest(host.payload_digest) || device_id < 0
        || allocator.ops == nullptr || allocator.ops->allocate == nullptr
        || allocator.ops->deallocate == nullptr
        || out == nullptr) {
        return cudaErrorInvalidValue;
    }
    cudaError_t status = out->reset();
    if (status != cudaSuccess) return status;
    int original = -1;
    status = cudaGetDevice(&original);
    if (status != cudaSuccess) return status;
    if (original != device_id) status = cudaSetDevice(device_id);
    void *allocation = nullptr;
    if (status == cudaSuccess) {
        status = allocator.ops->allocate(allocator.context, device_id,
                                         host.payload_bytes, host.alignment,
                                         &allocation);
    }
    if (status == cudaSuccess) {
        status = cudaMemcpyAsync(allocation, host.payload, host.payload_bytes,
                                 cudaMemcpyHostToDevice, caller_stream);
    }
    if (status != cudaSuccess && allocation != nullptr) {
        const cudaError_t release_status = allocator.ops->deallocate(
            allocator.context, device_id, allocation);
        if (status == cudaSuccess) status = release_status;
    }
    if (original != device_id) status = restore_device(original, status);
    if (status != cudaSuccess) return status;
    out->image_ = host.image;
    out->allocation_ = allocation;
    out->payload_bytes_ = host.payload_bytes;
    out->device_id_ = device_id;
    out->digest_ = host.payload_digest;
    out->allocator_ = allocator;
    return cudaSuccess;
}

} // namespace cellshard
