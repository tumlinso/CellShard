#include <CellShard/runtime/v2/device_transport.cuh>

namespace cellshard::runtime_v2 {

status_code same_device_alias(const const_device_region &source,
                              int destination_device,
                              const_device_region *out) noexcept {
    if (!valid_device_region(source) || out == nullptr
        || destination_device != source.device_id) {
        return status_code::invalid_input;
    }
    *out = source;
    return status_code::success;
}

status_code prepare_cuda_p2p(int source_device,
                             int destination_device) noexcept {
    if (source_device < 0 || destination_device < 0
        || source_device == destination_device) {
        return status_code::invalid_input;
    }
    int can_access = 0;
    if (cudaDeviceCanAccessPeer(&can_access, destination_device, source_device)
            != cudaSuccess
        || can_access == 0) {
        return status_code::unsupported_capability;
    }
    int previous_device = -1;
    if (cudaGetDevice(&previous_device) != cudaSuccess
        || cudaSetDevice(destination_device) != cudaSuccess) {
        return status_code::cuda_failure;
    }
    const cudaError_t enable = cudaDeviceEnablePeerAccess(source_device, 0);
    const cudaError_t restore = cudaSetDevice(previous_device);
    if ((enable != cudaSuccess && enable != cudaErrorPeerAccessAlreadyEnabled)
        || restore != cudaSuccess) {
        return status_code::cuda_failure;
    }
    if (enable == cudaErrorPeerAccessAlreadyEnabled) {
        (void)cudaGetLastError();
    }
    return status_code::success;
}

status_code cuda_p2p_copy_async(
    const const_device_region &source, const mutable_device_region &destination,
    std::uint64_t bytes, cudaStream_t stream) noexcept {
    if (!valid_device_region(source) || destination.data == nullptr
        || destination.bytes == 0 || destination.device_id < 0 || bytes == 0
        || bytes > source.bytes || bytes > destination.bytes
        || source.device_id == destination.device_id || stream == nullptr) {
        return status_code::invalid_input;
    }
    return cudaMemcpyPeerAsync(destination.data, destination.device_id,
                               source.data, source.device_id,
                               static_cast<std::size_t>(bytes), stream)
            == cudaSuccess
        ? status_code::success
        : status_code::cuda_failure;
}

} // namespace cellshard::runtime_v2
