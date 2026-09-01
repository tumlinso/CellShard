#include <CellShard/runtime/v2/host_staged_transport.cuh>

namespace cellshard::runtime_v2 {

status_code cuda_host_staged_copy_async(
    const const_device_region &source, const mutable_device_region &destination,
    pinned_staging_lease staging, std::uint64_t bytes,
    cudaStream_t source_stream, cudaStream_t destination_stream,
    cudaEvent_t source_complete_event) noexcept {
    if (!valid_device_region(source) || destination.data == nullptr
        || destination.device_id < 0 || destination.bytes == 0
        || source.device_id == destination.device_id || staging.data == nullptr
        || bytes == 0 || bytes > source.bytes || bytes > destination.bytes
        || bytes > staging.bytes || source_stream == nullptr
        || destination_stream == nullptr || source_complete_event == nullptr) {
        return status_code::invalid_input;
    }
    int previous = -1;
    if (cudaGetDevice(&previous) != cudaSuccess
        || cudaSetDevice(source.device_id) != cudaSuccess
        || cudaMemcpyAsync(staging.data, source.data,
                           static_cast<std::size_t>(bytes),
                           cudaMemcpyDeviceToHost, source_stream) != cudaSuccess
        || cudaEventRecord(source_complete_event, source_stream) != cudaSuccess
        || cudaSetDevice(destination.device_id) != cudaSuccess
        || cudaStreamWaitEvent(destination_stream, source_complete_event, 0)
               != cudaSuccess
        || cudaMemcpyAsync(destination.data, staging.data,
                           static_cast<std::size_t>(bytes),
                           cudaMemcpyHostToDevice, destination_stream)
               != cudaSuccess) {
        (void)cudaSetDevice(previous);
        return status_code::cuda_failure;
    }
    return cudaSetDevice(previous) == cudaSuccess ? status_code::success
                                                   : status_code::cuda_failure;
}

} // namespace cellshard::runtime_v2
