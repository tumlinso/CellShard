#pragma once

#include <CellShard/runtime/v2/device_transport.cuh>
#include <CellShard/runtime/v2/pinned_staging_pool.hh>

#include <cuda_runtime_api.h>

namespace cellshard::runtime_v2 {

struct numa_transfer_record {
    std::uint32_t source_numa = 0;
    std::uint32_t destination_numa = 0;
    std::uint64_t bytes = 0;
    bool crosses_numa_fabric = false;
};

[[nodiscard]] status_code numa_copy_exact(
    const std::byte *source, std::uint64_t source_bytes,
    std::uint32_t source_numa, std::byte *destination,
    std::uint64_t destination_bytes, std::uint32_t destination_numa,
    std::uint64_t bytes, numa_transfer_record *record) noexcept;

[[nodiscard]] status_code cuda_host_staged_copy_async(
    const const_device_region &source, const mutable_device_region &destination,
    pinned_staging_lease staging, std::uint64_t bytes,
    cudaStream_t source_stream, cudaStream_t destination_stream,
    cudaEvent_t source_complete_event) noexcept;

} // namespace cellshard::runtime_v2
