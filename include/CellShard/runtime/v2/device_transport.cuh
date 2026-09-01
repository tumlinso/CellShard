#pragma once

#include <CellShard/identity.hh>

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace cellshard::runtime_v2 {

struct const_device_region {
    const std::byte *data = nullptr;
    std::uint64_t bytes = 0;
    int device_id = -1;
};

struct mutable_device_region {
    std::byte *data = nullptr;
    std::uint64_t bytes = 0;
    int device_id = -1;
};

[[nodiscard]] constexpr bool valid_device_region(
    const const_device_region &region) noexcept {
    return region.data != nullptr && region.bytes != 0 && region.device_id >= 0;
}

[[nodiscard]] status_code same_device_alias(
    const const_device_region &source, int destination_device,
    const_device_region *out) noexcept;

[[nodiscard]] status_code prepare_cuda_p2p(int source_device,
                                           int destination_device) noexcept;

[[nodiscard]] status_code cuda_p2p_copy_async(
    const const_device_region &source, const mutable_device_region &destination,
    std::uint64_t bytes, cudaStream_t caller_stream) noexcept;

static_assert(std::is_trivially_copyable_v<const_device_region>);
static_assert(std::is_trivially_copyable_v<mutable_device_region>);

} // namespace cellshard::runtime_v2
