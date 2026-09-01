#pragma once

#include <CellShard/identity.hh>

#include <cstdint>
#include <type_traits>

namespace cellshard::runtime_v2 {

struct numabraid_plan_request {
    int source_device = -1;
    int destination_device = -1;
    std::uint64_t bytes = 0;
};

struct numabraid_transport_plan {
    content_digest topology_identity{};
    int source_device = -1;
    int relay_device = -1;
    int destination_device = -1;
    std::uint64_t capacity_bytes = 0;
    std::uint64_t chunk_bytes = 0;
    std::uint32_t staging_buffers = 0;
    std::uint64_t provider_cookie = 0;
};

enum class numabraid_transfer_state : std::uint8_t {
    invalid = 0,
    pending,
    complete,
    failed,
};

struct numabraid_transport_ops {
    status_code (*prepare)(void *, const numabraid_plan_request &,
                           numabraid_transport_plan *) noexcept = nullptr;
    status_code (*launch)(void *, const numabraid_transport_plan &, const void *,
                          void *, std::uint64_t) noexcept = nullptr;
    numabraid_transfer_state (*query)(void *,
                                      const numabraid_transport_plan &) noexcept = nullptr;
    status_code (*synchronize)(void *,
                               const numabraid_transport_plan &) noexcept = nullptr;
};

struct numabraid_transport_ref {
    void *context = nullptr;
    const numabraid_transport_ops *ops = nullptr;
    std::uint32_t abi_version = 0;
    content_digest topology_identity{};
};

[[nodiscard]] constexpr bool valid_numabraid_transport_plan(
    const numabraid_transport_plan &plan) noexcept {
    return plan.topology_identity.algorithm != digest_algorithm::none
        && valid_content_digest(plan.topology_identity)
        && plan.source_device >= 0 && plan.relay_device >= 0
        && plan.destination_device >= 0
        && plan.source_device != plan.relay_device
        && plan.relay_device != plan.destination_device
        && plan.source_device != plan.destination_device
        && plan.capacity_bytes != 0 && plan.chunk_bytes != 0
        && plan.chunk_bytes <= plan.capacity_bytes && plan.staging_buffers != 0
        && plan.provider_cookie != 0;
}

[[nodiscard]] status_code plan_numabraid_transport(
    numabraid_transport_ref provider, const numabraid_plan_request &request,
    numabraid_transport_plan *out) noexcept;

[[nodiscard]] status_code launch_numabraid_transport(
    numabraid_transport_ref provider, const numabraid_transport_plan &plan,
    const void *source, void *destination, std::uint64_t bytes) noexcept;

static_assert(std::is_trivially_copyable_v<numabraid_transport_plan>);
static_assert(std::is_trivially_copyable_v<numabraid_transport_ref>);

} // namespace cellshard::runtime_v2
