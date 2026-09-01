#pragma once

#include <CellShard/identity.hh>

#include <cstdint>
#include <type_traits>

namespace cellshard::runtime_v2 {

enum class storage_endpoint_kind : std::uint8_t {
    invalid = 0,
    local_file,
    block_device,
    remote_object,
};

enum class storage_access_mode : std::uint8_t {
    invalid = 0,
    buffered,
    memory_mapped,
    direct,
};

struct storage_access_policy {
    storage_access_mode mode = storage_access_mode::invalid;
    std::uint32_t required_alignment = 0;
    std::uint32_t max_in_flight = 0;
    std::uint64_t preferred_request_bytes = 0;
    bool read_only = true;
};

struct storage_endpoint {
    source_provider_id provider{};
    source_location_id location{};
    storage_endpoint_kind kind = storage_endpoint_kind::invalid;
    std::uint32_t logical_node_id = 0;
    storage_access_policy access{};
};

[[nodiscard]] constexpr bool valid_storage_access_policy(
    const storage_access_policy &policy) noexcept {
    if (policy.mode == storage_access_mode::invalid
        || policy.max_in_flight == 0 || policy.preferred_request_bytes == 0
        || !policy.read_only) {
        return false;
    }
    const bool power_of_two = policy.required_alignment != 0
        && (policy.required_alignment & (policy.required_alignment - 1)) == 0;
    if (!power_of_two
        || policy.preferred_request_bytes % policy.required_alignment != 0) {
        return false;
    }
    return policy.mode != storage_access_mode::direct
        || policy.required_alignment >= 512;
}

[[nodiscard]] constexpr bool valid_storage_endpoint(
    const storage_endpoint &endpoint) noexcept {
    return endpoint.provider.valid() && endpoint.location.valid()
        && endpoint.kind != storage_endpoint_kind::invalid
        && endpoint.logical_node_id != 0
        && valid_storage_access_policy(endpoint.access);
}

static_assert(std::is_trivially_copyable_v<storage_access_policy>);
static_assert(std::is_trivially_copyable_v<storage_endpoint>);

} // namespace cellshard::runtime_v2
