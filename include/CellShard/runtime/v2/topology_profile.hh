#pragma once

#include <CellShard/identity.hh>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace cellshard::runtime_v2 {

enum class topology_node_kind : std::uint8_t {
    invalid = 0,
    host_numa,
    cuda_device,
};

enum class topology_link_kind : std::uint8_t {
    invalid = 0,
    host_memory,
    pcie,
    nvlink,
};

struct topology_node {
    std::uint32_t id = 0;
    topology_node_kind kind = topology_node_kind::invalid;
    std::int32_t numa_node = -1;
    std::int32_t device_ordinal = -1;
    std::uint64_t capacity_bytes = 0;
};

struct topology_link {
    std::uint32_t source = 0;
    std::uint32_t destination = 0;
    topology_link_kind kind = topology_link_kind::invalid;
    std::uint64_t bandwidth_bytes_per_second = 0;
    std::uint64_t latency_nanoseconds = 0;
    bool direct_access = false;
};

// A topology profile is an immutable, caller-owned snapshot. Runtime decisions
// bind its identity, rather than rediscovering machine topology during launch.
struct topology_profile {
    content_digest identity{};
    array_view<topology_node> nodes{};
    array_view<topology_link> links{};
};

[[nodiscard]] constexpr bool valid_topology_node(
    const topology_node &node) noexcept {
    if (node.id == 0 || node.capacity_bytes == 0) {
        return false;
    }
    switch (node.kind) {
    case topology_node_kind::host_numa:
        return node.numa_node >= 0 && node.device_ordinal == -1;
    case topology_node_kind::cuda_device:
        return node.numa_node >= 0 && node.device_ordinal >= 0;
    case topology_node_kind::invalid:
        return false;
    }
    return false;
}

[[nodiscard]] constexpr bool valid_topology_link_kind(
    topology_link_kind kind) noexcept {
    return kind == topology_link_kind::host_memory
        || kind == topology_link_kind::pcie
        || kind == topology_link_kind::nvlink;
}

[[nodiscard]] inline bool valid_topology_profile(
    const topology_profile &profile) noexcept {
    if (profile.identity.algorithm == digest_algorithm::none
        || !valid_content_digest(profile.identity) || profile.nodes.empty()) {
        return false;
    }
    for (std::size_t i = 0; i < profile.nodes.size; ++i) {
        if (!valid_topology_node(profile.nodes[i])) {
            return false;
        }
        for (std::size_t j = 0; j < i; ++j) {
            if (profile.nodes[i].id == profile.nodes[j].id) {
                return false;
            }
        }
    }
    for (std::size_t i = 0; i < profile.links.size; ++i) {
        const auto &link = profile.links[i];
        if (link.source == link.destination
            || !valid_topology_link_kind(link.kind)
            || link.bandwidth_bytes_per_second == 0
            || link.latency_nanoseconds == 0) {
            return false;
        }
        bool source_found = false;
        bool destination_found = false;
        for (std::size_t node = 0; node < profile.nodes.size; ++node) {
            source_found |= profile.nodes[node].id == link.source;
            destination_found |= profile.nodes[node].id == link.destination;
        }
        if (!source_found || !destination_found) {
            return false;
        }
        for (std::size_t j = 0; j < i; ++j) {
            if (profile.links[j].source == link.source
                && profile.links[j].destination == link.destination) {
                return false;
            }
        }
    }
    return true;
}

static_assert(std::is_trivially_copyable_v<topology_node>);
static_assert(std::is_trivially_copyable_v<topology_link>);
static_assert(std::is_trivially_copyable_v<topology_profile>);

} // namespace cellshard::runtime_v2
