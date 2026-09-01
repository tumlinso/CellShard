#pragma once
#include "fixtures.hpp"
#include <array>
namespace cellshard::jbc::validation {
struct route_evidence {
    global_id route_id = 0;
    global_id source_node_id = 0;
    global_id destination_node_id = 0;
    global_id topology_epoch = 0;
    const global_id* hop_ids = nullptr;
    std::uint32_t hop_count = 0;
    std::uint64_t requested_bytes = 0;
    std::uint64_t delivered_bytes = 0;
    std::array<std::uint64_t, 4> source_digest{};
    std::array<std::uint64_t, 4> destination_digest{};
    bool resource_reserved = false;
};
inline bool valid_route(const route_evidence& route) noexcept {
    if (route.route_id == 0 || route.source_node_id == 0 || route.destination_node_id == 0 ||
        route.source_node_id == route.destination_node_id || route.topology_epoch == 0 ||
        route.hop_ids == nullptr || route.hop_count == 0 || route.hop_count > 64 || !route.resource_reserved) return false;
    if (route.requested_bytes == 0 || route.requested_bytes != route.delivered_bytes ||
        route.source_digest != route.destination_digest) return false;
    for (std::uint32_t i = 0; i < route.hop_count; ++i)
        if (route.hop_ids[i] == 0 || (i != 0 && route.hop_ids[i] == route.hop_ids[i - 1])) return false;
    return true;
}
}  // namespace cellshard::jbc::validation
