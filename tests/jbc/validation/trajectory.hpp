#pragma once
#include "fixtures.hpp"
#include <limits>
namespace cellshard::jbc::validation {
inline constexpr std::uint32_t no_parent = std::numeric_limits<std::uint32_t>::max();
struct trajectory_node { global_id state_id = 0; std::uint32_t parent = no_parent; std::uint32_t prefix_length = 0; global_id branch_id = 0; };
inline bool valid_trajectory(const trajectory_node* nodes, std::uint32_t count) noexcept {
    if (nodes == nullptr || count == 0) return false;
    for (std::uint32_t i = 0; i < count; ++i) {
        const auto& node = nodes[i];
        if (node.state_id == 0 || node.branch_id == 0) return false;
        if (node.parent == no_parent) { if (i != 0 || node.prefix_length != 0) return false; continue; }
        if (node.parent >= i || node.prefix_length <= nodes[node.parent].prefix_length) return false;
    }
    return true;
}
}  // namespace cellshard::jbc::validation
