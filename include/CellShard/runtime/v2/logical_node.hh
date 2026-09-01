#pragma once

#include <CellShard/identity.hh>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace cellshard::runtime_v2 {

struct numa_ownership {
    std::uint32_t numa_id = std::numeric_limits<std::uint32_t>::max();
    std::uint32_t cpu_count = 0;
    std::uint64_t local_capacity_bytes = 0;
    content_digest cpu_set_identity{};
};

struct logical_node {
    std::uint32_t id = 0;
    numa_ownership owner{};
};

[[nodiscard]] constexpr bool valid_numa_ownership(
    const numa_ownership &owner) noexcept {
    return owner.numa_id != std::numeric_limits<std::uint32_t>::max()
        && owner.cpu_count != 0 && owner.local_capacity_bytes != 0
        && owner.cpu_set_identity.algorithm != digest_algorithm::none
        && valid_content_digest(owner.cpu_set_identity);
}

[[nodiscard]] inline bool valid_logical_nodes(
    array_view<logical_node> nodes) noexcept {
    if (nodes.empty()) {
        return false;
    }
    for (std::size_t i = 0; i < nodes.size; ++i) {
        if (nodes[i].id == 0 || !valid_numa_ownership(nodes[i].owner)) {
            return false;
        }
        for (std::size_t j = 0; j < i; ++j) {
            if (nodes[i].id == nodes[j].id
                || nodes[i].owner.numa_id == nodes[j].owner.numa_id
                || nodes[i].owner.cpu_set_identity
                       == nodes[j].owner.cpu_set_identity) {
                return false;
            }
        }
    }
    return true;
}

static_assert(std::is_trivially_copyable_v<numa_ownership>);
static_assert(std::is_trivially_copyable_v<logical_node>);

} // namespace cellshard::runtime_v2
