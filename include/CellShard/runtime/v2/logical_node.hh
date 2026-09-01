#pragma once

#include <CellShard/identity.hh>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace cellshard::runtime_v2 {

struct numa_ownership {
    std::uint32_t numa_id = std::numeric_limits<std::uint32_t>::max();
    std::uint32_t first_cpu = 0;
    std::uint32_t cpu_count = 0;
    std::uint64_t local_capacity_bytes = 0;
};

struct logical_node {
    std::uint32_t id = 0;
    numa_ownership owner{};
};

[[nodiscard]] constexpr bool valid_numa_ownership(
    const numa_ownership &owner) noexcept {
    return owner.numa_id != std::numeric_limits<std::uint32_t>::max()
        && owner.cpu_count != 0 && owner.local_capacity_bytes != 0
        && owner.first_cpu <= std::numeric_limits<std::uint32_t>::max()
                               - owner.cpu_count;
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
        const std::uint32_t begin = nodes[i].owner.first_cpu;
        const std::uint32_t end = begin + nodes[i].owner.cpu_count;
        for (std::size_t j = 0; j < i; ++j) {
            if (nodes[i].id == nodes[j].id
                || nodes[i].owner.numa_id == nodes[j].owner.numa_id) {
                return false;
            }
            const std::uint32_t previous_begin = nodes[j].owner.first_cpu;
            const std::uint32_t previous_end =
                previous_begin + nodes[j].owner.cpu_count;
            if (begin < previous_end && previous_begin < end) {
                return false;
            }
        }
    }
    return true;
}

static_assert(std::is_trivially_copyable_v<numa_ownership>);
static_assert(std::is_trivially_copyable_v<logical_node>);

} // namespace cellshard::runtime_v2
