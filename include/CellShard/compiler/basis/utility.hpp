#pragma once

#include "CellShard/compiler/basis/input.hpp"

#include <array>
#include <cstdint>
#include <limits>

namespace cellshard::compiler::basis {

enum class utility_dimension : std::uint8_t {
    latency_saved,
    bytes_saved,
    launches_saved,
    reuse_value,
    count
};

inline constexpr std::size_t utility_dimension_count =
    static_cast<std::size_t>(utility_dimension::count);

struct atom_utility {
    global_id atom_id = 0;
    std::array<std::uint64_t, utility_dimension_count> benefit{};
};

struct utility_weights {
    std::array<std::uint64_t, utility_dimension_count> value{};
};

struct scored_utility {
    std::uint64_t value = 0;
    bool saturated = false;
};

inline scored_utility score(const atom_utility& utility,
                            const utility_weights& weights) noexcept {
    std::uint64_t total = 0;
    for (std::size_t i = 0; i < utility_dimension_count; ++i) {
        const std::uint64_t benefit = utility.benefit[i];
        const std::uint64_t weight = weights.value[i];
        if (benefit != 0 && weight > std::numeric_limits<std::uint64_t>::max() / benefit) {
            return {std::numeric_limits<std::uint64_t>::max(), true};
        }
        const std::uint64_t term = benefit * weight;
        if (term > std::numeric_limits<std::uint64_t>::max() - total) {
            return {std::numeric_limits<std::uint64_t>::max(), true};
        }
        total += term;
    }
    return {total, false};
}

}  // namespace cellshard::compiler::basis
