#pragma once

#include "CellShard/compiler/composition/superatom/statistics.hpp"

namespace cellshard::compiler::composition::superatom {

struct lifecycle_cost {
    std::uint64_t build = 0;
    std::uint64_t storage = 0;
    std::uint64_t maintenance = 0;
    std::uint64_t invalidation = 0;
};
struct promotion_value { std::uint64_t benefit = 0; std::uint64_t cost = 0; bool saturated = false; };

inline promotion_value evaluate_value(const composition_statistics& statistics,
                                      std::uint64_t savings_per_use,
                                      const lifecycle_cost& costs) noexcept {
    promotion_value result{};
    if (savings_per_use != 0 && statistics.frequency > UINT64_MAX / savings_per_use) {
        result.benefit = UINT64_MAX; result.saturated = true;
    } else result.benefit = statistics.frequency * savings_per_use;
    result.cost = saturating_add(costs.build, costs.storage, result.saturated);
    result.cost = saturating_add(result.cost, costs.maintenance, result.saturated);
    result.cost = saturating_add(result.cost, costs.invalidation, result.saturated);
    result.saturated = result.saturated || statistics.saturated;
    return result;
}

inline bool promotion_profitable(const promotion_value& value) noexcept {
    return !value.saturated && value.benefit > value.cost;
}

}  // namespace cellshard::compiler::composition::superatom
