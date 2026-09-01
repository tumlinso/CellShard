#pragma once

#include "CellShard/compiler/basis/overlap.hpp"

namespace cellshard::compiler::basis {

inline basis_solution weighted_set_cover(const coverage_view& view,
                                         const std::uint64_t* atom_cost,
                                         bool* covered, local_index* output,
                                         local_index capacity) noexcept {
    basis_solution result{output, capacity, 0, 0, false};
    if (atom_cost == nullptr || covered == nullptr || output == nullptr) return result;
    while (result.count < capacity) {
        local_index best = invalid_local_index;
        workload_weight best_ratio{};
        for (local_index atom = 0; atom < view.atom_count; ++atom) {
            if (already_selected(result, atom) || atom_cost[atom] == 0) continue;
            bool saturated = false;
            const auto gain = marginal_utility(view, atom, covered, saturated);
            const workload_weight ratio{gain, atom_cost[atom]};
            if (gain != 0 && (best == invalid_local_index || compare_weight(ratio, best_ratio) > 0 ||
                (compare_weight(ratio, best_ratio) == 0 && view.atoms[atom].atom_id < view.atoms[best].atom_id))) {
                best = atom; best_ratio = ratio;
            }
            result.saturated = result.saturated || saturated;
        }
        if (best == invalid_local_index) break;
        output[result.count++] = best;
        const auto& chosen = view.atoms[best];
        const std::uint64_t end = static_cast<std::uint64_t>(chosen.first_family) + chosen.family_count;
        for (std::uint64_t i = chosen.first_family; i < end; ++i) {
            if (view.families[i] < view.family_count) covered[view.families[i]] = true;
        }
    }
    return result;
}

}  // namespace cellshard::compiler::basis
