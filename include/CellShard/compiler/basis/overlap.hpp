#pragma once

#include "CellShard/compiler/basis/greedy.hpp"

#include <cstdint>

namespace cellshard::compiler::basis {

struct atom_coverage {
    global_id atom_id = 0;
    local_index first_family = 0;
    local_index family_count = 0;
};

struct coverage_view {
    const atom_coverage* atoms = nullptr;
    local_index atom_count = 0;
    const local_index* families = nullptr;
    local_index family_reference_count = 0;
    const std::uint64_t* family_frequency = nullptr;
    local_index family_count = 0;
};

inline std::uint64_t marginal_utility(const coverage_view& view, local_index atom,
                                      const bool* covered, bool& saturated) noexcept {
    if (atom >= view.atom_count || view.atoms == nullptr) return 0;
    const auto& coverage = view.atoms[atom];
    const std::uint64_t end = static_cast<std::uint64_t>(coverage.first_family) + coverage.family_count;
    if (end > view.family_reference_count || view.families == nullptr || view.family_frequency == nullptr) return 0;
    std::uint64_t total = 0;
    for (std::uint64_t i = coverage.first_family; i < end; ++i) {
        const local_index family = view.families[i];
        if (family >= view.family_count || covered[family]) continue;
        if (view.family_frequency[family] > UINT64_MAX - total) { saturated = true; return UINT64_MAX; }
        total += view.family_frequency[family];
    }
    return total;
}

inline basis_solution overlap_greedy_select(const coverage_view& view, bool* covered,
                                            local_index* output, local_index capacity) noexcept {
    basis_solution result{output, capacity, 0, 0, false};
    if (covered == nullptr || output == nullptr || capacity > view.atom_count) return result;
    while (result.count < capacity) {
        local_index best = invalid_local_index;
        std::uint64_t best_value = 0;
        for (local_index atom = 0; atom < view.atom_count; ++atom) {
            if (already_selected(result, atom)) continue;
            bool saturated = false;
            const auto value = marginal_utility(view, atom, covered, saturated);
            result.saturated = result.saturated || saturated;
            if (value > best_value || (value == best_value && value != 0 &&
                (best == invalid_local_index || view.atoms[atom].atom_id < view.atoms[best].atom_id))) {
                best = atom; best_value = value;
            }
        }
        if (best == invalid_local_index || best_value == 0) break;
        output[result.count++] = best;
        const auto& chosen = view.atoms[best];
        const std::uint64_t end = static_cast<std::uint64_t>(chosen.first_family) + chosen.family_count;
        for (std::uint64_t i = chosen.first_family; i < end; ++i) {
            const local_index family = view.families[i];
            if (family < view.family_count) covered[family] = true;
        }
    }
    return result;
}

}  // namespace cellshard::compiler::basis
