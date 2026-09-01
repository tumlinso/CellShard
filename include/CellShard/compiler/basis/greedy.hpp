#pragma once

#include "CellShard/compiler/basis/baseline.hpp"
#include "CellShard/compiler/basis/utility.hpp"

namespace cellshard::compiler::basis {

inline bool already_selected(const basis_solution& solution, local_index atom) noexcept {
    for (local_index i = 0; i < solution.count; ++i) {
        if (solution.selected_atoms[i] == atom) {
            return true;
        }
    }
    return false;
}

inline basis_solution greedy_select(const atom_utility* utilities,
                                    local_index utility_count,
                                    const utility_weights& weights,
                                    local_index* output,
                                    local_index capacity) noexcept {
    basis_solution solution{output, capacity, 0, 0, false};
    if (utilities == nullptr || output == nullptr) {
        return solution;
    }
    while (solution.count < capacity && solution.count < utility_count) {
        local_index best = invalid_local_index;
        scored_utility best_score{};
        for (local_index i = 0; i < utility_count; ++i) {
            if (already_selected(solution, i)) {
                continue;
            }
            const auto candidate_score = score(utilities[i], weights);
            if (best == invalid_local_index || candidate_score.value > best_score.value ||
                (candidate_score.value == best_score.value &&
                 utilities[i].atom_id < utilities[best].atom_id)) {
                best = i;
                best_score = candidate_score;
            }
        }
        if (best == invalid_local_index || best_score.value == 0) {
            break;
        }
        output[solution.count++] = best;
        solution.saturated = solution.saturated || best_score.saturated;
    }
    return solution;
}

}  // namespace cellshard::compiler::basis
