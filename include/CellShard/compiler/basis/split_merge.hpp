#pragma once

#include "CellShard/compiler/basis/input.hpp"

namespace cellshard::compiler::basis {

enum class refinement_kind : std::uint8_t { none, split, merge };

struct split_merge_candidate {
    refinement_kind kind = refinement_kind::none;
    global_id first_basis_id = 0;
    global_id second_basis_id = 0;
    std::uint64_t old_cost = 0;
    std::uint64_t new_cost = 0;
};

inline bool valid(const split_merge_candidate& candidate) noexcept {
    if (candidate.kind == refinement_kind::none || candidate.first_basis_id == 0 ||
        candidate.new_cost >= candidate.old_cost) return false;
    return candidate.kind == refinement_kind::split
               ? candidate.second_basis_id != 0 && candidate.second_basis_id != candidate.first_basis_id
               : candidate.second_basis_id != 0;
}

inline const split_merge_candidate* best_refinement(const split_merge_candidate* candidates,
                                                     local_index count) noexcept {
    const split_merge_candidate* best = nullptr;
    std::uint64_t best_gain = 0;
    for (local_index i = 0; i < count; ++i) {
        if (!valid(candidates[i])) continue;
        const auto gain = candidates[i].old_cost - candidates[i].new_cost;
        if (best == nullptr || gain > best_gain ||
            (gain == best_gain && candidates[i].first_basis_id < best->first_basis_id)) {
            best = &candidates[i]; best_gain = gain;
        }
    }
    return best;
}

}  // namespace cellshard::compiler::basis
