#pragma once

#include "CellShard/compiler/basis/baseline.hpp"

namespace cellshard::compiler::basis {

struct refinement_candidate {
    global_id atom_id = 0;
    std::uint64_t add_gain = 0;
    std::uint64_t remove_loss = 0;
};

inline bool refine_add_remove(const refinement_candidate* candidates,
                              local_index candidate_count, bool* selected,
                              local_index selection_capacity,
                              local_index& selected_count) noexcept {
    if (candidates == nullptr || selected == nullptr) return false;
    bool changed = false;
    for (local_index i = 0; i < candidate_count; ++i) {
        if (selected[i] && candidates[i].remove_loss == 0) {
            selected[i] = false;
            --selected_count;
            changed = true;
        }
    }
    while (selected_count < selection_capacity) {
        local_index best = invalid_local_index;
        for (local_index i = 0; i < candidate_count; ++i) {
            if (selected[i] || candidates[i].add_gain == 0) continue;
            if (best == invalid_local_index || candidates[i].add_gain > candidates[best].add_gain ||
                (candidates[i].add_gain == candidates[best].add_gain &&
                 candidates[i].atom_id < candidates[best].atom_id)) best = i;
        }
        if (best == invalid_local_index) break;
        selected[best] = true;
        ++selected_count;
        changed = true;
    }
    return changed;
}

}  // namespace cellshard::compiler::basis
