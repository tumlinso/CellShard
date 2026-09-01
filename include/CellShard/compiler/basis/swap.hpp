#pragma once

#include "CellShard/compiler/basis/input.hpp"

namespace cellshard::compiler::basis {

struct swap_proposal {
    local_index remove_atom = invalid_local_index;
    const local_index* add_atoms = nullptr;
    local_index add_count = 0;
    std::uint64_t removed_utility = 0;
    std::uint64_t added_utility = 0;
};

inline bool improving(const swap_proposal& proposal) noexcept {
    return proposal.remove_atom != invalid_local_index && proposal.add_atoms != nullptr &&
           proposal.add_count != 0 && proposal.added_utility > proposal.removed_utility;
}

inline bool apply_swap(const swap_proposal& proposal, bool* selected,
                       local_index atom_count) noexcept {
    if (!improving(proposal) || selected == nullptr || proposal.remove_atom >= atom_count) return false;
    for (local_index i = 0; i < proposal.add_count; ++i) {
        if (proposal.add_atoms[i] >= atom_count || proposal.add_atoms[i] == proposal.remove_atom) return false;
        for (local_index j = 0; j < i; ++j) if (proposal.add_atoms[j] == proposal.add_atoms[i]) return false;
    }
    selected[proposal.remove_atom] = false;
    for (local_index i = 0; i < proposal.add_count; ++i) selected[proposal.add_atoms[i]] = true;
    return true;
}

}  // namespace cellshard::compiler::basis
