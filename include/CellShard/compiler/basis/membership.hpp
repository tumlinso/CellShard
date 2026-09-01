#pragma once

#include "CellShard/compiler/basis/input.hpp"

namespace cellshard::compiler::basis {

struct atom_membership {
    global_id atom_id = 0;
    global_id redundancy_class_id = 0;
    local_index first_basis = 0;
    local_index basis_count = 0;
};

struct membership_view {
    const atom_membership* atoms = nullptr;
    local_index atom_count = 0;
    const global_id* basis_ids = nullptr;
    local_index basis_reference_count = 0;
};

inline bool valid_memberships(const membership_view& view) noexcept {
    if ((view.atom_count != 0 && view.atoms == nullptr) ||
        (view.basis_reference_count != 0 && view.basis_ids == nullptr)) return false;
    for (local_index i = 0; i < view.atom_count; ++i) {
        const auto& atom = view.atoms[i];
        const std::uint64_t end = static_cast<std::uint64_t>(atom.first_basis) + atom.basis_count;
        if (atom.atom_id == 0 || atom.redundancy_class_id == 0 || end > view.basis_reference_count) return false;
        global_id previous = 0;
        for (std::uint64_t j = atom.first_basis; j < end; ++j) {
            if (view.basis_ids[j] == 0 || (j != atom.first_basis && view.basis_ids[j] <= previous)) return false;
            previous = view.basis_ids[j];
        }
    }
    return true;
}

inline global_id canonical_redundant_atom(const membership_view& view,
                                          global_id redundancy_class_id) noexcept {
    global_id best = 0;
    for (local_index i = 0; i < view.atom_count; ++i) {
        const auto& atom = view.atoms[i];
        if (atom.redundancy_class_id == redundancy_class_id && (best == 0 || atom.atom_id < best)) best = atom.atom_id;
    }
    return best;
}

}  // namespace cellshard::compiler::basis
