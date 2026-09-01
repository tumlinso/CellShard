#pragma once

#include "CellShard/compiler/basis/input.hpp"

namespace cellshard::compiler::basis {

struct family_basis_offer {
    local_index family = 0;
    global_id basis_id = 0;
    std::uint64_t utility = 0;
};

inline bool assign_family_bases(const family_basis_offer* offers, local_index offer_count,
                                global_id* assignment, std::uint64_t* assigned_utility,
                                local_index family_count) noexcept {
    if (offers == nullptr || assignment == nullptr || assigned_utility == nullptr) return false;
    for (local_index family = 0; family < family_count; ++family) {
        assignment[family] = 0; assigned_utility[family] = 0;
    }
    for (local_index i = 0; i < offer_count; ++i) {
        const auto& offer = offers[i];
        if (offer.family >= family_count || offer.basis_id == 0) return false;
        if (offer.utility > assigned_utility[offer.family] ||
            (offer.utility == assigned_utility[offer.family] && offer.utility != 0 &&
             (assignment[offer.family] == 0 || offer.basis_id < assignment[offer.family]))) {
            assignment[offer.family] = offer.basis_id;
            assigned_utility[offer.family] = offer.utility;
        }
    }
    return true;
}

}  // namespace cellshard::compiler::basis
