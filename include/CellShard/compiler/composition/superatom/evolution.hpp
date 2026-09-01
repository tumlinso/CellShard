#pragma once

#include "CellShard/compiler/composition/superatom/lifecycle.hpp"

namespace cellshard::compiler::composition::superatom {

struct evolution_evidence {
    std::uint64_t old_utility = 0;
    std::uint64_t new_utility = 0;
    bool independently_verified = false;
};

inline transition_result split(const superatom_record& parent,
                               promotion_identity left_identity,
                               promotion_identity right_identity,
                               const evolution_evidence& evidence,
                               superatom_record& left,
                               superatom_record& right) noexcept {
    if (parent.state != lifecycle_state::promoted) return transition_result::wrong_state;
    if (!evidence.independently_verified) return transition_result::not_verified;
    if (evidence.new_utility <= evidence.old_utility) return transition_result::not_profitable;
    if (parent.generation == UINT64_MAX) return transition_result::generation_exhausted;
    if (left_identity.superatom_id == 0 || right_identity.superatom_id == 0 ||
        left_identity.superatom_id == right_identity.superatom_id ||
        left_identity.structure_epoch != parent.identity.structure_epoch ||
        right_identity.structure_epoch != parent.identity.structure_epoch) return transition_result::invalid_identity;
    left = {left_identity, parent.lineage_id, parent.generation + 1, lifecycle_state::promoted};
    right = {right_identity, parent.lineage_id, parent.generation + 1, lifecycle_state::promoted};
    return transition_result::applied;
}

inline transition_result merge(const superatom_record& left,
                               const superatom_record& right,
                               promotion_identity merged_identity,
                               const evolution_evidence& evidence,
                               superatom_record& merged) noexcept {
    if (left.state != lifecycle_state::promoted || right.state != lifecycle_state::promoted) return transition_result::wrong_state;
    if (!evidence.independently_verified) return transition_result::not_verified;
    if (evidence.new_utility <= evidence.old_utility) return transition_result::not_profitable;
    if (left.generation == UINT64_MAX || right.generation == UINT64_MAX) return transition_result::generation_exhausted;
    if (merged_identity.superatom_id == 0 || left.identity.structure_epoch != right.identity.structure_epoch ||
        merged_identity.structure_epoch != left.identity.structure_epoch) return transition_result::invalid_identity;
    const global_id lineage = left.lineage_id < right.lineage_id ? left.lineage_id : right.lineage_id;
    const global_id generation = (left.generation > right.generation ? left.generation : right.generation) + 1;
    merged = {merged_identity, lineage, generation, lifecycle_state::promoted};
    return transition_result::applied;
}

}  // namespace cellshard::compiler::composition::superatom
