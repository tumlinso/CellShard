#pragma once

#include "CellShard/compiler/composition/superatom/cost.hpp"

namespace cellshard::compiler::composition::superatom {

enum class lifecycle_state : std::uint8_t { candidate, promoted, demoted };
struct superatom_record {
    promotion_identity identity{};
    global_id lineage_id = 0;
    global_id generation = 0;
    lifecycle_state state = lifecycle_state::candidate;
};
struct promotion_evidence {
    global_id composition_id = 0;
    global_id structure_epoch = 0;
    promotion_value independently_verified_value{};
    bool composition_verified = false;
};
enum class transition_result : std::uint8_t { applied, invalid_identity, stale_structure, not_verified, not_profitable, wrong_state };

inline transition_result promote(superatom_record& record,
                                 const promotion_evidence& evidence) noexcept {
    if (record.identity.superatom_id == 0 || record.identity.composition_id == 0) return transition_result::invalid_identity;
    if (record.state != lifecycle_state::candidate) return transition_result::wrong_state;
    if (record.identity.structure_epoch != evidence.structure_epoch) return transition_result::stale_structure;
    if (!evidence.composition_verified || evidence.composition_id != record.identity.composition_id) return transition_result::not_verified;
    if (!promotion_profitable(evidence.independently_verified_value)) return transition_result::not_profitable;
    record.state = lifecycle_state::promoted;
    record.generation = 1;
    record.lineage_id = record.identity.superatom_id;
    return transition_result::applied;
}

}  // namespace cellshard::compiler::composition::superatom
