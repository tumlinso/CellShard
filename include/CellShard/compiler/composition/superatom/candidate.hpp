#pragma once

#include <cstdint>
#include <limits>

namespace cellshard::compiler::composition::superatom {

using global_id = std::uint64_t;
using local_index = std::uint32_t;
inline constexpr local_index invalid_local_index = std::numeric_limits<local_index>::max();

struct promotion_identity {
    global_id superatom_id = 0;
    global_id composition_id = 0;
    global_id structure_epoch = 0;
    global_id basis_id = 0;
};

struct candidate {
    promotion_identity identity{};
    local_index first_atom = 0;
    local_index atom_count = 0;
};

struct candidate_view {
    const candidate* candidates = nullptr;
    local_index candidate_count = 0;
    const global_id* atom_ids = nullptr;
    local_index atom_reference_count = 0;
};

enum class candidate_error : std::uint8_t { none, missing_data, invalid_identity, invalid_range, invalid_atoms };

inline candidate_error validate_candidates(const candidate_view& view) noexcept {
    if ((view.candidate_count != 0 && view.candidates == nullptr) ||
        (view.atom_reference_count != 0 && view.atom_ids == nullptr)) return candidate_error::missing_data;
    for (local_index i = 0; i < view.candidate_count; ++i) {
        const auto& item = view.candidates[i];
        if (item.identity.superatom_id == 0 || item.identity.composition_id == 0 ||
            item.identity.structure_epoch == 0 || item.identity.basis_id == 0) return candidate_error::invalid_identity;
        const std::uint64_t end = static_cast<std::uint64_t>(item.first_atom) + item.atom_count;
        if (item.atom_count < 2 || end > view.atom_reference_count) return candidate_error::invalid_range;
        global_id previous = 0;
        for (std::uint64_t j = item.first_atom; j < end; ++j) {
            if (view.atom_ids[j] == 0 || (j != item.first_atom && view.atom_ids[j] <= previous)) return candidate_error::invalid_atoms;
            previous = view.atom_ids[j];
        }
    }
    return candidate_error::none;
}

}  // namespace cellshard::compiler::composition::superatom
