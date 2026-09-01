#pragma once

#include "CellShard/compiler/composition/superatom/lifecycle.hpp"

namespace cellshard::compiler::composition::superatom {

struct lineage_membership {
    global_id superatom_id = 0;
    global_id lineage_id = 0;
    global_id parent_superatom_id = 0;
    global_id generation = 0;
    local_index first_basis = 0;
    local_index basis_count = 0;
};
struct lineage_view {
    const lineage_membership* records = nullptr;
    local_index record_count = 0;
    const global_id* basis_ids = nullptr;
    local_index basis_reference_count = 0;
};

inline bool validate_lineage(const lineage_view& view) noexcept {
    if ((view.record_count != 0 && view.records == nullptr) ||
        (view.basis_reference_count != 0 && view.basis_ids == nullptr)) return false;
    for (local_index i = 0; i < view.record_count; ++i) {
        const auto& record = view.records[i];
        if (record.superatom_id == 0 || record.lineage_id == 0 || record.basis_count == 0) return false;
        if (record.generation == 0 && record.parent_superatom_id != 0) return false;
        if (record.generation != 0 && record.parent_superatom_id == 0) return false;
        const std::uint64_t end = static_cast<std::uint64_t>(record.first_basis) + record.basis_count;
        if (end > view.basis_reference_count) return false;
        global_id previous = 0;
        for (std::uint64_t j = record.first_basis; j < end; ++j) {
            if (view.basis_ids[j] == 0 || (j != record.first_basis && view.basis_ids[j] <= previous)) return false;
            previous = view.basis_ids[j];
        }
    }
    return true;
}

inline bool belongs_to_basis(const lineage_view& view, local_index record,
                             global_id basis_id) noexcept {
    if (record >= view.record_count || basis_id == 0) return false;
    const auto& item = view.records[record];
    local_index low = item.first_basis;
    local_index high = item.first_basis + item.basis_count;
    while (low < high) {
        const local_index middle = low + (high - low) / 2;
        if (view.basis_ids[middle] < basis_id) low = middle + 1;
        else high = middle;
    }
    return low < item.first_basis + item.basis_count && view.basis_ids[low] == basis_id;
}

}  // namespace cellshard::compiler::composition::superatom
