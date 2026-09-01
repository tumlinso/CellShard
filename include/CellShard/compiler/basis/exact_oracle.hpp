#pragma once

#include "CellShard/compiler/basis/overlap.hpp"

namespace cellshard::compiler::basis {

inline constexpr local_index exact_oracle_max_atoms = 20;

enum class oracle_status : std::uint8_t { success, invalid_input, instance_too_large };
struct oracle_result { oracle_status status = oracle_status::invalid_input; std::uint64_t mask = 0; std::uint64_t utility = 0; };

inline oracle_result exact_coverage_oracle(const coverage_view& view,
                                           const std::uint64_t* atom_cost,
                                           bool* covered_scratch) noexcept {
    if (view.atom_count > exact_oracle_max_atoms) return {oracle_status::instance_too_large, 0, 0};
    if (view.atoms == nullptr || view.families == nullptr || view.family_frequency == nullptr ||
        atom_cost == nullptr || covered_scratch == nullptr) return {};
    oracle_result best{oracle_status::success, 0, 0};
    const std::uint64_t subset_count = UINT64_C(1) << view.atom_count;
    for (std::uint64_t mask = 0; mask < subset_count; ++mask) {
        for (local_index f = 0; f < view.family_count; ++f) covered_scratch[f] = false;
        std::uint64_t cost = 0;
        for (local_index atom = 0; atom < view.atom_count; ++atom) {
            if ((mask & (UINT64_C(1) << atom)) == 0) continue;
            cost = atom_cost[atom] > UINT64_MAX - cost ? UINT64_MAX : cost + atom_cost[atom];
            const auto& item = view.atoms[atom];
            const std::uint64_t end = static_cast<std::uint64_t>(item.first_family) + item.family_count;
            if (end > view.family_reference_count) return {};
            for (std::uint64_t i = item.first_family; i < end; ++i) {
                if (view.families[i] >= view.family_count) return {};
                covered_scratch[view.families[i]] = true;
            }
        }
        std::uint64_t benefit = 0;
        for (local_index f = 0; f < view.family_count; ++f) if (covered_scratch[f]) {
            benefit = view.family_frequency[f] > UINT64_MAX - benefit ? UINT64_MAX : benefit + view.family_frequency[f];
        }
        const std::uint64_t utility = benefit > cost ? benefit - cost : 0;
        if (utility > best.utility || (utility == best.utility && mask < best.mask)) best = {oracle_status::success, mask, utility};
    }
    return best;
}

}  // namespace cellshard::compiler::basis
