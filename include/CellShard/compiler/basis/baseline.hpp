#pragma once

#include "CellShard/compiler/basis/input.hpp"

#include <cstdint>
#include <limits>

namespace cellshard::compiler::basis {

struct basis_solution {
    local_index* selected_atoms = nullptr;
    local_index capacity = 0;
    local_index count = 0;
    std::uint64_t uncovered_frequency = 0;
    bool saturated = false;
};

inline basis_solution canonical_no_basis(const basis_input_view& input,
                                         local_index* output = nullptr,
                                         local_index capacity = 0) noexcept {
    basis_solution result{output, capacity, 0, 0, false};
    for (local_index i = 0; i < input.family_count; ++i) {
        const auto frequency = input.families[i].frequency;
        if (frequency > std::numeric_limits<std::uint64_t>::max() -
                            result.uncovered_frequency) {
            result.uncovered_frequency = std::numeric_limits<std::uint64_t>::max();
            result.saturated = true;
            break;
        }
        result.uncovered_frequency += frequency;
    }
    return result;
}

}  // namespace cellshard::compiler::basis
