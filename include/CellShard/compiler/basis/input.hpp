#pragma once

#include <cstdint>
#include <limits>

namespace cellshard::compiler::basis {

using global_id = std::uint64_t;
using local_index = std::uint32_t;

inline constexpr local_index invalid_local_index =
    std::numeric_limits<local_index>::max();

struct workload_family_input {
    global_id family_id = 0;
    std::uint64_t frequency = 0;
    local_index first_required_atom = 0;
    local_index required_atom_count = 0;
};

struct workload_weight {
    std::uint64_t numerator = 0;
    std::uint64_t denominator = 1;
};

inline bool valid_weight(workload_weight weight) noexcept {
    return weight.denominator != 0;
}

inline int compare_weight(workload_weight left, workload_weight right) noexcept {
    // Exact comparison without multiplication overflow.
    bool reverse = false;
    for (;;) {
        const std::uint64_t left_q = left.numerator / left.denominator;
        const std::uint64_t right_q = right.numerator / right.denominator;
        if (left_q != right_q) {
            const int result = left_q < right_q ? -1 : 1;
            return reverse ? -result : result;
        }
        const std::uint64_t left_r = left.numerator % left.denominator;
        const std::uint64_t right_r = right.numerator % right.denominator;
        if (left_r == 0 || right_r == 0) {
            const int result = left_r == right_r ? 0 : (left_r == 0 ? -1 : 1);
            return reverse ? -result : result;
        }
        left = {left.denominator, left_r};
        right = {right.denominator, right_r};
        reverse = !reverse;
    }
}

struct atom_input {
    global_id atom_id = 0;
    std::uint64_t storage_bytes = 0;
    std::uint64_t build_cost = 0;
    std::uint64_t resident_bytes = 0;
    std::uint64_t mutation_cost = 0;
};

struct basis_input_view {
    const workload_family_input* families = nullptr;
    local_index family_count = 0;
    const atom_input* atoms = nullptr;
    local_index atom_count = 0;
    const local_index* required_atoms = nullptr;
    local_index required_atom_count = 0;
};

enum class input_error : std::uint8_t {
    none,
    missing_families,
    missing_atoms,
    missing_requirements,
    invalid_atom_reference,
    duplicate_family_id,
    duplicate_atom_id,
    zero_frequency
};

inline input_error validate_input(const basis_input_view& input) noexcept {
    if (input.family_count != 0 && input.families == nullptr) {
        return input_error::missing_families;
    }
    if (input.atom_count != 0 && input.atoms == nullptr) {
        return input_error::missing_atoms;
    }
    if (input.required_atom_count != 0 && input.required_atoms == nullptr) {
        return input_error::missing_requirements;
    }
    for (local_index i = 0; i < input.atom_count; ++i) {
        for (local_index j = 0; j < i; ++j) {
            if (input.atoms[i].atom_id == input.atoms[j].atom_id) {
                return input_error::duplicate_atom_id;
            }
        }
    }
    for (local_index i = 0; i < input.family_count; ++i) {
        const auto& family = input.families[i];
        if (family.frequency == 0) {
            return input_error::zero_frequency;
        }
        for (local_index j = 0; j < i; ++j) {
            if (family.family_id == input.families[j].family_id) {
                return input_error::duplicate_family_id;
            }
        }
        const std::uint64_t end = static_cast<std::uint64_t>(family.first_required_atom) +
                                  static_cast<std::uint64_t>(family.required_atom_count);
        if (end > input.required_atom_count) {
            return input_error::missing_requirements;
        }
        for (std::uint64_t r = family.first_required_atom; r < end; ++r) {
            if (input.required_atoms[r] >= input.atom_count) {
                return input_error::invalid_atom_reference;
            }
        }
    }
    return input_error::none;
}

}  // namespace cellshard::compiler::basis
