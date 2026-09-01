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
