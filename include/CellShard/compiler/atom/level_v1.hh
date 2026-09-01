#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace cellshard::compiler::atom {

inline constexpr std::uint32_t atom_level_contract_version_v1 = 1;

// Levels describe how an atom is interpreted by the compiler. They do not
// encode device, topology route, physical format, replica, or residency
// identity. An atom path may start or stop at any valid level and may skip
// levels that do not apply to that atom.
enum class atom_level_v1 : std::uint32_t {
    invalid = 0,
    evidence = 1,
    semantic = 2,
    structural = 3,
    materialized = 4,
    partial = 5,
    executable = 6,
    graph_family = 7,
    schedule = 8,
    topology = 9,
    resident = 10,
};

struct atom_level_path_view_v1 {
    const atom_level_v1 *levels = nullptr;
    std::size_t level_count = 0;
};

enum class atom_level_validation_code_v1 : std::uint32_t {
    valid = 0,
    empty_path,
    null_levels,
    invalid_level,
    duplicate_level,
    non_monotonic_level,
};

struct atom_level_validation_v1 {
    atom_level_validation_code_v1 code = atom_level_validation_code_v1::valid;
    std::size_t index = 0;
    atom_level_v1 level = atom_level_v1::invalid;

    [[nodiscard]] constexpr bool valid() const noexcept {
        return code == atom_level_validation_code_v1::valid;
    }
};

static_assert(sizeof(atom_level_v1) == sizeof(std::uint32_t),
              "atom levels must retain a stable 32-bit representation");
static_assert(std::is_standard_layout<atom_level_path_view_v1>::value,
              "atom level path views must remain standard-layout values");
static_assert(std::is_trivially_copyable<atom_level_path_view_v1>::value,
              "atom level path views must remain trivially copyable");
static_assert(offsetof(atom_level_path_view_v1, levels) == 0,
              "atom level path views must remain pointer-first");
static_assert(std::is_standard_layout<atom_level_validation_v1>::value,
              "atom level diagnostics must remain standard-layout values");
static_assert(std::is_trivially_copyable<atom_level_validation_v1>::value,
              "atom level diagnostics must remain trivially copyable");

[[nodiscard]] constexpr std::uint32_t atom_level_rank_v1(
    atom_level_v1 level) noexcept {
    switch (level) {
    case atom_level_v1::evidence:
        return 1;
    case atom_level_v1::semantic:
        return 2;
    case atom_level_v1::structural:
        return 3;
    case atom_level_v1::materialized:
        return 4;
    case atom_level_v1::partial:
        return 5;
    case atom_level_v1::executable:
        return 6;
    case atom_level_v1::graph_family:
        return 7;
    case atom_level_v1::schedule:
        return 8;
    case atom_level_v1::topology:
        return 9;
    case atom_level_v1::resident:
        return 10;
    case atom_level_v1::invalid:
        return 0;
    }
    return 0;
}

[[nodiscard]] constexpr bool valid_atom_level_v1(atom_level_v1 level) noexcept {
    return atom_level_rank_v1(level) != 0;
}

[[nodiscard]] constexpr const char *atom_level_name_v1(
    atom_level_v1 level) noexcept {
    switch (level) {
    case atom_level_v1::evidence:
        return "evidence";
    case atom_level_v1::semantic:
        return "semantic";
    case atom_level_v1::structural:
        return "structural";
    case atom_level_v1::materialized:
        return "materialized";
    case atom_level_v1::partial:
        return "partial";
    case atom_level_v1::executable:
        return "executable";
    case atom_level_v1::graph_family:
        return "graph_family";
    case atom_level_v1::schedule:
        return "schedule";
    case atom_level_v1::topology:
        return "topology";
    case atom_level_v1::resident:
        return "resident";
    case atom_level_v1::invalid:
        return "invalid";
    }
    return "invalid";
}

// A transition may skip inapplicable levels, but it may not reinterpret a
// later-level atom as an earlier-level atom.
[[nodiscard]] constexpr bool valid_atom_level_transition_v1(
    atom_level_v1 from, atom_level_v1 to) noexcept {
    const auto from_rank = atom_level_rank_v1(from);
    const auto to_rank = atom_level_rank_v1(to);
    return from_rank != 0 && to_rank != 0 && from_rank < to_rank;
}

// Validation is O(level_count) time and O(1) auxiliary storage.
[[nodiscard]] constexpr atom_level_validation_v1 validate_atom_level_path_v1(
    atom_level_path_view_v1 path) noexcept {
    if (path.level_count == 0) {
        return {atom_level_validation_code_v1::empty_path, 0,
                atom_level_v1::invalid};
    }
    if (path.levels == nullptr) {
        return {atom_level_validation_code_v1::null_levels, 0,
                atom_level_v1::invalid};
    }

    std::uint32_t previous_rank = 0;
    for (std::size_t index = 0; index < path.level_count; ++index) {
        const auto level = path.levels[index];
        const auto rank = atom_level_rank_v1(level);
        if (rank == 0) {
            return {atom_level_validation_code_v1::invalid_level, index, level};
        }
        if (rank == previous_rank) {
            return {atom_level_validation_code_v1::duplicate_level, index, level};
        }
        if (rank < previous_rank) {
            return {atom_level_validation_code_v1::non_monotonic_level, index,
                    level};
        }
        previous_rank = rank;
    }
    return {atom_level_validation_code_v1::valid, path.level_count,
            atom_level_v1::invalid};
}

} // namespace cellshard::compiler::atom
