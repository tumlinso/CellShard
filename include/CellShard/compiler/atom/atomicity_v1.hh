#pragma once

#include <cstdint>
#include <type_traits>

namespace cellshard::compiler::atom {

inline constexpr std::uint32_t atom_atomicity_contract_version_v1 = 1;

// Atomicity classes are independent capabilities, not mutually exclusive
// kinds. They describe which boundaries an atom can preserve without implying
// an atom level, species, physical format, or execution backend.
enum class atom_atomicity_capability_v1 : std::uint64_t {
    semantic = UINT64_C(1) << 0,
    ownership = UINT64_C(1) << 1,
    materialization = UINT64_C(1) << 2,
    transfer = UINT64_C(1) << 3,
    cache_reuse = UINT64_C(1) << 4,
    order = UINT64_C(1) << 5,
    algebraic = UINT64_C(1) << 6,
    executable = UINT64_C(1) << 7,
    grammar = UINT64_C(1) << 8,
    cellerator_local = UINT64_C(1) << 9,
    cellshard_global = UINT64_C(1) << 10,
};

inline constexpr std::uint64_t atom_atomicity_known_mask_v1 =
    (UINT64_C(1) << 11) - 1;

struct atom_atomicity_set_v1 {
    std::uint64_t capabilities = 0;
};

enum class atom_atomicity_validation_code_v1 : std::uint32_t {
    valid = 0,
    empty,
    unknown_capability,
};

struct atom_atomicity_validation_v1 {
    atom_atomicity_validation_code_v1 code =
        atom_atomicity_validation_code_v1::valid;
    std::uint64_t unknown_capabilities = 0;

    [[nodiscard]] constexpr bool valid() const noexcept {
        return code == atom_atomicity_validation_code_v1::valid;
    }
};

static_assert(sizeof(atom_atomicity_capability_v1) == sizeof(std::uint64_t),
              "atomicity capabilities must retain a stable 64-bit representation");
static_assert(sizeof(atom_atomicity_set_v1) == sizeof(std::uint64_t),
              "atomicity sets must remain one 64-bit field");
static_assert(std::is_standard_layout<atom_atomicity_set_v1>::value,
              "atomicity sets must remain standard-layout values");
static_assert(std::is_trivially_copyable<atom_atomicity_set_v1>::value,
              "atomicity sets must remain trivially copyable");
static_assert(std::is_standard_layout<atom_atomicity_validation_v1>::value,
              "atomicity diagnostics must remain standard-layout values");
static_assert(std::is_trivially_copyable<atom_atomicity_validation_v1>::value,
              "atomicity diagnostics must remain trivially copyable");

[[nodiscard]] constexpr std::uint64_t atom_atomicity_bit_v1(
    atom_atomicity_capability_v1 capability) noexcept {
    return static_cast<std::uint64_t>(capability);
}

[[nodiscard]] constexpr bool valid_atom_atomicity_capability_v1(
    atom_atomicity_capability_v1 capability) noexcept {
    const auto bit = atom_atomicity_bit_v1(capability);
    return bit != 0 && (bit & (bit - 1)) == 0
        && (bit & ~atom_atomicity_known_mask_v1) == 0;
}

[[nodiscard]] constexpr atom_atomicity_set_v1 operator|(
    atom_atomicity_capability_v1 lhs,
    atom_atomicity_capability_v1 rhs) noexcept {
    return {atom_atomicity_bit_v1(lhs) | atom_atomicity_bit_v1(rhs)};
}

[[nodiscard]] constexpr atom_atomicity_set_v1 operator|(
    atom_atomicity_set_v1 set,
    atom_atomicity_capability_v1 capability) noexcept {
    return {set.capabilities | atom_atomicity_bit_v1(capability)};
}

[[nodiscard]] constexpr bool atom_has_atomicity_v1(
    atom_atomicity_set_v1 set,
    atom_atomicity_capability_v1 capability) noexcept {
    return valid_atom_atomicity_capability_v1(capability)
        && (set.capabilities & atom_atomicity_bit_v1(capability)) != 0;
}

[[nodiscard]] constexpr const char *atom_atomicity_name_v1(
    atom_atomicity_capability_v1 capability) noexcept {
    switch (capability) {
    case atom_atomicity_capability_v1::semantic:
        return "semantic";
    case atom_atomicity_capability_v1::ownership:
        return "ownership";
    case atom_atomicity_capability_v1::materialization:
        return "materialization";
    case atom_atomicity_capability_v1::transfer:
        return "transfer";
    case atom_atomicity_capability_v1::cache_reuse:
        return "cache_reuse";
    case atom_atomicity_capability_v1::order:
        return "order";
    case atom_atomicity_capability_v1::algebraic:
        return "algebraic";
    case atom_atomicity_capability_v1::executable:
        return "executable";
    case atom_atomicity_capability_v1::grammar:
        return "grammar";
    case atom_atomicity_capability_v1::cellerator_local:
        return "cellerator_local";
    case atom_atomicity_capability_v1::cellshard_global:
        return "cellshard_global";
    }
    return "invalid";
}

// Validation is O(1) time and O(1) auxiliary storage. All combinations of
// known capabilities are permitted; policy about required combinations belongs
// to the consuming atom contract rather than this taxonomy.
[[nodiscard]] constexpr atom_atomicity_validation_v1 validate_atom_atomicity_v1(
    atom_atomicity_set_v1 set) noexcept {
    if (set.capabilities == 0) {
        return {atom_atomicity_validation_code_v1::empty, 0};
    }
    const auto unknown = set.capabilities & ~atom_atomicity_known_mask_v1;
    if (unknown != 0) {
        return {atom_atomicity_validation_code_v1::unknown_capability, unknown};
    }
    return {atom_atomicity_validation_code_v1::valid, 0};
}

} // namespace cellshard::compiler::atom
