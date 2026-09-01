#pragma once

#include <CellShard/compiler/atom/persistent_identity_v1.hh>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace cellshard::compiler::atom {

inline constexpr std::uint32_t atom_port_contract_version_v1 = 1;

enum class atom_port_direction_v1 : std::uint8_t {
    input = 1,
    output = 2,
    inout = 3,
};

enum class atom_port_axis_role_v1 : std::uint8_t {
    source = 1,
    destination = 2,
};

enum class atom_port_mutability_v1 : std::uint8_t {
    immutable = 1,
    mutable_value = 2,
    mutable_state = 3,
};

enum class atom_port_requirement_v1 : std::uint8_t {
    required = 1,
    optional = 2,
};

enum atom_port_extent_form_v1 : std::uint32_t {
    atom_single_contiguous_extent_v1 = 1u << 0u,
    atom_multiple_contiguous_extents_v1 = 1u << 1u,
    atom_regular_stride_extent_v1 = 1u << 2u,
    atom_segmented_extent_v1 = 1u << 3u,
    atom_provider_defined_extent_v1 = 1u << 4u,
};

inline constexpr std::uint32_t atom_known_port_extent_forms_v1 =
    atom_single_contiguous_extent_v1
    | atom_multiple_contiguous_extents_v1
    | atom_regular_stride_extent_v1
    | atom_segmented_extent_v1
    | atom_provider_defined_extent_v1;

// Numeric identities are namespace-qualified so new encodings do not require
// extending a universal enum. Storage, logical, and accumulation identities
// remain separate because equal byte widths do not imply equal arithmetic.
struct atom_port_numeric_v1 {
    atom_persistent_identity_v1 storage_type{};
    atom_persistent_identity_v1 logical_type{};
    atom_persistent_identity_v1 accumulation_type{};
};

struct atom_port_v1 {
    atom_persistent_identity_v1 port_identity{};
    atom_persistent_identity_v1 domain_identity{};
    atom_persistent_identity_v1 axis_identity{};
    atom_persistent_identity_v1 order_identity{};
    atom_persistent_identity_v1 plane_kind{};
    atom_port_numeric_v1 numeric{};
    std::uint64_t generation = 0;
    std::uint32_t accepted_extent_forms = 0;
    std::uint32_t minimum_extent_count = 0;
    std::uint32_t maximum_extent_count = 0;
    atom_port_direction_v1 direction = atom_port_direction_v1::input;
    atom_port_axis_role_v1 axis_role = atom_port_axis_role_v1::source;
    atom_port_mutability_v1 mutability = atom_port_mutability_v1::immutable;
    atom_port_requirement_v1 requirement = atom_port_requirement_v1::required;
    std::uint32_t reserved = 0;
};

struct atom_port_table_view_v1 {
    const atom_port_v1 *ports = nullptr;
    std::uint64_t port_count = 0;
};

enum class atom_port_validation_code_v1 : std::uint32_t {
    valid = 0,
    empty_table,
    missing_ports,
    invalid_port_identity,
    unordered_or_duplicate_port,
    invalid_domain_identity,
    invalid_axis_identity,
    invalid_order_identity,
    invalid_plane_kind,
    invalid_storage_type,
    invalid_logical_type,
    invalid_accumulation_type,
    missing_generation,
    invalid_extent_form,
    invalid_extent_count,
    invalid_direction,
    invalid_axis_role,
    invalid_mutability,
    invalid_requirement,
    immutable_inout,
    nonzero_reserved,
};

struct atom_port_validation_v1 {
    atom_port_validation_code_v1 code = atom_port_validation_code_v1::valid;
    std::uint64_t index = 0;

    [[nodiscard]] constexpr bool valid() const noexcept {
        return code == atom_port_validation_code_v1::valid;
    }
};

static_assert(std::is_standard_layout<atom_port_numeric_v1>::value);
static_assert(std::is_trivially_copyable<atom_port_numeric_v1>::value);
static_assert(std::is_standard_layout<atom_port_v1>::value);
static_assert(std::is_trivially_copyable<atom_port_v1>::value);
static_assert(offsetof(atom_port_table_view_v1, ports) == 0,
              "port tables must remain pointer-first");
static_assert(std::is_standard_layout<atom_port_table_view_v1>::value);
static_assert(std::is_trivially_copyable<atom_port_table_view_v1>::value);

[[nodiscard]] constexpr bool valid_atom_port_direction_v1(
    atom_port_direction_v1 value) noexcept {
    return value == atom_port_direction_v1::input
        || value == atom_port_direction_v1::output
        || value == atom_port_direction_v1::inout;
}

[[nodiscard]] constexpr bool valid_atom_port_axis_role_v1(
    atom_port_axis_role_v1 value) noexcept {
    return value == atom_port_axis_role_v1::source
        || value == atom_port_axis_role_v1::destination;
}

[[nodiscard]] constexpr bool valid_atom_port_mutability_v1(
    atom_port_mutability_v1 value) noexcept {
    return value == atom_port_mutability_v1::immutable
        || value == atom_port_mutability_v1::mutable_value
        || value == atom_port_mutability_v1::mutable_state;
}

[[nodiscard]] constexpr bool valid_atom_port_requirement_v1(
    atom_port_requirement_v1 value) noexcept {
    return value == atom_port_requirement_v1::required
        || value == atom_port_requirement_v1::optional;
}

// Tables are sorted by namespace-qualified port identity. Validation is O(N)
// time and O(1) storage; it performs no allocation or canonicalization.
[[nodiscard]] constexpr atom_port_validation_v1 validate_atom_port_table_v1(
    atom_port_table_view_v1 table) noexcept {
    if (table.port_count == 0) {
        return {atom_port_validation_code_v1::empty_table, 0};
    }
    if (table.ports == nullptr) {
        return {atom_port_validation_code_v1::missing_ports, 0};
    }
    for (std::uint64_t index = 0; index < table.port_count; ++index) {
        const auto &port = table.ports[index];
        if (!validate_atom_persistent_identity_v1(port.port_identity).valid()) {
            return {atom_port_validation_code_v1::invalid_port_identity, index};
        }
        if (index != 0
            && !atom_persistent_identity_less_v1(
                table.ports[index - 1].port_identity, port.port_identity)) {
            return {atom_port_validation_code_v1::unordered_or_duplicate_port,
                    index};
        }
        if (!validate_atom_persistent_identity_v1(port.domain_identity).valid()) {
            return {atom_port_validation_code_v1::invalid_domain_identity,
                    index};
        }
        if (!validate_atom_persistent_identity_v1(port.axis_identity).valid()) {
            return {atom_port_validation_code_v1::invalid_axis_identity, index};
        }
        if (!validate_atom_persistent_identity_v1(port.order_identity).valid()) {
            return {atom_port_validation_code_v1::invalid_order_identity, index};
        }
        if (!validate_atom_persistent_identity_v1(port.plane_kind).valid()) {
            return {atom_port_validation_code_v1::invalid_plane_kind, index};
        }
        if (!validate_atom_persistent_identity_v1(port.numeric.storage_type)
                 .valid()) {
            return {atom_port_validation_code_v1::invalid_storage_type, index};
        }
        if (!validate_atom_persistent_identity_v1(port.numeric.logical_type)
                 .valid()) {
            return {atom_port_validation_code_v1::invalid_logical_type, index};
        }
        if (!validate_atom_persistent_identity_v1(
                 port.numeric.accumulation_type)
                 .valid()) {
            return {atom_port_validation_code_v1::invalid_accumulation_type,
                    index};
        }
        if (port.generation == 0) {
            return {atom_port_validation_code_v1::missing_generation, index};
        }
        if (port.accepted_extent_forms == 0
            || (port.accepted_extent_forms
                & ~atom_known_port_extent_forms_v1) != 0) {
            return {atom_port_validation_code_v1::invalid_extent_form, index};
        }
        if (port.minimum_extent_count == 0
            || port.maximum_extent_count < port.minimum_extent_count
            || ((port.accepted_extent_forms
                 & atom_single_contiguous_extent_v1) != 0
                && port.accepted_extent_forms
                       == atom_single_contiguous_extent_v1
                && (port.minimum_extent_count != 1
                    || port.maximum_extent_count != 1))) {
            return {atom_port_validation_code_v1::invalid_extent_count, index};
        }
        if (!valid_atom_port_direction_v1(port.direction)) {
            return {atom_port_validation_code_v1::invalid_direction, index};
        }
        if (!valid_atom_port_axis_role_v1(port.axis_role)) {
            return {atom_port_validation_code_v1::invalid_axis_role, index};
        }
        if (!valid_atom_port_mutability_v1(port.mutability)) {
            return {atom_port_validation_code_v1::invalid_mutability, index};
        }
        if (!valid_atom_port_requirement_v1(port.requirement)) {
            return {atom_port_validation_code_v1::invalid_requirement, index};
        }
        if (port.direction == atom_port_direction_v1::inout
            && port.mutability == atom_port_mutability_v1::immutable) {
            return {atom_port_validation_code_v1::immutable_inout, index};
        }
        if (port.reserved != 0) {
            return {atom_port_validation_code_v1::nonzero_reserved, index};
        }
    }
    return {atom_port_validation_code_v1::valid, table.port_count};
}

} // namespace cellshard::compiler::atom
