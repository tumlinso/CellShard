#pragma once

#include <CellShard/compiler/atom/port_v1.hh>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace cellshard::compiler::atom {

inline constexpr std::uint32_t
    atom_executable_affordance_plane_schema_version_v1 = 1;

enum class atom_lowering_entry_kind_v1 : std::uint32_t {
    cellerator_native = 1,
    external = 2,
};

enum class atom_target_restriction_kind_v1 : std::uint32_t {
    unrestricted = 1,
    exact_target = 2,
};

enum atom_output_affordance_v1 : std::uint32_t {
    atom_complete_output_affordance_v1 = 1u << 0u,
    atom_partial_output_affordance_v1 = 1u << 1u,
};

inline constexpr std::uint32_t atom_known_output_affordances_v1 =
    atom_complete_output_affordance_v1
    | atom_partial_output_affordance_v1;

struct atom_executable_affordance_v1 {
    atom_persistent_identity_v1 operation_identity{};
    atom_persistent_identity_v1 lowering_entry_identity{};
    atom_persistent_identity_v1 target_identity{};
    const atom_persistent_identity_v1 *required_mutable_ports = nullptr;
    std::uint64_t required_mutable_port_count = 0;
    std::uint32_t output_affordances = 0;
    atom_lowering_entry_kind_v1 lowering_kind =
        atom_lowering_entry_kind_v1::cellerator_native;
    atom_target_restriction_kind_v1 target_restriction =
        atom_target_restriction_kind_v1::unrestricted;
    std::uint32_t reserved = 0;
};

// Preparation identity is independent of mutable port generations and launch
// pointers. Changing a port binding does not redefine the prepared lowering.
struct atom_executable_affordance_plane_v1 {
    const atom_executable_affordance_v1 *affordances = nullptr;
    std::uint64_t affordance_count = 0;
    atom_port_table_view_v1 ports{};
    atom_persistent_identity_v1 plane_identity{};
    atom_persistent_identity_v1 preparation_identity{};
    std::uint64_t preparation_generation = 0;
};

enum class atom_executable_affordance_validation_code_v1 : std::uint32_t {
    valid = 0,
    empty_affordances,
    missing_affordances,
    invalid_ports,
    invalid_plane_identity,
    invalid_preparation_identity,
    missing_preparation_generation,
    invalid_operation_identity,
    unordered_or_duplicate_operation,
    invalid_lowering_entry,
    invalid_lowering_kind,
    invalid_target_restriction,
    unexpected_target_identity,
    missing_target_identity,
    missing_required_mutable_ports,
    invalid_required_mutable_port,
    unordered_or_duplicate_required_port,
    unknown_required_mutable_port,
    referenced_port_not_required_mutable,
    invalid_output_affordance,
    nonzero_reserved,
};

struct atom_executable_affordance_validation_v1 {
    atom_executable_affordance_validation_code_v1 code =
        atom_executable_affordance_validation_code_v1::valid;
    std::uint64_t affordance_index = 0;
    std::uint64_t port_index = 0;
    std::uint32_t nested_code = 0;

    [[nodiscard]] constexpr bool valid() const noexcept {
        return code == atom_executable_affordance_validation_code_v1::valid;
    }
};

static_assert(std::is_standard_layout<atom_executable_affordance_v1>::value);
static_assert(std::is_trivially_copyable<atom_executable_affordance_v1>::value);
static_assert(offsetof(atom_executable_affordance_plane_v1, affordances) == 0,
              "executable-affordance planes must remain pointer-first");
static_assert(
    std::is_standard_layout<atom_executable_affordance_plane_v1>::value);
static_assert(
    std::is_trivially_copyable<atom_executable_affordance_plane_v1>::value);

[[nodiscard]] constexpr bool valid_atom_lowering_entry_kind_v1(
    atom_lowering_entry_kind_v1 kind) noexcept {
    return kind == atom_lowering_entry_kind_v1::cellerator_native
        || kind == atom_lowering_entry_kind_v1::external;
}

[[nodiscard]] constexpr bool valid_atom_target_restriction_kind_v1(
    atom_target_restriction_kind_v1 kind) noexcept {
    return kind == atom_target_restriction_kind_v1::unrestricted
        || kind == atom_target_restriction_kind_v1::exact_target;
}

namespace detail {

[[nodiscard]] inline const atom_port_v1 *find_atom_port_v1(
    atom_port_table_view_v1 table,
    atom_persistent_identity_v1 identity) noexcept {
    std::uint64_t first = 0;
    std::uint64_t last = table.port_count;
    while (first < last) {
        const auto middle = first + (last - first) / 2;
        const auto &candidate = table.ports[middle];
        if (atom_persistent_identity_less_v1(candidate.port_identity, identity)) {
            first = middle + 1;
        } else {
            last = middle;
        }
    }
    return first < table.port_count
            && table.ports[first].port_identity == identity
        ? &table.ports[first]
        : nullptr;
}

} // namespace detail

// O(port_count + affordance_count + R log(port_count)), where R is the total
// required-mutable-port count. Validation allocates nothing and performs no
// lowering discovery or preparation.
[[nodiscard]] inline atom_executable_affordance_validation_v1
validate_atom_executable_affordance_plane_v1(
    const atom_executable_affordance_plane_v1 &plane) noexcept {
    if (plane.affordance_count == 0) {
        return {atom_executable_affordance_validation_code_v1::
                    empty_affordances,
                0, 0, 0};
    }
    if (plane.affordances == nullptr) {
        return {atom_executable_affordance_validation_code_v1::
                    missing_affordances,
                0, 0, 0};
    }
    const auto port_result = validate_atom_port_table_v1(plane.ports);
    if (!port_result.valid()) {
        return {atom_executable_affordance_validation_code_v1::invalid_ports,
                0, port_result.index,
                static_cast<std::uint32_t>(port_result.code)};
    }
    if (!validate_atom_persistent_identity_v1(plane.plane_identity).valid()) {
        return {atom_executable_affordance_validation_code_v1::
                    invalid_plane_identity,
                0, 0, 0};
    }
    if (!validate_atom_persistent_identity_v1(plane.preparation_identity)
             .valid()) {
        return {atom_executable_affordance_validation_code_v1::
                    invalid_preparation_identity,
                0, 0, 0};
    }
    if (plane.preparation_generation == 0) {
        return {atom_executable_affordance_validation_code_v1::
                    missing_preparation_generation,
                0, 0, 0};
    }
    for (std::uint64_t index = 0; index < plane.affordance_count; ++index) {
        const auto &affordance = plane.affordances[index];
        if (!validate_atom_persistent_identity_v1(affordance.operation_identity)
                 .valid()) {
            return {atom_executable_affordance_validation_code_v1::
                        invalid_operation_identity,
                    index, 0, 0};
        }
        if (index != 0
            && !atom_persistent_identity_less_v1(
                plane.affordances[index - 1].operation_identity,
                affordance.operation_identity)) {
            return {atom_executable_affordance_validation_code_v1::
                        unordered_or_duplicate_operation,
                    index, 0, 0};
        }
        if (!validate_atom_persistent_identity_v1(
                 affordance.lowering_entry_identity)
                 .valid()) {
            return {atom_executable_affordance_validation_code_v1::
                        invalid_lowering_entry,
                    index, 0, 0};
        }
        if (!valid_atom_lowering_entry_kind_v1(affordance.lowering_kind)) {
            return {atom_executable_affordance_validation_code_v1::
                        invalid_lowering_kind,
                    index, 0, 0};
        }
        if (!valid_atom_target_restriction_kind_v1(
                affordance.target_restriction)) {
            return {atom_executable_affordance_validation_code_v1::
                        invalid_target_restriction,
                    index, 0, 0};
        }
        const bool target_valid = validate_atom_persistent_identity_v1(
            affordance.target_identity).valid();
        if (affordance.target_restriction
                == atom_target_restriction_kind_v1::unrestricted
            && target_valid) {
            return {atom_executable_affordance_validation_code_v1::
                        unexpected_target_identity,
                    index, 0, 0};
        }
        if (affordance.target_restriction
                == atom_target_restriction_kind_v1::exact_target
            && !target_valid) {
            return {atom_executable_affordance_validation_code_v1::
                        missing_target_identity,
                    index, 0, 0};
        }
        if (affordance.required_mutable_port_count == 0
            || affordance.required_mutable_ports == nullptr) {
            return {atom_executable_affordance_validation_code_v1::
                        missing_required_mutable_ports,
                    index, 0, 0};
        }
        for (std::uint64_t port_index = 0;
             port_index < affordance.required_mutable_port_count;
             ++port_index) {
            const auto port_identity =
                affordance.required_mutable_ports[port_index];
            if (!validate_atom_persistent_identity_v1(port_identity).valid()) {
                return {atom_executable_affordance_validation_code_v1::
                            invalid_required_mutable_port,
                        index, port_index, 0};
            }
            if (port_index != 0
                && !atom_persistent_identity_less_v1(
                    affordance.required_mutable_ports[port_index - 1],
                    port_identity)) {
                return {atom_executable_affordance_validation_code_v1::
                            unordered_or_duplicate_required_port,
                        index, port_index, 0};
            }
            const auto *port = detail::find_atom_port_v1(
                plane.ports, port_identity);
            if (port == nullptr) {
                return {atom_executable_affordance_validation_code_v1::
                            unknown_required_mutable_port,
                        index, port_index, 0};
            }
            if (port->requirement != atom_port_requirement_v1::required
                || port->mutability == atom_port_mutability_v1::immutable) {
                return {atom_executable_affordance_validation_code_v1::
                            referenced_port_not_required_mutable,
                        index, port_index, 0};
            }
        }
        if (affordance.output_affordances == 0
            || (affordance.output_affordances
                & ~atom_known_output_affordances_v1) != 0) {
            return {atom_executable_affordance_validation_code_v1::
                        invalid_output_affordance,
                    index, 0, 0};
        }
        if (affordance.reserved != 0) {
            return {atom_executable_affordance_validation_code_v1::
                        nonzero_reserved,
                    index, 0, 0};
        }
    }
    return {atom_executable_affordance_validation_code_v1::valid,
            plane.affordance_count, 0, 0};
}

} // namespace cellshard::compiler::atom
