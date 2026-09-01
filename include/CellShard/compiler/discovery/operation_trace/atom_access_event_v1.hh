#pragma once

#include <CellShard/compiler/atom/persistent_identity_v1.hh>
#include <CellShard/compiler/evidence/atom_evidence_record_v1.hh>

#include <cstdint>
#include <type_traits>

namespace cellshard::compiler::discovery::operation_trace {

inline constexpr std::uint32_t atom_access_event_schema_version_v1 = 1;

enum class atom_access_mode_v1 : std::uint8_t {
    read = 1,
    write = 2,
    read_write = 3,
};

enum class atom_access_role_v1 : std::uint8_t {
    input = 1,
    output = 2,
    intermediate = 3,
    state = 4,
};

// One pointer-free observation from an operation trace. All persistent
// identities are namespace-qualified global values. sequence_number is local
// only to trace_identity and is intentionally u64 so long-lived traces cannot
// silently wrap. Physical addresses, paths, devices, and placement epochs are
// absent: none of them establishes biological or operation identity.
struct atom_access_event_v1 {
    std::uint32_t schema_version = atom_access_event_schema_version_v1;
    std::uint32_t record_bytes = sizeof(atom_access_event_v1);
    evidence::evidence_identity_v1 event_identity{};
    evidence::evidence_identity_v1 trace_identity{};
    evidence::evidence_identity_v1 source_identity{};
    atom::atom_persistent_identity_v1 workload_identity{};
    atom::atom_persistent_identity_v1 graph_identity{};
    atom::atom_persistent_identity_v1 operation_identity{};
    atom::atom_persistent_identity_v1 stage_identity{};
    atom::atom_persistent_identity_v1 atom_identity{};
    atom::atom_persistent_identity_v1 port_identity{};
    std::uint64_t trace_generation = 0;
    std::uint64_t graph_generation = 0;
    std::uint64_t operation_generation = 0;
    std::uint64_t stage_generation = 0;
    std::uint64_t atom_generation = 0;
    std::uint64_t sequence_number = 0;
    std::uint64_t logical_byte_count = 0;
    atom_access_mode_v1 mode = atom_access_mode_v1::read;
    atom_access_role_v1 role = atom_access_role_v1::input;
    std::uint16_t reserved16 = 0;
    std::uint32_t reserved32 = 0;
};

enum class atom_access_event_validation_code_v1 : std::uint32_t {
    valid = 0,
    unsupported_schema,
    invalid_record_bytes,
    invalid_event_identity,
    invalid_trace_identity,
    invalid_source_identity,
    invalid_workload_identity,
    invalid_graph_identity,
    invalid_operation_identity,
    invalid_stage_identity,
    invalid_atom_identity,
    invalid_port_identity,
    missing_trace_generation,
    missing_graph_generation,
    missing_operation_generation,
    missing_stage_generation,
    missing_atom_generation,
    missing_sequence_number,
    empty_logical_access,
    invalid_mode,
    invalid_role,
    nonzero_reserved,
};

struct atom_access_event_validation_v1 {
    atom_access_event_validation_code_v1 code =
        atom_access_event_validation_code_v1::valid;

    [[nodiscard]] constexpr bool valid() const noexcept {
        return code == atom_access_event_validation_code_v1::valid;
    }
};

[[nodiscard]] constexpr bool valid_atom_access_mode_v1(
    atom_access_mode_v1 mode) noexcept {
    return mode == atom_access_mode_v1::read
        || mode == atom_access_mode_v1::write
        || mode == atom_access_mode_v1::read_write;
}

[[nodiscard]] constexpr bool valid_atom_access_role_v1(
    atom_access_role_v1 role) noexcept {
    return role == atom_access_role_v1::input
        || role == atom_access_role_v1::output
        || role == atom_access_role_v1::intermediate
        || role == atom_access_role_v1::state;
}

// O(1) time and storage. This validates a stable proposal observation only;
// exact coverage and contribution ownership remain independent contracts.
[[nodiscard]] constexpr atom_access_event_validation_v1
validate_atom_access_event_v1(const atom_access_event_v1 &event) noexcept {
    if (event.schema_version != atom_access_event_schema_version_v1) {
        return {atom_access_event_validation_code_v1::unsupported_schema};
    }
    if (event.record_bytes != sizeof(atom_access_event_v1)) {
        return {atom_access_event_validation_code_v1::invalid_record_bytes};
    }
    if (!evidence::valid_evidence_identity_v1(event.event_identity)) {
        return {atom_access_event_validation_code_v1::invalid_event_identity};
    }
    if (!evidence::valid_evidence_identity_v1(event.trace_identity)) {
        return {atom_access_event_validation_code_v1::invalid_trace_identity};
    }
    if (!evidence::valid_evidence_identity_v1(event.source_identity)) {
        return {atom_access_event_validation_code_v1::invalid_source_identity};
    }
    if (!atom::validate_atom_persistent_identity_v1(event.workload_identity)
             .valid()) {
        return {atom_access_event_validation_code_v1::invalid_workload_identity};
    }
    if (!atom::validate_atom_persistent_identity_v1(event.graph_identity)
             .valid()) {
        return {atom_access_event_validation_code_v1::invalid_graph_identity};
    }
    if (!atom::validate_atom_persistent_identity_v1(event.operation_identity)
             .valid()) {
        return {atom_access_event_validation_code_v1::invalid_operation_identity};
    }
    if (!atom::validate_atom_persistent_identity_v1(event.stage_identity)
             .valid()) {
        return {atom_access_event_validation_code_v1::invalid_stage_identity};
    }
    if (!atom::validate_atom_persistent_identity_v1(event.atom_identity)
             .valid()) {
        return {atom_access_event_validation_code_v1::invalid_atom_identity};
    }
    if (!atom::validate_atom_persistent_identity_v1(event.port_identity)
             .valid()) {
        return {atom_access_event_validation_code_v1::invalid_port_identity};
    }
    if (event.trace_generation == 0) {
        return {atom_access_event_validation_code_v1::missing_trace_generation};
    }
    if (event.graph_generation == 0) {
        return {atom_access_event_validation_code_v1::missing_graph_generation};
    }
    if (event.operation_generation == 0) {
        return {atom_access_event_validation_code_v1::missing_operation_generation};
    }
    if (event.stage_generation == 0) {
        return {atom_access_event_validation_code_v1::missing_stage_generation};
    }
    if (event.atom_generation == 0) {
        return {atom_access_event_validation_code_v1::missing_atom_generation};
    }
    if (event.sequence_number == 0) {
        return {atom_access_event_validation_code_v1::missing_sequence_number};
    }
    if (event.logical_byte_count == 0) {
        return {atom_access_event_validation_code_v1::empty_logical_access};
    }
    if (!valid_atom_access_mode_v1(event.mode)) {
        return {atom_access_event_validation_code_v1::invalid_mode};
    }
    if (!valid_atom_access_role_v1(event.role)) {
        return {atom_access_event_validation_code_v1::invalid_role};
    }
    if (event.reserved16 != 0 || event.reserved32 != 0) {
        return {atom_access_event_validation_code_v1::nonzero_reserved};
    }
    return {};
}

[[nodiscard]] constexpr bool authorizes_execution(
    const atom_access_event_v1 &) noexcept {
    return false;
}

static_assert(sizeof(atom_access_mode_v1) == sizeof(std::uint8_t));
static_assert(sizeof(atom_access_role_v1) == sizeof(std::uint8_t));
static_assert(std::is_standard_layout<atom_access_event_v1>::value);
static_assert(std::is_trivially_copyable<atom_access_event_v1>::value);
static_assert(std::is_standard_layout<atom_access_event_validation_v1>::value);
static_assert(std::is_trivially_copyable<atom_access_event_validation_v1>::value);

} // namespace cellshard::compiler::discovery::operation_trace
