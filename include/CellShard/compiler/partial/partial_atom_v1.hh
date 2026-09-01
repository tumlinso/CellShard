#pragma once

#include <CellShard/compiler/atom/partial_result_plane_v1.hh>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace cellshard::compiler::partial {

using atom::atom_partial_result_plane_v1;
using atom::atom_partial_result_status_v1;
using atom::atom_persistent_identity_v1;

inline constexpr std::uint32_t partial_atom_schema_version_v1 = 1;

// A persistent partial is never an opaque scratch buffer. Only an exact,
// independently reconstructible contribution may cross this boundary.
enum class partial_persistence_class_v1 : std::uint32_t {
    exact_reconstructible = 1,
};

// Persistence is field-oriented. Encoders serialize these fields in their
// declared artifact byte order; native struct bytes are not a wire format.
struct partial_atom_header_v1 {
    std::uint32_t schema_version = partial_atom_schema_version_v1;
    std::uint32_t record_bytes = sizeof(partial_atom_header_v1);
    atom_persistent_identity_v1 partial_identity{};
    atom_persistent_identity_v1 source_atom_semantic_identity{};
    atom_persistent_identity_v1 partial_kind_identity{};
    atom_persistent_identity_v1 payload_schema_identity{};
    atom_persistent_identity_v1 contribution_coverage_identity{};
    atom_persistent_identity_v1 dependency_closure_identity{};
    atom_persistent_identity_v1 reconstruction_algebra_identity{};
    atom_persistent_identity_v1 numerical_policy_identity{};
    atom_persistent_identity_v1 complete_cost_evidence_identity{};
    std::uint64_t structure_generation = 0;
    std::uint64_t value_generation = 0;
    std::uint64_t state_generation = 0;
    std::uint64_t materialization_generation = 0;
    std::uint64_t cost_model_generation = 0;
    partial_persistence_class_v1 persistence_class =
        partial_persistence_class_v1::exact_reconstructible;
    std::uint32_t reserved = 0;
};

// The envelope is a nonowning, pointer-first view over exact payload bytes and
// the common atom partial-result plane. Payload ownership and serialization
// are deliberately left to later materialization tasks.
struct partial_atom_view_v1 {
    const void *payload = nullptr;
    std::uint64_t payload_bytes = 0;
    std::uint32_t payload_alignment = 0;
    std::uint32_t reserved = 0;
    partial_atom_header_v1 header{};
    atom_partial_result_plane_v1 result{};
};

enum class partial_atom_envelope_validation_code_v1 : std::uint32_t {
    valid = 0,
    unsupported_schema,
    invalid_record_bytes,
    invalid_partial_identity,
    invalid_source_atom_identity,
    invalid_partial_kind,
    invalid_payload_schema,
    invalid_contribution_coverage_identity,
    invalid_dependency_closure_identity,
    invalid_reconstruction_algebra,
    invalid_numerical_policy,
    invalid_complete_cost_evidence,
    missing_structure_generation,
    missing_value_generation,
    missing_state_generation,
    missing_materialization_generation,
    missing_cost_model_generation,
    scientifically_unknown_partial,
    missing_payload,
    empty_payload,
    invalid_payload_alignment,
    misaligned_payload,
    payload_binding_mismatch,
    coverage_binding_mismatch,
    algebra_binding_mismatch,
    numerical_policy_binding_mismatch,
    incomplete_partial,
    nonzero_reserved,
};

struct partial_atom_envelope_validation_v1 {
    partial_atom_envelope_validation_code_v1 code =
        partial_atom_envelope_validation_code_v1::valid;

    [[nodiscard]] constexpr bool valid() const noexcept {
        return code == partial_atom_envelope_validation_code_v1::valid;
    }
};

static_assert(std::is_standard_layout<partial_atom_header_v1>::value);
static_assert(std::is_trivially_copyable<partial_atom_header_v1>::value);
static_assert(offsetof(partial_atom_view_v1, payload) == 0,
              "partial atom views must remain pointer-first");
static_assert(std::is_standard_layout<partial_atom_view_v1>::value);
static_assert(std::is_trivially_copyable<partial_atom_view_v1>::value);

[[nodiscard]] inline partial_atom_envelope_validation_v1
validate_partial_atom_envelope_v1(const partial_atom_view_v1 &partial) noexcept {
    const auto &header = partial.header;
    if (header.schema_version != partial_atom_schema_version_v1) {
        return {partial_atom_envelope_validation_code_v1::unsupported_schema};
    }
    if (header.record_bytes != sizeof(partial_atom_header_v1)) {
        return {partial_atom_envelope_validation_code_v1::invalid_record_bytes};
    }
#define CELLSHARD_PARTIAL_CHECK_ID(field, code) \
    if (!atom::validate_atom_persistent_identity_v1(header.field).valid()) { \
        return {partial_atom_envelope_validation_code_v1::code}; \
    }
    CELLSHARD_PARTIAL_CHECK_ID(partial_identity, invalid_partial_identity)
    CELLSHARD_PARTIAL_CHECK_ID(source_atom_semantic_identity,
                               invalid_source_atom_identity)
    CELLSHARD_PARTIAL_CHECK_ID(partial_kind_identity, invalid_partial_kind)
    CELLSHARD_PARTIAL_CHECK_ID(payload_schema_identity, invalid_payload_schema)
    CELLSHARD_PARTIAL_CHECK_ID(contribution_coverage_identity,
                               invalid_contribution_coverage_identity)
    CELLSHARD_PARTIAL_CHECK_ID(dependency_closure_identity,
                               invalid_dependency_closure_identity)
    CELLSHARD_PARTIAL_CHECK_ID(reconstruction_algebra_identity,
                               invalid_reconstruction_algebra)
    CELLSHARD_PARTIAL_CHECK_ID(numerical_policy_identity,
                               invalid_numerical_policy)
    CELLSHARD_PARTIAL_CHECK_ID(complete_cost_evidence_identity,
                               invalid_complete_cost_evidence)
#undef CELLSHARD_PARTIAL_CHECK_ID
#define CELLSHARD_PARTIAL_CHECK_GENERATION(field, code) \
    if (header.field == 0) { \
        return {partial_atom_envelope_validation_code_v1::code}; \
    }
    CELLSHARD_PARTIAL_CHECK_GENERATION(structure_generation,
                                       missing_structure_generation)
    CELLSHARD_PARTIAL_CHECK_GENERATION(value_generation,
                                       missing_value_generation)
    CELLSHARD_PARTIAL_CHECK_GENERATION(state_generation,
                                       missing_state_generation)
    CELLSHARD_PARTIAL_CHECK_GENERATION(materialization_generation,
                                       missing_materialization_generation)
    CELLSHARD_PARTIAL_CHECK_GENERATION(cost_model_generation,
                                       missing_cost_model_generation)
#undef CELLSHARD_PARTIAL_CHECK_GENERATION
    if (header.persistence_class
        != partial_persistence_class_v1::exact_reconstructible) {
        return {partial_atom_envelope_validation_code_v1::
                    scientifically_unknown_partial};
    }
    if (partial.payload == nullptr) {
        return {partial_atom_envelope_validation_code_v1::missing_payload};
    }
    if (partial.payload_bytes == 0) {
        return {partial_atom_envelope_validation_code_v1::empty_payload};
    }
    if (partial.payload_alignment == 0
        || (partial.payload_alignment & (partial.payload_alignment - 1)) != 0) {
        return {partial_atom_envelope_validation_code_v1::
                    invalid_payload_alignment};
    }
    if (reinterpret_cast<std::uintptr_t>(partial.payload)
        % partial.payload_alignment != 0) {
        return {partial_atom_envelope_validation_code_v1::misaligned_payload};
    }
    if (partial.payload != partial.result.partial_layout.values
        || partial.payload_bytes != partial.result.partial_layout.value_bytes
        || partial.payload_alignment
            != partial.result.partial_layout.value_alignment) {
        return {partial_atom_envelope_validation_code_v1::
                    payload_binding_mismatch};
    }
    if (header.contribution_coverage_identity
        != partial.result.exact_contribution_coverage.coverage_identity) {
        return {partial_atom_envelope_validation_code_v1::
                    coverage_binding_mismatch};
    }
    if (header.reconstruction_algebra_identity
        != partial.result.reconstruction_algebra_identity) {
        return {partial_atom_envelope_validation_code_v1::
                    algebra_binding_mismatch};
    }
    if (header.numerical_policy_identity
        != partial.result.numerical_policy_identity) {
        return {partial_atom_envelope_validation_code_v1::
                    numerical_policy_binding_mismatch};
    }
    if (partial.result.status == atom_partial_result_status_v1::accumulating) {
        return {partial_atom_envelope_validation_code_v1::incomplete_partial};
    }
    if (header.reserved != 0 || partial.reserved != 0) {
        return {partial_atom_envelope_validation_code_v1::nonzero_reserved};
    }
    return {};
}

} // namespace cellshard::compiler::partial
