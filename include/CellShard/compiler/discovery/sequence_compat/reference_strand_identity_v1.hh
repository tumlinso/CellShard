#pragma once

#include <CellShard/compiler/atom/persistent_identity_v1.hh>

#include <cstdint>
#include <type_traits>

namespace cellshard::compiler::discovery::sequence_compat {

inline constexpr std::uint32_t reference_strand_identity_schema_version_v1 = 1;

enum class strand_identity_v1 : std::uint8_t {
    forward = 1,
    reverse = 2,
    both = 3,
    unknown = 4,
};

// Assembly and sequence identities are separate: a coordinate and extent do
// not establish which biological reference was used. Unknown strand is a
// deliberate provider statement, not an omitted value.
struct reference_strand_identity_v1 {
    std::uint32_t schema_version =
        reference_strand_identity_schema_version_v1;
    std::uint32_t record_bytes = sizeof(reference_strand_identity_v1);
    atom::atom_persistent_identity_v1 assembly_identity{};
    atom::atom_persistent_identity_v1 sequence_identity{};
    strand_identity_v1 strand = strand_identity_v1::unknown;
    std::uint8_t reserved[7]{};
};

enum class reference_strand_identity_validation_code_v1 : std::uint32_t {
    valid = 0,
    unsupported_schema,
    invalid_record_bytes,
    invalid_assembly_identity,
    invalid_sequence_identity,
    collapsed_assembly_sequence_identity,
    invalid_strand,
    nonzero_reserved,
};

struct reference_strand_identity_validation_v1 {
    reference_strand_identity_validation_code_v1 code =
        reference_strand_identity_validation_code_v1::valid;

    [[nodiscard]] constexpr bool valid() const noexcept {
        return code == reference_strand_identity_validation_code_v1::valid;
    }
};

static_assert(std::is_standard_layout<reference_strand_identity_v1>::value);
static_assert(
    std::is_trivially_copyable<reference_strand_identity_v1>::value);

[[nodiscard]] constexpr bool valid_strand_identity_v1(
    strand_identity_v1 strand) noexcept {
    const auto value = static_cast<std::uint8_t>(strand);
    return value >= static_cast<std::uint8_t>(strand_identity_v1::forward)
        && value <= static_cast<std::uint8_t>(strand_identity_v1::unknown);
}

[[nodiscard]] constexpr reference_strand_identity_validation_v1
validate_reference_strand_identity_v1(
    const reference_strand_identity_v1 &identity) noexcept {
    if (identity.schema_version
        != reference_strand_identity_schema_version_v1) {
        return {reference_strand_identity_validation_code_v1::
                    unsupported_schema};
    }
    if (identity.record_bytes != sizeof(reference_strand_identity_v1)) {
        return {reference_strand_identity_validation_code_v1::
                    invalid_record_bytes};
    }
    if (!atom::validate_atom_persistent_identity_v1(
             identity.assembly_identity)
             .valid()) {
        return {reference_strand_identity_validation_code_v1::
                    invalid_assembly_identity};
    }
    if (!atom::validate_atom_persistent_identity_v1(
             identity.sequence_identity)
             .valid()) {
        return {reference_strand_identity_validation_code_v1::
                    invalid_sequence_identity};
    }
    if (identity.assembly_identity == identity.sequence_identity) {
        return {reference_strand_identity_validation_code_v1::
                    collapsed_assembly_sequence_identity};
    }
    if (!valid_strand_identity_v1(identity.strand)) {
        return {reference_strand_identity_validation_code_v1::invalid_strand};
    }
    for (const auto item : identity.reserved) {
        if (item != 0) {
            return {reference_strand_identity_validation_code_v1::
                        nonzero_reserved};
        }
    }
    return {};
}

} // namespace cellshard::compiler::discovery::sequence_compat
