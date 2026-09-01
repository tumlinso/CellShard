#pragma once

#include <CellShard/compiler/atom/persistent_identity_v1.hh>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace cellshard::compiler::atom {

inline constexpr std::uint32_t atom_logical_coverage_contract_version_v1 = 1;
inline constexpr std::uint32_t cellerator_logical_coverage_schema_version_v1 = 1;
inline constexpr std::uint32_t cellerator_logical_coverage_record_bytes_v1 = 248;

// Values intentionally match Cellerator logical_coverage_kind_v1. CellShard
// records the source kind; it does not reinterpret membership semantics.
enum class atom_logical_coverage_kind_v1 : std::uint16_t {
    canonical_intervals = 1,
    explicit_ids = 2,
    relation_edge_ids = 3,
    semantic_components = 4,
    segment_set = 5,
    coverage_union = 6,
    provider_defined = 7,
};

enum atom_coverage_role_flag_v1 : std::uint32_t {
    atom_certified_exact_coverage_role_v1 = 1u << 0u,
    atom_approximate_proposal_membership_role_v1 = 1u << 1u,
    atom_exact_read_requirement_role_v1 = 1u << 2u,
    atom_read_only_halo_role_v1 = 1u << 3u,
    atom_physical_replica_role_v1 = 1u << 4u,
    atom_exclusive_output_owner_role_v1 = 1u << 5u,
    atom_partial_contribution_owner_role_v1 = 1u << 6u,
};

inline constexpr std::uint32_t atom_known_coverage_role_flags_v1 =
    atom_certified_exact_coverage_role_v1
    | atom_approximate_proposal_membership_role_v1
    | atom_exact_read_requirement_role_v1
    | atom_read_only_halo_role_v1
    | atom_physical_replica_role_v1
    | atom_exclusive_output_owner_role_v1
    | atom_partial_contribution_owner_role_v1;

// Pointer-first, non-owning binding to one independently validated Cellerator
// logical_coverage_view_v1. Membership stays owned by Cellerator/the caller;
// CellShard retains exact identity, kind, roles, and logical cardinality and
// never substitutes proposal or physical-overlap membership for certification.
struct atom_logical_coverage_ref_v1 {
    const void *cellerator_coverage = nullptr;
    atom_persistent_identity_v1 coverage_identity{};
    std::uint64_t logical_count = 0;
    std::uint32_t source_schema_version = 0;
    std::uint32_t source_record_bytes = 0;
    std::uint32_t role_flags = 0;
    atom_logical_coverage_kind_v1 kind =
        atom_logical_coverage_kind_v1::canonical_intervals;
    std::uint16_t reserved = 0;
};

enum class atom_logical_coverage_validation_code_v1 : std::uint32_t {
    valid = 0,
    missing_source,
    misaligned_source,
    unsupported_schema,
    invalid_record_bytes,
    invalid_coverage_identity,
    invalid_kind,
    missing_exact_certification,
    proposal_execution_mixture,
    unknown_role,
    empty_coverage,
    nonzero_reserved,
    source_validation_failed,
};

struct atom_logical_coverage_validation_v1 {
    atom_logical_coverage_validation_code_v1 code =
        atom_logical_coverage_validation_code_v1::valid;
    std::uint32_t source_validation_code = 0;

    [[nodiscard]] constexpr bool valid() const noexcept {
        return code == atom_logical_coverage_validation_code_v1::valid;
    }
};

static_assert(offsetof(atom_logical_coverage_ref_v1, cellerator_coverage) == 0,
              "logical coverage references must remain pointer-first");
static_assert(std::is_standard_layout<atom_logical_coverage_ref_v1>::value);
static_assert(std::is_trivially_copyable<atom_logical_coverage_ref_v1>::value);

[[nodiscard]] constexpr bool valid_atom_logical_coverage_kind_v1(
    atom_logical_coverage_kind_v1 kind) noexcept {
    const auto value = static_cast<std::uint16_t>(kind);
    return value >= 1 && value <= 7;
}

// O(1) time and O(1) storage. Exact member validation remains Cellerator's
// responsibility and its result is carried explicitly as source_validation.
[[nodiscard]] inline atom_logical_coverage_validation_v1
validate_atom_logical_coverage_ref_v1(
    const atom_logical_coverage_ref_v1 &coverage,
    std::uint32_t source_validation) noexcept {
    if (coverage.cellerator_coverage == nullptr) {
        return {atom_logical_coverage_validation_code_v1::missing_source,
                source_validation};
    }
    if (reinterpret_cast<std::uintptr_t>(coverage.cellerator_coverage)
        % alignof(std::uint64_t) != 0) {
        return {atom_logical_coverage_validation_code_v1::misaligned_source,
                source_validation};
    }
    if (coverage.source_schema_version
        != cellerator_logical_coverage_schema_version_v1) {
        return {atom_logical_coverage_validation_code_v1::unsupported_schema,
                source_validation};
    }
    if (coverage.source_record_bytes
        != cellerator_logical_coverage_record_bytes_v1) {
        return {atom_logical_coverage_validation_code_v1::invalid_record_bytes,
                source_validation};
    }
    if (!validate_atom_persistent_identity_v1(coverage.coverage_identity)
             .valid()) {
        return {
            atom_logical_coverage_validation_code_v1::invalid_coverage_identity,
            source_validation};
    }
    if (!valid_atom_logical_coverage_kind_v1(coverage.kind)) {
        return {atom_logical_coverage_validation_code_v1::invalid_kind,
                source_validation};
    }
    if ((coverage.role_flags & ~atom_known_coverage_role_flags_v1) != 0) {
        return {atom_logical_coverage_validation_code_v1::unknown_role,
                source_validation};
    }
    if ((coverage.role_flags & atom_certified_exact_coverage_role_v1) == 0) {
        return {
            atom_logical_coverage_validation_code_v1::missing_exact_certification,
            source_validation};
    }
    if ((coverage.role_flags
         & atom_approximate_proposal_membership_role_v1) != 0) {
        return {
            atom_logical_coverage_validation_code_v1::proposal_execution_mixture,
            source_validation};
    }
    if (coverage.logical_count == 0) {
        return {atom_logical_coverage_validation_code_v1::empty_coverage,
                source_validation};
    }
    if (coverage.reserved != 0) {
        return {atom_logical_coverage_validation_code_v1::nonzero_reserved,
                source_validation};
    }
    if (source_validation != 0) {
        return {
            atom_logical_coverage_validation_code_v1::source_validation_failed,
            source_validation};
    }
    return {};
}

} // namespace cellshard::compiler::atom
