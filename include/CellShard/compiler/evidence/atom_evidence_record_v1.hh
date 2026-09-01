#pragma once

#include <CellShard/compiler/evidence/evidence_kind.hh>

#include <cstdint>
#include <type_traits>

namespace cellshard::compiler::evidence {

inline constexpr std::uint32_t atom_evidence_record_schema_version_v1 = 1;

// Namespace-qualified identity used by the standalone evidence atlas. The
// integration adapter maps atom identities field-by-field; values are never
// inferred from storage locations, pointers, ordinals, or payload digests.
struct evidence_identity_v1 {
    std::uint64_t producer_namespace = 0;
    std::uint64_t local_identity = 0;
};

enum class evidence_disposition_v1 : std::uint32_t {
    proposal_only = 1,
};

// Fixed, pointer-free record spine. Later evidence tables attach provenance,
// strata, confidence, membership, and mechanism-specific payloads by this
// record identity. No field is an exact-coverage certificate.
struct atom_evidence_record_v1 {
    std::uint32_t schema_version = atom_evidence_record_schema_version_v1;
    std::uint32_t record_bytes = sizeof(atom_evidence_record_v1);
    evidence_identity_v1 evidence_identity{};
    evidence_identity_v1 subject_atom_identity{};
    evidence_identity_v1 source_identity{};
    std::uint64_t observation_generation = 0;
    std::uint64_t observation_count = 0;
    evidence_kind kind = evidence_kind::invalid;
    evidence_disposition_v1 disposition =
        evidence_disposition_v1::proposal_only;
};

enum class atom_evidence_record_validation_code_v1 : std::uint32_t {
    valid = 0,
    unsupported_schema,
    invalid_record_bytes,
    invalid_evidence_identity,
    invalid_subject_atom_identity,
    invalid_source_identity,
    missing_observation_generation,
    empty_observations,
    invalid_kind,
    non_proposal_disposition,
};

struct atom_evidence_record_validation_v1 {
    atom_evidence_record_validation_code_v1 code =
        atom_evidence_record_validation_code_v1::valid;

    [[nodiscard]] constexpr bool valid() const noexcept {
        return code == atom_evidence_record_validation_code_v1::valid;
    }
};

[[nodiscard]] constexpr bool valid_evidence_identity_v1(
    evidence_identity_v1 identity) noexcept {
    return identity.producer_namespace != 0 && identity.local_identity != 0;
}

[[nodiscard]] constexpr bool operator==(
    evidence_identity_v1 lhs, evidence_identity_v1 rhs) noexcept {
    return lhs.producer_namespace == rhs.producer_namespace
        && lhs.local_identity == rhs.local_identity;
}

[[nodiscard]] constexpr bool evidence_identity_less_v1(
    evidence_identity_v1 lhs, evidence_identity_v1 rhs) noexcept {
    return lhs.producer_namespace < rhs.producer_namespace
        || (lhs.producer_namespace == rhs.producer_namespace
            && lhs.local_identity < rhs.local_identity);
}

[[nodiscard]] constexpr atom_evidence_record_validation_v1
validate_atom_evidence_record_v1(
    const atom_evidence_record_v1 &record) noexcept {
    if (record.schema_version != atom_evidence_record_schema_version_v1) {
        return {atom_evidence_record_validation_code_v1::unsupported_schema};
    }
    if (record.record_bytes != sizeof(atom_evidence_record_v1)) {
        return {atom_evidence_record_validation_code_v1::invalid_record_bytes};
    }
    if (!valid_evidence_identity_v1(record.evidence_identity)) {
        return {atom_evidence_record_validation_code_v1::
                    invalid_evidence_identity};
    }
    if (!valid_evidence_identity_v1(record.subject_atom_identity)) {
        return {atom_evidence_record_validation_code_v1::
                    invalid_subject_atom_identity};
    }
    if (!valid_evidence_identity_v1(record.source_identity)) {
        return {atom_evidence_record_validation_code_v1::
                    invalid_source_identity};
    }
    if (record.observation_generation == 0) {
        return {atom_evidence_record_validation_code_v1::
                    missing_observation_generation};
    }
    if (record.observation_count == 0) {
        return {atom_evidence_record_validation_code_v1::empty_observations};
    }
    if (!valid_evidence_kind(record.kind)) {
        return {atom_evidence_record_validation_code_v1::invalid_kind};
    }
    if (record.disposition != evidence_disposition_v1::proposal_only) {
        return {atom_evidence_record_validation_code_v1::
                    non_proposal_disposition};
    }
    return {};
}

static_assert(sizeof(evidence_identity_v1) == 16,
              "evidence identity ABI must remain two explicit u64 fields");
static_assert(std::is_standard_layout<evidence_identity_v1>::value);
static_assert(std::is_trivially_copyable<evidence_identity_v1>::value);
static_assert(sizeof(atom_evidence_record_v1) == 80,
              "atom evidence record ABI v1 size changed");
static_assert(std::is_standard_layout<atom_evidence_record_v1>::value);
static_assert(std::is_trivially_copyable<atom_evidence_record_v1>::value);

} // namespace cellshard::compiler::evidence
