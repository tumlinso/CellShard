#pragma once

#include <CellShard/compiler/atom/persistent_identity_v1.hh>
#include <CellShard/identity/digest.hh>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace cellshard::compiler::atom {

inline constexpr std::uint32_t atom_evidence_plane_schema_version_v1 = 1;

enum class atom_evidence_kind_v1 : std::uint32_t {
    biological_observation = 1,
    statistical_proposal = 2,
    structural_support = 3,
    performance_measurement = 4,
    provider_defined = 5,
};

// Evidence is source-linked descriptive input. Even a confidence of 1/1 does
// not certify exact logical coverage or assign execution-contribution ownership.
struct atom_evidence_record_ref_v1 {
    const void *record = nullptr;
    std::uint64_t record_bytes = 0;
    atom_persistent_identity_v1 record_identity{};
    atom_persistent_identity_v1 provenance_identity{};
    atom_persistent_identity_v1 provenance_schema{};
    atom_persistent_identity_v1 method_identity{};
    atom_persistent_identity_v1 subject_identity{};
    content_digest record_digest{};
    std::uint64_t observation_generation = 0;
    std::uint64_t confidence_numerator = 0;
    std::uint64_t confidence_denominator = 0;
    std::uint32_t record_alignment = 0;
    atom_evidence_kind_v1 kind = atom_evidence_kind_v1::biological_observation;
};

struct atom_evidence_plane_v1 {
    const atom_evidence_record_ref_v1 *records = nullptr;
    std::uint64_t record_count = 0;
    atom_persistent_identity_v1 plane_identity{};
    std::uint64_t evidence_generation = 0;
};

enum class atom_evidence_validation_code_v1 : std::uint32_t {
    valid = 0,
    empty_records,
    missing_records,
    invalid_plane_identity,
    missing_evidence_generation,
    missing_record,
    empty_record,
    invalid_record_alignment,
    misaligned_record,
    invalid_record_identity,
    unordered_or_duplicate_record,
    invalid_provenance_identity,
    invalid_provenance_schema,
    invalid_method_identity,
    invalid_subject_identity,
    invalid_record_digest,
    missing_record_digest,
    missing_observation_generation,
    invalid_confidence,
    invalid_evidence_kind,
};

struct atom_evidence_validation_v1 {
    atom_evidence_validation_code_v1 code =
        atom_evidence_validation_code_v1::valid;
    std::uint64_t index = 0;

    [[nodiscard]] constexpr bool valid() const noexcept {
        return code == atom_evidence_validation_code_v1::valid;
    }
};

static_assert(offsetof(atom_evidence_record_ref_v1, record) == 0,
              "evidence record references must remain pointer-first");
static_assert(std::is_standard_layout<atom_evidence_record_ref_v1>::value);
static_assert(std::is_trivially_copyable<atom_evidence_record_ref_v1>::value);
static_assert(offsetof(atom_evidence_plane_v1, records) == 0,
              "evidence planes must remain pointer-first");
static_assert(std::is_standard_layout<atom_evidence_plane_v1>::value);
static_assert(std::is_trivially_copyable<atom_evidence_plane_v1>::value);

[[nodiscard]] constexpr bool valid_atom_evidence_kind_v1(
    atom_evidence_kind_v1 kind) noexcept {
    const auto value = static_cast<std::uint32_t>(kind);
    return value >= 1 && value <= 5;
}

// Records are sorted by namespace-qualified record identity. Validation is
// O(record_count), O(1) storage, allocation-free, and never interprets payloads.
[[nodiscard]] inline atom_evidence_validation_v1
validate_atom_evidence_plane_v1(const atom_evidence_plane_v1 &plane) noexcept {
    if (plane.record_count == 0) {
        return {atom_evidence_validation_code_v1::empty_records, 0};
    }
    if (plane.records == nullptr) {
        return {atom_evidence_validation_code_v1::missing_records, 0};
    }
    if (!validate_atom_persistent_identity_v1(plane.plane_identity).valid()) {
        return {atom_evidence_validation_code_v1::invalid_plane_identity, 0};
    }
    if (plane.evidence_generation == 0) {
        return {atom_evidence_validation_code_v1::
                    missing_evidence_generation,
                0};
    }
    for (std::uint64_t index = 0; index < plane.record_count; ++index) {
        const auto &record = plane.records[index];
        if (record.record == nullptr) {
            return {atom_evidence_validation_code_v1::missing_record, index};
        }
        if (record.record_bytes == 0) {
            return {atom_evidence_validation_code_v1::empty_record, index};
        }
        if (record.record_alignment == 0
            || (record.record_alignment & (record.record_alignment - 1)) != 0) {
            return {atom_evidence_validation_code_v1::
                        invalid_record_alignment,
                    index};
        }
        if (reinterpret_cast<std::uintptr_t>(record.record)
            % record.record_alignment != 0) {
            return {atom_evidence_validation_code_v1::misaligned_record,
                    index};
        }
        if (!validate_atom_persistent_identity_v1(record.record_identity)
                 .valid()) {
            return {atom_evidence_validation_code_v1::invalid_record_identity,
                    index};
        }
        if (index != 0
            && !atom_persistent_identity_less_v1(
                plane.records[index - 1].record_identity,
                record.record_identity)) {
            return {atom_evidence_validation_code_v1::
                        unordered_or_duplicate_record,
                    index};
        }
#define CELLSHARD_ATOM_EVIDENCE_CHECK_ID(field, code) \
    if (!validate_atom_persistent_identity_v1(record.field).valid()) { \
        return {atom_evidence_validation_code_v1::code, index}; \
    }
        CELLSHARD_ATOM_EVIDENCE_CHECK_ID(provenance_identity,
                                         invalid_provenance_identity)
        CELLSHARD_ATOM_EVIDENCE_CHECK_ID(provenance_schema,
                                         invalid_provenance_schema)
        CELLSHARD_ATOM_EVIDENCE_CHECK_ID(method_identity,
                                         invalid_method_identity)
        CELLSHARD_ATOM_EVIDENCE_CHECK_ID(subject_identity,
                                         invalid_subject_identity)
#undef CELLSHARD_ATOM_EVIDENCE_CHECK_ID
        if (!valid_content_digest(record.record_digest)) {
            return {atom_evidence_validation_code_v1::invalid_record_digest,
                    index};
        }
        if (record.record_digest.algorithm == digest_algorithm::none) {
            return {atom_evidence_validation_code_v1::missing_record_digest,
                    index};
        }
        if (record.observation_generation == 0) {
            return {atom_evidence_validation_code_v1::
                        missing_observation_generation,
                    index};
        }
        if (record.confidence_denominator == 0
            || record.confidence_numerator > record.confidence_denominator) {
            return {atom_evidence_validation_code_v1::invalid_confidence,
                    index};
        }
        if (!valid_atom_evidence_kind_v1(record.kind)) {
            return {atom_evidence_validation_code_v1::invalid_evidence_kind,
                    index};
        }
    }
    return {atom_evidence_validation_code_v1::valid, plane.record_count};
}

} // namespace cellshard::compiler::atom
