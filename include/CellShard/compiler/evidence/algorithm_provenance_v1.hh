#pragma once

#include <CellShard/compiler/evidence/atom_evidence_record_v1.hh>
#include <CellShard/identity/digest.hh>

#include <cstdint>
#include <type_traits>

namespace cellshard::compiler::evidence {

inline constexpr std::uint32_t algorithm_provenance_schema_version_v1 = 1;

struct algorithm_provenance_v1 {
    std::uint32_t schema_version = algorithm_provenance_schema_version_v1;
    std::uint32_t record_bytes = sizeof(algorithm_provenance_v1);
    evidence_identity_v1 provenance_identity{};
    evidence_identity_v1 algorithm_identity{};
    evidence_identity_v1 execution_environment_identity{};
    content_digest implementation_digest{};
    content_digest parameter_digest{};
    std::uint64_t algorithm_revision = 0;
    std::uint64_t random_seed = 0;
};

enum class algorithm_provenance_validation_code_v1 : std::uint32_t {
    valid = 0,
    unsupported_schema,
    invalid_record_bytes,
    invalid_provenance_identity,
    invalid_algorithm_identity,
    invalid_environment_identity,
    invalid_implementation_digest,
    missing_implementation_digest,
    invalid_parameter_digest,
    missing_parameter_digest,
    missing_algorithm_revision,
};

struct algorithm_provenance_validation_v1 {
    algorithm_provenance_validation_code_v1 code =
        algorithm_provenance_validation_code_v1::valid;

    [[nodiscard]] constexpr bool valid() const noexcept {
        return code == algorithm_provenance_validation_code_v1::valid;
    }
};

[[nodiscard]] constexpr algorithm_provenance_validation_v1
validate_algorithm_provenance_v1(
    const algorithm_provenance_v1 &record) noexcept {
    if (record.schema_version != algorithm_provenance_schema_version_v1) {
        return {algorithm_provenance_validation_code_v1::unsupported_schema};
    }
    if (record.record_bytes != sizeof(algorithm_provenance_v1)) {
        return {algorithm_provenance_validation_code_v1::invalid_record_bytes};
    }
    if (!valid_evidence_identity_v1(record.provenance_identity)) {
        return {algorithm_provenance_validation_code_v1::
                    invalid_provenance_identity};
    }
    if (!valid_evidence_identity_v1(record.algorithm_identity)) {
        return {algorithm_provenance_validation_code_v1::
                    invalid_algorithm_identity};
    }
    if (!valid_evidence_identity_v1(record.execution_environment_identity)) {
        return {algorithm_provenance_validation_code_v1::
                    invalid_environment_identity};
    }
    if (!valid_content_digest(record.implementation_digest)) {
        return {algorithm_provenance_validation_code_v1::
                    invalid_implementation_digest};
    }
    if (record.implementation_digest.algorithm == digest_algorithm::none) {
        return {algorithm_provenance_validation_code_v1::
                    missing_implementation_digest};
    }
    if (!valid_content_digest(record.parameter_digest)) {
        return {algorithm_provenance_validation_code_v1::
                    invalid_parameter_digest};
    }
    if (record.parameter_digest.algorithm == digest_algorithm::none) {
        return {algorithm_provenance_validation_code_v1::
                    missing_parameter_digest};
    }
    if (record.algorithm_revision == 0) {
        return {algorithm_provenance_validation_code_v1::
                    missing_algorithm_revision};
    }
    return {};
}

static_assert(std::is_standard_layout<algorithm_provenance_v1>::value);
static_assert(std::is_trivially_copyable<algorithm_provenance_v1>::value);

} // namespace cellshard::compiler::evidence
