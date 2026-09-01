#pragma once

#include <CellShard/compiler/evidence/evidence_atlas_v1.hh>

#include <cstdint>

namespace cellshard::compiler::evidence {

enum class evidence_atlas_merge_code_v1 : std::uint32_t {
    merged = 0,
    invalid_left,
    invalid_right,
    output_limit_exceeded,
    conflicting_duplicate,
    allocation_failure,
    build_failure,
};

struct evidence_atlas_merge_result_v1 {
    evidence_atlas_merge_code_v1 code = evidence_atlas_merge_code_v1::merged;
    std::uint64_t index = 0;
    evidence_atlas_build_result_v1 build{};
    [[nodiscard]] constexpr bool merged() const noexcept {
        return code == evidence_atlas_merge_code_v1::merged;
    }
};

[[nodiscard]] constexpr bool atom_evidence_record_equal_v1(
    const atom_evidence_record_v1 &lhs,
    const atom_evidence_record_v1 &rhs) noexcept {
    return lhs.schema_version == rhs.schema_version
        && lhs.record_bytes == rhs.record_bytes
        && lhs.evidence_identity == rhs.evidence_identity
        && lhs.subject_atom_identity == rhs.subject_atom_identity
        && lhs.source_identity == rhs.source_identity
        && lhs.observation_generation == rhs.observation_generation
        && lhs.observation_count == rhs.observation_count
        && lhs.kind == rhs.kind
        && lhs.disposition == rhs.disposition;
}

[[nodiscard]] evidence_atlas_merge_result_v1 merge_evidence_atlases_v1(
    evidence_atlas_view_v1 left,
    evidence_atlas_view_v1 right,
    evidence_identity_v1 output_identity,
    std::uint64_t output_generation,
    std::uint64_t maximum_records,
    evidence_atlas_builder_v1 *output) noexcept;

} // namespace cellshard::compiler::evidence
