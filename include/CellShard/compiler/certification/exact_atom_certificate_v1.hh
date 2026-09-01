#pragma once

#include <CellShard/compiler/certification/atom_certification_v1.hh>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace cellshard::compiler::certification {

inline constexpr std::uint32_t exact_atom_certificate_schema_version_v1 = 1;

[[nodiscard]] constexpr std::uint64_t certification_stage_bit_v1(
    atom_certification_stage_v1 stage) noexcept {
    return UINT64_C(1) << static_cast<std::uint32_t>(stage);
}

inline constexpr std::uint64_t required_atom_certificate_stage_mask_v1 =
    certification_stage_bit_v1(atom_certification_stage_v1::canonical_domains)
    | certification_stage_bit_v1(atom_certification_stage_v1::entity_coverage)
    | certification_stage_bit_v1(
        atom_certification_stage_v1::relation_edge_coverage)
    | certification_stage_bit_v1(
        atom_certification_stage_v1::duplicate_detection)
    | certification_stage_bit_v1(atom_certification_stage_v1::local_maps)
    | certification_stage_bit_v1(
        atom_certification_stage_v1::read_only_halos)
    | certification_stage_bit_v1(
        atom_certification_stage_v1::physical_replicas)
    | certification_stage_bit_v1(
        atom_certification_stage_v1::contribution_owners)
    | certification_stage_bit_v1(
        atom_certification_stage_v1::residual_coverage)
    | certification_stage_bit_v1(
        atom_certification_stage_v1::multimodal_identity)
    | certification_stage_bit_v1(
        atom_certification_stage_v1::trajectory_lineage)
    | certification_stage_bit_v1(
        atom_certification_stage_v1::partial_result_algebra)
    | certification_stage_bit_v1(
        atom_certification_stage_v1::dependency_closure);

// Pointer-free certificate payload. Persistence encodes named fields in the
// artifact byte order; native struct bytes are never the wire representation.
struct exact_atom_certificate_v1 {
    std::uint32_t schema_version = exact_atom_certificate_schema_version_v1;
    std::uint32_t record_bytes = sizeof(exact_atom_certificate_v1);
    atom::atom_persistent_identity_v1 certificate_identity{};
    atom::atom_persistent_identity_v1 request_identity{};
    atom::atom_persistent_identity_v1 proposal_provider_identity{};
    atom::atom_persistent_identity_v1 certification_authority_identity{};
    atom::atom_persistent_identity_v1 canonical_source_identity{};
    atom::atom_semantic_family_id_v1 semantic_family{};
    atom::atom_materialization_id_v1 materialization{};
    atom::atom_persistent_identity_v1 exact_coverage_identity{};
    atom::atom_content_id_v1 content{};
    std::uint64_t canonical_source_generation = 0;
    std::uint64_t lineage_generation = 0;
    std::uint64_t certified_entity_count = 0;
    std::uint64_t certified_relation_edge_count = 0;
    std::uint64_t contribution_owner_count = 0;
    std::uint64_t residual_count = 0;
    std::uint64_t completed_stage_mask = 0;
    std::uint64_t certification_generation = 0;
    std::uint64_t atom_index = 0;
};

struct exact_atom_certificate_summary_v1 {
    atom::atom_persistent_identity_v1 certificate_identity{};
    std::uint64_t contribution_owner_count = 0;
    std::uint64_t residual_count = 0;
    std::uint64_t completed_stage_mask = 0;
    std::uint64_t certification_generation = 0;
};

enum class exact_atom_certificate_emit_code_v1 : std::uint32_t {
    emitted = 0,
    null_destination,
    invalid_request,
    result_not_certified,
    result_request_mismatch,
    result_authority_mismatch,
    atom_index_out_of_range,
    invalid_atom_identity,
    invalid_exact_coverage,
    invalid_certificate_identity,
    incomplete_stages,
    missing_certification_generation,
    count_mismatch,
};

struct exact_atom_certificate_emit_result_v1 {
    exact_atom_certificate_emit_code_v1 code =
        exact_atom_certificate_emit_code_v1::emitted;
    std::uint32_t nested_code = 0;

    [[nodiscard]] constexpr bool emitted() const noexcept {
        return code == exact_atom_certificate_emit_code_v1::emitted;
    }
};

static_assert(std::is_standard_layout<exact_atom_certificate_v1>::value);
static_assert(std::is_trivially_copyable<exact_atom_certificate_v1>::value);
static_assert(std::is_standard_layout<exact_atom_certificate_summary_v1>::value);
static_assert(
    std::is_trivially_copyable<exact_atom_certificate_summary_v1>::value);

[[nodiscard]] inline exact_atom_certificate_emit_result_v1
emit_exact_atom_certificate_v1(
    const atom_certification_request_v1 &request,
    const atom_certification_result_v1 &result,
    std::uint64_t atom_index,
    exact_atom_certificate_summary_v1 summary,
    exact_atom_certificate_v1 *destination) noexcept {
    if (destination == nullptr) {
        return {exact_atom_certificate_emit_code_v1::null_destination};
    }
    *destination = {};
    const auto request_validation =
        validate_atom_certification_request_v1(request);
    if (!request_validation.valid()) {
        return {exact_atom_certificate_emit_code_v1::invalid_request,
                static_cast<std::uint32_t>(request_validation.code)};
    }
    if (!result.certified()) {
        return {exact_atom_certificate_emit_code_v1::result_not_certified};
    }
    if (result.request_identity != request.request_identity) {
        return {exact_atom_certificate_emit_code_v1::result_request_mismatch};
    }
    if (result.certification_authority_identity
        != request.certification_authority_identity) {
        return {exact_atom_certificate_emit_code_v1::
                    result_authority_mismatch};
    }
    if (atom_index >= request.proposed_atom_count
        || atom_index >= result.certified_atom_count) {
        return {exact_atom_certificate_emit_code_v1::atom_index_out_of_range};
    }
    const auto &common = request.proposed_atoms[atom_index];
    const auto identity_validation =
        atom::validate_atom_identity_binding_v1(common.identities);
    if (!identity_validation.valid()) {
        return {exact_atom_certificate_emit_code_v1::invalid_atom_identity,
                static_cast<std::uint32_t>(identity_validation.code)};
    }
    const auto coverage_validation =
        atom::validate_atom_logical_coverage_ref_v1(common.exact_coverage, 0);
    if (!coverage_validation.valid()) {
        return {exact_atom_certificate_emit_code_v1::invalid_exact_coverage,
                static_cast<std::uint32_t>(coverage_validation.code)};
    }
    if (!atom::validate_atom_persistent_identity_v1(summary.certificate_identity)
             .valid()) {
        return {exact_atom_certificate_emit_code_v1::
                    invalid_certificate_identity};
    }
    if ((summary.completed_stage_mask & required_atom_certificate_stage_mask_v1)
        != required_atom_certificate_stage_mask_v1) {
        return {exact_atom_certificate_emit_code_v1::incomplete_stages};
    }
    if (summary.certification_generation == 0) {
        return {exact_atom_certificate_emit_code_v1::
                    missing_certification_generation};
    }
    if (summary.contribution_owner_count
            > UINT64_MAX - summary.residual_count
        || summary.contribution_owner_count + summary.residual_count
               != common.exact_coverage.logical_count) {
        return {exact_atom_certificate_emit_code_v1::count_mismatch};
    }
    *destination = {exact_atom_certificate_schema_version_v1,
                    sizeof(exact_atom_certificate_v1),
                    summary.certificate_identity,
                    request.request_identity,
                    request.proposal_provider_identity,
                    request.certification_authority_identity,
                    request.canonical_source_identity,
                    common.identities.semantic_family,
                    common.identities.materialization,
                    common.exact_coverage.coverage_identity,
                    common.identities.content,
                    request.canonical_source_generation,
                    common.lineage_generation,
                    result.certified_entity_count,
                    result.certified_relation_edge_count,
                    summary.contribution_owner_count,
                    summary.residual_count,
                    summary.completed_stage_mask,
                    summary.certification_generation,
                    atom_index};
    return {};
}

} // namespace cellshard::compiler::certification
