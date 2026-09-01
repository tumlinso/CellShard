#include <CellShard/compiler/certification/exact_atom_certificate_v1.hh>

#include <cassert>
#include <cstddef>
#include <cstdint>

using namespace cellshard;
using namespace cellshard::compiler;

int main() {
    alignas(8) std::byte coverage_record[248]{};
    atom::common_atom_view_v1 common{};
    common.identities.semantic_family = {{10, 1}};
    common.identities.materialization = {{11, 1}};
    common.identities.replica = {{12, 1}};
    common.identities.resident = {13, 1};
    common.identities.content.digest.algorithm =
        digest_algorithm::legacy_fnv1a64;
    common.identities.content.digest.used_bytes = 8;
    common.exact_coverage = {
        coverage_record,
        {14, 1},
        8,
        1,
        248,
        atom::atom_certified_exact_coverage_role_v1,
        atom::atom_logical_coverage_kind_v1::explicit_ids,
        0};
    common.lineage_generation = 4;

    std::uint64_t workspace = 0;
    certification::atom_certification_request_v1 request{};
    request.proposed_atoms = &common;
    request.workspace = &workspace;
    request.proposed_atom_count = 1;
    request.workspace_bytes = sizeof(workspace);
    request.request_identity = {1, 1};
    request.proposal_provider_identity = {2, 1};
    request.certification_authority_identity = {3, 1};
    request.canonical_source_identity = {4, 1};
    request.canonical_source_generation = 5;

    certification::atom_certification_result_v1 result{};
    result.request_identity = request.request_identity;
    result.certification_authority_identity =
        request.certification_authority_identity;
    result.proposed_atom_count = 1;
    result.certified_atom_count = 1;
    result.certified_entity_count = 8;
    result.outcome = certification::atom_certification_outcome_v1::certified;
    certification::exact_atom_certificate_summary_v1 summary{
        {20, 1},
        6,
        2,
        certification::required_atom_certificate_stage_mask_v1,
        7};
    certification::exact_atom_certificate_v1 certificate{};
    assert(certification::emit_exact_atom_certificate_v1(
               request, result, 0, summary, &certificate)
               .emitted());
    assert(certificate.contribution_owner_count == 6);
    assert(certificate.residual_count == 2);

    summary.completed_stage_mask &= ~certification::certification_stage_bit_v1(
        certification::atom_certification_stage_v1::dependency_closure);
    assert(certification::emit_exact_atom_certificate_v1(
               request, result, 0, summary, &certificate)
               .code
           == certification::exact_atom_certificate_emit_code_v1::
               incomplete_stages);

    summary.completed_stage_mask =
        certification::required_atom_certificate_stage_mask_v1;
    result.certification_authority_identity = request.proposal_provider_identity;
    assert(certification::emit_exact_atom_certificate_v1(
               request, result, 0, summary, &certificate)
               .code
           == certification::exact_atom_certificate_emit_code_v1::
               result_authority_mismatch);
}
