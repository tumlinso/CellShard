#include <CellShard/compiler/certification/independent_verifier_v1.hh>

#include <array>
#include <cassert>
#include <cstdint>

using namespace cellshard;
using namespace cellshard::compiler;

namespace {

std::uint64_t oracle_hash_byte(std::uint64_t digest, std::uint8_t value) {
    return (digest ^ value) * UINT64_C(1099511628211);
}

std::uint64_t oracle_hash_u64(std::uint64_t digest, std::uint64_t value) {
    for (std::uint8_t byte = 0; byte < 8; ++byte) {
        digest = oracle_hash_byte(
            digest, static_cast<std::uint8_t>(value >> (byte * 8)));
    }
    return digest;
}

std::uint64_t oracle_member_digest(
    const certification::certified_member_record_v1 *members,
    std::uint64_t count) {
    std::uint64_t digest = UINT64_C(14695981039346656037);
    for (std::uint64_t index = 0; index < count; ++index) {
        const auto &member = members[index];
        digest = oracle_hash_byte(
            digest, static_cast<std::uint8_t>(member.kind));
        digest = oracle_hash_u64(
            digest, member.owner_space_identity.producer_namespace);
        digest = oracle_hash_u64(
            digest, member.owner_space_identity.local_identity);
        digest = oracle_hash_u64(digest, member.global_identity);
        digest = oracle_hash_byte(
            digest, static_cast<std::uint8_t>(member.disposition));
        digest = oracle_hash_u64(
            digest, member.contribution_owner_atom_index);
    }
    return digest;
}

certification::exact_atom_certificate_v1 make_certificate() {
    certification::exact_atom_certificate_v1 certificate{};
    certificate.certificate_identity = {1, 1};
    certificate.request_identity = {2, 1};
    certificate.proposal_provider_identity = {3, 1};
    certificate.certification_authority_identity = {4, 1};
    certificate.canonical_source_identity = {5, 1};
    certificate.semantic_family = {{6, 1}};
    certificate.materialization = {{7, 1}};
    certificate.exact_coverage_identity = {8, 1};
    certificate.content.digest.algorithm = digest_algorithm::legacy_fnv1a64;
    certificate.content.digest.used_bytes = 8;
    certificate.content.digest.bytes[0] = std::byte{0x42};
    certificate.canonical_source_generation = 9;
    certificate.lineage_generation = 10;
    certificate.certified_entity_count = 3;
    certificate.certified_relation_edge_count = 2;
    certificate.contribution_owner_count = 3;
    certificate.residual_count = 2;
    certificate.completed_stage_mask =
        certification::required_atom_certificate_stage_mask_v1;
    certificate.certification_generation = 11;
    certificate.atom_index = 0;
    return certificate;
}

certification::exact_atom_verification_expectation_v1 make_expectation(
    const certification::exact_atom_certificate_v1 &certificate,
    std::uint64_t digest) {
    return {certificate.certificate_identity,
            certificate.request_identity,
            certificate.proposal_provider_identity,
            certificate.certification_authority_identity,
            certificate.canonical_source_identity,
            certificate.semantic_family,
            certificate.materialization,
            certificate.exact_coverage_identity,
            certificate.content,
            certificate.canonical_source_generation,
            certificate.lineage_generation,
            certificate.certified_entity_count,
            certificate.certified_relation_edge_count,
            certificate.contribution_owner_count,
            certificate.residual_count,
            certificate.completed_stage_mask,
            certificate.certification_generation,
            certificate.atom_index,
            digest};
}

} // namespace

int main() {
    using certification::certification_member_kind_v1;
    using certification::certified_member_disposition_v1;
    using certification::certified_member_record_v1;
    const std::array<certified_member_record_v1, 5> members{{
        {{20, 1},
         100,
         0,
         certification_member_kind_v1::entity,
         certified_member_disposition_v1::contribution_owner,
         {}},
        {{20, 1},
         200,
         certification::no_failed_certification_index_v1,
         certification_member_kind_v1::entity,
         certified_member_disposition_v1::residual,
         {}},
        {{20, 1},
         300,
         1,
         certification_member_kind_v1::entity,
         certified_member_disposition_v1::contribution_owner,
         {}},
        {{30, 1},
         1000,
         0,
         certification_member_kind_v1::relation_edge,
         certified_member_disposition_v1::contribution_owner,
         {}},
        {{30, 1},
         2000,
         certification::no_failed_certification_index_v1,
         certification_member_kind_v1::relation_edge,
         certified_member_disposition_v1::residual,
         {}}}};
    const auto oracle_digest =
        oracle_member_digest(members.data(), members.size());
    const auto certificate = make_certificate();
    const auto expectation = make_expectation(certificate, oracle_digest);
    certification::independent_atom_verifier_v1 verifier{};
    assert(certification::begin_independent_atom_verification_v1(
               certificate, expectation, &verifier)
               .valid());
    for (const auto member : members) {
        assert(certification::update_independent_atom_verification_v1(
                   &verifier, member)
                   .valid());
    }
    assert(certification::finish_independent_atom_verification_v1(&verifier)
               .valid());

    auto corrupt = certificate;
    corrupt.schema_version = 2;
    assert(!certification::begin_independent_atom_verification_v1(
                corrupt, expectation, &verifier)
                .valid());
    corrupt = certificate;
    corrupt.proposal_provider_identity =
        corrupt.certification_authority_identity;
    assert(certification::begin_independent_atom_verification_v1(
               corrupt, expectation, &verifier)
               .code
           == certification::independent_verification_code_v1::
               provider_self_certification);
    corrupt = certificate;
    corrupt.completed_stage_mask ^= UINT64_C(1) << 63;
    assert(certification::begin_independent_atom_verification_v1(
               corrupt, expectation, &verifier)
               .code
           == certification::independent_verification_code_v1::
               incomplete_or_unknown_stages);
    corrupt = certificate;
    ++corrupt.canonical_source_generation;
    assert(certification::begin_independent_atom_verification_v1(
               corrupt, expectation, &verifier)
               .code
           == certification::independent_verification_code_v1::
               expectation_mismatch);

    assert(certification::begin_independent_atom_verification_v1(
               certificate, expectation, &verifier)
               .valid());
    assert(certification::update_independent_atom_verification_v1(
               &verifier, members[0])
               .valid());
    assert(certification::update_independent_atom_verification_v1(
               &verifier, members[0])
               .code
           == certification::independent_verification_code_v1::
               unordered_or_duplicate_member);

    auto bad_expectation = expectation;
    bad_expectation.member_stream_digest ^= 1;
    assert(certification::begin_independent_atom_verification_v1(
               certificate, bad_expectation, &verifier)
               .valid());
    for (const auto member : members) {
        assert(certification::update_independent_atom_verification_v1(
                   &verifier, member)
                   .valid());
    }
    assert(certification::finish_independent_atom_verification_v1(&verifier)
               .code
           == certification::independent_verification_code_v1::
               stream_digest_mismatch);
}
