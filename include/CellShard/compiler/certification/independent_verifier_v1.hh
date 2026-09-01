#pragma once

#include <CellShard/compiler/certification/duplicate_detection_v1.hh>
#include <CellShard/compiler/certification/exact_atom_certificate_v1.hh>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace cellshard::compiler::certification {

inline constexpr std::uint32_t independent_verifier_contract_version_v1 = 1;
inline constexpr std::uint64_t certification_stream_fnv_offset_v1 =
    UINT64_C(14695981039346656037);
inline constexpr std::uint64_t certification_stream_fnv_prime_v1 =
    UINT64_C(1099511628211);

enum class certified_member_disposition_v1 : std::uint8_t {
    contribution_owner = 1,
    residual = 2,
};

struct certified_member_record_v1 {
    atom::atom_persistent_identity_v1 owner_space_identity{};
    std::uint64_t global_identity = 0;
    std::uint64_t contribution_owner_atom_index =
        no_failed_certification_index_v1;
    certification_member_kind_v1 kind =
        certification_member_kind_v1::entity;
    certified_member_disposition_v1 disposition =
        certified_member_disposition_v1::residual;
    std::uint8_t reserved[6]{};
};

// Built from the independent canonical oracle, not from proposal-builder output
// or the certificate emitter. It binds every certificate field and the exact
// digest of the sorted certified-member stream.
struct exact_atom_verification_expectation_v1 {
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
    std::uint64_t member_stream_digest = 0;
};

enum class independent_verification_code_v1 : std::uint32_t {
    valid = 0,
    invalid_certificate_schema,
    invalid_certificate_identity,
    provider_self_certification,
    incomplete_or_unknown_stages,
    invalid_generation,
    invalid_content_digest,
    certificate_count_overflow,
    certificate_count_mismatch,
    expectation_mismatch,
    stream_not_active,
    zero_member_identity,
    invalid_member_kind,
    invalid_disposition,
    invalid_owner_atom_index,
    nonzero_reserved,
    unordered_or_duplicate_member,
    streamed_count_mismatch,
    stream_digest_mismatch,
};

struct independent_verification_result_v1 {
    independent_verification_code_v1 code =
        independent_verification_code_v1::valid;
    std::uint64_t index = 0;

    [[nodiscard]] constexpr bool valid() const noexcept {
        return code == independent_verification_code_v1::valid;
    }
};

struct independent_atom_verifier_v1 {
    exact_atom_certificate_v1 certificate{};
    exact_atom_verification_expectation_v1 expectation{};
    certified_member_record_v1 previous{};
    std::uint64_t stream_digest = certification_stream_fnv_offset_v1;
    std::uint64_t member_count = 0;
    std::uint64_t entity_count = 0;
    std::uint64_t relation_edge_count = 0;
    std::uint64_t contribution_owner_count = 0;
    std::uint64_t residual_count = 0;
    bool active = false;
    bool has_previous = false;
};

static_assert(std::is_standard_layout<certified_member_record_v1>::value);
static_assert(std::is_trivially_copyable<certified_member_record_v1>::value);

[[nodiscard]] constexpr bool certified_member_less_v1(
    const certified_member_record_v1 &lhs,
    const certified_member_record_v1 &rhs) noexcept {
    if (lhs.kind != rhs.kind) {
        return static_cast<std::uint8_t>(lhs.kind)
            < static_cast<std::uint8_t>(rhs.kind);
    }
    if (lhs.owner_space_identity != rhs.owner_space_identity) {
        return atom::atom_persistent_identity_less_v1(
            lhs.owner_space_identity, rhs.owner_space_identity);
    }
    return lhs.global_identity < rhs.global_identity;
}

inline void verification_hash_byte_v1(
    std::uint64_t &digest,
    std::uint8_t value) noexcept {
    digest ^= value;
    digest *= certification_stream_fnv_prime_v1;
}

inline void verification_hash_u64_v1(
    std::uint64_t &digest,
    std::uint64_t value) noexcept {
    for (std::uint8_t byte = 0; byte < 8; ++byte) {
        verification_hash_byte_v1(
            digest, static_cast<std::uint8_t>(value >> (byte * 8)));
    }
}

inline void verification_hash_member_v1(
    std::uint64_t &digest,
    const certified_member_record_v1 &member) noexcept {
    verification_hash_byte_v1(digest, static_cast<std::uint8_t>(member.kind));
    verification_hash_u64_v1(
        digest, member.owner_space_identity.producer_namespace);
    verification_hash_u64_v1(digest, member.owner_space_identity.local_identity);
    verification_hash_u64_v1(digest, member.global_identity);
    verification_hash_byte_v1(
        digest, static_cast<std::uint8_t>(member.disposition));
    verification_hash_u64_v1(digest, member.contribution_owner_atom_index);
}

[[nodiscard]] inline bool certificate_matches_expectation_v1(
    const exact_atom_certificate_v1 &certificate,
    const exact_atom_verification_expectation_v1 &expected) noexcept {
    return certificate.certificate_identity == expected.certificate_identity
        && certificate.request_identity == expected.request_identity
        && certificate.proposal_provider_identity
               == expected.proposal_provider_identity
        && certificate.certification_authority_identity
               == expected.certification_authority_identity
        && certificate.canonical_source_identity
               == expected.canonical_source_identity
        && certificate.semantic_family == expected.semantic_family
        && certificate.materialization == expected.materialization
        && certificate.exact_coverage_identity
               == expected.exact_coverage_identity
        && certificate.content == expected.content
        && certificate.canonical_source_generation
               == expected.canonical_source_generation
        && certificate.lineage_generation == expected.lineage_generation
        && certificate.certified_entity_count
               == expected.certified_entity_count
        && certificate.certified_relation_edge_count
               == expected.certified_relation_edge_count
        && certificate.contribution_owner_count
               == expected.contribution_owner_count
        && certificate.residual_count == expected.residual_count
        && certificate.completed_stage_mask == expected.completed_stage_mask
        && certificate.certification_generation
               == expected.certification_generation
        && certificate.atom_index == expected.atom_index;
}

[[nodiscard]] inline independent_verification_result_v1
begin_independent_atom_verification_v1(
    const exact_atom_certificate_v1 &certificate,
    const exact_atom_verification_expectation_v1 &expectation,
    independent_atom_verifier_v1 *verifier) noexcept {
    if (verifier == nullptr) {
        return {independent_verification_code_v1::stream_not_active};
    }
    *verifier = {};
    if (certificate.schema_version != exact_atom_certificate_schema_version_v1
        || certificate.record_bytes != sizeof(exact_atom_certificate_v1)) {
        return {independent_verification_code_v1::invalid_certificate_schema};
    }
    if (!atom::validate_atom_persistent_identity_v1(
             certificate.certificate_identity)
             .valid()) {
        return {independent_verification_code_v1::
                    invalid_certificate_identity};
    }
    if (certificate.proposal_provider_identity
        == certificate.certification_authority_identity) {
        return {independent_verification_code_v1::provider_self_certification};
    }
    if (certificate.completed_stage_mask
        != required_atom_certificate_stage_mask_v1) {
        return {independent_verification_code_v1::
                    incomplete_or_unknown_stages};
    }
    if (certificate.canonical_source_generation == 0
        || certificate.lineage_generation == 0
        || certificate.certification_generation == 0) {
        return {independent_verification_code_v1::invalid_generation};
    }
    if (!valid_content_digest(certificate.content.digest)
        || certificate.content.digest.algorithm == digest_algorithm::none) {
        return {independent_verification_code_v1::invalid_content_digest};
    }
    if (certificate.certified_entity_count
            > UINT64_MAX - certificate.certified_relation_edge_count
        || certificate.contribution_owner_count
            > UINT64_MAX - certificate.residual_count) {
        return {independent_verification_code_v1::certificate_count_overflow};
    }
    if (certificate.certified_entity_count
            + certificate.certified_relation_edge_count
        != certificate.contribution_owner_count + certificate.residual_count) {
        return {independent_verification_code_v1::certificate_count_mismatch};
    }
    if (!certificate_matches_expectation_v1(certificate, expectation)) {
        return {independent_verification_code_v1::expectation_mismatch};
    }
    verifier->certificate = certificate;
    verifier->expectation = expectation;
    verifier->stream_digest = certification_stream_fnv_offset_v1;
    verifier->active = true;
    return {};
}

[[nodiscard]] inline independent_verification_result_v1
update_independent_atom_verification_v1(
    independent_atom_verifier_v1 *verifier,
    certified_member_record_v1 member) noexcept {
    if (verifier == nullptr || !verifier->active) {
        return {independent_verification_code_v1::stream_not_active};
    }
    const auto index = verifier->member_count;
    if (!atom::validate_atom_persistent_identity_v1(
             member.owner_space_identity)
             .valid()
        || member.global_identity == 0) {
        return {independent_verification_code_v1::zero_member_identity, index};
    }
    if (member.kind != certification_member_kind_v1::entity
        && member.kind != certification_member_kind_v1::relation_edge) {
        return {independent_verification_code_v1::invalid_member_kind, index};
    }
    if (member.disposition != certified_member_disposition_v1::contribution_owner
        && member.disposition != certified_member_disposition_v1::residual) {
        return {independent_verification_code_v1::invalid_disposition, index};
    }
    if ((member.disposition
             == certified_member_disposition_v1::contribution_owner
         && member.contribution_owner_atom_index
                == no_failed_certification_index_v1)
        || (member.disposition == certified_member_disposition_v1::residual
            && member.contribution_owner_atom_index
                   != no_failed_certification_index_v1)) {
        return {independent_verification_code_v1::invalid_owner_atom_index,
                index};
    }
    for (const auto reserved : member.reserved) {
        if (reserved != 0) {
            return {independent_verification_code_v1::nonzero_reserved, index};
        }
    }
    if (verifier->has_previous
        && !certified_member_less_v1(verifier->previous, member)) {
        return {independent_verification_code_v1::
                    unordered_or_duplicate_member,
                index};
    }
    verification_hash_member_v1(verifier->stream_digest, member);
    verifier->previous = member;
    verifier->has_previous = true;
    ++verifier->member_count;
    if (member.kind == certification_member_kind_v1::entity) {
        ++verifier->entity_count;
    } else {
        ++verifier->relation_edge_count;
    }
    if (member.disposition
        == certified_member_disposition_v1::contribution_owner) {
        ++verifier->contribution_owner_count;
    } else {
        ++verifier->residual_count;
    }
    return {independent_verification_code_v1::valid, index};
}

[[nodiscard]] inline independent_verification_result_v1
finish_independent_atom_verification_v1(
    independent_atom_verifier_v1 *verifier) noexcept {
    if (verifier == nullptr || !verifier->active) {
        return {independent_verification_code_v1::stream_not_active};
    }
    verifier->active = false;
    if (verifier->entity_count != verifier->certificate.certified_entity_count
        || verifier->relation_edge_count
               != verifier->certificate.certified_relation_edge_count
        || verifier->contribution_owner_count
               != verifier->certificate.contribution_owner_count
        || verifier->residual_count != verifier->certificate.residual_count) {
        return {independent_verification_code_v1::streamed_count_mismatch,
                verifier->member_count};
    }
    if (verifier->stream_digest
        != verifier->expectation.member_stream_digest) {
        return {independent_verification_code_v1::stream_digest_mismatch,
                verifier->member_count};
    }
    return {independent_verification_code_v1::valid,
            verifier->member_count};
}

} // namespace cellshard::compiler::certification
