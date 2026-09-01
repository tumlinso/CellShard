#pragma once

#include <CellShard/compiler/certification/atom_certification_v1.hh>
#include <CellShard/compiler/discovery/overlap/membership_v1.hh>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace cellshard::compiler::discovery::overlap {

inline constexpr std::uint32_t overlap_certification_candidate_contract_v1 = 1;

struct exact_overlap_rescan_member_v1 {
    std::uint64_t global_member_id = 0;
    std::uint32_t local_community_index = 0;
    std::uint32_t reserved = 0;
};

struct exact_overlap_rescan_view_v1 {
    const exact_overlap_rescan_member_v1 *members = nullptr;
    std::uint64_t member_count = 0;
    atom::atom_persistent_identity_v1 canonical_source_identity{};
    std::uint64_t canonical_source_generation = 0;
};

struct exact_overlap_candidate_v1 {
    const std::uint64_t *global_member_ids = nullptr;
    std::uint64_t member_count = 0;
    atom::atom_persistent_identity_v1 candidate_identity{};
    evidence::evidence_identity_v1 community_identity{};
};

struct exact_overlap_candidate_table_v1 {
    const exact_overlap_candidate_v1 *candidates = nullptr;
    std::uint64_t candidate_count = 0;
    atom::atom_persistent_identity_v1 proposal_provider_identity{};
    atom::atom_persistent_identity_v1 certification_authority_identity{};
    atom::atom_persistent_identity_v1 canonical_source_identity{};
    std::uint64_t canonical_source_generation = 0;
};

struct exact_overlap_candidate_buffers_v1 {
    exact_overlap_candidate_v1 *candidates = nullptr;
    std::uint64_t candidate_capacity = 0;
    std::uint64_t *global_member_ids = nullptr;
    std::uint64_t member_capacity = 0;
};

enum class overlap_candidate_build_code_v1 : std::uint32_t {
    built = 0,
    invalid_proposal,
    empty_rescan,
    missing_rescan,
    invalid_canonical_source_identity,
    missing_canonical_source_generation,
    invalid_proposal_provider_identity,
    invalid_certification_authority_identity,
    provider_self_certification,
    missing_candidate_identities,
    invalid_candidate_identity,
    unordered_or_duplicate_candidate_identity,
    zero_member_identity,
    community_out_of_range,
    nonzero_reserved,
    unordered_or_duplicate_rescan_member,
    member_not_in_proposal,
    community_not_proposed_for_member,
    missing_output,
    insufficient_output,
};

struct overlap_candidate_build_result_v1 {
    overlap_candidate_build_code_v1 code =
        overlap_candidate_build_code_v1::built;
    exact_overlap_candidate_table_v1 table{};
    std::uint64_t index = 0;
    std::uint64_t required_candidate_capacity = 0;
    std::uint64_t required_member_capacity = 0;
    std::uint32_t nested_code = 0;

    [[nodiscard]] constexpr bool built() const noexcept {
        return code == overlap_candidate_build_code_v1::built;
    }
};

static_assert(offsetof(exact_overlap_rescan_view_v1, members) == 0);
static_assert(std::is_standard_layout<exact_overlap_rescan_member_v1>::value);
static_assert(std::is_trivially_copyable<exact_overlap_rescan_member_v1>::value);
static_assert(std::is_standard_layout<exact_overlap_candidate_v1>::value);
static_assert(std::is_trivially_copyable<exact_overlap_candidate_v1>::value);

// Converts only an independently rescanned, exact subset of the proposal into
// certification candidates. Proposal evidence is checked for provenance but
// never certifies itself. Work is O(R(log M + log L)), where L is the explicit
// overlap bound, and outputs are caller-owned.
[[nodiscard]] inline overlap_candidate_build_result_v1
build_exact_overlap_certification_candidates_v1(
    bounded_overlap_membership_view_v1 proposal,
    exact_overlap_rescan_view_v1 rescan,
    const atom::atom_persistent_identity_v1
        *candidate_identities_by_community,
    atom::atom_persistent_identity_v1 proposal_provider_identity,
    atom::atom_persistent_identity_v1
        certification_authority_identity,
    exact_overlap_candidate_buffers_v1 buffers) noexcept {
    const auto proposal_validation =
        validate_bounded_overlap_membership_v1(proposal);
    if (!proposal_validation.valid()) {
        return {overlap_candidate_build_code_v1::invalid_proposal,
                {},
                proposal_validation.member_index,
                0,
                0,
                static_cast<std::uint32_t>(proposal_validation.code)};
    }
    if (rescan.member_count == 0) {
        return {overlap_candidate_build_code_v1::empty_rescan};
    }
    if (rescan.members == nullptr) {
        return {overlap_candidate_build_code_v1::missing_rescan};
    }
    if (!atom::validate_atom_persistent_identity_v1(
             rescan.canonical_source_identity)
             .valid()) {
        return {overlap_candidate_build_code_v1::
                    invalid_canonical_source_identity};
    }
    if (rescan.canonical_source_generation == 0) {
        return {overlap_candidate_build_code_v1::
                    missing_canonical_source_generation};
    }
    if (!atom::validate_atom_persistent_identity_v1(
             proposal_provider_identity)
             .valid()) {
        return {overlap_candidate_build_code_v1::
                    invalid_proposal_provider_identity};
    }
    if (!atom::validate_atom_persistent_identity_v1(
             certification_authority_identity)
             .valid()) {
        return {overlap_candidate_build_code_v1::
                    invalid_certification_authority_identity};
    }
    if (proposal_provider_identity == certification_authority_identity) {
        return {overlap_candidate_build_code_v1::provider_self_certification};
    }
    if (candidate_identities_by_community == nullptr) {
        return {overlap_candidate_build_code_v1::
                    missing_candidate_identities};
    }
    for (std::uint64_t index = 0; index < proposal.community_count; ++index) {
        if (!atom::validate_atom_persistent_identity_v1(
                 candidate_identities_by_community[index])
                 .valid()) {
            return {overlap_candidate_build_code_v1::invalid_candidate_identity,
                    {},
                    index};
        }
        if (index != 0
            && !atom::atom_persistent_identity_less_v1(
                candidate_identities_by_community[index - 1],
                candidate_identities_by_community[index])) {
            return {overlap_candidate_build_code_v1::
                        unordered_or_duplicate_candidate_identity,
                    {},
                    index};
        }
    }
    std::uint64_t nonempty_communities = 0;
    std::uint32_t previous_community = UINT32_MAX;
    for (std::uint64_t index = 0; index < rescan.member_count; ++index) {
        const auto &member = rescan.members[index];
        if (member.global_member_id == 0) {
            return {overlap_candidate_build_code_v1::zero_member_identity,
                    {},
                    index};
        }
        if (member.local_community_index >= proposal.community_count) {
            return {overlap_candidate_build_code_v1::community_out_of_range,
                    {},
                    index};
        }
        if (member.reserved != 0) {
            return {overlap_candidate_build_code_v1::nonzero_reserved,
                    {},
                    index};
        }
        if (index != 0
            && (rescan.members[index - 1].local_community_index
                    > member.local_community_index
                || (rescan.members[index - 1].local_community_index
                            == member.local_community_index
                    && rescan.members[index - 1].global_member_id
                           >= member.global_member_id))) {
            return {overlap_candidate_build_code_v1::
                        unordered_or_duplicate_rescan_member,
                    {},
                    index};
        }
        if (member.local_community_index != previous_community) {
            ++nonempty_communities;
            previous_community = member.local_community_index;
        }
        std::uint64_t begin = 0;
        std::uint64_t end = proposal.member_count;
        while (begin < end) {
            const auto middle = begin + (end - begin) / 2;
            if (proposal.global_member_ids[middle] < member.global_member_id) {
                begin = middle + 1;
            } else {
                end = middle;
            }
        }
        if (begin == proposal.member_count
            || proposal.global_member_ids[begin] != member.global_member_id) {
            return {overlap_candidate_build_code_v1::member_not_in_proposal,
                    {},
                    index};
        }
        auto membership_begin = proposal.member_offsets[begin];
        auto membership_end = proposal.member_offsets[begin + 1];
        while (membership_begin < membership_end) {
            const auto middle = membership_begin
                + (membership_end - membership_begin) / 2;
            if (proposal.memberships[middle].local_community_index
                < member.local_community_index) {
                membership_begin = middle + 1;
            } else {
                membership_end = middle;
            }
        }
        if (membership_begin == proposal.member_offsets[begin + 1]
            || proposal.memberships[membership_begin].local_community_index
                   != member.local_community_index) {
            return {overlap_candidate_build_code_v1::
                        community_not_proposed_for_member,
                    {},
                    index};
        }
    }
    if (buffers.candidates == nullptr || buffers.global_member_ids == nullptr) {
        return {overlap_candidate_build_code_v1::missing_output,
                {},
                0,
                nonempty_communities,
                rescan.member_count};
    }
    if (buffers.candidate_capacity < nonempty_communities
        || buffers.member_capacity < rescan.member_count) {
        return {overlap_candidate_build_code_v1::insufficient_output,
                {},
                0,
                nonempty_communities,
                rescan.member_count};
    }
    std::uint64_t candidate_index = 0;
    std::uint64_t group_begin = 0;
    while (group_begin < rescan.member_count) {
        const auto community = rescan.members[group_begin].local_community_index;
        auto group_end = group_begin + 1;
        while (group_end < rescan.member_count
               && rescan.members[group_end].local_community_index == community) {
            ++group_end;
        }
        for (auto index = group_begin; index < group_end; ++index) {
            buffers.global_member_ids[index] =
                rescan.members[index].global_member_id;
        }
        buffers.candidates[candidate_index++] = {
            buffers.global_member_ids + group_begin,
            group_end - group_begin,
            candidate_identities_by_community[community],
            proposal.community_identities[community]};
        group_begin = group_end;
    }
    return {overlap_candidate_build_code_v1::built,
            {buffers.candidates,
             candidate_index,
             proposal_provider_identity,
             certification_authority_identity,
             rescan.canonical_source_identity,
             rescan.canonical_source_generation},
            rescan.member_count,
            nonempty_communities,
            rescan.member_count,
            0};
}

[[nodiscard]] constexpr bool authorizes_execution(
    exact_overlap_candidate_table_v1) noexcept {
    return false;
}

} // namespace cellshard::compiler::discovery::overlap
