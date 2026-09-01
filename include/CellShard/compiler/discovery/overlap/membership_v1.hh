#pragma once

#include <CellShard/compiler/evidence/atom_evidence_record_v1.hh>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace cellshard::compiler::discovery::overlap {

inline constexpr std::uint32_t overlap_membership_contract_version_v1 = 1;

struct overlap_membership_v1 {
    std::uint32_t local_community_index = 0;
    std::uint32_t reserved = 0;
    std::uint64_t weight_numerator = 0;
    std::uint64_t weight_denominator = 0;
};

// CSR-like proposal evidence. Global member and community identities stay u64
// namespace-qualified values; only the bounded table-local community ordinal is
// compact. Offsets are u64 so atlas-scale membership counts cannot truncate.
struct bounded_overlap_membership_view_v1 {
    const std::uint64_t *global_member_ids = nullptr;
    const std::uint64_t *member_offsets = nullptr;
    const overlap_membership_v1 *memberships = nullptr;
    const evidence::evidence_identity_v1 *community_identities = nullptr;
    std::uint64_t member_count = 0;
    std::uint64_t membership_count = 0;
    std::uint64_t community_count = 0;
    std::uint32_t maximum_memberships_per_member = 0;
    std::uint32_t reserved = 0;
    evidence::evidence_identity_v1 evidence_identity{};
    evidence::evidence_identity_v1 source_identity{};
    std::uint64_t observation_generation = 0;
};

enum class bounded_overlap_validation_code_v1 : std::uint32_t {
    valid = 0,
    invalid_evidence_identity,
    invalid_source_identity,
    missing_observation_generation,
    empty_members,
    missing_member_ids,
    missing_offsets,
    empty_communities,
    missing_community_identities,
    too_many_communities,
    missing_memberships,
    invalid_overlap_bound,
    zero_global_member_identity,
    unordered_or_duplicate_global_member,
    invalid_community_identity,
    unordered_or_duplicate_community,
    invalid_initial_offset,
    unordered_or_out_of_range_offset,
    membership_count_mismatch,
    empty_member_assignment,
    overlap_bound_exceeded,
    community_index_out_of_range,
    unordered_or_duplicate_member_community,
    invalid_weight,
    nonzero_reserved,
};

struct bounded_overlap_validation_v1 {
    bounded_overlap_validation_code_v1 code =
        bounded_overlap_validation_code_v1::valid;
    std::uint64_t member_index = 0;
    std::uint64_t membership_index = 0;

    [[nodiscard]] constexpr bool valid() const noexcept {
        return code == bounded_overlap_validation_code_v1::valid;
    }
};

static_assert(std::is_standard_layout<overlap_membership_v1>::value);
static_assert(std::is_trivially_copyable<overlap_membership_v1>::value);
static_assert(offsetof(bounded_overlap_membership_view_v1,
                       global_member_ids)
              == 0);
static_assert(
    std::is_standard_layout<bounded_overlap_membership_view_v1>::value);
static_assert(
    std::is_trivially_copyable<bounded_overlap_membership_view_v1>::value);

// O(members + communities + memberships), O(1) storage. This validates bounded
// proposal evidence only and never converts it into exact execution coverage.
[[nodiscard]] constexpr bounded_overlap_validation_v1
validate_bounded_overlap_membership_v1(
    bounded_overlap_membership_view_v1 view) noexcept {
    if (!evidence::valid_evidence_identity_v1(view.evidence_identity)) {
        return {bounded_overlap_validation_code_v1::invalid_evidence_identity};
    }
    if (!evidence::valid_evidence_identity_v1(view.source_identity)) {
        return {bounded_overlap_validation_code_v1::invalid_source_identity};
    }
    if (view.observation_generation == 0) {
        return {bounded_overlap_validation_code_v1::
                    missing_observation_generation};
    }
    if (view.member_count == 0) {
        return {bounded_overlap_validation_code_v1::empty_members};
    }
    if (view.global_member_ids == nullptr) {
        return {bounded_overlap_validation_code_v1::missing_member_ids};
    }
    if (view.member_offsets == nullptr) {
        return {bounded_overlap_validation_code_v1::missing_offsets};
    }
    if (view.community_count == 0) {
        return {bounded_overlap_validation_code_v1::empty_communities};
    }
    if (view.community_identities == nullptr) {
        return {bounded_overlap_validation_code_v1::
                    missing_community_identities};
    }
    if (view.community_count > UINT32_MAX) {
        return {bounded_overlap_validation_code_v1::too_many_communities};
    }
    if (view.membership_count == 0 || view.memberships == nullptr) {
        return {bounded_overlap_validation_code_v1::missing_memberships};
    }
    if (view.maximum_memberships_per_member == 0) {
        return {bounded_overlap_validation_code_v1::invalid_overlap_bound};
    }
    if (view.reserved != 0) {
        return {bounded_overlap_validation_code_v1::nonzero_reserved};
    }
    for (std::uint64_t index = 0; index < view.member_count; ++index) {
        if (view.global_member_ids[index] == 0) {
            return {bounded_overlap_validation_code_v1::
                        zero_global_member_identity,
                    index};
        }
        if (index != 0
            && view.global_member_ids[index - 1]
                >= view.global_member_ids[index]) {
            return {bounded_overlap_validation_code_v1::
                        unordered_or_duplicate_global_member,
                    index};
        }
    }
    for (std::uint64_t index = 0; index < view.community_count; ++index) {
        if (!evidence::valid_evidence_identity_v1(
                view.community_identities[index])) {
            return {bounded_overlap_validation_code_v1::
                        invalid_community_identity,
                    0,
                    index};
        }
        if (index != 0
            && !evidence::evidence_identity_less_v1(
                view.community_identities[index - 1],
                view.community_identities[index])) {
            return {bounded_overlap_validation_code_v1::
                        unordered_or_duplicate_community,
                    0,
                    index};
        }
    }
    if (view.member_offsets[0] != 0) {
        return {bounded_overlap_validation_code_v1::invalid_initial_offset};
    }
    for (std::uint64_t member_index = 0;
         member_index < view.member_count;
         ++member_index) {
        const auto begin = view.member_offsets[member_index];
        const auto end = view.member_offsets[member_index + 1];
        if (end < begin || end > view.membership_count) {
            return {bounded_overlap_validation_code_v1::
                        unordered_or_out_of_range_offset,
                    member_index,
                    end};
        }
        if (begin == end) {
            return {bounded_overlap_validation_code_v1::empty_member_assignment,
                    member_index,
                    begin};
        }
        if (end - begin > view.maximum_memberships_per_member) {
            return {bounded_overlap_validation_code_v1::overlap_bound_exceeded,
                    member_index,
                    end - begin};
        }
        for (std::uint64_t index = begin; index < end; ++index) {
            const auto &membership = view.memberships[index];
            if (membership.local_community_index >= view.community_count) {
                return {bounded_overlap_validation_code_v1::
                            community_index_out_of_range,
                        member_index,
                        index};
            }
            if (index != begin
                && view.memberships[index - 1].local_community_index
                    >= membership.local_community_index) {
                return {bounded_overlap_validation_code_v1::
                            unordered_or_duplicate_member_community,
                        member_index,
                        index};
            }
            if (membership.weight_denominator == 0
                || membership.weight_numerator == 0
                || membership.weight_numerator
                       > membership.weight_denominator) {
                return {bounded_overlap_validation_code_v1::invalid_weight,
                        member_index,
                        index};
            }
            if (membership.reserved != 0) {
                return {bounded_overlap_validation_code_v1::nonzero_reserved,
                        member_index,
                        index};
            }
        }
    }
    if (view.member_offsets[view.member_count] != view.membership_count) {
        return {bounded_overlap_validation_code_v1::
                    membership_count_mismatch,
                view.member_count,
                view.member_offsets[view.member_count]};
    }
    return {bounded_overlap_validation_code_v1::valid,
            view.member_count,
            view.membership_count};
}

[[nodiscard]] constexpr bool authorizes_execution(
    bounded_overlap_membership_view_v1) noexcept {
    return false;
}

} // namespace cellshard::compiler::discovery::overlap
