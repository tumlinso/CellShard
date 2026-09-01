#pragma once

#include <CellShard/compiler/discovery/overlap/membership_v1.hh>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace cellshard::compiler::discovery::overlap {

inline constexpr std::uint32_t disjoint_baseline_contract_version_v1 = 1;

struct disjoint_assignment_view_v1 {
    const std::uint64_t *global_member_ids = nullptr;
    const std::uint32_t *local_community_indices = nullptr;
    const evidence::evidence_identity_v1 *community_identities = nullptr;
    std::uint64_t member_count = 0;
    std::uint64_t community_count = 0;
    evidence::evidence_identity_v1 evidence_identity{};
    evidence::evidence_identity_v1 source_identity{};
    std::uint64_t observation_generation = 0;
};

struct disjoint_baseline_buffers_v1 {
    std::uint64_t *member_offsets = nullptr;
    std::uint64_t offset_capacity = 0;
    overlap_membership_v1 *memberships = nullptr;
    std::uint64_t membership_capacity = 0;
};

enum class disjoint_baseline_build_code_v1 : std::uint32_t {
    built = 0,
    empty_members,
    missing_member_ids,
    missing_assignments,
    empty_communities,
    missing_communities,
    too_many_communities,
    invalid_evidence_identity,
    invalid_source_identity,
    missing_observation_generation,
    unordered_or_duplicate_member,
    unordered_or_duplicate_community,
    assignment_out_of_range,
    missing_output,
    insufficient_output,
    invalid_built_view,
};

struct disjoint_baseline_build_result_v1 {
    disjoint_baseline_build_code_v1 code =
        disjoint_baseline_build_code_v1::built;
    bounded_overlap_membership_view_v1 view{};
    std::uint64_t index = 0;
    std::uint64_t required_offset_capacity = 0;
    std::uint64_t required_membership_capacity = 0;
    std::uint32_t nested_code = 0;

    [[nodiscard]] constexpr bool built() const noexcept {
        return code == disjoint_baseline_build_code_v1::built;
    }
};

static_assert(offsetof(disjoint_assignment_view_v1, global_member_ids) == 0);
static_assert(std::is_standard_layout<disjoint_assignment_view_v1>::value);
static_assert(std::is_trivially_copyable<disjoint_assignment_view_v1>::value);

// Deterministically lifts one exact input label per member into the common
// bounded proposal representation. It allocates nothing and writes exactly
// member_count offsets/memberships in O(members + communities).
[[nodiscard]] inline disjoint_baseline_build_result_v1
build_disjoint_community_baseline_v1(
    disjoint_assignment_view_v1 source,
    disjoint_baseline_buffers_v1 buffers) noexcept {
    if (source.member_count == 0) {
        return {disjoint_baseline_build_code_v1::empty_members};
    }
    if (source.global_member_ids == nullptr) {
        return {disjoint_baseline_build_code_v1::missing_member_ids};
    }
    if (source.local_community_indices == nullptr) {
        return {disjoint_baseline_build_code_v1::missing_assignments};
    }
    if (source.community_count == 0) {
        return {disjoint_baseline_build_code_v1::empty_communities};
    }
    if (source.community_identities == nullptr) {
        return {disjoint_baseline_build_code_v1::missing_communities};
    }
    if (source.community_count > UINT32_MAX) {
        return {disjoint_baseline_build_code_v1::too_many_communities};
    }
    if (!evidence::valid_evidence_identity_v1(source.evidence_identity)) {
        return {disjoint_baseline_build_code_v1::invalid_evidence_identity};
    }
    if (!evidence::valid_evidence_identity_v1(source.source_identity)) {
        return {disjoint_baseline_build_code_v1::invalid_source_identity};
    }
    if (source.observation_generation == 0) {
        return {disjoint_baseline_build_code_v1::
                    missing_observation_generation};
    }
    for (std::uint64_t index = 0; index < source.member_count; ++index) {
        if (source.global_member_ids[index] == 0
            || (index != 0
                && source.global_member_ids[index - 1]
                    >= source.global_member_ids[index])) {
            return {disjoint_baseline_build_code_v1::
                        unordered_or_duplicate_member,
                    {},
                    index};
        }
        if (source.local_community_indices[index] >= source.community_count) {
            return {disjoint_baseline_build_code_v1::assignment_out_of_range,
                    {},
                    index};
        }
    }
    for (std::uint64_t index = 0; index < source.community_count; ++index) {
        if (!evidence::valid_evidence_identity_v1(
                source.community_identities[index])
            || (index != 0
                && !evidence::evidence_identity_less_v1(
                    source.community_identities[index - 1],
                    source.community_identities[index]))) {
            return {disjoint_baseline_build_code_v1::
                        unordered_or_duplicate_community,
                    {},
                    index};
        }
    }
    if (buffers.member_offsets == nullptr || buffers.memberships == nullptr) {
        return {disjoint_baseline_build_code_v1::missing_output,
                {},
                0,
                source.member_count + 1,
                source.member_count};
    }
    if (buffers.offset_capacity < source.member_count + 1
        || buffers.membership_capacity < source.member_count) {
        return {disjoint_baseline_build_code_v1::insufficient_output,
                {},
                0,
                source.member_count + 1,
                source.member_count};
    }
    for (std::uint64_t index = 0; index < source.member_count; ++index) {
        buffers.member_offsets[index] = index;
        buffers.memberships[index] = {
            source.local_community_indices[index], 0, 1, 1};
    }
    buffers.member_offsets[source.member_count] = source.member_count;
    bounded_overlap_membership_view_v1 view{
        source.global_member_ids,
        buffers.member_offsets,
        buffers.memberships,
        source.community_identities,
        source.member_count,
        source.member_count,
        source.community_count,
        1,
        0,
        source.evidence_identity,
        source.source_identity,
        source.observation_generation};
    const auto validation = validate_bounded_overlap_membership_v1(view);
    if (!validation.valid()) {
        return {disjoint_baseline_build_code_v1::invalid_built_view,
                {},
                validation.member_index,
                source.member_count + 1,
                source.member_count,
                static_cast<std::uint32_t>(validation.code)};
    }
    return {disjoint_baseline_build_code_v1::built,
            view,
            source.member_count,
            source.member_count + 1,
            source.member_count,
            0};
}

} // namespace cellshard::compiler::discovery::overlap
