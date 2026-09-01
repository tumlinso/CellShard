#pragma once

#include <CellShard/compiler/discovery/overlap/membership_v1.hh>

#include <cstdint>
#include <limits>

namespace cellshard::compiler::discovery::overlap {

inline constexpr std::uint32_t overlap_stability_cost_contract_version_v1 = 1;

struct overlap_duplication_cost_model_v1 {
    std::uint64_t bytes_per_membership = 0;
    std::uint64_t persistent_replica_count = 0;
    std::uint64_t expected_transfer_count = 0;
};

struct overlap_stability_cost_v1 {
    std::uint64_t stability_intersection_count = 0;
    std::uint64_t stability_union_count = 0;
    std::uint64_t additional_membership_count = 0;
    std::uint64_t persistent_duplication_bytes = 0;
    std::uint64_t expected_movement_bytes = 0;
    std::uint64_t complete_duplication_cost_bytes = 0;
};

enum class overlap_stability_cost_code_v1 : std::uint32_t {
    scored = 0,
    invalid_candidate,
    empty_resamples,
    missing_resamples,
    invalid_resample,
    member_spine_mismatch,
    community_spine_mismatch,
    source_mismatch,
    invalid_cost_model,
    count_overflow,
    byte_overflow,
};

struct overlap_stability_cost_result_v1 {
    overlap_stability_cost_code_v1 code =
        overlap_stability_cost_code_v1::scored;
    overlap_stability_cost_v1 score{};
    std::uint64_t resample_index = 0;
    std::uint64_t member_index = 0;
    std::uint32_t nested_code = 0;

    [[nodiscard]] constexpr bool scored() const noexcept {
        return code == overlap_stability_cost_code_v1::scored;
    }
};

[[nodiscard]] inline bool same_overlap_spines_v1(
    bounded_overlap_membership_view_v1 lhs,
    bounded_overlap_membership_view_v1 rhs,
    std::uint64_t *mismatch_index,
    bool *community_mismatch) noexcept {
    if (lhs.member_count != rhs.member_count) {
        *mismatch_index = lhs.member_count < rhs.member_count
            ? lhs.member_count
            : rhs.member_count;
        *community_mismatch = false;
        return false;
    }
    for (std::uint64_t index = 0; index < lhs.member_count; ++index) {
        if (lhs.global_member_ids[index] != rhs.global_member_ids[index]) {
            *mismatch_index = index;
            *community_mismatch = false;
            return false;
        }
    }
    if (lhs.community_count != rhs.community_count) {
        *mismatch_index = lhs.community_count < rhs.community_count
            ? lhs.community_count
            : rhs.community_count;
        *community_mismatch = true;
        return false;
    }
    for (std::uint64_t index = 0; index < lhs.community_count; ++index) {
        if (!(lhs.community_identities[index]
              == rhs.community_identities[index])) {
            *mismatch_index = index;
            *community_mismatch = true;
            return false;
        }
    }
    return true;
}

// Stability is aggregate exact Jaccard membership support over bounded
// resamples. Each pair of sorted member slices is merged once, giving linear
// work in observed memberships. The cost includes persistent copies and
// expected movement, with every multiplication checked for overflow.
[[nodiscard]] inline overlap_stability_cost_result_v1
score_overlap_stability_and_duplication_v1(
    bounded_overlap_membership_view_v1 candidate,
    const bounded_overlap_membership_view_v1 *resamples,
    std::uint64_t resample_count,
    overlap_duplication_cost_model_v1 cost_model) noexcept {
    const auto candidate_validation =
        validate_bounded_overlap_membership_v1(candidate);
    if (!candidate_validation.valid()) {
        return {overlap_stability_cost_code_v1::invalid_candidate,
                {},
                0,
                candidate_validation.member_index,
                static_cast<std::uint32_t>(candidate_validation.code)};
    }
    if (resample_count == 0) {
        return {overlap_stability_cost_code_v1::empty_resamples};
    }
    if (resamples == nullptr) {
        return {overlap_stability_cost_code_v1::missing_resamples};
    }
    if (cost_model.bytes_per_membership == 0
        || cost_model.persistent_replica_count == 0) {
        return {overlap_stability_cost_code_v1::invalid_cost_model};
    }
    overlap_stability_cost_v1 score{};
    for (std::uint64_t resample_index = 0;
         resample_index < resample_count;
         ++resample_index) {
        const auto &resample = resamples[resample_index];
        const auto validation = validate_bounded_overlap_membership_v1(resample);
        if (!validation.valid()) {
            return {overlap_stability_cost_code_v1::invalid_resample,
                    {},
                    resample_index,
                    validation.member_index,
                    static_cast<std::uint32_t>(validation.code)};
        }
        std::uint64_t mismatch_index = 0;
        bool community_mismatch = false;
        if (!same_overlap_spines_v1(candidate,
                                    resample,
                                    &mismatch_index,
                                    &community_mismatch)) {
            return {community_mismatch
                        ? overlap_stability_cost_code_v1::
                              community_spine_mismatch
                        : overlap_stability_cost_code_v1::member_spine_mismatch,
                    {},
                    resample_index,
                    mismatch_index};
        }
        if (!(candidate.source_identity == resample.source_identity)) {
            return {overlap_stability_cost_code_v1::source_mismatch,
                    {},
                    resample_index};
        }
        for (std::uint64_t member_index = 0;
             member_index < candidate.member_count;
             ++member_index) {
            auto candidate_index = candidate.member_offsets[member_index];
            const auto candidate_end = candidate.member_offsets[member_index + 1];
            auto resample_member_index = resample.member_offsets[member_index];
            const auto resample_end = resample.member_offsets[member_index + 1];
            while (candidate_index < candidate_end
                   || resample_member_index < resample_end) {
                if (score.stability_union_count == UINT64_MAX) {
                    return {overlap_stability_cost_code_v1::count_overflow,
                            score,
                            resample_index,
                            member_index};
                }
                ++score.stability_union_count;
                if (candidate_index < candidate_end
                    && resample_member_index < resample_end
                    && candidate.memberships[candidate_index]
                           .local_community_index
                       == resample.memberships[resample_member_index]
                              .local_community_index) {
                    if (score.stability_intersection_count == UINT64_MAX) {
                        return {overlap_stability_cost_code_v1::count_overflow,
                                score,
                                resample_index,
                                member_index};
                    }
                    ++score.stability_intersection_count;
                    ++candidate_index;
                    ++resample_member_index;
                } else if (resample_member_index == resample_end
                           || (candidate_index < candidate_end
                               && candidate.memberships[candidate_index]
                                          .local_community_index
                                      < resample
                                            .memberships[resample_member_index]
                                            .local_community_index)) {
                    ++candidate_index;
                } else {
                    ++resample_member_index;
                }
            }
        }
    }
    score.additional_membership_count =
        candidate.membership_count - candidate.member_count;
    if (score.additional_membership_count
        > UINT64_MAX / cost_model.bytes_per_membership) {
        return {overlap_stability_cost_code_v1::byte_overflow, score};
    }
    const auto bytes_per_replica =
        score.additional_membership_count * cost_model.bytes_per_membership;
    if (bytes_per_replica
        > UINT64_MAX / cost_model.persistent_replica_count) {
        return {overlap_stability_cost_code_v1::byte_overflow, score};
    }
    score.persistent_duplication_bytes =
        bytes_per_replica * cost_model.persistent_replica_count;
    if (cost_model.expected_transfer_count != 0
        && score.persistent_duplication_bytes
               > UINT64_MAX / cost_model.expected_transfer_count) {
        return {overlap_stability_cost_code_v1::byte_overflow, score};
    }
    score.expected_movement_bytes =
        score.persistent_duplication_bytes
        * cost_model.expected_transfer_count;
    if (score.persistent_duplication_bytes
        > UINT64_MAX - score.expected_movement_bytes) {
        return {overlap_stability_cost_code_v1::byte_overflow, score};
    }
    score.complete_duplication_cost_bytes =
        score.persistent_duplication_bytes + score.expected_movement_bytes;
    return {overlap_stability_cost_code_v1::scored,
            score,
            resample_count,
            candidate.member_count,
            0};
}

} // namespace cellshard::compiler::discovery::overlap
