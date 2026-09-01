#pragma once

#include <CellShard/compiler/discovery/factor_topic/candidate_generation_v1.hh>

#include <cstdint>
#include <type_traits>

namespace cellshard::compiler::discovery::factor_topic {

struct factor_canonical_members_v1 {
    const evidence::evidence_identity_v1 *members = nullptr;
    std::uint64_t member_count = 0;
    evidence::evidence_identity_v1 domain_identity{};
    std::uint64_t source_generation = 0;
};

struct factor_exact_membership_view_v1 {
    const evidence::evidence_identity_v1 *members = nullptr;
    std::uint64_t member_count = 0;
    evidence::evidence_identity_v1 candidate_identity{};
};

struct factor_exact_owner_v1 {
    evidence::evidence_identity_v1 member_identity{};
    evidence::evidence_identity_v1 candidate_identity{};
};

struct factor_exact_coverage_span_v1 {
    evidence::evidence_identity_v1 candidate_identity{};
    std::uint64_t owner_begin = 0;
    std::uint64_t owner_count = 0;
};

enum class factor_exact_coverage_code_v1 : std::uint32_t {
    constructed = 0,
    invalid_canonical,
    missing_candidates,
    missing_candidate_members,
    missing_exact_views,
    invalid_candidate,
    duplicate_candidate_identity,
    exact_identity_mismatch,
    invalid_exact_membership,
    exact_member_not_proposed,
    exact_member_not_canonical,
    missing_coverage_spans,
    insufficient_owner_capacity,
    insufficient_residual_capacity,
};

struct factor_exact_coverage_result_v1 {
    factor_exact_coverage_code_v1 code = factor_exact_coverage_code_v1::constructed;
    std::uint64_t coverage_span_count = 0;
    std::uint64_t owner_count = 0;
    std::uint64_t residual_count = 0;
    std::uint64_t candidate_index = 0;
    std::uint64_t member_index = 0;

    [[nodiscard]] constexpr bool constructed() const noexcept {
        return code == factor_exact_coverage_code_v1::constructed;
    }
};

[[nodiscard]] inline bool contains_identity_v1(
    const evidence::evidence_identity_v1 *members,
    std::uint64_t count,
    evidence::evidence_identity_v1 target) noexcept {
    std::uint64_t begin = 0;
    std::uint64_t end = count;
    while (begin < end) {
        const auto middle = begin + (end - begin) / 2;
        if (evidence::evidence_identity_less_v1(members[middle], target)) {
            begin = middle + 1;
        } else {
            end = middle;
        }
    }
    return begin < count && members[begin] == target;
}

[[nodiscard]] inline factor_exact_coverage_result_v1
construct_factor_exact_coverage_v1(
    factor_canonical_members_v1 canonical,
    const factor_candidate_v1 *candidates,
    std::uint64_t candidate_count,
    const factor_candidate_member_v1 *candidate_members,
    std::uint64_t candidate_member_count,
    const factor_exact_membership_view_v1 *exact_views,
    factor_exact_coverage_span_v1 *coverage_spans,
    std::uint64_t coverage_span_capacity,
    factor_exact_owner_v1 *owners,
    std::uint64_t owner_capacity,
    evidence::evidence_identity_v1 *residual,
    std::uint64_t residual_capacity) noexcept {
    factor_exact_coverage_result_v1 result{};
    if (canonical.member_count == 0 || canonical.members == nullptr
        || !evidence::valid_evidence_identity_v1(canonical.domain_identity)
        || canonical.source_generation == 0) {
        result.code = factor_exact_coverage_code_v1::invalid_canonical;
        return result;
    }
    for (std::uint64_t index = 0; index < canonical.member_count; ++index) {
        if (!evidence::valid_evidence_identity_v1(canonical.members[index])
            || (index != 0 && !evidence::evidence_identity_less_v1(
                    canonical.members[index - 1], canonical.members[index]))) {
            result.code = factor_exact_coverage_code_v1::invalid_canonical;
            result.member_index = index;
            return result;
        }
    }
    if (candidate_count != 0 && candidates == nullptr) {
        result.code = factor_exact_coverage_code_v1::missing_candidates;
        return result;
    }
    if (candidate_member_count != 0 && candidate_members == nullptr) {
        result.code = factor_exact_coverage_code_v1::missing_candidate_members;
        return result;
    }
    if (candidate_count != 0 && exact_views == nullptr) {
        result.code = factor_exact_coverage_code_v1::missing_exact_views;
        return result;
    }
    if (candidate_count > coverage_span_capacity
        || (candidate_count != 0 && coverage_spans == nullptr)) {
        result.code = factor_exact_coverage_code_v1::missing_coverage_spans;
        result.coverage_span_count = candidate_count;
        return result;
    }

    for (std::uint64_t candidate_index = 0;
         candidate_index < candidate_count;
         ++candidate_index) {
        result.candidate_index = candidate_index;
        const auto &candidate = candidates[candidate_index];
        if (!evidence::valid_evidence_identity_v1(candidate.evidence_identity)
            || candidate.member_count == 0
            || candidate.member_begin > candidate_member_count
            || candidate.member_count > candidate_member_count - candidate.member_begin
            || !valid_external_factor_topic_kind_v1(candidate.kind)
            || candidate.reserved != 0) {
            result.code = factor_exact_coverage_code_v1::invalid_candidate;
            return result;
        }
        if (candidate_index != 0
            && !(evidence::evidence_identity_less_v1(
                candidates[candidate_index - 1].evidence_identity,
                candidate.evidence_identity))) {
            result.code = factor_exact_coverage_code_v1::duplicate_candidate_identity;
            return result;
        }
        const auto &exact = exact_views[candidate_index];
        if (!(exact.candidate_identity == candidate.evidence_identity)) {
            result.code = factor_exact_coverage_code_v1::exact_identity_mismatch;
            return result;
        }
        if (exact.member_count != 0 && exact.members == nullptr) {
            result.code = factor_exact_coverage_code_v1::invalid_exact_membership;
            return result;
        }
        for (std::uint64_t member_index = 0;
             member_index < exact.member_count;
             ++member_index) {
            result.member_index = member_index;
            const auto member = exact.members[member_index];
            if (!evidence::valid_evidence_identity_v1(member)
                || (member_index != 0 && !evidence::evidence_identity_less_v1(
                    exact.members[member_index - 1], member))) {
                result.code = factor_exact_coverage_code_v1::invalid_exact_membership;
                return result;
            }
            bool proposed = false;
            for (std::uint64_t proposal_index = candidate.member_begin;
                 proposal_index < candidate.member_begin + candidate.member_count;
                 ++proposal_index) {
                if (candidate_members[proposal_index].member_identity == member) {
                    proposed = true;
                    break;
                }
            }
            if (!proposed) {
                result.code = factor_exact_coverage_code_v1::exact_member_not_proposed;
                return result;
            }
            if (!contains_identity_v1(canonical.members, canonical.member_count, member)) {
                result.code = factor_exact_coverage_code_v1::exact_member_not_canonical;
                return result;
            }
            bool previously_owned = false;
            for (std::uint64_t prior = 0; prior < candidate_index; ++prior) {
                if (contains_identity_v1(exact_views[prior].members,
                                         exact_views[prior].member_count,
                                         member)) {
                    previously_owned = true;
                    break;
                }
            }
            if (!previously_owned) {
                ++result.owner_count;
            }
        }
    }
    for (std::uint64_t index = 0; index < canonical.member_count; ++index) {
        bool owned = false;
        for (std::uint64_t candidate_index = 0;
             candidate_index < candidate_count;
             ++candidate_index) {
            if (contains_identity_v1(exact_views[candidate_index].members,
                                     exact_views[candidate_index].member_count,
                                     canonical.members[index])) {
                owned = true;
                break;
            }
        }
        if (!owned) {
            ++result.residual_count;
        }
    }
    if (result.owner_count > owner_capacity
        || (result.owner_count != 0 && owners == nullptr)) {
        result.code = factor_exact_coverage_code_v1::insufficient_owner_capacity;
        return result;
    }
    if (result.residual_count > residual_capacity
        || (result.residual_count != 0 && residual == nullptr)) {
        result.code = factor_exact_coverage_code_v1::insufficient_residual_capacity;
        return result;
    }

    std::uint64_t owner_output = 0;
    for (std::uint64_t candidate_index = 0;
         candidate_index < candidate_count;
         ++candidate_index) {
        const auto begin = owner_output;
        const auto &exact = exact_views[candidate_index];
        for (std::uint64_t member_index = 0;
             member_index < exact.member_count;
             ++member_index) {
            const auto member = exact.members[member_index];
            bool previously_owned = false;
            for (std::uint64_t prior = 0; prior < candidate_index; ++prior) {
                if (contains_identity_v1(exact_views[prior].members,
                                         exact_views[prior].member_count,
                                         member)) {
                    previously_owned = true;
                    break;
                }
            }
            if (!previously_owned) {
                owners[owner_output++] = {member, candidates[candidate_index].evidence_identity};
            }
        }
        coverage_spans[candidate_index] = {candidates[candidate_index].evidence_identity,
                                           begin,
                                           owner_output - begin};
    }
    std::uint64_t residual_output = 0;
    for (std::uint64_t index = 0; index < canonical.member_count; ++index) {
        bool owned = false;
        for (std::uint64_t candidate_index = 0;
             candidate_index < candidate_count;
             ++candidate_index) {
            if (contains_identity_v1(exact_views[candidate_index].members,
                                     exact_views[candidate_index].member_count,
                                     canonical.members[index])) {
                owned = true;
                break;
            }
        }
        if (!owned) {
            residual[residual_output++] = canonical.members[index];
        }
    }
    result.coverage_span_count = candidate_count;
    result.candidate_index = candidate_count;
    result.member_index = 0;
    return result;
}

[[nodiscard]] constexpr bool authorizes_execution(
    const factor_exact_coverage_result_v1 &) noexcept {
    return false;
}

static_assert(std::is_standard_layout<factor_exact_owner_v1>::value);
static_assert(std::is_trivially_copyable<factor_exact_owner_v1>::value);

} // namespace cellshard::compiler::discovery::factor_topic
