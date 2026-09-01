#pragma once

#include <CellShard/compiler/discovery/factor_topic/soft_membership_store_v1.hh>

#include <cstdint>
#include <type_traits>

namespace cellshard::compiler::discovery::factor_topic {

struct factor_candidate_member_v1 {
    evidence::evidence_identity_v1 member_identity{};
    std::uint64_t weight_numerator = 0;
    std::uint64_t weight_denominator = 0;
};

struct factor_candidate_v1 {
    evidence::evidence_identity_v1 evidence_identity{};
    std::uint64_t member_begin = 0;
    std::uint64_t member_count = 0;
    external_factor_topic_kind_v1 kind = external_factor_topic_kind_v1::factor;
    std::uint32_t reserved = 0;
};

struct factor_candidate_generation_config_v1 {
    std::uint64_t threshold_numerator = 0;
    std::uint64_t threshold_denominator = 1;
    std::uint64_t minimum_members = 1;
    std::uint64_t maximum_overlap = 1;
    std::uint64_t maximum_pair_checks = 1;
};

enum class factor_candidate_generation_code_v1 : std::uint32_t {
    generated = 0,
    invalid_config,
    missing_stores,
    invalid_store,
    missing_candidates,
    missing_members,
    insufficient_candidate_capacity,
    insufficient_member_capacity,
    work_limit_exceeded,
};

struct factor_candidate_generation_result_v1 {
    factor_candidate_generation_code_v1 code =
        factor_candidate_generation_code_v1::generated;
    std::uint64_t candidate_count = 0;
    std::uint64_t member_count = 0;
    std::uint64_t pair_checks = 0;
    std::uint64_t input_index = 0;

    [[nodiscard]] constexpr bool generated() const noexcept {
        return code == factor_candidate_generation_code_v1::generated;
    }
};

// Exact overflow-free rational comparison by continued fractions.
[[nodiscard]] constexpr bool fraction_at_least_v1(
    std::uint64_t lhs_numerator,
    std::uint64_t lhs_denominator,
    std::uint64_t rhs_numerator,
    std::uint64_t rhs_denominator) noexcept {
    bool reversed = false;
    for (;;) {
        const auto lhs_quotient = lhs_numerator / lhs_denominator;
        const auto rhs_quotient = rhs_numerator / rhs_denominator;
        if (lhs_quotient != rhs_quotient) {
            return reversed ? lhs_quotient < rhs_quotient
                            : lhs_quotient > rhs_quotient;
        }
        lhs_numerator %= lhs_denominator;
        rhs_numerator %= rhs_denominator;
        if (lhs_numerator == 0 || rhs_numerator == 0) {
            if (lhs_numerator == rhs_numerator) {
                return true;
            }
            return reversed ? lhs_numerator != 0 : rhs_numerator == 0;
        }
        const auto next_lhs_numerator = lhs_denominator;
        const auto next_rhs_numerator = rhs_denominator;
        lhs_denominator = lhs_numerator;
        rhs_denominator = rhs_numerator;
        lhs_numerator = next_lhs_numerator;
        rhs_numerator = next_rhs_numerator;
        reversed = !reversed;
    }
}

[[nodiscard]] inline factor_candidate_generation_result_v1
generate_factor_candidates_v1(
    const soft_membership_store_v1 *stores,
    std::uint64_t store_count,
    factor_candidate_generation_config_v1 config,
    factor_candidate_v1 *candidates,
    std::uint64_t candidate_capacity,
    factor_candidate_member_v1 *members,
    std::uint64_t member_capacity) noexcept {
    if (config.threshold_denominator == 0
        || config.threshold_numerator > config.threshold_denominator
        || config.minimum_members == 0 || config.maximum_overlap == 0
        || config.maximum_pair_checks == 0) {
        return {factor_candidate_generation_code_v1::invalid_config};
    }
    if (store_count != 0 && stores == nullptr) {
        return {factor_candidate_generation_code_v1::missing_stores};
    }
    if (candidate_capacity != 0 && candidates == nullptr) {
        return {factor_candidate_generation_code_v1::missing_candidates};
    }
    if (member_capacity != 0 && members == nullptr) {
        return {factor_candidate_generation_code_v1::missing_members};
    }

    factor_candidate_generation_result_v1 result{};
    for (std::uint64_t input_index = 0; input_index < store_count; ++input_index) {
        result.input_index = input_index;
        const auto &store = stores[input_index];
        if (!evidence::validate_atom_evidence_record_v1(store.evidence_record).valid()
            || !evidence::validate_approximate_membership_v1(store.membership).valid()
            || !(store.evidence_record.evidence_identity
                 == store.membership.evidence_identity)
            || !valid_external_factor_topic_kind_v1(store.kind)
            || store.reserved != 0) {
            result.code = factor_candidate_generation_code_v1::invalid_store;
            return result;
        }

        const auto candidate_member_begin = result.member_count;
        for (std::uint64_t index = 0; index < store.membership.member_count; ++index) {
            const auto &member = store.membership.members[index];
            if (!fraction_at_least_v1(member.weight_numerator,
                                      member.weight_denominator,
                                      config.threshold_numerator,
                                      config.threshold_denominator)) {
                continue;
            }
            std::uint64_t overlap = 0;
            for (std::uint64_t prior = 0; prior < candidate_member_begin; ++prior) {
                if (result.pair_checks == config.maximum_pair_checks) {
                    result.member_count = candidate_member_begin;
                    result.code = factor_candidate_generation_code_v1::work_limit_exceeded;
                    return result;
                }
                ++result.pair_checks;
                if (members[prior].member_identity == member.member_identity) {
                    ++overlap;
                }
            }
            if (overlap >= config.maximum_overlap) {
                continue;
            }
            if (result.member_count == member_capacity) {
                result.member_count = candidate_member_begin;
                result.code =
                    factor_candidate_generation_code_v1::insufficient_member_capacity;
                return result;
            }
            members[result.member_count++] = {member.member_identity,
                                              member.weight_numerator,
                                              member.weight_denominator};
        }
        const auto accepted_count = result.member_count - candidate_member_begin;
        if (accepted_count < config.minimum_members) {
            result.member_count = candidate_member_begin;
            continue;
        }
        if (result.candidate_count == candidate_capacity) {
            result.member_count = candidate_member_begin;
            result.code =
                factor_candidate_generation_code_v1::insufficient_candidate_capacity;
            return result;
        }
        candidates[result.candidate_count++] = {store.evidence_record.evidence_identity,
                                                candidate_member_begin,
                                                accepted_count,
                                                store.kind,
                                                0};
    }
    result.input_index = store_count;
    return result;
}

static_assert(std::is_standard_layout<factor_candidate_v1>::value);
static_assert(std::is_trivially_copyable<factor_candidate_v1>::value);
static_assert(std::is_standard_layout<factor_candidate_member_v1>::value);
static_assert(std::is_trivially_copyable<factor_candidate_member_v1>::value);

} // namespace cellshard::compiler::discovery::factor_topic
