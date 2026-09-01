#pragma once

#include <CellShard/compiler/evidence/atom_evidence_record_v1.hh>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace cellshard::compiler::evidence {

struct approximate_member_v1 {
    evidence_identity_v1 member_identity{};
    std::uint64_t weight_numerator = 0;
    std::uint64_t weight_denominator = 0;
};

struct approximate_membership_view_v1 {
    const approximate_member_v1 *members = nullptr;
    std::uint64_t member_count = 0;
    std::uint64_t member_capacity = 0;
    evidence_identity_v1 evidence_identity{};
};

enum class approximate_membership_validation_code_v1 : std::uint32_t {
    valid = 0,
    invalid_evidence_identity,
    empty_membership,
    missing_members,
    capacity_overflow,
    invalid_member_identity,
    invalid_weight,
    unordered_or_duplicate_member,
};

struct approximate_membership_validation_v1 {
    approximate_membership_validation_code_v1 code =
        approximate_membership_validation_code_v1::valid;
    std::uint64_t index = 0;
    [[nodiscard]] constexpr bool valid() const noexcept {
        return code == approximate_membership_validation_code_v1::valid;
    }
};

[[nodiscard]] constexpr approximate_membership_validation_v1
validate_approximate_membership_v1(
    approximate_membership_view_v1 view) noexcept {
    if (!valid_evidence_identity_v1(view.evidence_identity))
        return {approximate_membership_validation_code_v1::invalid_evidence_identity, 0};
    if (view.member_count == 0)
        return {approximate_membership_validation_code_v1::empty_membership, 0};
    if (view.members == nullptr)
        return {approximate_membership_validation_code_v1::missing_members, 0};
    if (view.member_count > view.member_capacity)
        return {approximate_membership_validation_code_v1::capacity_overflow, 0};
    for (std::uint64_t index = 0; index < view.member_count; ++index) {
        const auto &member = view.members[index];
        if (!valid_evidence_identity_v1(member.member_identity))
            return {approximate_membership_validation_code_v1::invalid_member_identity, index};
        if (member.weight_denominator == 0
            || member.weight_numerator > member.weight_denominator)
            return {approximate_membership_validation_code_v1::invalid_weight, index};
        if (index != 0 && !evidence_identity_less_v1(
                view.members[index - 1].member_identity,
                member.member_identity))
            return {approximate_membership_validation_code_v1::unordered_or_duplicate_member, index};
    }
    return {approximate_membership_validation_code_v1::valid, view.member_count};
}

[[nodiscard]] constexpr bool is_exact_membership(
    approximate_membership_view_v1) noexcept {
    return false;
}

static_assert(std::is_standard_layout<approximate_member_v1>::value);
static_assert(std::is_trivially_copyable<approximate_member_v1>::value);
static_assert(offsetof(approximate_membership_view_v1, members) == 0);
static_assert(std::is_standard_layout<approximate_membership_view_v1>::value);
static_assert(std::is_trivially_copyable<approximate_membership_view_v1>::value);

} // namespace cellshard::compiler::evidence
