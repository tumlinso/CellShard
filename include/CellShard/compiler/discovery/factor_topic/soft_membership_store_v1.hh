#pragma once

#include <CellShard/compiler/discovery/factor_topic/external_evidence_adapter_v1.hh>
#include <CellShard/compiler/evidence/approximate_membership_v1.hh>

#include <cstdint>
#include <type_traits>

namespace cellshard::compiler::discovery::factor_topic {

struct soft_membership_store_v1 {
    evidence::atom_evidence_record_v1 evidence_record{};
    evidence::approximate_membership_view_v1 membership{};
    external_factor_topic_kind_v1 kind = external_factor_topic_kind_v1::factor;
    std::uint32_t reserved = 0;
};

enum class soft_membership_store_code_v1 : std::uint32_t {
    stored = 0,
    null_destination,
    invalid_external_evidence,
    invalid_membership,
    evidence_identity_mismatch,
    insufficient_capacity,
};

struct soft_membership_store_result_v1 {
    soft_membership_store_code_v1 code = soft_membership_store_code_v1::stored;
    std::uint64_t required_capacity = 0;
    std::uint64_t nested_code = 0;

    [[nodiscard]] constexpr bool stored() const noexcept {
        return code == soft_membership_store_code_v1::stored;
    }
};

// Cold, allocation-free copy into caller-owned storage. Rational weights stay
// exact and portable; neither their magnitude nor factor rank establishes
// exact biological coverage.
[[nodiscard]] inline soft_membership_store_result_v1
store_soft_membership_evidence_v1(
    const external_factor_topic_evidence_v1 &external,
    evidence::approximate_membership_view_v1 input,
    evidence::approximate_member_v1 *member_storage,
    std::uint64_t member_capacity,
    soft_membership_store_v1 *destination) noexcept {
    if (destination == nullptr) {
        return {soft_membership_store_code_v1::null_destination};
    }
    *destination = {};
    evidence::atom_evidence_record_v1 adapted{};
    const auto adaptation =
        adapt_external_factor_topic_evidence_v1(external, &adapted);
    if (!adaptation.adapted()) {
        return {soft_membership_store_code_v1::invalid_external_evidence,
                0,
                static_cast<std::uint64_t>(adaptation.code)};
    }
    const auto membership_validation =
        evidence::validate_approximate_membership_v1(input);
    if (!membership_validation.valid()) {
        return {soft_membership_store_code_v1::invalid_membership,
                input.member_count,
                static_cast<std::uint64_t>(membership_validation.code)};
    }
    if (!(input.evidence_identity == external.evidence_identity)) {
        return {soft_membership_store_code_v1::evidence_identity_mismatch,
                input.member_count};
    }
    if (input.member_count > member_capacity
        || (input.member_count != 0 && member_storage == nullptr)) {
        return {soft_membership_store_code_v1::insufficient_capacity,
                input.member_count};
    }
    for (std::uint64_t index = 0; index < input.member_count; ++index) {
        member_storage[index] = input.members[index];
    }
    destination->evidence_record = adapted;
    destination->membership = {member_storage,
                               input.member_count,
                               member_capacity,
                               external.evidence_identity};
    destination->kind = external.kind;
    return {soft_membership_store_code_v1::stored, input.member_count};
}

static_assert(std::is_standard_layout<soft_membership_store_v1>::value);
static_assert(std::is_trivially_copyable<soft_membership_store_v1>::value);

} // namespace cellshard::compiler::discovery::factor_topic
