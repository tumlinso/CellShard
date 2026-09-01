#pragma once

#include <CellShard/compiler/certification/duplicate_detection_v1.hh>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace cellshard::compiler::certification {

inline constexpr std::uint32_t contribution_owner_contract_version_v1 = 1;

struct exact_contribution_owner_v1 {
    atom::atom_persistent_identity_v1 owner_identity{};
    std::uint64_t global_identity = 0;
    std::uint64_t owner_atom_index = 0;
    certification_member_kind_v1 kind =
        certification_member_kind_v1::entity;
    std::uint8_t reserved[7]{};
};

enum class contribution_owner_assignment_code_v1 : std::uint32_t {
    assigned = 0,
    missing_members,
    missing_output,
    insufficient_output,
    invalid_member_kind,
    invalid_owner_identity,
    zero_global_identity,
    unordered_member,
    duplicate_contribution_owner,
};

struct contribution_owner_assignment_result_v1 {
    contribution_owner_assignment_code_v1 code =
        contribution_owner_assignment_code_v1::assigned;
    std::uint64_t index = 0;
    std::uint64_t required_capacity = 0;

    [[nodiscard]] constexpr bool assigned() const noexcept {
        return code == contribution_owner_assignment_code_v1::assigned;
    }
};

static_assert(std::is_standard_layout<exact_contribution_owner_v1>::value);
static_assert(std::is_trivially_copyable<exact_contribution_owner_v1>::value);

[[nodiscard]] constexpr bool same_contribution_key_v1(
    const certification_member_key_v1 &lhs,
    const certification_member_key_v1 &rhs) noexcept {
    return lhs.kind == rhs.kind && lhs.owner_identity == rhs.owner_identity
        && lhs.global_identity == rhs.global_identity;
}

// Input is the sorted key workspace produced by scalable duplicate detection.
// One output record is emitted per unique global member; replicas and halos do
// not enter this surface. Runtime execution can therefore name one exact owner.
[[nodiscard]] inline contribution_owner_assignment_result_v1
assign_exact_contribution_owners_v1(
    const certification_member_key_v1 *members,
    std::uint64_t member_count,
    exact_contribution_owner_v1 *owners,
    std::uint64_t owner_capacity) noexcept {
    if (member_count != 0 && members == nullptr) {
        return {contribution_owner_assignment_code_v1::missing_members};
    }
    if (member_count != 0 && owners == nullptr) {
        return {contribution_owner_assignment_code_v1::missing_output,
                0,
                member_count};
    }
    if (owner_capacity < member_count) {
        return {contribution_owner_assignment_code_v1::insufficient_output,
                0,
                member_count};
    }
    for (std::uint64_t index = 0; index < member_count; ++index) {
        const auto &member = members[index];
        if (member.kind != certification_member_kind_v1::entity
            && member.kind != certification_member_kind_v1::relation_edge) {
            return {contribution_owner_assignment_code_v1::invalid_member_kind,
                    index,
                    member_count};
        }
        if (!atom::validate_atom_persistent_identity_v1(member.owner_identity)
                 .valid()) {
            return {contribution_owner_assignment_code_v1::
                        invalid_owner_identity,
                    index,
                    member_count};
        }
        if (member.global_identity == 0) {
            return {contribution_owner_assignment_code_v1::zero_global_identity,
                    index,
                    member_count};
        }
        if (index != 0) {
            if (same_contribution_key_v1(members[index - 1], member)) {
                return {contribution_owner_assignment_code_v1::
                            duplicate_contribution_owner,
                        index,
                        member_count};
            }
            if (!certification_member_key_less_v1(
                    members[index - 1], member)) {
                return {contribution_owner_assignment_code_v1::unordered_member,
                        index,
                        member_count};
            }
        }
        owners[index] = {member.owner_identity,
                         member.global_identity,
                         member.atom_index,
                         member.kind,
                         {}};
    }
    return {contribution_owner_assignment_code_v1::assigned,
            member_count,
            member_count};
}

} // namespace cellshard::compiler::certification
