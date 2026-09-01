#pragma once

#include <CellShard/compiler/atom/persistent_identity_v1.hh>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace cellshard::compiler::atom {

inline constexpr std::uint32_t atom_overlap_role_contract_version_v1 = 1;

enum class atom_overlap_role_v1 : std::uint32_t {
    proposal_membership = 1,
    physical_replica = 2,
    read_halo = 3,
    partial_contribution = 4,
    exclusive_contribution_owner = 5,
};

// An overlap group is a certified statement that this member overlaps another
// member in the named category. Categories never imply one another.
struct atom_overlap_role_record_v1 {
    atom_persistent_identity_v1 member_identity{};
    atom_persistent_identity_v1 membership_identity{};
    atom_persistent_identity_v1 overlap_group_identity{};
    atom_persistent_identity_v1 partial_algebra_identity{};
    atom_overlap_role_v1 role = atom_overlap_role_v1::proposal_membership;
    std::uint8_t overlaps_other_members = 0;
    std::uint8_t reserved[3]{};
};

struct atom_overlap_role_table_v1 {
    const atom_overlap_role_record_v1 *records = nullptr;
    std::uint64_t record_count = 0;
    atom_persistent_identity_v1 table_identity{};
};

enum class atom_overlap_role_validation_code_v1 : std::uint32_t {
    valid = 0,
    empty_table,
    missing_records,
    invalid_table_identity,
    invalid_member_identity,
    invalid_membership_identity,
    invalid_role,
    invalid_overlap_flag,
    unordered_or_duplicate_role,
    missing_overlap_group,
    unexpected_overlap_group,
    exclusive_owner_overlap,
    missing_partial_algebra,
    unexpected_partial_algebra,
    nonzero_reserved,
};

struct atom_overlap_role_validation_v1 {
    atom_overlap_role_validation_code_v1 code =
        atom_overlap_role_validation_code_v1::valid;
    std::uint64_t index = 0;

    [[nodiscard]] constexpr bool valid() const noexcept {
        return code == atom_overlap_role_validation_code_v1::valid;
    }
};

static_assert(std::is_standard_layout<atom_overlap_role_record_v1>::value);
static_assert(std::is_trivially_copyable<atom_overlap_role_record_v1>::value);
static_assert(offsetof(atom_overlap_role_table_v1, records) == 0,
              "overlap role tables must remain pointer-first");
static_assert(std::is_standard_layout<atom_overlap_role_table_v1>::value);
static_assert(std::is_trivially_copyable<atom_overlap_role_table_v1>::value);

[[nodiscard]] constexpr bool valid_atom_overlap_role_v1(
    atom_overlap_role_v1 role) noexcept {
    const auto value = static_cast<std::uint32_t>(role);
    return value >= 1 && value <= 5;
}

[[nodiscard]] constexpr bool atom_overlap_role_record_less_v1(
    const atom_overlap_role_record_v1 &lhs,
    const atom_overlap_role_record_v1 &rhs) noexcept {
    return static_cast<std::uint32_t>(lhs.role)
            < static_cast<std::uint32_t>(rhs.role)
        || (lhs.role == rhs.role
            && atom_persistent_identity_less_v1(
                lhs.member_identity, rhs.member_identity));
}

[[nodiscard]] constexpr bool atom_overlap_permitted_v1(
    atom_overlap_role_v1 role) noexcept {
    return role == atom_overlap_role_v1::proposal_membership
        || role == atom_overlap_role_v1::physical_replica
        || role == atom_overlap_role_v1::read_halo
        || role == atom_overlap_role_v1::partial_contribution;
}

// O(record_count), O(1) storage and allocation-free. Exact set intersection is
// produced upstream; this validator enforces the semantics of that declaration.
[[nodiscard]] constexpr atom_overlap_role_validation_v1
validate_atom_overlap_role_table_v1(atom_overlap_role_table_v1 table) noexcept {
    if (table.record_count == 0) {
        return {atom_overlap_role_validation_code_v1::empty_table, 0};
    }
    if (table.records == nullptr) {
        return {atom_overlap_role_validation_code_v1::missing_records, 0};
    }
    if (!validate_atom_persistent_identity_v1(table.table_identity).valid()) {
        return {atom_overlap_role_validation_code_v1::invalid_table_identity,
                0};
    }
    for (std::uint64_t index = 0; index < table.record_count; ++index) {
        const auto &record = table.records[index];
        if (!validate_atom_persistent_identity_v1(record.member_identity)
                 .valid()) {
            return {atom_overlap_role_validation_code_v1::
                        invalid_member_identity,
                    index};
        }
        if (!validate_atom_persistent_identity_v1(record.membership_identity)
                 .valid()) {
            return {atom_overlap_role_validation_code_v1::
                        invalid_membership_identity,
                    index};
        }
        if (!valid_atom_overlap_role_v1(record.role)) {
            return {atom_overlap_role_validation_code_v1::invalid_role, index};
        }
        if (record.overlaps_other_members > 1) {
            return {atom_overlap_role_validation_code_v1::invalid_overlap_flag,
                    index};
        }
        if (index != 0
            && !atom_overlap_role_record_less_v1(
                table.records[index - 1], record)) {
            return {atom_overlap_role_validation_code_v1::
                        unordered_or_duplicate_role,
                    index};
        }
        const bool group_valid = validate_atom_persistent_identity_v1(
            record.overlap_group_identity).valid();
        if (record.overlaps_other_members != 0 && !group_valid) {
            return {atom_overlap_role_validation_code_v1::
                        missing_overlap_group,
                    index};
        }
        if (record.overlaps_other_members == 0 && group_valid) {
            return {atom_overlap_role_validation_code_v1::
                        unexpected_overlap_group,
                    index};
        }
        if (record.overlaps_other_members != 0
            && !atom_overlap_permitted_v1(record.role)) {
            return {atom_overlap_role_validation_code_v1::
                        exclusive_owner_overlap,
                    index};
        }
        const bool algebra_valid = validate_atom_persistent_identity_v1(
            record.partial_algebra_identity).valid();
        if (record.role == atom_overlap_role_v1::partial_contribution
            && !algebra_valid) {
            return {atom_overlap_role_validation_code_v1::
                        missing_partial_algebra,
                    index};
        }
        if (record.role != atom_overlap_role_v1::partial_contribution
            && algebra_valid) {
            return {atom_overlap_role_validation_code_v1::
                        unexpected_partial_algebra,
                    index};
        }
        for (const auto reserved : record.reserved) {
            if (reserved != 0) {
                return {atom_overlap_role_validation_code_v1::
                            nonzero_reserved,
                        index};
            }
        }
    }
    return {atom_overlap_role_validation_code_v1::valid, table.record_count};
}

} // namespace cellshard::compiler::atom
