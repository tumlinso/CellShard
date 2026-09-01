#pragma once

#include <CellShard/compiler/partial/partial_atom_v1.hh>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace cellshard::compiler::partial {

inline constexpr std::uint32_t relation_contribution_schema_version_v1 = 1;

struct relation_contribution_v1 {
    atom_persistent_identity_v1 logical_edge_identity{};
    atom_persistent_identity_v1 output_identity{};
    std::uint64_t output_canonical_ordinal = 0;
    double contribution = 0.0;
};

struct relation_contribution_view_v1 {
    const relation_contribution_v1 *contributions = nullptr;
    std::uint64_t contribution_count = 0;
    atom_persistent_identity_v1 relation_identity{};
    atom_persistent_identity_v1 algebra_identity{};
    atom_persistent_identity_v1 numerical_policy_identity{};
    atom_persistent_identity_v1 output_order_identity{};
    std::uint64_t structure_generation = 0;
    std::uint64_t value_generation = 0;
    std::uint32_t schema_version = relation_contribution_schema_version_v1;
    std::uint32_t reserved = 0;
};

enum class relation_contribution_code_v1 : std::uint32_t {
    valid = 0,
    unsupported_schema,
    invalid_identity,
    missing_generation,
    missing_contributions,
    nonfinite_contribution,
    unordered_or_duplicate_edge,
    nonzero_reserved,
    incompatible_contract,
    capacity_overflow,
    duplicate_edge_contribution,
};

struct relation_contribution_result_v1 {
    relation_contribution_code_v1 code = relation_contribution_code_v1::valid;
    std::uint64_t output_count = 0;
    std::uint64_t index = 0;
    [[nodiscard]] constexpr bool valid() const noexcept {
        return code == relation_contribution_code_v1::valid;
    }
};

static_assert(offsetof(relation_contribution_view_v1, contributions) == 0);
static_assert(std::is_standard_layout<relation_contribution_v1>::value);
static_assert(std::is_trivially_copyable<relation_contribution_v1>::value);

[[nodiscard]] inline relation_contribution_result_v1
validate_relation_contributions_v1(
    const relation_contribution_view_v1 &view) noexcept {
    if (view.schema_version != relation_contribution_schema_version_v1) {
        return {relation_contribution_code_v1::unsupported_schema, 0, 0};
    }
    if (!atom::validate_atom_persistent_identity_v1(view.relation_identity)
             .valid()
        || !atom::validate_atom_persistent_identity_v1(view.algebra_identity)
                .valid()
        || !atom::validate_atom_persistent_identity_v1(
                view.numerical_policy_identity)
                .valid()
        || !atom::validate_atom_persistent_identity_v1(view.output_order_identity)
                .valid()) {
        return {relation_contribution_code_v1::invalid_identity, 0, 0};
    }
    if (view.structure_generation == 0 || view.value_generation == 0) {
        return {relation_contribution_code_v1::missing_generation, 0, 0};
    }
    if (view.contribution_count == 0 || view.contributions == nullptr) {
        return {relation_contribution_code_v1::missing_contributions, 0, 0};
    }
    for (std::uint64_t index = 0; index < view.contribution_count; ++index) {
        const auto &item = view.contributions[index];
        if (!atom::validate_atom_persistent_identity_v1(item.logical_edge_identity)
                 .valid()
            || !atom::validate_atom_persistent_identity_v1(item.output_identity)
                    .valid()) {
            return {relation_contribution_code_v1::invalid_identity, 0, index};
        }
        if (!std::isfinite(item.contribution)) {
            return {relation_contribution_code_v1::nonfinite_contribution, 0,
                    index};
        }
        if (index != 0
            && !atom::atom_persistent_identity_less_v1(
                view.contributions[index - 1].logical_edge_identity,
                item.logical_edge_identity)) {
            return {relation_contribution_code_v1::unordered_or_duplicate_edge,
                    0, index};
        }
    }
    if (view.reserved != 0) {
        return {relation_contribution_code_v1::nonzero_reserved, 0, 0};
    }
    return {relation_contribution_code_v1::valid, view.contribution_count,
            view.contribution_count};
}

[[nodiscard]] inline bool relation_contribution_contract_equal_v1(
    const relation_contribution_view_v1 &left,
    const relation_contribution_view_v1 &right) noexcept {
    return left.relation_identity == right.relation_identity
        && left.algebra_identity == right.algebra_identity
        && left.numerical_policy_identity == right.numerical_policy_identity
        && left.output_order_identity == right.output_order_identity
        && left.structure_generation == right.structure_generation
        && left.value_generation == right.value_generation;
}

[[nodiscard]] inline relation_contribution_result_v1
merge_relation_contributions_v1(
    const relation_contribution_view_v1 &left,
    const relation_contribution_view_v1 &right,
    relation_contribution_v1 *output,
    std::uint64_t output_capacity) noexcept {
    if (!validate_relation_contributions_v1(left).valid()) {
        return {relation_contribution_code_v1::invalid_identity, 0, 0};
    }
    if (!validate_relation_contributions_v1(right).valid()) {
        return {relation_contribution_code_v1::invalid_identity, 0, 1};
    }
    if (!relation_contribution_contract_equal_v1(left, right)) {
        return {relation_contribution_code_v1::incompatible_contract, 0, 0};
    }
    if (output == nullptr
        || output_capacity < left.contribution_count + right.contribution_count) {
        return {relation_contribution_code_v1::capacity_overflow, 0,
                left.contribution_count + right.contribution_count};
    }
    std::uint64_t lhs = 0, rhs = 0, out = 0;
    while (lhs < left.contribution_count && rhs < right.contribution_count) {
        const auto &l = left.contributions[lhs];
        const auto &r = right.contributions[rhs];
        if (atom::atom_persistent_identity_less_v1(
                l.logical_edge_identity, r.logical_edge_identity)) {
            output[out++] = l;
            ++lhs;
        } else if (atom::atom_persistent_identity_less_v1(
                       r.logical_edge_identity, l.logical_edge_identity)) {
            output[out++] = r;
            ++rhs;
        } else {
            return {relation_contribution_code_v1::duplicate_edge_contribution,
                    out, lhs};
        }
    }
    while (lhs < left.contribution_count) output[out++] = left.contributions[lhs++];
    while (rhs < right.contribution_count) output[out++] = right.contributions[rhs++];
    return {relation_contribution_code_v1::valid, out, out};
}

} // namespace cellshard::compiler::partial
