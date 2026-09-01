#pragma once

#include <CellShard/compiler/partial/partial_atom_v1.hh>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace cellshard::compiler::partial {

inline constexpr std::uint32_t trajectory_prefix_schema_version_v1 = 1;

struct trajectory_prefix_state_view_v1 {
    const void *state = nullptr;
    std::uint64_t state_bytes = 0;
    std::uint32_t state_alignment = 0;
    std::uint32_t reserved = 0;
    atom_persistent_identity_v1 trajectory_identity{};
    atom_persistent_identity_v1 begin_node_identity{};
    atom_persistent_identity_v1 end_node_identity{};
    atom_persistent_identity_v1 state_schema_identity{};
    atom_persistent_identity_v1 algebra_identity{};
    atom_persistent_identity_v1 numerical_policy_identity{};
    std::uint64_t begin_step = 0;
    std::uint64_t end_step = 0;
    std::uint64_t structure_generation = 0;
    std::uint64_t value_generation = 0;
    std::uint64_t state_generation = 0;
    std::uint32_t schema_version = trajectory_prefix_schema_version_v1;
    std::uint32_t trailing_reserved = 0;
};

enum class trajectory_prefix_code_v1 : std::uint32_t {
    valid = 0,
    unsupported_schema,
    invalid_identity,
    invalid_range,
    missing_generation,
    missing_state,
    invalid_alignment,
    misaligned_state,
    nonzero_reserved,
    incompatible_contract,
    noncontiguous_prefix,
    node_boundary_mismatch,
};

struct trajectory_prefix_result_v1 {
    trajectory_prefix_code_v1 code = trajectory_prefix_code_v1::valid;
    [[nodiscard]] constexpr bool valid() const noexcept {
        return code == trajectory_prefix_code_v1::valid;
    }
};

static_assert(offsetof(trajectory_prefix_state_view_v1, state) == 0);
static_assert(std::is_standard_layout<trajectory_prefix_state_view_v1>::value);
static_assert(std::is_trivially_copyable<trajectory_prefix_state_view_v1>::value);

[[nodiscard]] inline trajectory_prefix_result_v1
validate_trajectory_prefix_v1(
    const trajectory_prefix_state_view_v1 &prefix) noexcept {
    if (prefix.schema_version != trajectory_prefix_schema_version_v1) {
        return {trajectory_prefix_code_v1::unsupported_schema};
    }
#define CELLSHARD_TRAJECTORY_PREFIX_ID(field) \
    if (!atom::validate_atom_persistent_identity_v1(prefix.field).valid()) \
        return {trajectory_prefix_code_v1::invalid_identity}
    CELLSHARD_TRAJECTORY_PREFIX_ID(trajectory_identity);
    CELLSHARD_TRAJECTORY_PREFIX_ID(begin_node_identity);
    CELLSHARD_TRAJECTORY_PREFIX_ID(end_node_identity);
    CELLSHARD_TRAJECTORY_PREFIX_ID(state_schema_identity);
    CELLSHARD_TRAJECTORY_PREFIX_ID(algebra_identity);
    CELLSHARD_TRAJECTORY_PREFIX_ID(numerical_policy_identity);
#undef CELLSHARD_TRAJECTORY_PREFIX_ID
    if (prefix.begin_step >= prefix.end_step) {
        return {trajectory_prefix_code_v1::invalid_range};
    }
    if (prefix.structure_generation == 0 || prefix.value_generation == 0
        || prefix.state_generation == 0) {
        return {trajectory_prefix_code_v1::missing_generation};
    }
    if (prefix.state == nullptr || prefix.state_bytes == 0) {
        return {trajectory_prefix_code_v1::missing_state};
    }
    if (prefix.state_alignment == 0
        || (prefix.state_alignment & (prefix.state_alignment - 1)) != 0) {
        return {trajectory_prefix_code_v1::invalid_alignment};
    }
    if (reinterpret_cast<std::uintptr_t>(prefix.state) % prefix.state_alignment
        != 0) {
        return {trajectory_prefix_code_v1::misaligned_state};
    }
    return prefix.reserved == 0 && prefix.trailing_reserved == 0
        ? trajectory_prefix_result_v1{}
        : trajectory_prefix_result_v1{trajectory_prefix_code_v1::nonzero_reserved};
}

// This proves only structural composability. The Cellerator-owned algebra must
// still combine the opaque states; CellShard never infers trajectory science.
[[nodiscard]] inline trajectory_prefix_result_v1
validate_trajectory_prefix_composition_v1(
    const trajectory_prefix_state_view_v1 &left,
    const trajectory_prefix_state_view_v1 &right) noexcept {
    const auto left_result = validate_trajectory_prefix_v1(left);
    if (!left_result.valid()) return left_result;
    const auto right_result = validate_trajectory_prefix_v1(right);
    if (!right_result.valid()) return right_result;
    if (left.trajectory_identity != right.trajectory_identity
        || left.state_schema_identity != right.state_schema_identity
        || left.algebra_identity != right.algebra_identity
        || left.numerical_policy_identity != right.numerical_policy_identity
        || left.structure_generation != right.structure_generation
        || left.value_generation != right.value_generation) {
        return {trajectory_prefix_code_v1::incompatible_contract};
    }
    if (left.end_step != right.begin_step) {
        return {trajectory_prefix_code_v1::noncontiguous_prefix};
    }
    if (left.end_node_identity != right.begin_node_identity) {
        return {trajectory_prefix_code_v1::node_boundary_mismatch};
    }
    return {};
}

} // namespace cellshard::compiler::partial
