#pragma once

#include <CellShard/compiler/atom/persistent_identity_v1.hh>
#include <CellShard/compiler/evidence/atom_evidence_record_v1.hh>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace cellshard::compiler::discovery::trajectory {

struct trajectory_state_v1 {
    std::uint64_t global_state_id = 0;
    evidence::evidence_identity_v1 biological_state_identity{};
    std::uint64_t state_generation = 0;
    std::uint64_t time_tick = 0;
};

struct lineage_edge_v1 {
    std::uint64_t parent_state_id = 0;
    std::uint64_t child_state_id = 0;
    atom::atom_persistent_identity_v1 branch_identity{};
    std::uint64_t transition_generation = 0;
};

struct trajectory_lineage_view_v1 {
    const trajectory_state_v1 *states = nullptr;
    const lineage_edge_v1 *edges = nullptr;
    std::uint64_t state_count = 0;
    std::uint64_t edge_count = 0;
    std::uint64_t maximum_state_count = 0;
    std::uint64_t maximum_edge_count = 0;
    atom::atom_persistent_identity_v1 trajectory_identity{};
    atom::atom_persistent_identity_v1 state_domain_identity{};
    atom::atom_persistent_identity_v1 state_order_identity{};
    std::uint64_t observation_generation = 0;
};

enum class trajectory_input_code_v1 : std::uint32_t {
    valid = 0,
    empty_states,
    missing_states,
    missing_edges,
    invalid_bound,
    bound_exceeded,
    invalid_trajectory_identity,
    invalid_domain_identity,
    invalid_order_identity,
    missing_observation_generation,
    invalid_state,
    unordered_or_duplicate_state,
    invalid_edge,
    unordered_or_duplicate_edge,
    endpoint_not_found,
    nonforward_time,
    stale_transition_generation,
};

struct trajectory_input_validation_v1 {
    trajectory_input_code_v1 code = trajectory_input_code_v1::valid;
    std::uint64_t index = 0;
    [[nodiscard]] constexpr bool valid() const noexcept {
        return code == trajectory_input_code_v1::valid;
    }
};

static_assert(std::is_standard_layout<trajectory_state_v1>::value);
static_assert(std::is_trivially_copyable<trajectory_state_v1>::value);
static_assert(std::is_standard_layout<lineage_edge_v1>::value);
static_assert(std::is_trivially_copyable<lineage_edge_v1>::value);
static_assert(offsetof(trajectory_lineage_view_v1, states) == 0);
static_assert(std::is_standard_layout<trajectory_lineage_view_v1>::value);

[[nodiscard]] constexpr std::uint64_t find_trajectory_state_v1(
    trajectory_lineage_view_v1 view, std::uint64_t state_id) noexcept {
    std::uint64_t begin = 0;
    std::uint64_t end = view.state_count;
    while (begin < end) {
        const auto middle = begin + (end - begin) / 2;
        if (view.states[middle].global_state_id < state_id) begin = middle + 1;
        else end = middle;
    }
    return begin < view.state_count && view.states[begin].global_state_id == state_id
        ? begin : view.state_count;
}

// O((S+E) log S), O(1) storage. Strictly increasing edge time proves the
// lineage graph acyclic without shape- or ordinal-derived identity.
[[nodiscard]] constexpr trajectory_input_validation_v1
validate_trajectory_lineage_v1(trajectory_lineage_view_v1 view) noexcept {
    if (view.state_count == 0) return {trajectory_input_code_v1::empty_states};
    if (view.states == nullptr) return {trajectory_input_code_v1::missing_states};
    if (view.edge_count != 0 && view.edges == nullptr)
        return {trajectory_input_code_v1::missing_edges};
    if (view.maximum_state_count == 0 || view.maximum_edge_count == 0)
        return {trajectory_input_code_v1::invalid_bound};
    if (view.state_count > view.maximum_state_count
        || view.edge_count > view.maximum_edge_count)
        return {trajectory_input_code_v1::bound_exceeded};
    if (!atom::validate_atom_persistent_identity_v1(view.trajectory_identity).valid())
        return {trajectory_input_code_v1::invalid_trajectory_identity};
    if (!atom::validate_atom_persistent_identity_v1(view.state_domain_identity).valid())
        return {trajectory_input_code_v1::invalid_domain_identity};
    if (!atom::validate_atom_persistent_identity_v1(view.state_order_identity).valid())
        return {trajectory_input_code_v1::invalid_order_identity};
    if (view.observation_generation == 0)
        return {trajectory_input_code_v1::missing_observation_generation};
    for (std::uint64_t index = 0; index < view.state_count; ++index) {
        const auto &state = view.states[index];
        if (state.global_state_id == 0
            || !evidence::valid_evidence_identity_v1(state.biological_state_identity)
            || state.state_generation == 0)
            return {trajectory_input_code_v1::invalid_state, index};
        if (index != 0
            && view.states[index - 1].global_state_id >= state.global_state_id)
            return {trajectory_input_code_v1::unordered_or_duplicate_state, index};
    }
    for (std::uint64_t index = 0; index < view.edge_count; ++index) {
        const auto &edge = view.edges[index];
        if (edge.parent_state_id == 0 || edge.child_state_id == 0
            || edge.parent_state_id == edge.child_state_id
            || !atom::validate_atom_persistent_identity_v1(edge.branch_identity).valid())
            return {trajectory_input_code_v1::invalid_edge, index};
        if (index != 0
            && (view.edges[index - 1].parent_state_id > edge.parent_state_id
                || (view.edges[index - 1].parent_state_id == edge.parent_state_id
                    && view.edges[index - 1].child_state_id >= edge.child_state_id)))
            return {trajectory_input_code_v1::unordered_or_duplicate_edge, index};
        const auto parent = find_trajectory_state_v1(view, edge.parent_state_id);
        const auto child = find_trajectory_state_v1(view, edge.child_state_id);
        if (parent == view.state_count || child == view.state_count)
            return {trajectory_input_code_v1::endpoint_not_found, index};
        if (view.states[parent].time_tick >= view.states[child].time_tick)
            return {trajectory_input_code_v1::nonforward_time, index};
        if (edge.transition_generation != view.observation_generation)
            return {trajectory_input_code_v1::stale_transition_generation, index};
    }
    return {trajectory_input_code_v1::valid, view.edge_count};
}

[[nodiscard]] constexpr bool authorizes_execution(
    trajectory_lineage_view_v1) noexcept { return false; }

} // namespace cellshard::compiler::discovery::trajectory
