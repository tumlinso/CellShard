#pragma once

#include <CellShard/compiler/discovery/operation_trace/partial_result_recurrence_v1.hh>

#include <cstdint>
#include <limits>
#include <type_traits>

namespace cellshard::compiler::discovery::operation_trace {

struct graph_trace_view_v1 {
    const atom_access_event_v1 *events = nullptr;
    std::uint64_t event_count = 0;
    atom::atom_persistent_identity_v1 graph_identity{};
    std::uint64_t graph_generation = 0;
};

struct graph_fragment_seed_v1 {
    std::uint64_t left_begin = 0;
    std::uint64_t right_begin = 0;
};

struct graph_family_fragment_v1 {
    evidence::evidence_identity_v1 fragment_identity{};
    evidence::evidence_identity_v1 evidence_identity{};
    atom::atom_persistent_identity_v1 left_graph_identity{};
    atom::atom_persistent_identity_v1 right_graph_identity{};
    std::uint64_t left_graph_generation = 0;
    std::uint64_t right_graph_generation = 0;
    std::uint64_t left_begin = 0;
    std::uint64_t right_begin = 0;
    std::uint64_t event_count = 0;
};

enum class graph_family_fragment_code_v1 : std::uint32_t {
    discovered = 0,
    invalid_left_graph_identity,
    invalid_right_graph_identity,
    same_graph,
    missing_left_generation,
    missing_right_generation,
    empty_left_trace,
    empty_right_trace,
    missing_left_events,
    missing_right_events,
    seed_out_of_range,
    invalid_fragment_bound,
    invalid_fragment_identity,
    invalid_evidence_identity,
    missing_output,
    invalid_left_event,
    invalid_right_event,
    event_graph_mismatch,
    no_common_fragment,
};

struct graph_family_fragment_result_v1 {
    graph_family_fragment_code_v1 code =
        graph_family_fragment_code_v1::discovered;
    std::uint64_t event_count = 0;
    std::uint64_t mismatch_offset = 0;

    [[nodiscard]] constexpr bool discovered() const noexcept {
        return code == graph_family_fragment_code_v1::discovered;
    }
};

// Graph identity is deliberately excluded. This compares the semantic access
// signature that may recur across distinct graph instances; exact graph and
// trace identities remain recorded separately in the output provenance.
[[nodiscard]] constexpr bool graph_fragment_event_equal_v1(
    const atom_access_event_v1 &lhs,
    const atom_access_event_v1 &rhs) noexcept {
    return lhs.operation_identity == rhs.operation_identity
        && lhs.stage_identity == rhs.stage_identity
        && lhs.atom_identity == rhs.atom_identity
        && lhs.port_identity == rhs.port_identity
        && lhs.operation_generation == rhs.operation_generation
        && lhs.stage_generation == rhs.stage_generation
        && lhs.atom_generation == rhs.atom_generation
        && lhs.logical_byte_count == rhs.logical_byte_count
        && lhs.mode == rhs.mode
        && lhs.role == rhs.role;
}

// Exact bounded extension of one externally proposed alignment. Runtime is
// O(maximum_fragment_events), storage O(1). The discovery mechanism does not
// enumerate graph pairs or trace offsets, and exact matching remains proposal
// evidence rather than coverage certification.
[[nodiscard]] constexpr graph_family_fragment_result_v1
discover_graph_family_fragment_v1(
    graph_trace_view_v1 left,
    graph_trace_view_v1 right,
    graph_fragment_seed_v1 seed,
    std::uint32_t maximum_fragment_events,
    evidence::evidence_identity_v1 fragment_identity,
    evidence::evidence_identity_v1 evidence_identity,
    graph_family_fragment_v1 *output) noexcept {
    if (!atom::validate_atom_persistent_identity_v1(left.graph_identity)
             .valid()) {
        return {graph_family_fragment_code_v1::invalid_left_graph_identity};
    }
    if (!atom::validate_atom_persistent_identity_v1(right.graph_identity)
             .valid()) {
        return {graph_family_fragment_code_v1::invalid_right_graph_identity};
    }
    if (left.graph_identity == right.graph_identity
        && left.graph_generation == right.graph_generation) {
        return {graph_family_fragment_code_v1::same_graph};
    }
    if (left.graph_generation == 0) {
        return {graph_family_fragment_code_v1::missing_left_generation};
    }
    if (right.graph_generation == 0) {
        return {graph_family_fragment_code_v1::missing_right_generation};
    }
    if (left.event_count == 0) {
        return {graph_family_fragment_code_v1::empty_left_trace};
    }
    if (right.event_count == 0) {
        return {graph_family_fragment_code_v1::empty_right_trace};
    }
    if (left.events == nullptr) {
        return {graph_family_fragment_code_v1::missing_left_events};
    }
    if (right.events == nullptr) {
        return {graph_family_fragment_code_v1::missing_right_events};
    }
    if (seed.left_begin >= left.event_count
        || seed.right_begin >= right.event_count) {
        return {graph_family_fragment_code_v1::seed_out_of_range};
    }
    if (maximum_fragment_events == 0) {
        return {graph_family_fragment_code_v1::invalid_fragment_bound};
    }
    if (!evidence::valid_evidence_identity_v1(fragment_identity)) {
        return {graph_family_fragment_code_v1::invalid_fragment_identity};
    }
    if (!evidence::valid_evidence_identity_v1(evidence_identity)) {
        return {graph_family_fragment_code_v1::invalid_evidence_identity};
    }
    if (output == nullptr) {
        return {graph_family_fragment_code_v1::missing_output};
    }
    *output = {};

    const auto left_remaining = left.event_count - seed.left_begin;
    const auto right_remaining = right.event_count - seed.right_begin;
    std::uint64_t limit = maximum_fragment_events;
    if (left_remaining < limit) limit = left_remaining;
    if (right_remaining < limit) limit = right_remaining;
    std::uint64_t matched = 0;
    for (; matched < limit; ++matched) {
        const auto &left_event = left.events[seed.left_begin + matched];
        const auto &right_event = right.events[seed.right_begin + matched];
        if (!validate_atom_access_event_v1(left_event).valid()) {
            return {graph_family_fragment_code_v1::invalid_left_event,
                    matched,
                    matched};
        }
        if (!validate_atom_access_event_v1(right_event).valid()) {
            return {graph_family_fragment_code_v1::invalid_right_event,
                    matched,
                    matched};
        }
        if (left_event.graph_identity != left.graph_identity
            || left_event.graph_generation != left.graph_generation
            || right_event.graph_identity != right.graph_identity
            || right_event.graph_generation != right.graph_generation) {
            return {graph_family_fragment_code_v1::event_graph_mismatch,
                    matched,
                    matched};
        }
        if (!graph_fragment_event_equal_v1(left_event, right_event)) break;
    }
    if (matched == 0) {
        return {graph_family_fragment_code_v1::no_common_fragment, 0, 0};
    }
    *output = {
        fragment_identity,
        evidence_identity,
        left.graph_identity,
        right.graph_identity,
        left.graph_generation,
        right.graph_generation,
        seed.left_begin,
        seed.right_begin,
        matched};
    return {graph_family_fragment_code_v1::discovered, matched, matched};
}

[[nodiscard]] constexpr bool authorizes_execution(
    const graph_family_fragment_v1 &) noexcept {
    return false;
}

static_assert(std::is_standard_layout<graph_trace_view_v1>::value);
static_assert(std::is_trivially_copyable<graph_trace_view_v1>::value);
static_assert(std::is_standard_layout<graph_family_fragment_v1>::value);
static_assert(std::is_trivially_copyable<graph_family_fragment_v1>::value);

} // namespace cellshard::compiler::discovery::operation_trace
