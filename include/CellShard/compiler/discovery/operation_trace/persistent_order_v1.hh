#pragma once

#include <CellShard/compiler/discovery/operation_trace/coaccess_heavy_hitters_v1.hh>

#include <cstdint>
#include <limits>
#include <type_traits>

namespace cellshard::compiler::discovery::operation_trace {

struct persistent_order_key_v1 {
    atom::atom_persistent_identity_v1 predecessor{};
    atom::atom_persistent_identity_v1 successor{};
};

struct persistent_order_counter_v1 {
    persistent_order_key_v1 key{};
    std::uint64_t estimated_occurrences = 0;
    std::uint64_t maximum_overestimate = 0;
    std::uint64_t total_sequence_distance = 0;
    std::uint64_t maximum_sequence_distance = 0;
};

struct persistent_order_state_v1 {
    persistent_order_counter_v1 *counters = nullptr;
    std::uint32_t counter_capacity = 0;
    std::uint32_t counter_count = 0;
    std::uint32_t maximum_sequence_gap = 0;
    std::uint32_t reserved = 0;
    std::uint64_t observed_pairs = 0;
    evidence::evidence_identity_v1 evidence_identity{};
    evidence::evidence_identity_v1 source_identity{};
    std::uint64_t observation_generation = 0;
};

enum class persistent_order_code_v1 : std::uint32_t {
    initialized = 0,
    observed,
    invalid_evidence_identity,
    invalid_source_identity,
    missing_observation_generation,
    invalid_capacity,
    missing_counters,
    invalid_sequence_gap,
    nonzero_reserved,
    invalid_predecessor,
    invalid_successor,
    incompatible_event_context,
    non_increasing_sequence,
    sequence_gap_exceeded,
    self_order,
    count_overflow,
    distance_overflow,
};

struct persistent_order_result_v1 {
    persistent_order_code_v1 code = persistent_order_code_v1::initialized;
    std::uint32_t counter_index = 0;

    [[nodiscard]] constexpr bool ok() const noexcept {
        return code == persistent_order_code_v1::initialized
            || code == persistent_order_code_v1::observed;
    }
};

[[nodiscard]] constexpr bool persistent_order_key_equal_v1(
    persistent_order_key_v1 lhs,
    persistent_order_key_v1 rhs) noexcept {
    return lhs.predecessor == rhs.predecessor
        && lhs.successor == rhs.successor;
}

[[nodiscard]] constexpr persistent_order_result_v1
initialize_persistent_order_v1(persistent_order_state_v1 *state) noexcept {
    if (state == nullptr || state->counter_capacity == 0) {
        return {persistent_order_code_v1::invalid_capacity};
    }
    if (state->counters == nullptr) {
        return {persistent_order_code_v1::missing_counters};
    }
    if (!evidence::valid_evidence_identity_v1(state->evidence_identity)) {
        return {persistent_order_code_v1::invalid_evidence_identity};
    }
    if (!evidence::valid_evidence_identity_v1(state->source_identity)) {
        return {persistent_order_code_v1::invalid_source_identity};
    }
    if (state->observation_generation == 0) {
        return {persistent_order_code_v1::missing_observation_generation};
    }
    if (state->maximum_sequence_gap == 0) {
        return {persistent_order_code_v1::invalid_sequence_gap};
    }
    if (state->reserved != 0) {
        return {persistent_order_code_v1::nonzero_reserved};
    }
    for (std::uint32_t index = 0; index < state->counter_capacity; ++index) {
        state->counters[index] = {};
    }
    state->counter_count = 0;
    state->observed_pairs = 0;
    return {};
}

// Updates one directed, caller-bounded pair. The caller selects pairs within a
// local trace window; this function never sorts traces or constructs all-pairs
// order relations. Deterministic Space-Saving retains at most K candidates.
[[nodiscard]] constexpr persistent_order_result_v1
observe_persistent_order_v1(
    persistent_order_state_v1 *state,
    const atom_access_event_v1 &predecessor,
    const atom_access_event_v1 &successor) noexcept {
    if (state == nullptr || state->counter_capacity == 0
        || state->counter_count > state->counter_capacity) {
        return {persistent_order_code_v1::invalid_capacity};
    }
    if (state->counters == nullptr) {
        return {persistent_order_code_v1::missing_counters};
    }
    if (!validate_atom_access_event_v1(predecessor).valid()) {
        return {persistent_order_code_v1::invalid_predecessor};
    }
    if (!validate_atom_access_event_v1(successor).valid()) {
        return {persistent_order_code_v1::invalid_successor};
    }
    if (!same_event_context_v1(predecessor, successor)) {
        return {persistent_order_code_v1::incompatible_event_context};
    }
    if (successor.sequence_number <= predecessor.sequence_number) {
        return {persistent_order_code_v1::non_increasing_sequence};
    }
    const auto distance =
        successor.sequence_number - predecessor.sequence_number;
    if (distance > state->maximum_sequence_gap) {
        return {persistent_order_code_v1::sequence_gap_exceeded};
    }
    if (predecessor.atom_identity == successor.atom_identity) {
        return {persistent_order_code_v1::self_order};
    }
    if (state->observed_pairs == std::numeric_limits<std::uint64_t>::max()) {
        return {persistent_order_code_v1::count_overflow};
    }

    const persistent_order_key_v1 key{
        predecessor.atom_identity, successor.atom_identity};
    for (std::uint32_t index = 0; index < state->counter_count; ++index) {
        auto &counter = state->counters[index];
        if (!persistent_order_key_equal_v1(counter.key, key)) continue;
        if (counter.estimated_occurrences
                == std::numeric_limits<std::uint64_t>::max()) {
            return {persistent_order_code_v1::count_overflow, index};
        }
        if (counter.total_sequence_distance
            > std::numeric_limits<std::uint64_t>::max() - distance) {
            return {persistent_order_code_v1::distance_overflow, index};
        }
        ++counter.estimated_occurrences;
        counter.total_sequence_distance += distance;
        if (distance > counter.maximum_sequence_distance) {
            counter.maximum_sequence_distance = distance;
        }
        ++state->observed_pairs;
        return {persistent_order_code_v1::observed, index};
    }

    std::uint32_t index = state->counter_count;
    if (index < state->counter_capacity) {
        state->counters[index] = {key, 1, 0, distance, distance};
        ++state->counter_count;
    } else {
        index = 0;
        for (std::uint32_t candidate = 1;
             candidate < state->counter_count;
             ++candidate) {
            if (state->counters[candidate].estimated_occurrences
                < state->counters[index].estimated_occurrences) {
                index = candidate;
            }
        }
        const auto minimum = state->counters[index].estimated_occurrences;
        state->counters[index] = {
            key, minimum + 1, minimum, distance, distance};
    }
    ++state->observed_pairs;
    return {persistent_order_code_v1::observed, index};
}

[[nodiscard]] constexpr bool authorizes_execution(
    const persistent_order_state_v1 &) noexcept {
    return false;
}

static_assert(std::is_standard_layout<persistent_order_key_v1>::value);
static_assert(std::is_trivially_copyable<persistent_order_key_v1>::value);
static_assert(std::is_standard_layout<persistent_order_counter_v1>::value);
static_assert(std::is_trivially_copyable<persistent_order_counter_v1>::value);
static_assert(std::is_standard_layout<persistent_order_state_v1>::value);
static_assert(std::is_trivially_copyable<persistent_order_state_v1>::value);

} // namespace cellshard::compiler::discovery::operation_trace
