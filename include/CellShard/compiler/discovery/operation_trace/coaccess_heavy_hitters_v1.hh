#pragma once

#include <CellShard/compiler/discovery/operation_trace/atom_access_event_v1.hh>

#include <cstdint>
#include <limits>
#include <type_traits>

namespace cellshard::compiler::discovery::operation_trace {

struct atom_coaccess_key_v1 {
    atom::atom_persistent_identity_v1 first_atom{};
    atom::atom_persistent_identity_v1 second_atom{};
};

struct atom_coaccess_counter_v1 {
    atom_coaccess_key_v1 key{};
    std::uint64_t estimated_weight = 0;
    std::uint64_t maximum_overestimate = 0;
};

struct coaccess_heavy_hitter_state_v1 {
    atom_coaccess_counter_v1 *counters = nullptr;
    std::uint32_t counter_capacity = 0;
    std::uint32_t counter_count = 0;
    std::uint64_t observed_weight = 0;
    evidence::evidence_identity_v1 evidence_identity{};
    evidence::evidence_identity_v1 source_identity{};
    std::uint64_t observation_generation = 0;
};

enum class coaccess_heavy_hitter_code_v1 : std::uint32_t {
    initialized = 0,
    observed,
    invalid_evidence_identity,
    invalid_source_identity,
    missing_observation_generation,
    invalid_capacity,
    missing_counters,
    invalid_left_event,
    invalid_right_event,
    incompatible_event_context,
    duplicate_event,
    self_coaccess,
    empty_weight,
    weight_overflow,
};

struct coaccess_heavy_hitter_result_v1 {
    coaccess_heavy_hitter_code_v1 code =
        coaccess_heavy_hitter_code_v1::initialized;
    std::uint32_t counter_index = 0;

    [[nodiscard]] constexpr bool ok() const noexcept {
        return code == coaccess_heavy_hitter_code_v1::initialized
            || code == coaccess_heavy_hitter_code_v1::observed;
    }
};

[[nodiscard]] constexpr bool atom_coaccess_key_equal_v1(
    atom_coaccess_key_v1 lhs, atom_coaccess_key_v1 rhs) noexcept {
    return lhs.first_atom == rhs.first_atom
        && lhs.second_atom == rhs.second_atom;
}

[[nodiscard]] constexpr bool same_event_context_v1(
    const atom_access_event_v1 &lhs,
    const atom_access_event_v1 &rhs) noexcept {
    return lhs.trace_identity == rhs.trace_identity
        && lhs.source_identity == rhs.source_identity
        && lhs.workload_identity == rhs.workload_identity
        && lhs.graph_identity == rhs.graph_identity
        && lhs.operation_identity == rhs.operation_identity
        && lhs.stage_identity == rhs.stage_identity
        && lhs.trace_generation == rhs.trace_generation
        && lhs.graph_generation == rhs.graph_generation
        && lhs.operation_generation == rhs.operation_generation
        && lhs.stage_generation == rhs.stage_generation;
}

[[nodiscard]] constexpr atom_coaccess_key_v1 make_atom_coaccess_key_v1(
    atom::atom_persistent_identity_v1 lhs,
    atom::atom_persistent_identity_v1 rhs) noexcept {
    return atom::atom_persistent_identity_less_v1(lhs, rhs)
        ? atom_coaccess_key_v1{lhs, rhs}
        : atom_coaccess_key_v1{rhs, lhs};
}

// Clears caller-owned O(K) storage. K is an explicit u32 local bound; global
// identities and cumulative weights remain u64. No atlas-sized structure is
// allocated or scanned.
[[nodiscard]] constexpr coaccess_heavy_hitter_result_v1
initialize_coaccess_heavy_hitters_v1(
    coaccess_heavy_hitter_state_v1 *state) noexcept {
    if (state == nullptr || state->counter_capacity == 0) {
        return {coaccess_heavy_hitter_code_v1::invalid_capacity};
    }
    if (state->counters == nullptr) {
        return {coaccess_heavy_hitter_code_v1::missing_counters};
    }
    if (!evidence::valid_evidence_identity_v1(state->evidence_identity)) {
        return {coaccess_heavy_hitter_code_v1::invalid_evidence_identity};
    }
    if (!evidence::valid_evidence_identity_v1(state->source_identity)) {
        return {coaccess_heavy_hitter_code_v1::invalid_source_identity};
    }
    if (state->observation_generation == 0) {
        return {coaccess_heavy_hitter_code_v1::
                    missing_observation_generation};
    }
    for (std::uint32_t index = 0; index < state->counter_capacity; ++index) {
        state->counters[index] = {};
    }
    state->counter_count = 0;
    state->observed_weight = 0;
    return {};
}

// Deterministic weighted Space-Saving update. Each supplied pair is local to
// one already-bounded operation/stage context. Runtime is O(K), storage O(K),
// and no all-pairs atlas scan is performed.
[[nodiscard]] constexpr coaccess_heavy_hitter_result_v1
observe_atom_coaccess_v1(
    coaccess_heavy_hitter_state_v1 *state,
    const atom_access_event_v1 &left,
    const atom_access_event_v1 &right,
    std::uint64_t weight = 1) noexcept {
    if (state == nullptr || state->counter_capacity == 0) {
        return {coaccess_heavy_hitter_code_v1::invalid_capacity};
    }
    if (state->counters == nullptr) {
        return {coaccess_heavy_hitter_code_v1::missing_counters};
    }
    if (state->counter_count > state->counter_capacity) {
        return {coaccess_heavy_hitter_code_v1::invalid_capacity};
    }
    if (!validate_atom_access_event_v1(left).valid()) {
        return {coaccess_heavy_hitter_code_v1::invalid_left_event};
    }
    if (!validate_atom_access_event_v1(right).valid()) {
        return {coaccess_heavy_hitter_code_v1::invalid_right_event};
    }
    if (!same_event_context_v1(left, right)) {
        return {coaccess_heavy_hitter_code_v1::incompatible_event_context};
    }
    if (left.event_identity == right.event_identity
        || left.sequence_number == right.sequence_number) {
        return {coaccess_heavy_hitter_code_v1::duplicate_event};
    }
    if (left.atom_identity == right.atom_identity) {
        return {coaccess_heavy_hitter_code_v1::self_coaccess};
    }
    if (weight == 0) {
        return {coaccess_heavy_hitter_code_v1::empty_weight};
    }
    if (state->observed_weight
        > std::numeric_limits<std::uint64_t>::max() - weight) {
        return {coaccess_heavy_hitter_code_v1::weight_overflow};
    }

    const auto key = make_atom_coaccess_key_v1(
        left.atom_identity, right.atom_identity);
    for (std::uint32_t index = 0; index < state->counter_count; ++index) {
        auto &counter = state->counters[index];
        if (atom_coaccess_key_equal_v1(counter.key, key)) {
            if (counter.estimated_weight
                > std::numeric_limits<std::uint64_t>::max() - weight) {
                return {coaccess_heavy_hitter_code_v1::weight_overflow, index};
            }
            counter.estimated_weight += weight;
            state->observed_weight += weight;
            return {coaccess_heavy_hitter_code_v1::observed, index};
        }
    }

    std::uint32_t index = state->counter_count;
    if (index < state->counter_capacity) {
        state->counters[index] = {key, weight, 0};
        ++state->counter_count;
    } else {
        index = 0;
        for (std::uint32_t candidate = 1;
             candidate < state->counter_count;
             ++candidate) {
            if (state->counters[candidate].estimated_weight
                < state->counters[index].estimated_weight) {
                index = candidate;
            }
        }
        const auto minimum = state->counters[index].estimated_weight;
        state->counters[index] = {key, minimum + weight, minimum};
    }
    state->observed_weight += weight;
    return {coaccess_heavy_hitter_code_v1::observed, index};
}

[[nodiscard]] constexpr bool authorizes_execution(
    const coaccess_heavy_hitter_state_v1 &) noexcept {
    return false;
}

static_assert(std::is_standard_layout<atom_coaccess_key_v1>::value);
static_assert(std::is_trivially_copyable<atom_coaccess_key_v1>::value);
static_assert(std::is_standard_layout<atom_coaccess_counter_v1>::value);
static_assert(std::is_trivially_copyable<atom_coaccess_counter_v1>::value);
static_assert(std::is_standard_layout<coaccess_heavy_hitter_state_v1>::value);
static_assert(std::is_trivially_copyable<coaccess_heavy_hitter_state_v1>::value);

} // namespace cellshard::compiler::discovery::operation_trace
