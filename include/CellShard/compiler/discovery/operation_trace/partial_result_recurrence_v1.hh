#pragma once

#include <CellShard/compiler/discovery/operation_trace/persistent_order_v1.hh>

#include <cstdint>
#include <limits>
#include <type_traits>

namespace cellshard::compiler::discovery::operation_trace {

struct partial_result_key_v1 {
    atom::atom_persistent_identity_v1 partial_atom_identity{};
    atom::atom_persistent_identity_v1 producer_operation_identity{};
    atom::atom_persistent_identity_v1 producer_stage_identity{};
    atom::atom_persistent_identity_v1 consumer_operation_identity{};
    atom::atom_persistent_identity_v1 consumer_stage_identity{};
    std::uint64_t partial_generation = 0;
    std::uint64_t producer_operation_generation = 0;
    std::uint64_t producer_stage_generation = 0;
    std::uint64_t consumer_operation_generation = 0;
    std::uint64_t consumer_stage_generation = 0;
};

struct partial_result_counter_v1 {
    partial_result_key_v1 key{};
    std::uint64_t estimated_reuses = 0;
    std::uint64_t maximum_overestimate = 0;
    std::uint64_t logical_byte_count = 0;
};

struct partial_result_recurrence_state_v1 {
    partial_result_counter_v1 *counters = nullptr;
    std::uint32_t counter_capacity = 0;
    std::uint32_t counter_count = 0;
    std::uint32_t maximum_sequence_gap = 0;
    std::uint32_t reserved = 0;
    std::uint64_t observed_reuses = 0;
    evidence::evidence_identity_v1 evidence_identity{};
    evidence::evidence_identity_v1 source_identity{};
    std::uint64_t observation_generation = 0;
};

enum class partial_result_recurrence_code_v1 : std::uint32_t {
    initialized = 0,
    observed,
    invalid_state,
    invalid_evidence_identity,
    invalid_source_identity,
    missing_observation_generation,
    invalid_capacity,
    missing_counters,
    invalid_sequence_gap,
    nonzero_reserved,
    invalid_producer,
    invalid_consumer,
    incompatible_trace_context,
    non_increasing_sequence,
    sequence_gap_exceeded,
    mismatched_partial_identity,
    stale_partial_generation,
    mismatched_logical_bytes,
    invalid_producer_access,
    invalid_consumer_access,
    same_operation_stage,
    count_overflow,
};

struct partial_result_recurrence_result_v1 {
    partial_result_recurrence_code_v1 code =
        partial_result_recurrence_code_v1::initialized;
    std::uint32_t counter_index = 0;

    [[nodiscard]] constexpr bool ok() const noexcept {
        return code == partial_result_recurrence_code_v1::initialized
            || code == partial_result_recurrence_code_v1::observed;
    }
};

[[nodiscard]] constexpr bool partial_result_key_equal_v1(
    const partial_result_key_v1 &lhs,
    const partial_result_key_v1 &rhs) noexcept {
    return lhs.partial_atom_identity == rhs.partial_atom_identity
        && lhs.producer_operation_identity == rhs.producer_operation_identity
        && lhs.producer_stage_identity == rhs.producer_stage_identity
        && lhs.consumer_operation_identity == rhs.consumer_operation_identity
        && lhs.consumer_stage_identity == rhs.consumer_stage_identity
        && lhs.partial_generation == rhs.partial_generation
        && lhs.producer_operation_generation
               == rhs.producer_operation_generation
        && lhs.producer_stage_generation == rhs.producer_stage_generation
        && lhs.consumer_operation_generation
               == rhs.consumer_operation_generation
        && lhs.consumer_stage_generation == rhs.consumer_stage_generation;
}

[[nodiscard]] constexpr bool partial_trace_context_equal_v1(
    const atom_access_event_v1 &producer,
    const atom_access_event_v1 &consumer) noexcept {
    return producer.trace_identity == consumer.trace_identity
        && producer.source_identity == consumer.source_identity
        && producer.workload_identity == consumer.workload_identity
        && producer.graph_identity == consumer.graph_identity
        && producer.trace_generation == consumer.trace_generation
        && producer.graph_generation == consumer.graph_generation;
}

[[nodiscard]] constexpr partial_result_recurrence_result_v1
initialize_partial_result_recurrence_v1(
    partial_result_recurrence_state_v1 *state) noexcept {
    if (state == nullptr) {
        return {partial_result_recurrence_code_v1::invalid_state};
    }
    if (state->counter_capacity == 0) {
        return {partial_result_recurrence_code_v1::invalid_capacity};
    }
    if (state->counters == nullptr) {
        return {partial_result_recurrence_code_v1::missing_counters};
    }
    if (!evidence::valid_evidence_identity_v1(state->evidence_identity)) {
        return {partial_result_recurrence_code_v1::invalid_evidence_identity};
    }
    if (!evidence::valid_evidence_identity_v1(state->source_identity)) {
        return {partial_result_recurrence_code_v1::invalid_source_identity};
    }
    if (state->observation_generation == 0) {
        return {partial_result_recurrence_code_v1::
                    missing_observation_generation};
    }
    if (state->maximum_sequence_gap == 0) {
        return {partial_result_recurrence_code_v1::invalid_sequence_gap};
    }
    if (state->reserved != 0) {
        return {partial_result_recurrence_code_v1::nonzero_reserved};
    }
    for (std::uint32_t index = 0; index < state->counter_capacity; ++index) {
        state->counters[index] = {};
    }
    state->counter_count = 0;
    state->observed_reuses = 0;
    return {};
}

// Observes one exact producer-to-consumer access relation, then updates only a
// bounded proposal summary. Equal bytes alone never establish a partial; atom
// identity and generation must agree exactly.
[[nodiscard]] constexpr partial_result_recurrence_result_v1
observe_partial_result_reuse_v1(
    partial_result_recurrence_state_v1 *state,
    const atom_access_event_v1 &producer,
    const atom_access_event_v1 &consumer) noexcept {
    if (state == nullptr) {
        return {partial_result_recurrence_code_v1::invalid_state};
    }
    if (state->counter_capacity == 0
        || state->counter_count > state->counter_capacity) {
        return {partial_result_recurrence_code_v1::invalid_capacity};
    }
    if (state->counters == nullptr) {
        return {partial_result_recurrence_code_v1::missing_counters};
    }
    if (!validate_atom_access_event_v1(producer).valid()) {
        return {partial_result_recurrence_code_v1::invalid_producer};
    }
    if (!validate_atom_access_event_v1(consumer).valid()) {
        return {partial_result_recurrence_code_v1::invalid_consumer};
    }
    if (!partial_trace_context_equal_v1(producer, consumer)) {
        return {partial_result_recurrence_code_v1::incompatible_trace_context};
    }
    if (consumer.sequence_number <= producer.sequence_number) {
        return {partial_result_recurrence_code_v1::non_increasing_sequence};
    }
    if (consumer.sequence_number - producer.sequence_number
        > state->maximum_sequence_gap) {
        return {partial_result_recurrence_code_v1::sequence_gap_exceeded};
    }
    if (producer.atom_identity != consumer.atom_identity) {
        return {partial_result_recurrence_code_v1::mismatched_partial_identity};
    }
    if (producer.atom_generation != consumer.atom_generation) {
        return {partial_result_recurrence_code_v1::stale_partial_generation};
    }
    if (producer.logical_byte_count != consumer.logical_byte_count) {
        return {partial_result_recurrence_code_v1::mismatched_logical_bytes};
    }
    if ((producer.mode != atom_access_mode_v1::write
         && producer.mode != atom_access_mode_v1::read_write)
        || (producer.role != atom_access_role_v1::output
            && producer.role != atom_access_role_v1::intermediate)) {
        return {partial_result_recurrence_code_v1::invalid_producer_access};
    }
    if ((consumer.mode != atom_access_mode_v1::read
         && consumer.mode != atom_access_mode_v1::read_write)
        || (consumer.role != atom_access_role_v1::input
            && consumer.role != atom_access_role_v1::intermediate)) {
        return {partial_result_recurrence_code_v1::invalid_consumer_access};
    }
    if (producer.operation_identity == consumer.operation_identity
        && producer.stage_identity == consumer.stage_identity) {
        return {partial_result_recurrence_code_v1::same_operation_stage};
    }
    if (state->observed_reuses == std::numeric_limits<std::uint64_t>::max()) {
        return {partial_result_recurrence_code_v1::count_overflow};
    }

    const partial_result_key_v1 key{
        producer.atom_identity,
        producer.operation_identity,
        producer.stage_identity,
        consumer.operation_identity,
        consumer.stage_identity,
        producer.atom_generation,
        producer.operation_generation,
        producer.stage_generation,
        consumer.operation_generation,
        consumer.stage_generation};
    for (std::uint32_t index = 0; index < state->counter_count; ++index) {
        auto &counter = state->counters[index];
        if (!partial_result_key_equal_v1(counter.key, key)) continue;
        if (counter.estimated_reuses
            == std::numeric_limits<std::uint64_t>::max()) {
            return {partial_result_recurrence_code_v1::count_overflow, index};
        }
        ++counter.estimated_reuses;
        ++state->observed_reuses;
        return {partial_result_recurrence_code_v1::observed, index};
    }

    std::uint32_t index = state->counter_count;
    if (index < state->counter_capacity) {
        state->counters[index] = {
            key, 1, 0, producer.logical_byte_count};
        ++state->counter_count;
    } else {
        index = 0;
        for (std::uint32_t candidate = 1;
             candidate < state->counter_count;
             ++candidate) {
            if (state->counters[candidate].estimated_reuses
                < state->counters[index].estimated_reuses) {
                index = candidate;
            }
        }
        const auto minimum = state->counters[index].estimated_reuses;
        state->counters[index] = {
            key, minimum + 1, minimum, producer.logical_byte_count};
    }
    ++state->observed_reuses;
    return {partial_result_recurrence_code_v1::observed, index};
}

[[nodiscard]] constexpr bool authorizes_execution(
    const partial_result_recurrence_state_v1 &) noexcept {
    return false;
}

static_assert(std::is_standard_layout<partial_result_key_v1>::value);
static_assert(std::is_trivially_copyable<partial_result_key_v1>::value);
static_assert(std::is_standard_layout<partial_result_counter_v1>::value);
static_assert(std::is_trivially_copyable<partial_result_counter_v1>::value);

} // namespace cellshard::compiler::discovery::operation_trace
