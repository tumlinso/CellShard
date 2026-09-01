#pragma once

#include <CellShard/compiler/discovery/operation_trace/graph_family_fragment_v1.hh>
#include <CellShard/compiler/evidence/negative_evidence_v1.hh>

#include <cstdint>
#include <limits>
#include <type_traits>

namespace cellshard::compiler::discovery::operation_trace {

inline constexpr std::uint32_t negative_trace_summary_schema_version_v1 = 1;
inline constexpr std::uint32_t trace_mechanism_count_v1 = 4;

enum class trace_discovery_mechanism_v1 : std::uint32_t {
    coaccess = 0,
    persistent_order = 1,
    partial_result = 2,
    graph_family_fragment = 3,
};

struct negative_trace_summary_input_v1 {
    evidence::evidence_identity_v1 summary_identity{};
    evidence::evidence_identity_v1 evidence_identity{};
    evidence::evidence_identity_v1 trace_scope_identity{};
    std::uint64_t observation_generation = 0;
    std::uint64_t attempted_observations[trace_mechanism_count_v1]{};
    std::uint64_t rejected_observations[trace_mechanism_count_v1]{};
    std::uint64_t retained_candidates[trace_mechanism_count_v1]{};
    std::uint32_t candidate_capacities[trace_mechanism_count_v1]{};
    std::uint32_t maximum_sequence_gap = 0;
    std::uint32_t maximum_fragment_events = 0;
    evidence::negative_evidence_reason_v1 reason =
        evidence::negative_evidence_reason_v1::not_observed;
    std::uint32_t reserved = 0;
};

// Pointer-free persistence record. The fixed four-mechanism table is stable in
// v1 and avoids allocator-owned summary graphs. Persistence encodes fields in
// the artifact byte order; native struct bytes are not a wire format.
struct negative_trace_summary_v1 {
    std::uint32_t schema_version = negative_trace_summary_schema_version_v1;
    std::uint32_t record_bytes = sizeof(negative_trace_summary_v1);
    evidence::evidence_identity_v1 summary_identity{};
    evidence::negative_evidence_v1 negative{};
    std::uint64_t attempted_observations[trace_mechanism_count_v1]{};
    std::uint64_t rejected_observations[trace_mechanism_count_v1]{};
    std::uint64_t retained_candidates[trace_mechanism_count_v1]{};
    std::uint32_t candidate_capacities[trace_mechanism_count_v1]{};
    std::uint32_t maximum_sequence_gap = 0;
    std::uint32_t maximum_fragment_events = 0;
    std::uint32_t reserved = 0;
};

enum class negative_trace_summary_code_v1 : std::uint32_t {
    built = 0,
    missing_output,
    invalid_summary_identity,
    invalid_evidence_identity,
    invalid_trace_scope_identity,
    missing_observation_generation,
    empty_attempts,
    rejection_overflow,
    candidate_capacity_overflow,
    missing_mechanism_bound,
    count_overflow,
    inconsistent_no_observation_reason,
    inconsistent_cap_reason,
    invalid_negative_evidence,
    nonzero_reserved,
};

struct negative_trace_summary_result_v1 {
    negative_trace_summary_code_v1 code =
        negative_trace_summary_code_v1::built;
    std::uint32_t mechanism_index = 0;

    [[nodiscard]] constexpr bool built() const noexcept {
        return code == negative_trace_summary_code_v1::built;
    }
};

[[nodiscard]] constexpr negative_trace_summary_result_v1
build_negative_trace_summary_v1(
    const negative_trace_summary_input_v1 &input,
    negative_trace_summary_v1 *output) noexcept {
    if (output == nullptr) {
        return {negative_trace_summary_code_v1::missing_output};
    }
    *output = {};
    if (!evidence::valid_evidence_identity_v1(input.summary_identity)) {
        return {negative_trace_summary_code_v1::invalid_summary_identity};
    }
    if (!evidence::valid_evidence_identity_v1(input.evidence_identity)) {
        return {negative_trace_summary_code_v1::invalid_evidence_identity};
    }
    if (!evidence::valid_evidence_identity_v1(input.trace_scope_identity)) {
        return {negative_trace_summary_code_v1::invalid_trace_scope_identity};
    }
    if (input.observation_generation == 0) {
        return {negative_trace_summary_code_v1::
                    missing_observation_generation};
    }
    if (input.maximum_sequence_gap == 0
        || input.maximum_fragment_events == 0) {
        return {negative_trace_summary_code_v1::missing_mechanism_bound};
    }
    if (input.reserved != 0) {
        return {negative_trace_summary_code_v1::nonzero_reserved};
    }

    std::uint64_t total_attempted = 0;
    std::uint64_t total_rejected = 0;
    std::uint64_t total_retained = 0;
    bool capacity_reached = false;
    for (std::uint32_t index = 0;
         index < trace_mechanism_count_v1;
         ++index) {
        if (input.rejected_observations[index]
            > input.attempted_observations[index]) {
            return {negative_trace_summary_code_v1::rejection_overflow, index};
        }
        if (input.retained_candidates[index]
            > input.candidate_capacities[index]) {
            return {negative_trace_summary_code_v1::
                        candidate_capacity_overflow,
                    index};
        }
        if (input.candidate_capacities[index] == 0) {
            return {negative_trace_summary_code_v1::missing_mechanism_bound,
                    index};
        }
        if (total_attempted
                > std::numeric_limits<std::uint64_t>::max()
                      - input.attempted_observations[index]
            || total_rejected
                   > std::numeric_limits<std::uint64_t>::max()
                         - input.rejected_observations[index]
            || total_retained
                   > std::numeric_limits<std::uint64_t>::max()
                         - input.retained_candidates[index]) {
            return {negative_trace_summary_code_v1::count_overflow, index};
        }
        total_attempted += input.attempted_observations[index];
        total_rejected += input.rejected_observations[index];
        total_retained += input.retained_candidates[index];
        capacity_reached = capacity_reached
            || input.retained_candidates[index]
                   == input.candidate_capacities[index];
    }
    if (total_attempted == 0) {
        return {negative_trace_summary_code_v1::empty_attempts};
    }
    if (input.reason == evidence::negative_evidence_reason_v1::not_observed
        && total_retained != 0) {
        return {negative_trace_summary_code_v1::
                    inconsistent_no_observation_reason};
    }
    if (input.reason
            == evidence::negative_evidence_reason_v1::candidate_cap_reached
        && !capacity_reached) {
        return {negative_trace_summary_code_v1::inconsistent_cap_reason};
    }

    output->summary_identity = input.summary_identity;
    output->negative.evidence_identity = input.evidence_identity;
    output->negative.subject_identity = input.summary_identity;
    output->negative.observation_scope_identity = input.trace_scope_identity;
    output->negative.observation_generation = input.observation_generation;
    output->negative.attempted_observations = total_attempted;
    output->negative.contradictory_observations = total_rejected;
    output->negative.reason = input.reason;
    for (std::uint32_t index = 0;
         index < trace_mechanism_count_v1;
         ++index) {
        output->attempted_observations[index] =
            input.attempted_observations[index];
        output->rejected_observations[index] =
            input.rejected_observations[index];
        output->retained_candidates[index] =
            input.retained_candidates[index];
        output->candidate_capacities[index] =
            input.candidate_capacities[index];
    }
    output->maximum_sequence_gap = input.maximum_sequence_gap;
    output->maximum_fragment_events = input.maximum_fragment_events;
    if (!evidence::validate_negative_evidence_v1(output->negative).valid()) {
        *output = {};
        return {negative_trace_summary_code_v1::invalid_negative_evidence};
    }
    return {};
}

[[nodiscard]] constexpr bool certifies_absence(
    const negative_trace_summary_v1 &) noexcept {
    return false;
}

[[nodiscard]] constexpr bool authorizes_execution(
    const negative_trace_summary_v1 &) noexcept {
    return false;
}

static_assert(std::is_standard_layout<negative_trace_summary_input_v1>::value);
static_assert(std::is_trivially_copyable<negative_trace_summary_input_v1>::value);
static_assert(std::is_standard_layout<negative_trace_summary_v1>::value);
static_assert(std::is_trivially_copyable<negative_trace_summary_v1>::value);

} // namespace cellshard::compiler::discovery::operation_trace
