#pragma once

#include <CellShard/compiler/atom/persistent_identity_v1.hh>
#include <CellShard/compiler/evidence/biological_stratum_v1.hh>

#include <cstdint>
#include <limits>
#include <type_traits>

namespace cellshard::compiler::discovery::motif {

struct motif_recurrence_key_v1 {
    evidence::evidence_identity_v1 motif_identity{};
    atom::atom_persistent_identity_v1 relation_identity{};
    evidence::evidence_identity_v1 stratum_identity{};
    atom::atom_persistent_identity_v1 graph_family_identity{};
    std::uint64_t graph_generation = 0;
    std::uint64_t stratum_selection_generation = 0;
};

struct motif_recurrence_observation_v1 {
    motif_recurrence_key_v1 key{};
    std::uint64_t graph_count = 0;
    std::uint64_t occurrence_count = 0;
    std::uint64_t opportunity_count = 0;
};

struct motif_recurrence_measurement_v1 {
    motif_recurrence_key_v1 key{};
    std::uint64_t graph_count = 0;
    std::uint64_t occurrence_count = 0;
    std::uint64_t opportunity_count = 0;
};

enum class motif_recurrence_code_v1 : std::uint32_t {
    measured = 0,
    empty_observations,
    missing_observations,
    missing_output,
    invalid_motif_identity,
    invalid_relation_identity,
    invalid_stratum_identity,
    invalid_graph_family_identity,
    missing_graph_generation,
    missing_stratum_generation,
    empty_graphs,
    empty_opportunities,
    occurrences_exceed_opportunities,
    key_mismatch,
    count_overflow,
};

struct motif_recurrence_result_v1 {
    motif_recurrence_code_v1 code = motif_recurrence_code_v1::measured;
    std::uint64_t index = 0;
    [[nodiscard]] constexpr bool measured() const noexcept {
        return code == motif_recurrence_code_v1::measured;
    }
};

static_assert(std::is_standard_layout<motif_recurrence_key_v1>::value);
static_assert(std::is_trivially_copyable<motif_recurrence_key_v1>::value);
static_assert(std::is_standard_layout<motif_recurrence_observation_v1>::value);
static_assert(
    std::is_trivially_copyable<motif_recurrence_observation_v1>::value);

[[nodiscard]] constexpr bool operator==(
    const motif_recurrence_key_v1 &lhs,
    const motif_recurrence_key_v1 &rhs) noexcept {
    return lhs.motif_identity == rhs.motif_identity
        && lhs.relation_identity == rhs.relation_identity
        && lhs.stratum_identity == rhs.stratum_identity
        && lhs.graph_family_identity == rhs.graph_family_identity
        && lhs.graph_generation == rhs.graph_generation
        && lhs.stratum_selection_generation
               == rhs.stratum_selection_generation;
}

[[nodiscard]] constexpr motif_recurrence_result_v1
validate_motif_recurrence_observation_v1(
    const motif_recurrence_observation_v1 &observation,
    std::uint64_t index = 0) noexcept {
    if (!evidence::valid_evidence_identity_v1(
            observation.key.motif_identity)) {
        return {motif_recurrence_code_v1::invalid_motif_identity, index};
    }
    if (!atom::validate_atom_persistent_identity_v1(
             observation.key.relation_identity).valid()) {
        return {motif_recurrence_code_v1::invalid_relation_identity, index};
    }
    if (!evidence::valid_evidence_identity_v1(
            observation.key.stratum_identity)) {
        return {motif_recurrence_code_v1::invalid_stratum_identity, index};
    }
    if (!atom::validate_atom_persistent_identity_v1(
             observation.key.graph_family_identity).valid()) {
        return {motif_recurrence_code_v1::invalid_graph_family_identity, index};
    }
    if (observation.key.graph_generation == 0) {
        return {motif_recurrence_code_v1::missing_graph_generation, index};
    }
    if (observation.key.stratum_selection_generation == 0) {
        return {motif_recurrence_code_v1::missing_stratum_generation, index};
    }
    if (observation.graph_count == 0) {
        return {motif_recurrence_code_v1::empty_graphs, index};
    }
    if (observation.opportunity_count == 0) {
        return {motif_recurrence_code_v1::empty_opportunities, index};
    }
    if (observation.occurrence_count > observation.opportunity_count) {
        return {motif_recurrence_code_v1::occurrences_exceed_opportunities,
                index};
    }
    return {motif_recurrence_code_v1::measured, index};
}

// O(observation_count), O(1) storage. All records must have the exact same
// relation, biological stratum, graph family, and both relevant generations.
[[nodiscard]] constexpr motif_recurrence_result_v1 measure_motif_recurrence_v1(
    const motif_recurrence_observation_v1 *observations,
    std::uint64_t observation_count,
    motif_recurrence_measurement_v1 *output) noexcept {
    if (observation_count == 0) {
        return {motif_recurrence_code_v1::empty_observations};
    }
    if (observations == nullptr) {
        return {motif_recurrence_code_v1::missing_observations};
    }
    if (output == nullptr) {
        return {motif_recurrence_code_v1::missing_output};
    }
    *output = {};
    output->key = observations[0].key;
    for (std::uint64_t index = 0; index < observation_count; ++index) {
        const auto valid =
            validate_motif_recurrence_observation_v1(observations[index], index);
        if (!valid.measured()) {
            return valid;
        }
        if (!(observations[index].key == output->key)) {
            return {motif_recurrence_code_v1::key_mismatch, index};
        }
        if (output->graph_count
                > std::numeric_limits<std::uint64_t>::max()
                      - observations[index].graph_count
            || output->occurrence_count
                > std::numeric_limits<std::uint64_t>::max()
                      - observations[index].occurrence_count
            || output->opportunity_count
                > std::numeric_limits<std::uint64_t>::max()
                      - observations[index].opportunity_count) {
            *output = {};
            return {motif_recurrence_code_v1::count_overflow, index};
        }
        output->graph_count += observations[index].graph_count;
        output->occurrence_count += observations[index].occurrence_count;
        output->opportunity_count += observations[index].opportunity_count;
    }
    return {motif_recurrence_code_v1::measured, observation_count};
}

[[nodiscard]] constexpr bool authorizes_execution(
    const motif_recurrence_measurement_v1 &) noexcept {
    return false;
}

} // namespace cellshard::compiler::discovery::motif
