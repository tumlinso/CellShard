#pragma once
#include <CellShard/compiler/discovery/trajectory/input_contract_v1.hh>
#include <cstdint>

namespace cellshard::compiler::discovery::trajectory {

struct state_neighbor_observation_v1 {
    std::uint64_t first_state_id = 0;
    std::uint64_t second_state_id = 0;
    std::uint64_t squared_distance_numerator = 0;
    evidence::evidence_identity_v1 evidence_identity{};
};
struct state_neighborhood_view_v1 {
    const state_neighbor_observation_v1 *neighbors = nullptr;
    std::uint64_t neighbor_count = 0;
    std::uint64_t distance_denominator = 0;
    std::uint64_t maximum_squared_distance_numerator = 0;
    atom::atom_persistent_identity_v1 trajectory_identity{};
    std::uint64_t observation_generation = 0;
};
enum class state_neighborhood_code_v1 : std::uint32_t {
    built, invalid_lineage, empty_observations, missing_observations,
    invalid_distance_policy, invalid_observation, unordered_or_duplicate_pair,
    endpoint_not_found, missing_output, insufficient_output
};
struct state_neighborhood_result_v1 {
    state_neighborhood_code_v1 code = state_neighborhood_code_v1::built;
    state_neighborhood_view_v1 view{};
    std::uint64_t index = 0, required = 0;
    [[nodiscard]] constexpr bool built() const noexcept { return code == state_neighborhood_code_v1::built; }
};

[[nodiscard]] constexpr state_neighborhood_result_v1 build_state_neighborhood_v1(
    trajectory_lineage_view_v1 lineage,
    const state_neighbor_observation_v1 *observations,
    std::uint64_t observation_count,
    std::uint64_t distance_denominator,
    std::uint64_t maximum_squared_distance_numerator,
    state_neighbor_observation_v1 *output,
    std::uint64_t output_capacity) noexcept {
    if (!validate_trajectory_lineage_v1(lineage).valid()) return {state_neighborhood_code_v1::invalid_lineage};
    if (observation_count == 0) return {state_neighborhood_code_v1::empty_observations};
    if (observations == nullptr) return {state_neighborhood_code_v1::missing_observations};
    if (distance_denominator == 0 || maximum_squared_distance_numerator == 0)
        return {state_neighborhood_code_v1::invalid_distance_policy};
    std::uint64_t required = 0;
    for (std::uint64_t i = 0; i < observation_count; ++i) {
        const auto &x = observations[i];
        if (x.first_state_id == 0 || x.first_state_id >= x.second_state_id
            || !evidence::valid_evidence_identity_v1(x.evidence_identity))
            return {state_neighborhood_code_v1::invalid_observation, {}, i};
        if (i && (observations[i-1].first_state_id > x.first_state_id
            || (observations[i-1].first_state_id == x.first_state_id
                && observations[i-1].second_state_id >= x.second_state_id)))
            return {state_neighborhood_code_v1::unordered_or_duplicate_pair, {}, i};
        if (find_trajectory_state_v1(lineage, x.first_state_id) == lineage.state_count
            || find_trajectory_state_v1(lineage, x.second_state_id) == lineage.state_count)
            return {state_neighborhood_code_v1::endpoint_not_found, {}, i};
        if (x.squared_distance_numerator <= maximum_squared_distance_numerator) ++required;
    }
    if (output == nullptr) return {state_neighborhood_code_v1::missing_output, {}, 0, required};
    if (output_capacity < required) return {state_neighborhood_code_v1::insufficient_output, {}, 0, required};
    std::uint64_t cursor = 0;
    for (std::uint64_t i = 0; i < observation_count; ++i)
        if (observations[i].squared_distance_numerator <= maximum_squared_distance_numerator)
            output[cursor++] = observations[i];
    return {state_neighborhood_code_v1::built,
            {output, cursor, distance_denominator, maximum_squared_distance_numerator,
             lineage.trajectory_identity, lineage.observation_generation}, observation_count, required};
}
[[nodiscard]] constexpr bool authorizes_execution(state_neighborhood_view_v1) noexcept { return false; }
} // namespace cellshard::compiler::discovery::trajectory
