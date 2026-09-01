#pragma once

#include <CellShard/compiler/discovery/bicluster/exact_census_v1.hh>

#include <cstdint>
#include <type_traits>

namespace cellshard::compiler::discovery::bicluster {

struct bicluster_proposal_v1 {
    expanded_bicluster_v1 rectangle{};
    bicluster_exact_census_v1 census{};
    bicluster_marginal_utility_v1 utility{};
    std::uint64_t proposal_index = 0;
};

struct bicluster_overlap_config_v1 {
    std::uint64_t maximum_proposals = 0;
    std::uint64_t maximum_pair_overlap = 0;
    std::uint64_t maximum_pair_checks = 0;
};

enum class bicluster_proposal_emit_code_v1 : std::uint32_t {
    emitted = 0,
    invalid_config,
    missing_inputs,
    invalid_input,
    missing_output,
    insufficient_capacity,
    work_limit_exceeded,
};

struct bicluster_proposal_emit_result_v1 {
    bicluster_proposal_emit_code_v1 code = bicluster_proposal_emit_code_v1::emitted;
    std::uint64_t proposal_count = 0;
    std::uint64_t rejected_overlap_count = 0;
    std::uint64_t pair_checks = 0;
    std::uint64_t input_index = 0;
    [[nodiscard]] constexpr bool emitted() const noexcept {
        return code == bicluster_proposal_emit_code_v1::emitted;
    }
};

[[nodiscard]] inline bool rectangle_contains_pair_v1(
    const expanded_bicluster_v1 &rectangle,
    evidence::evidence_identity_v1 source,
    evidence::evidence_identity_v1 destination,
    evidence::evidence_identity_v1 condition) noexcept {
    return rectangle.condition_identity == condition
        && identity_in_v1(rectangle.sources, rectangle.source_count, source)
        && identity_in_v1(rectangle.destinations, rectangle.destination_count,
                          destination);
}

[[nodiscard]] inline bicluster_proposal_emit_result_v1
emit_bounded_bicluster_proposals_v1(
    const expanded_bicluster_v1 *rectangles,
    const bicluster_exact_census_v1 *censuses,
    const bicluster_marginal_utility_v1 *utilities,
    std::uint64_t input_count,
    bicluster_overlap_config_v1 config,
    bicluster_proposal_v1 *proposals,
    std::uint64_t proposal_capacity) noexcept {
    if (config.maximum_proposals == 0 || config.maximum_pair_overlap == 0
        || config.maximum_pair_checks == 0)
        return {bicluster_proposal_emit_code_v1::invalid_config};
    if (input_count != 0
        && (rectangles == nullptr || censuses == nullptr || utilities == nullptr))
        return {bicluster_proposal_emit_code_v1::missing_inputs};
    if (proposal_capacity != 0 && proposals == nullptr)
        return {bicluster_proposal_emit_code_v1::missing_output};

    bicluster_proposal_emit_result_v1 result{};
    for (std::uint64_t input_index = 0; input_index < input_count; ++input_index) {
        result.input_index = input_index;
        const auto &rectangle = rectangles[input_index];
        const auto &census = censuses[input_index];
        const auto &utility = utilities[input_index];
        if (rectangle.sources == nullptr || rectangle.source_count == 0
            || rectangle.destinations == nullptr || rectangle.destination_count == 0
            || census.covered_edge_count == 0
            || census.observation_generation == 0 || utility.reserved != 0)
            return {bicluster_proposal_emit_code_v1::invalid_input,
                    result.proposal_count, result.rejected_overlap_count,
                    result.pair_checks, input_index};
        if (utility.disposition != bicluster_promotion_v1::promote_proposal)
            continue;
        bool overlap_rejected = false;
        for (std::uint64_t source_index = 0;
             source_index < rectangle.source_count && !overlap_rejected;
             ++source_index) {
            for (std::uint64_t destination_index = 0;
                 destination_index < rectangle.destination_count;
                 ++destination_index) {
                std::uint64_t overlap = 0;
                for (std::uint64_t prior = 0; prior < result.proposal_count; ++prior) {
                    if (result.pair_checks == config.maximum_pair_checks) {
                        result.code = bicluster_proposal_emit_code_v1::work_limit_exceeded;
                        return result;
                    }
                    ++result.pair_checks;
                    if (rectangle_contains_pair_v1(
                            proposals[prior].rectangle,
                            rectangle.sources[source_index],
                            rectangle.destinations[destination_index],
                            rectangle.condition_identity))
                        ++overlap;
                }
                if (overlap >= config.maximum_pair_overlap) {
                    overlap_rejected = true;
                    break;
                }
            }
        }
        if (overlap_rejected) {
            ++result.rejected_overlap_count;
            continue;
        }
        if (result.proposal_count == config.maximum_proposals) break;
        if (result.proposal_count == proposal_capacity) {
            result.code = bicluster_proposal_emit_code_v1::insufficient_capacity;
            return result;
        }
        proposals[result.proposal_count] = {
            rectangle, census, utility, result.proposal_count};
        ++result.proposal_count;
    }
    result.input_index = input_count;
    return result;
}

[[nodiscard]] constexpr bool authorizes_execution(
    const bicluster_proposal_v1 &) noexcept { return false; }

static_assert(std::is_standard_layout<bicluster_proposal_v1>::value);
static_assert(std::is_trivially_copyable<bicluster_proposal_v1>::value);

} // namespace cellshard::compiler::discovery::bicluster
