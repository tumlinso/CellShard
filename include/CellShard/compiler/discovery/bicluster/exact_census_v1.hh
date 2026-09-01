#pragma once

#include <CellShard/compiler/discovery/bicluster/marginal_cost_v1.hh>

#include <cstdint>

namespace cellshard::compiler::discovery::bicluster {

struct bicluster_exact_census_v1 {
    std::uint64_t rectangle_interaction_count = 0;
    std::uint64_t covered_edge_count = 0;
    std::uint64_t missing_rectangle_count = 0;
    std::uint64_t residual_edge_count = 0;
    std::uint64_t observation_generation = 0;
};

enum class bicluster_exact_census_code_v1 : std::uint32_t {
    constructed = 0,
    invalid_request,
    invalid_rectangle,
    duplicate_source,
    duplicate_destination,
    arithmetic_overflow,
    missing_residual_output,
    insufficient_residual_capacity,
    null_destination,
};

struct bicluster_exact_census_result_v1 {
    bicluster_exact_census_code_v1 code = bicluster_exact_census_code_v1::constructed;
    std::uint64_t residual_count = 0;
    std::uint64_t index = 0;
    [[nodiscard]] constexpr bool constructed() const noexcept {
        return code == bicluster_exact_census_code_v1::constructed;
    }
};

[[nodiscard]] inline bicluster_exact_census_result_v1
construct_bicluster_exact_census_v1(
    const bicluster_provider_request_v1 &request,
    const expanded_bicluster_v1 &rectangle,
    bicluster_edge_v1 *residual_edges,
    std::uint64_t residual_capacity,
    bicluster_exact_census_v1 *destination) noexcept {
    if (destination == nullptr)
        return {bicluster_exact_census_code_v1::null_destination};
    *destination = {};
    if (!validate_bicluster_provider_request_v1(request).valid())
        return {bicluster_exact_census_code_v1::invalid_request};
    if (!(rectangle.provider_evidence_identity == request.evidence_identity)
        || !evidence::valid_evidence_identity_v1(rectangle.condition_identity)
        || rectangle.sources == nullptr || rectangle.source_count == 0
        || rectangle.source_count > rectangle.source_capacity
        || rectangle.destinations == nullptr || rectangle.destination_count == 0
        || rectangle.destination_count > rectangle.destination_capacity) {
        return {bicluster_exact_census_code_v1::invalid_rectangle};
    }
    for (std::uint64_t left = 0; left < rectangle.source_count; ++left) {
        for (std::uint64_t right = left + 1; right < rectangle.source_count; ++right) {
            if (rectangle.sources[left] == rectangle.sources[right])
                return {bicluster_exact_census_code_v1::duplicate_source, 0, right};
        }
    }
    for (std::uint64_t left = 0; left < rectangle.destination_count; ++left) {
        for (std::uint64_t right = left + 1; right < rectangle.destination_count; ++right) {
            if (rectangle.destinations[left] == rectangle.destinations[right])
                return {bicluster_exact_census_code_v1::duplicate_destination, 0, right};
        }
    }
    std::uint64_t rectangle_count = 0;
    if (!bicluster_checked_multiply_v1(
            rectangle.source_count, rectangle.destination_count, &rectangle_count))
        return {bicluster_exact_census_code_v1::arithmetic_overflow};

    std::uint64_t covered = 0;
    std::uint64_t residual_count = 0;
    for (std::uint64_t index = 0; index < request.edge_count; ++index) {
        const auto &edge = request.edges[index];
        if (!(edge.condition_identity == rectangle.condition_identity))
            continue;
        const bool inside = identity_in_v1(
                rectangle.sources, rectangle.source_count, edge.source_identity)
            && identity_in_v1(rectangle.destinations, rectangle.destination_count,
                              edge.destination_identity);
        if (inside) ++covered;
        else ++residual_count;
    }
    if (residual_count != 0 && residual_edges == nullptr)
        return {bicluster_exact_census_code_v1::missing_residual_output,
                residual_count};
    if (residual_count > residual_capacity)
        return {bicluster_exact_census_code_v1::insufficient_residual_capacity,
                residual_count};
    std::uint64_t output = 0;
    for (std::uint64_t index = 0; index < request.edge_count; ++index) {
        const auto &edge = request.edges[index];
        if (!(edge.condition_identity == rectangle.condition_identity)) continue;
        if (!identity_in_v1(rectangle.sources, rectangle.source_count,
                            edge.source_identity)
            || !identity_in_v1(rectangle.destinations, rectangle.destination_count,
                               edge.destination_identity)) {
            residual_edges[output++] = edge;
        }
    }
    *destination = {rectangle_count, covered, rectangle_count - covered,
                    output, request.observation_generation};
    return {bicluster_exact_census_code_v1::constructed, output, request.edge_count};
}

[[nodiscard]] constexpr bool authorizes_execution(
    const bicluster_exact_census_v1 &) noexcept { return false; }

} // namespace cellshard::compiler::discovery::bicluster
