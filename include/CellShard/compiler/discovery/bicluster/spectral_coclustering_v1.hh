#pragma once

#include <CellShard/compiler/discovery/bicluster/overlapping_proposals_v1.hh>

#include <cstdint>
#include <limits>

namespace cellshard::compiler::discovery::bicluster {

struct bicluster_spectral_config_v1 {
    std::uint64_t fixed_point_scale = 0;
    std::uint64_t iteration_count = 0;
    std::uint64_t maximum_work_items = 0;
};

enum class bicluster_spectral_code_v1 : std::uint32_t {
    generated = 0,
    invalid_request,
    invalid_catalog,
    invalid_condition,
    invalid_config,
    missing_workspace,
    missing_output,
    insufficient_output_capacity,
    unknown_edge_identity,
    arithmetic_overflow,
    work_limit_exceeded,
    empty_partition,
};

struct bicluster_spectral_result_v1 {
    bicluster_spectral_code_v1 code = bicluster_spectral_code_v1::generated;
    std::uint64_t source_count = 0;
    std::uint64_t destination_count = 0;
    std::uint64_t work_items = 0;
    [[nodiscard]] constexpr bool generated() const noexcept {
        return code == bicluster_spectral_code_v1::generated;
    }
};

[[nodiscard]] inline std::uint64_t identity_index_v1(
    const evidence::evidence_identity_v1 *values,
    std::uint64_t count,
    evidence::evidence_identity_v1 target) noexcept {
    for (std::uint64_t index = 0; index < count; ++index)
        if (values[index] == target) return index;
    return count;
}

[[nodiscard]] inline bicluster_spectral_result_v1
generate_spectral_cocluster_v1(
    const bicluster_provider_request_v1 &request,
    const evidence::evidence_identity_v1 *source_catalog,
    std::uint64_t source_count,
    const evidence::evidence_identity_v1 *destination_catalog,
    std::uint64_t destination_count,
    evidence::evidence_identity_v1 condition,
    bicluster_spectral_config_v1 config,
    std::int64_t *source_scores,
    std::int64_t *destination_scores,
    evidence::evidence_identity_v1 *output_sources,
    std::uint64_t output_source_capacity,
    evidence::evidence_identity_v1 *output_destinations,
    std::uint64_t output_destination_capacity) noexcept {
    if (!validate_bicluster_provider_request_v1(request).valid())
        return {bicluster_spectral_code_v1::invalid_request};
    if (source_catalog == nullptr || source_count == 0
        || destination_catalog == nullptr || destination_count == 0)
        return {bicluster_spectral_code_v1::invalid_catalog};
    if (!evidence::valid_evidence_identity_v1(condition))
        return {bicluster_spectral_code_v1::invalid_condition};
    if (config.fixed_point_scale == 0
        || config.fixed_point_scale
            > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())
        || config.iteration_count == 0 || config.maximum_work_items == 0)
        return {bicluster_spectral_code_v1::invalid_config};
    if (source_scores == nullptr || destination_scores == nullptr)
        return {bicluster_spectral_code_v1::missing_workspace};
    if (output_sources == nullptr || output_destinations == nullptr)
        return {bicluster_spectral_code_v1::missing_output};
    for (std::uint64_t index = 0; index < source_count; ++index) {
        if (!evidence::valid_evidence_identity_v1(source_catalog[index]))
            return {bicluster_spectral_code_v1::invalid_catalog, 0, 0, index};
        source_scores[index] = 1;
    }
    for (std::uint64_t index = 0; index < destination_count; ++index) {
        if (!evidence::valid_evidence_identity_v1(destination_catalog[index]))
            return {bicluster_spectral_code_v1::invalid_catalog, 0, 0, index};
        destination_scores[index] =
            (destination_catalog[index].local_identity & UINT64_C(1)) != 0 ? 1 : -1;
    }

    bicluster_spectral_result_v1 result{};
    for (std::uint64_t iteration = 0; iteration < config.iteration_count; ++iteration) {
        for (std::uint64_t index = 0; index < source_count; ++index)
            source_scores[index] = 0;
        for (std::uint64_t edge_index = 0; edge_index < request.edge_count; ++edge_index) {
            if (result.work_items == config.maximum_work_items) {
                result.code = bicluster_spectral_code_v1::work_limit_exceeded;
                return result;
            }
            ++result.work_items;
            const auto &edge = request.edges[edge_index];
            if (!(edge.condition_identity == condition)) continue;
            const auto source_index = identity_index_v1(
                source_catalog, source_count, edge.source_identity);
            const auto destination_index = identity_index_v1(
                destination_catalog, destination_count, edge.destination_identity);
            if (source_index == source_count || destination_index == destination_count)
                return {bicluster_spectral_code_v1::unknown_edge_identity,
                        0, 0, result.work_items};
            if (edge.weight_numerator != 0
                && config.fixed_point_scale
                    > static_cast<std::uint64_t>(
                        std::numeric_limits<std::int64_t>::max())
                        / edge.weight_numerator)
                return {bicluster_spectral_code_v1::arithmetic_overflow};
            const auto magnitude = static_cast<std::int64_t>(
                (edge.weight_numerator * config.fixed_point_scale)
                / edge.weight_denominator);
            const auto contribution = destination_scores[destination_index] < 0
                ? -magnitude : magnitude;
            if ((contribution > 0
                 && source_scores[source_index]
                     > std::numeric_limits<std::int64_t>::max() - contribution)
                || (contribution < 0
                    && source_scores[source_index]
                        < std::numeric_limits<std::int64_t>::min() - contribution))
                return {bicluster_spectral_code_v1::arithmetic_overflow};
            source_scores[source_index] += contribution;
        }
        for (std::uint64_t index = 0; index < source_count; ++index)
            source_scores[index] = source_scores[index] >= 0 ? 1 : -1;

        for (std::uint64_t index = 0; index < destination_count; ++index)
            destination_scores[index] = 0;
        for (std::uint64_t edge_index = 0; edge_index < request.edge_count; ++edge_index) {
            if (result.work_items == config.maximum_work_items) {
                result.code = bicluster_spectral_code_v1::work_limit_exceeded;
                return result;
            }
            ++result.work_items;
            const auto &edge = request.edges[edge_index];
            if (!(edge.condition_identity == condition)) continue;
            const auto source_index = identity_index_v1(
                source_catalog, source_count, edge.source_identity);
            const auto destination_index = identity_index_v1(
                destination_catalog, destination_count, edge.destination_identity);
            if (source_index == source_count || destination_index == destination_count)
                return {bicluster_spectral_code_v1::unknown_edge_identity};
            const auto magnitude = static_cast<std::int64_t>(
                (edge.weight_numerator * config.fixed_point_scale)
                / edge.weight_denominator);
            const auto contribution = source_scores[source_index] < 0
                ? -magnitude : magnitude;
            if ((contribution > 0
                 && destination_scores[destination_index]
                     > std::numeric_limits<std::int64_t>::max() - contribution)
                || (contribution < 0
                    && destination_scores[destination_index]
                        < std::numeric_limits<std::int64_t>::min() - contribution))
                return {bicluster_spectral_code_v1::arithmetic_overflow};
            destination_scores[destination_index] += contribution;
        }
        for (std::uint64_t index = 0; index < destination_count; ++index)
            destination_scores[index] = destination_scores[index] >= 0 ? 1 : -1;
    }
    for (std::uint64_t index = 0; index < source_count; ++index) {
        if (source_scores[index] <= 0) continue;
        if (result.source_count == output_source_capacity)
            return {bicluster_spectral_code_v1::insufficient_output_capacity,
                    result.source_count, result.destination_count, result.work_items};
        output_sources[result.source_count++] = source_catalog[index];
    }
    for (std::uint64_t index = 0; index < destination_count; ++index) {
        if (destination_scores[index] <= 0) continue;
        if (result.destination_count == output_destination_capacity)
            return {bicluster_spectral_code_v1::insufficient_output_capacity,
                    result.source_count, result.destination_count, result.work_items};
        output_destinations[result.destination_count++] = destination_catalog[index];
    }
    if (result.source_count == 0 || result.destination_count == 0) {
        result.code = bicluster_spectral_code_v1::empty_partition;
        return result;
    }
    return result;
}

} // namespace cellshard::compiler::discovery::bicluster
