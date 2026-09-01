#pragma once

#include <CellShard/compiler/discovery/bicluster/provider_v1.hh>

#include <cstdint>
#include <type_traits>

namespace cellshard::compiler::discovery::bicluster {

struct bicluster_seed_rectangle_v1 {
    evidence::evidence_identity_v1 provider_evidence_identity{};
    evidence::evidence_identity_v1 source_anchor{};
    evidence::evidence_identity_v1 destination_anchor{};
    evidence::evidence_identity_v1 condition_identity{};
    std::uint64_t seed_index = 0;
    std::uint64_t edge_index = 0;
};

struct bicluster_seed_config_v1 {
    std::uint64_t minimum_weight_numerator = 0;
    std::uint64_t minimum_weight_denominator = 1;
    std::uint64_t maximum_seeds = 0;
};

enum class bicluster_seed_code_v1 : std::uint32_t {
    generated = 0,
    invalid_request,
    invalid_config,
    missing_output,
    insufficient_capacity,
    work_limit_exceeded,
};

struct bicluster_seed_result_v1 {
    bicluster_seed_code_v1 code = bicluster_seed_code_v1::generated;
    std::uint64_t seed_count = 0;
    std::uint64_t edges_scanned = 0;

    [[nodiscard]] constexpr bool generated() const noexcept {
        return code == bicluster_seed_code_v1::generated;
    }
};

[[nodiscard]] constexpr bool rational_at_least_v1(
    std::uint64_t lhs_numerator,
    std::uint64_t lhs_denominator,
    std::uint64_t rhs_numerator,
    std::uint64_t rhs_denominator) noexcept {
    bool reversed = false;
    for (;;) {
        const auto lhs_whole = lhs_numerator / lhs_denominator;
        const auto rhs_whole = rhs_numerator / rhs_denominator;
        if (lhs_whole != rhs_whole) {
            return reversed ? lhs_whole < rhs_whole : lhs_whole > rhs_whole;
        }
        lhs_numerator %= lhs_denominator;
        rhs_numerator %= rhs_denominator;
        if (lhs_numerator == 0 || rhs_numerator == 0) {
            if (lhs_numerator == rhs_numerator) {
                return true;
            }
            return reversed ? lhs_numerator != 0 : rhs_numerator == 0;
        }
        const auto old_lhs_denominator = lhs_denominator;
        const auto old_rhs_denominator = rhs_denominator;
        lhs_denominator = lhs_numerator;
        rhs_denominator = rhs_numerator;
        lhs_numerator = old_lhs_denominator;
        rhs_numerator = old_rhs_denominator;
        reversed = !reversed;
    }
}

[[nodiscard]] inline bicluster_seed_result_v1 generate_bicluster_seeds_v1(
    const bicluster_provider_request_v1 &request,
    bicluster_seed_config_v1 config,
    bicluster_seed_rectangle_v1 *seeds,
    std::uint64_t seed_capacity) noexcept {
    const auto request_validation = validate_bicluster_provider_request_v1(request);
    if (!request_validation.valid()) {
        return {bicluster_seed_code_v1::invalid_request, 0,
                request_validation.index};
    }
    if (config.minimum_weight_denominator == 0
        || config.minimum_weight_numerator > config.minimum_weight_denominator
        || config.maximum_seeds == 0
        || config.maximum_seeds > request.maximum_proposals) {
        return {bicluster_seed_code_v1::invalid_config};
    }
    if (seed_capacity != 0 && seeds == nullptr) {
        return {bicluster_seed_code_v1::missing_output};
    }
    bicluster_seed_result_v1 result{};
    for (std::uint64_t index = 0; index < request.edge_count; ++index) {
        if (result.edges_scanned == request.maximum_work_items) {
            result.code = bicluster_seed_code_v1::work_limit_exceeded;
            return result;
        }
        ++result.edges_scanned;
        const auto &edge = request.edges[index];
        if (!rational_at_least_v1(edge.weight_numerator,
                                  edge.weight_denominator,
                                  config.minimum_weight_numerator,
                                  config.minimum_weight_denominator)) {
            continue;
        }
        if (result.seed_count == config.maximum_seeds) {
            break;
        }
        if (result.seed_count == seed_capacity) {
            result.code = bicluster_seed_code_v1::insufficient_capacity;
            return result;
        }
        seeds[result.seed_count] = {request.evidence_identity,
                                    edge.source_identity,
                                    edge.destination_identity,
                                    edge.condition_identity,
                                    result.seed_count,
                                    index};
        ++result.seed_count;
    }
    return result;
}

static_assert(std::is_standard_layout<bicluster_seed_rectangle_v1>::value);
static_assert(std::is_trivially_copyable<bicluster_seed_rectangle_v1>::value);

} // namespace cellshard::compiler::discovery::bicluster
