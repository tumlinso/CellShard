#pragma once

#include <CellShard/compiler/discovery/bicluster/seed_rectangles_v1.hh>

#include <cstdint>
#include <type_traits>

namespace cellshard::compiler::discovery::bicluster {

struct expanded_bicluster_v1 {
    evidence::evidence_identity_v1 provider_evidence_identity{};
    evidence::evidence_identity_v1 condition_identity{};
    const evidence::evidence_identity_v1 *sources = nullptr;
    std::uint64_t source_count = 0;
    std::uint64_t source_capacity = 0;
    const evidence::evidence_identity_v1 *destinations = nullptr;
    std::uint64_t destination_count = 0;
    std::uint64_t destination_capacity = 0;
    std::uint64_t seed_index = 0;
    std::uint64_t completed_rounds = 0;
};

struct bicluster_expansion_config_v1 {
    std::uint64_t minimum_weight_numerator = 0;
    std::uint64_t minimum_weight_denominator = 1;
    std::uint64_t maximum_rounds = 0;
    std::uint64_t maximum_work_items = 0;
};

enum class bicluster_expansion_code_v1 : std::uint32_t {
    expanded = 0,
    invalid_request,
    invalid_seed,
    invalid_config,
    missing_source_storage,
    missing_destination_storage,
    insufficient_source_capacity,
    insufficient_destination_capacity,
    work_limit_exceeded,
    null_destination,
};

struct bicluster_expansion_result_v1 {
    bicluster_expansion_code_v1 code = bicluster_expansion_code_v1::expanded;
    std::uint64_t source_count = 0;
    std::uint64_t destination_count = 0;
    std::uint64_t work_items = 0;

    [[nodiscard]] constexpr bool expanded() const noexcept {
        return code == bicluster_expansion_code_v1::expanded;
    }
};

[[nodiscard]] inline bool identity_in_v1(
    const evidence::evidence_identity_v1 *values,
    std::uint64_t count,
    evidence::evidence_identity_v1 value) noexcept {
    for (std::uint64_t index = 0; index < count; ++index) {
        if (values[index] == value) {
            return true;
        }
    }
    return false;
}

[[nodiscard]] inline bool qualifying_edge_v1(
    const bicluster_provider_request_v1 &request,
    evidence::evidence_identity_v1 source,
    evidence::evidence_identity_v1 destination,
    evidence::evidence_identity_v1 condition,
    bicluster_expansion_config_v1 config,
    std::uint64_t *work_items,
    bool *work_exhausted) noexcept {
    for (std::uint64_t index = 0; index < request.edge_count; ++index) {
        if (*work_items == config.maximum_work_items) {
            *work_exhausted = true;
            return false;
        }
        ++*work_items;
        const auto &edge = request.edges[index];
        if (edge.source_identity == source
            && edge.destination_identity == destination
            && edge.condition_identity == condition) {
            return rational_at_least_v1(edge.weight_numerator,
                                        edge.weight_denominator,
                                        config.minimum_weight_numerator,
                                        config.minimum_weight_denominator);
        }
    }
    return false;
}

[[nodiscard]] inline bicluster_expansion_result_v1
expand_bicluster_alternating_v1(
    const bicluster_provider_request_v1 &request,
    const bicluster_seed_rectangle_v1 &seed,
    bicluster_expansion_config_v1 config,
    evidence::evidence_identity_v1 *source_storage,
    std::uint64_t source_capacity,
    evidence::evidence_identity_v1 *destination_storage,
    std::uint64_t destination_capacity,
    expanded_bicluster_v1 *destination) noexcept {
    if (destination == nullptr) {
        return {bicluster_expansion_code_v1::null_destination};
    }
    *destination = {};
    if (!validate_bicluster_provider_request_v1(request).valid()) {
        return {bicluster_expansion_code_v1::invalid_request};
    }
    if (!(seed.provider_evidence_identity == request.evidence_identity)
        || !evidence::valid_evidence_identity_v1(seed.source_anchor)
        || !evidence::valid_evidence_identity_v1(seed.destination_anchor)
        || !evidence::valid_evidence_identity_v1(seed.condition_identity)
        || seed.edge_index >= request.edge_count) {
        return {bicluster_expansion_code_v1::invalid_seed};
    }
    if (config.minimum_weight_denominator == 0
        || config.minimum_weight_numerator > config.minimum_weight_denominator
        || config.maximum_rounds == 0 || config.maximum_work_items == 0) {
        return {bicluster_expansion_code_v1::invalid_config};
    }
    if (source_storage == nullptr) {
        return {bicluster_expansion_code_v1::missing_source_storage};
    }
    if (destination_storage == nullptr) {
        return {bicluster_expansion_code_v1::missing_destination_storage};
    }
    if (source_capacity == 0) {
        return {bicluster_expansion_code_v1::insufficient_source_capacity};
    }
    if (destination_capacity == 0) {
        return {bicluster_expansion_code_v1::insufficient_destination_capacity};
    }
    source_storage[0] = seed.source_anchor;
    destination_storage[0] = seed.destination_anchor;
    bicluster_expansion_result_v1 result{};
    result.source_count = 1;
    result.destination_count = 1;

    std::uint64_t completed_rounds = 0;
    for (; completed_rounds < config.maximum_rounds; ++completed_rounds) {
        bool changed = false;
        for (std::uint64_t edge_index = 0; edge_index < request.edge_count; ++edge_index) {
            const auto candidate = request.edges[edge_index].source_identity;
            if (!(request.edges[edge_index].condition_identity == seed.condition_identity)
                || identity_in_v1(source_storage, result.source_count, candidate)) {
                continue;
            }
            bool connects = true;
            for (std::uint64_t index = 0; index < result.destination_count; ++index) {
                bool exhausted = false;
                if (!qualifying_edge_v1(request, candidate, destination_storage[index],
                                        seed.condition_identity, config,
                                        &result.work_items, &exhausted)) {
                    if (exhausted) {
                        result.code = bicluster_expansion_code_v1::work_limit_exceeded;
                        return result;
                    }
                    connects = false;
                    break;
                }
            }
            if (connects) {
                if (result.source_count == source_capacity) {
                    result.code = bicluster_expansion_code_v1::insufficient_source_capacity;
                    return result;
                }
                source_storage[result.source_count++] = candidate;
                changed = true;
            }
        }
        for (std::uint64_t edge_index = 0; edge_index < request.edge_count; ++edge_index) {
            const auto candidate = request.edges[edge_index].destination_identity;
            if (!(request.edges[edge_index].condition_identity == seed.condition_identity)
                || identity_in_v1(destination_storage, result.destination_count, candidate)) {
                continue;
            }
            bool connects = true;
            for (std::uint64_t index = 0; index < result.source_count; ++index) {
                bool exhausted = false;
                if (!qualifying_edge_v1(request, source_storage[index], candidate,
                                        seed.condition_identity, config,
                                        &result.work_items, &exhausted)) {
                    if (exhausted) {
                        result.code = bicluster_expansion_code_v1::work_limit_exceeded;
                        return result;
                    }
                    connects = false;
                    break;
                }
            }
            if (connects) {
                if (result.destination_count == destination_capacity) {
                    result.code = bicluster_expansion_code_v1::
                        insufficient_destination_capacity;
                    return result;
                }
                destination_storage[result.destination_count++] = candidate;
                changed = true;
            }
        }
        if (!changed) {
            ++completed_rounds;
            break;
        }
    }
    *destination = {request.evidence_identity,
                    seed.condition_identity,
                    source_storage,
                    result.source_count,
                    source_capacity,
                    destination_storage,
                    result.destination_count,
                    destination_capacity,
                    seed.seed_index,
                    completed_rounds};
    return result;
}

static_assert(std::is_standard_layout<expanded_bicluster_v1>::value);
static_assert(std::is_trivially_copyable<expanded_bicluster_v1>::value);

} // namespace cellshard::compiler::discovery::bicluster
