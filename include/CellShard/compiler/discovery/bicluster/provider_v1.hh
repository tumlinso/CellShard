#pragma once

#include <CellShard/compiler/evidence/atom_evidence_record_v1.hh>

#include <cstdint>
#include <type_traits>

namespace cellshard::compiler::discovery::bicluster {

struct bicluster_edge_v1 {
    evidence::evidence_identity_v1 source_identity{};
    evidence::evidence_identity_v1 destination_identity{};
    evidence::evidence_identity_v1 condition_identity{};
    std::uint64_t weight_numerator = 0;
    std::uint64_t weight_denominator = 0;
};

struct bicluster_provider_request_v1 {
    const bicluster_edge_v1 *edges = nullptr;
    std::uint64_t edge_count = 0;
    std::uint64_t edge_capacity = 0;
    evidence::evidence_identity_v1 evidence_identity{};
    evidence::evidence_identity_v1 source_domain_identity{};
    evidence::evidence_identity_v1 destination_domain_identity{};
    evidence::evidence_identity_v1 condition_domain_identity{};
    std::uint64_t observation_generation = 0;
    std::uint64_t maximum_proposals = 0;
    std::uint64_t maximum_work_items = 0;
};

enum class bicluster_provider_validation_code_v1 : std::uint32_t {
    valid = 0,
    invalid_evidence_identity,
    invalid_domain_identity,
    missing_generation,
    empty_edges,
    missing_edges,
    edge_capacity_overflow,
    invalid_edge_identity,
    domain_mismatch,
    invalid_weight,
    unordered_or_duplicate_edge,
    empty_proposal_budget,
    empty_work_budget,
};

struct bicluster_provider_validation_v1 {
    bicluster_provider_validation_code_v1 code =
        bicluster_provider_validation_code_v1::valid;
    std::uint64_t index = 0;

    [[nodiscard]] constexpr bool valid() const noexcept {
        return code == bicluster_provider_validation_code_v1::valid;
    }
};

[[nodiscard]] constexpr bool bicluster_edge_less_v1(
    const bicluster_edge_v1 &lhs, const bicluster_edge_v1 &rhs) noexcept {
    if (!(lhs.source_identity == rhs.source_identity)) {
        return evidence::evidence_identity_less_v1(
            lhs.source_identity, rhs.source_identity);
    }
    if (!(lhs.destination_identity == rhs.destination_identity)) {
        return evidence::evidence_identity_less_v1(
            lhs.destination_identity, rhs.destination_identity);
    }
    return evidence::evidence_identity_less_v1(
        lhs.condition_identity, rhs.condition_identity);
}

[[nodiscard]] inline bicluster_provider_validation_v1
validate_bicluster_provider_request_v1(
    const bicluster_provider_request_v1 &request) noexcept {
    if (!evidence::valid_evidence_identity_v1(request.evidence_identity)) {
        return {bicluster_provider_validation_code_v1::invalid_evidence_identity};
    }
    if (!evidence::valid_evidence_identity_v1(request.source_domain_identity)
        || !evidence::valid_evidence_identity_v1(
            request.destination_domain_identity)
        || !evidence::valid_evidence_identity_v1(request.condition_domain_identity)) {
        return {bicluster_provider_validation_code_v1::invalid_domain_identity};
    }
    if (request.observation_generation == 0) {
        return {bicluster_provider_validation_code_v1::missing_generation};
    }
    if (request.edge_count == 0) {
        return {bicluster_provider_validation_code_v1::empty_edges};
    }
    if (request.edges == nullptr) {
        return {bicluster_provider_validation_code_v1::missing_edges};
    }
    if (request.edge_count > request.edge_capacity) {
        return {bicluster_provider_validation_code_v1::edge_capacity_overflow};
    }
    for (std::uint64_t index = 0; index < request.edge_count; ++index) {
        const auto &edge = request.edges[index];
        if (!evidence::valid_evidence_identity_v1(edge.source_identity)
            || !evidence::valid_evidence_identity_v1(edge.destination_identity)
            || !evidence::valid_evidence_identity_v1(edge.condition_identity)) {
            return {bicluster_provider_validation_code_v1::invalid_edge_identity,
                    index};
        }
        if (edge.source_identity.producer_namespace
                != request.source_domain_identity.producer_namespace
            || edge.destination_identity.producer_namespace
                != request.destination_domain_identity.producer_namespace
            || edge.condition_identity.producer_namespace
                != request.condition_domain_identity.producer_namespace) {
            return {bicluster_provider_validation_code_v1::domain_mismatch, index};
        }
        if (edge.weight_denominator == 0
            || edge.weight_numerator > edge.weight_denominator) {
            return {bicluster_provider_validation_code_v1::invalid_weight, index};
        }
        if (index != 0 && !bicluster_edge_less_v1(request.edges[index - 1], edge)) {
            return {bicluster_provider_validation_code_v1::
                        unordered_or_duplicate_edge,
                    index};
        }
    }
    if (request.maximum_proposals == 0) {
        return {bicluster_provider_validation_code_v1::empty_proposal_budget};
    }
    if (request.maximum_work_items == 0) {
        return {bicluster_provider_validation_code_v1::empty_work_budget};
    }
    return {bicluster_provider_validation_code_v1::valid, request.edge_count};
}

[[nodiscard]] constexpr bool authorizes_execution(
    const bicluster_provider_request_v1 &) noexcept {
    return false;
}

static_assert(std::is_standard_layout<bicluster_edge_v1>::value);
static_assert(std::is_trivially_copyable<bicluster_edge_v1>::value);
static_assert(std::is_standard_layout<bicluster_provider_request_v1>::value);
static_assert(std::is_trivially_copyable<bicluster_provider_request_v1>::value);

} // namespace cellshard::compiler::discovery::bicluster
