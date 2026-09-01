#pragma once

#include <CellShard/compiler/discovery/support_signature/neighborhood_proposal_v1.hh>

#include <cstdint>
#include <limits>

namespace cellshard::compiler::discovery::support_signature {

struct exact_support_neighborhood_rescan_v1 {
    const std::uint64_t *shared_source_ids = nullptr;
    const std::uint64_t *first_residual_source_ids = nullptr;
    const std::uint64_t *second_residual_source_ids = nullptr;
    std::uint64_t shared_source_count = 0;
    std::uint64_t first_residual_count = 0;
    std::uint64_t second_residual_count = 0;
    atom::atom_persistent_identity_v1 proposal_identity{};
};

struct exact_support_neighborhood_rescan_table_v1 {
    const exact_support_neighborhood_rescan_v1 *rescans = nullptr;
    std::uint64_t rescan_count = 0;
    atom::atom_persistent_identity_v1 proposal_provider_identity{};
    atom::atom_persistent_identity_v1 certification_authority_identity{};
    atom::atom_persistent_identity_v1 canonical_source_identity{};
    std::uint64_t canonical_source_generation = 0;
    atom::atom_persistent_identity_v1 relation_identity{};
    std::uint64_t relation_generation = 0;
};

struct exact_support_neighborhood_rescan_buffers_v1 {
    exact_support_neighborhood_rescan_v1 *rescans = nullptr;
    std::uint64_t rescan_capacity = 0;
    std::uint64_t *source_ids = nullptr;
    std::uint64_t source_id_capacity = 0;
};

enum class full_relation_rescan_code_v1 : std::uint32_t {
    rescanned = 0,
    invalid_support,
    invalid_proposals,
    context_mismatch,
    invalid_source_identity,
    missing_source_generation,
    invalid_proposal_provider,
    invalid_certification_authority,
    provider_self_certification,
    destination_not_found,
    source_count_overflow,
    missing_output,
    insufficient_output,
};

struct full_relation_rescan_result_v1 {
    full_relation_rescan_code_v1 code = full_relation_rescan_code_v1::rescanned;
    exact_support_neighborhood_rescan_table_v1 table{};
    std::uint64_t index = 0;
    std::uint64_t required_rescans = 0;
    std::uint64_t required_source_ids = 0;
    [[nodiscard]] constexpr bool rescanned() const noexcept {
        return code == full_relation_rescan_code_v1::rescanned;
    }
};

namespace detail {

[[nodiscard]] constexpr std::uint64_t find_destination_v1(
    exact_destination_support_view_v1 support,
    std::uint64_t global_destination_id) noexcept {
    std::uint64_t begin = 0;
    std::uint64_t end = support.destination_count;
    while (begin < end) {
        const auto middle = begin + (end - begin) / 2;
        if (support.global_destination_ids[middle] < global_destination_id) {
            begin = middle + 1;
        } else {
            end = middle;
        }
    }
    return begin < support.destination_count
            && support.global_destination_ids[begin] == global_destination_id
        ? begin : support.destination_count;
}

struct support_partition_counts_v1 {
    std::uint64_t shared = 0;
    std::uint64_t first_residual = 0;
    std::uint64_t second_residual = 0;
};

constexpr support_partition_counts_v1 partition_support_v1(
    exact_destination_support_view_v1 support,
    std::uint64_t first_destination,
    std::uint64_t second_destination,
    std::uint64_t *shared_output,
    std::uint64_t *first_output,
    std::uint64_t *second_output) noexcept {
    auto first = support.destination_offsets[first_destination];
    const auto first_end = support.destination_offsets[first_destination + 1];
    auto second = support.destination_offsets[second_destination];
    const auto second_end = support.destination_offsets[second_destination + 1];
    support_partition_counts_v1 counts{};
    while (first < first_end || second < second_end) {
        if (second == second_end
            || (first < first_end
                && support.global_source_ids[first]
                    < support.global_source_ids[second])) {
            if (first_output != nullptr) {
                first_output[counts.first_residual] =
                    support.global_source_ids[first];
            }
            ++counts.first_residual;
            ++first;
        } else if (first == first_end
                   || support.global_source_ids[second]
                          < support.global_source_ids[first]) {
            if (second_output != nullptr) {
                second_output[counts.second_residual] =
                    support.global_source_ids[second];
            }
            ++counts.second_residual;
            ++second;
        } else {
            if (shared_output != nullptr) {
                shared_output[counts.shared] = support.global_source_ids[first];
            }
            ++counts.shared;
            ++first;
            ++second;
        }
    }
    return counts;
}

} // namespace detail

// Exact full-relation rescan independently partitions every proposed pair into
// shared support and both residuals. The proposal provider cannot certify its
// own result, and the emitted table still requires the certification lane.
[[nodiscard]] constexpr full_relation_rescan_result_v1
rescan_full_relation_support_neighborhoods_v1(
    exact_destination_support_view_v1 full_support,
    destination_support_neighborhood_view_v1 proposals,
    atom::atom_persistent_identity_v1 canonical_source_identity,
    std::uint64_t canonical_source_generation,
    atom::atom_persistent_identity_v1 proposal_provider_identity,
    atom::atom_persistent_identity_v1 certification_authority_identity,
    exact_support_neighborhood_rescan_buffers_v1 buffers) noexcept {
    if (!validate_exact_destination_support_view_v1(full_support)) {
        return {full_relation_rescan_code_v1::invalid_support};
    }
    if (proposals.proposals == nullptr || proposals.proposal_count == 0
        || !atom::validate_atom_persistent_identity_v1(
                proposals.relation_identity).valid()
        || proposals.relation_generation == 0) {
        return {full_relation_rescan_code_v1::invalid_proposals};
    }
    if (proposals.relation_identity != full_support.relation_identity
        || proposals.relation_generation != full_support.relation_generation) {
        return {full_relation_rescan_code_v1::context_mismatch};
    }
    if (!atom::validate_atom_persistent_identity_v1(
             canonical_source_identity).valid()) {
        return {full_relation_rescan_code_v1::invalid_source_identity};
    }
    if (canonical_source_generation == 0) {
        return {full_relation_rescan_code_v1::missing_source_generation};
    }
    if (!atom::validate_atom_persistent_identity_v1(
             proposal_provider_identity).valid()) {
        return {full_relation_rescan_code_v1::invalid_proposal_provider};
    }
    if (!atom::validate_atom_persistent_identity_v1(
             certification_authority_identity).valid()) {
        return {full_relation_rescan_code_v1::
                    invalid_certification_authority};
    }
    if (proposal_provider_identity == certification_authority_identity) {
        return {full_relation_rescan_code_v1::provider_self_certification};
    }
    std::uint64_t required_sources = 0;
    for (std::uint64_t index = 0; index < proposals.proposal_count; ++index) {
        const auto &proposal = proposals.proposals[index];
        if (!atom::validate_atom_persistent_identity_v1(
                 proposal.proposal_identity).valid()) {
            return {full_relation_rescan_code_v1::invalid_proposals, {}, index};
        }
        const auto first = detail::find_destination_v1(
            full_support, proposal.first_global_destination_id);
        const auto second = detail::find_destination_v1(
            full_support, proposal.second_global_destination_id);
        if (first == full_support.destination_count
            || second == full_support.destination_count || first == second) {
            return {full_relation_rescan_code_v1::destination_not_found,
                    {}, index};
        }
        const auto counts = detail::partition_support_v1(
            full_support, first, second, nullptr, nullptr, nullptr);
        const auto required = counts.shared + counts.first_residual
            + counts.second_residual;
        if (required_sources
            > std::numeric_limits<std::uint64_t>::max() - required) {
            return {full_relation_rescan_code_v1::source_count_overflow,
                    {}, index};
        }
        required_sources += required;
    }
    if (buffers.rescans == nullptr || buffers.source_ids == nullptr) {
        return {full_relation_rescan_code_v1::missing_output, {}, 0,
                proposals.proposal_count, required_sources};
    }
    if (buffers.rescan_capacity < proposals.proposal_count
        || buffers.source_id_capacity < required_sources) {
        return {full_relation_rescan_code_v1::insufficient_output, {}, 0,
                proposals.proposal_count, required_sources};
    }
    std::uint64_t cursor = 0;
    for (std::uint64_t index = 0; index < proposals.proposal_count; ++index) {
        const auto &proposal = proposals.proposals[index];
        const auto first = detail::find_destination_v1(
            full_support, proposal.first_global_destination_id);
        const auto second = detail::find_destination_v1(
            full_support, proposal.second_global_destination_id);
        const auto counts = detail::partition_support_v1(
            full_support, first, second, nullptr, nullptr, nullptr);
        auto *shared = buffers.source_ids + cursor;
        auto *first_residual = shared + counts.shared;
        auto *second_residual = first_residual + counts.first_residual;
        detail::partition_support_v1(
            full_support, first, second, shared, first_residual,
            second_residual);
        buffers.rescans[index] = {
            shared, first_residual, second_residual, counts.shared,
            counts.first_residual, counts.second_residual,
            proposal.proposal_identity};
        cursor += counts.shared + counts.first_residual
            + counts.second_residual;
    }
    return {full_relation_rescan_code_v1::rescanned,
            {buffers.rescans, proposals.proposal_count,
             proposal_provider_identity, certification_authority_identity,
             canonical_source_identity, canonical_source_generation,
             full_support.relation_identity, full_support.relation_generation},
            proposals.proposal_count, proposals.proposal_count,
            required_sources};
}

[[nodiscard]] constexpr bool authorizes_execution(
    exact_support_neighborhood_rescan_table_v1) noexcept {
    return false;
}

} // namespace cellshard::compiler::discovery::support_signature
