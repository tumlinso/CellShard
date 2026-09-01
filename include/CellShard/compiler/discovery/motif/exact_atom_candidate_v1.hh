#pragma once

#include <CellShard/compiler/discovery/motif/occurrence_enumeration_v1.hh>

#include <cstdint>
#include <limits>
#include <type_traits>

namespace cellshard::compiler::discovery::motif {

struct exact_motif_rescan_v1 {
    const std::uint64_t *global_node_ids = nullptr;
    std::uint64_t occurrence_count = 0;
    atom::atom_persistent_identity_v1 canonical_source_identity{};
    std::uint64_t canonical_source_generation = 0;
};

struct exact_motif_atom_candidate_v1 {
    const std::uint64_t *global_node_ids = nullptr;
    std::uint32_t node_count = 0;
    std::uint32_t reserved = 0;
    atom::atom_persistent_identity_v1 candidate_identity{};
};

struct exact_motif_candidate_table_v1 {
    const exact_motif_atom_candidate_v1 *candidates = nullptr;
    std::uint64_t candidate_count = 0;
    atom::atom_persistent_identity_v1 proposal_provider_identity{};
    atom::atom_persistent_identity_v1 certification_authority_identity{};
    atom::atom_persistent_identity_v1 canonical_source_identity{};
    std::uint64_t canonical_source_generation = 0;
};

struct exact_motif_candidate_buffers_v1 {
    exact_motif_atom_candidate_v1 *candidates = nullptr;
    std::uint64_t candidate_capacity = 0;
    std::uint64_t *global_node_ids = nullptr;
    std::uint64_t node_id_capacity = 0;
};

enum class exact_motif_candidate_code_v1 : std::uint32_t {
    built = 0,
    invalid_motif,
    empty_proposal,
    malformed_proposal,
    empty_rescan,
    missing_rescan,
    invalid_source_identity,
    missing_source_generation,
    invalid_proposal_provider,
    invalid_certification_authority,
    provider_self_certification,
    missing_candidate_identities,
    invalid_candidate_identity,
    unordered_or_duplicate_candidate_identity,
    rescan_size_overflow,
    zero_node_identity,
    unordered_or_duplicate_rescan_occurrence,
    occurrence_not_proposed,
    missing_output,
    insufficient_output,
};

struct exact_motif_candidate_result_v1 {
    exact_motif_candidate_code_v1 code = exact_motif_candidate_code_v1::built;
    exact_motif_candidate_table_v1 table{};
    std::uint64_t index = 0;
    std::uint64_t required_candidates = 0;
    std::uint64_t required_node_ids = 0;
    [[nodiscard]] constexpr bool built() const noexcept {
        return code == exact_motif_candidate_code_v1::built;
    }
};

static_assert(offsetof(exact_motif_rescan_v1, global_node_ids) == 0);
static_assert(std::is_standard_layout<exact_motif_atom_candidate_v1>::value);
static_assert(std::is_trivially_copyable<exact_motif_atom_candidate_v1>::value);

namespace detail {

[[nodiscard]] inline int occurrence_compare_v1(
    const std::uint64_t *lhs,
    const std::uint64_t *rhs,
    std::uint32_t node_count) noexcept {
    for (std::uint32_t index = 0; index < node_count; ++index) {
        if (lhs[index] < rhs[index]) return -1;
        if (lhs[index] > rhs[index]) return 1;
    }
    return 0;
}

} // namespace detail

// Only an independently rescanned exact subset may become certification
// candidates. This adapter proves proposal provenance but emits no certificate
// and grants no execution authority.
[[nodiscard]] inline exact_motif_candidate_result_v1
build_exact_motif_atom_candidates_v1(
    typed_motif_vocabulary_view_v1 motif,
    const motif_occurrence_output_v1 &proposal,
    exact_motif_rescan_v1 rescan,
    const atom::atom_persistent_identity_v1 *candidate_identities,
    atom::atom_persistent_identity_v1 proposal_provider_identity,
    atom::atom_persistent_identity_v1 certification_authority_identity,
    exact_motif_candidate_buffers_v1 buffers) noexcept {
    if (!validate_typed_motif_vocabulary_v1(motif).valid()) {
        return {exact_motif_candidate_code_v1::invalid_motif};
    }
    if (proposal.occurrence_count == 0) {
        return {exact_motif_candidate_code_v1::empty_proposal};
    }
    if (proposal.global_node_ids == nullptr
        || proposal.occurrence_count
               > proposal.node_id_capacity / motif.node_count) {
        return {exact_motif_candidate_code_v1::malformed_proposal};
    }
    if (rescan.occurrence_count == 0) {
        return {exact_motif_candidate_code_v1::empty_rescan};
    }
    if (rescan.global_node_ids == nullptr) {
        return {exact_motif_candidate_code_v1::missing_rescan};
    }
    if (!atom::validate_atom_persistent_identity_v1(
             rescan.canonical_source_identity).valid()) {
        return {exact_motif_candidate_code_v1::invalid_source_identity};
    }
    if (rescan.canonical_source_generation == 0) {
        return {exact_motif_candidate_code_v1::missing_source_generation};
    }
    if (!atom::validate_atom_persistent_identity_v1(
             proposal_provider_identity).valid()) {
        return {exact_motif_candidate_code_v1::invalid_proposal_provider};
    }
    if (!atom::validate_atom_persistent_identity_v1(
             certification_authority_identity).valid()) {
        return {exact_motif_candidate_code_v1::invalid_certification_authority};
    }
    if (proposal_provider_identity == certification_authority_identity) {
        return {exact_motif_candidate_code_v1::provider_self_certification};
    }
    if (candidate_identities == nullptr) {
        return {exact_motif_candidate_code_v1::missing_candidate_identities};
    }
    if (rescan.occurrence_count
        > std::numeric_limits<std::uint64_t>::max() / motif.node_count) {
        return {exact_motif_candidate_code_v1::rescan_size_overflow};
    }
    const auto required_nodes = rescan.occurrence_count * motif.node_count;
    for (std::uint64_t occurrence = 0;
         occurrence < rescan.occurrence_count;
         ++occurrence) {
        if (!atom::validate_atom_persistent_identity_v1(
                 candidate_identities[occurrence]).valid()) {
            return {exact_motif_candidate_code_v1::invalid_candidate_identity,
                    {}, occurrence};
        }
        if (occurrence != 0
            && !atom::atom_persistent_identity_less_v1(
                candidate_identities[occurrence - 1],
                candidate_identities[occurrence])) {
            return {exact_motif_candidate_code_v1::
                        unordered_or_duplicate_candidate_identity,
                    {}, occurrence};
        }
        const auto *current = rescan.global_node_ids
            + occurrence * motif.node_count;
        for (std::uint32_t node = 0; node < motif.node_count; ++node) {
            if (current[node] == 0) {
                return {exact_motif_candidate_code_v1::zero_node_identity,
                        {}, occurrence};
            }
        }
        if (occurrence != 0
            && detail::occurrence_compare_v1(
                   current - motif.node_count, current, motif.node_count) >= 0) {
            return {exact_motif_candidate_code_v1::
                        unordered_or_duplicate_rescan_occurrence,
                    {}, occurrence};
        }
        bool proposed = false;
        for (std::uint64_t candidate = 0;
             candidate < proposal.occurrence_count;
             ++candidate) {
            if (detail::occurrence_compare_v1(
                    current,
                    proposal.global_node_ids + candidate * motif.node_count,
                    motif.node_count) == 0) {
                proposed = true;
                break;
            }
        }
        if (!proposed) {
            return {exact_motif_candidate_code_v1::occurrence_not_proposed,
                    {}, occurrence};
        }
    }
    if (buffers.candidates == nullptr || buffers.global_node_ids == nullptr) {
        return {exact_motif_candidate_code_v1::missing_output,
                {}, 0, rescan.occurrence_count, required_nodes};
    }
    if (buffers.candidate_capacity < rescan.occurrence_count
        || buffers.node_id_capacity < required_nodes) {
        return {exact_motif_candidate_code_v1::insufficient_output,
                {}, 0, rescan.occurrence_count, required_nodes};
    }
    for (std::uint64_t index = 0; index < required_nodes; ++index) {
        buffers.global_node_ids[index] = rescan.global_node_ids[index];
    }
    for (std::uint64_t occurrence = 0;
         occurrence < rescan.occurrence_count;
         ++occurrence) {
        buffers.candidates[occurrence] = {
            buffers.global_node_ids + occurrence * motif.node_count,
            motif.node_count, 0, candidate_identities[occurrence]};
    }
    return {exact_motif_candidate_code_v1::built,
            {buffers.candidates, rescan.occurrence_count,
             proposal_provider_identity, certification_authority_identity,
             rescan.canonical_source_identity,
             rescan.canonical_source_generation},
            rescan.occurrence_count, rescan.occurrence_count, required_nodes};
}

[[nodiscard]] constexpr bool authorizes_execution(
    exact_motif_candidate_table_v1) noexcept {
    return false;
}

} // namespace cellshard::compiler::discovery::motif
