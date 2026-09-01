#pragma once

#include <CellShard/compiler/discovery/support_signature/candidate_pairs_v1.hh>

#include <cstdint>

namespace cellshard::compiler::discovery::support_signature {

struct exact_support_pair_score_v1 {
    std::uint32_t first_destination_index = 0;
    std::uint32_t second_destination_index = 0;
    std::uint64_t shared_support_count = 0;
    std::uint64_t union_support_count = 0;
    std::uint64_t first_support_count = 0;
    std::uint64_t second_support_count = 0;
};

struct exact_support_pair_score_view_v1 {
    const exact_support_pair_score_v1 *scores = nullptr;
    std::uint64_t score_count = 0;
    atom::atom_persistent_identity_v1 relation_identity{};
    std::uint64_t relation_generation = 0;
};

enum class exact_support_rescan_code_v1 : std::uint32_t {
    rescanned = 0,
    invalid_support,
    invalid_candidates,
    context_mismatch,
    invalid_pair,
    missing_output,
    insufficient_output,
};

struct exact_support_rescan_result_v1 {
    exact_support_rescan_code_v1 code = exact_support_rescan_code_v1::rescanned;
    exact_support_pair_score_view_v1 view{};
    std::uint64_t index = 0;
    [[nodiscard]] constexpr bool rescanned() const noexcept {
        return code == exact_support_rescan_code_v1::rescanned;
    }
};

[[nodiscard]] constexpr bool valid_candidate_pair_view_v1(
    support_candidate_pair_view_v1 pairs) noexcept {
    if (pairs.pairs == nullptr || pairs.pair_count == 0
        || pairs.destination_count == 0 || pairs.maximum_fan_out == 0
        || pairs.reserved != 0
        || !atom::validate_atom_persistent_identity_v1(
                pairs.relation_identity).valid()
        || pairs.relation_generation == 0) {
        return false;
    }
    for (std::uint64_t index = 0; index < pairs.pair_count; ++index) {
        const auto &pair = pairs.pairs[index];
        if (pair.first_destination_index >= pair.second_destination_index
            || pair.second_destination_index >= pairs.destination_count
            || pair.matching_band_count == 0 || pair.reserved != 0
            || (index != 0
                && !support_candidate_pair_less_v1(
                    pairs.pairs[index - 1], pair))) {
            return false;
        }
    }
    return true;
}

// Exact two-pointer rescan. Jaccard is shared/union; directional containment
// is shared/first_count and shared/second_count. Integer components are kept
// exact so no numerical tolerance or rounding policy is hidden here.
[[nodiscard]] constexpr exact_support_rescan_result_v1
rescan_exact_support_pairs_v1(
    exact_destination_support_view_v1 support,
    support_candidate_pair_view_v1 candidates,
    exact_support_pair_score_v1 *output,
    std::uint64_t output_capacity) noexcept {
    if (!validate_exact_destination_support_view_v1(support)) {
        return {exact_support_rescan_code_v1::invalid_support};
    }
    if (!valid_candidate_pair_view_v1(candidates)) {
        return {exact_support_rescan_code_v1::invalid_candidates};
    }
    if (candidates.destination_count != support.destination_count
        || candidates.relation_identity != support.relation_identity
        || candidates.relation_generation != support.relation_generation) {
        return {exact_support_rescan_code_v1::context_mismatch};
    }
    if (output == nullptr) {
        return {exact_support_rescan_code_v1::missing_output};
    }
    if (output_capacity < candidates.pair_count) {
        return {exact_support_rescan_code_v1::insufficient_output, {},
                candidates.pair_count};
    }
    for (std::uint64_t pair_index = 0;
         pair_index < candidates.pair_count;
         ++pair_index) {
        const auto &pair = candidates.pairs[pair_index];
        auto first = support.destination_offsets[pair.first_destination_index];
        const auto first_end =
            support.destination_offsets[pair.first_destination_index + 1];
        auto second =
            support.destination_offsets[pair.second_destination_index];
        const auto second_end =
            support.destination_offsets[pair.second_destination_index + 1];
        std::uint64_t shared = 0;
        while (first < first_end && second < second_end) {
            if (support.global_source_ids[first]
                < support.global_source_ids[second]) {
                ++first;
            } else if (support.global_source_ids[second]
                       < support.global_source_ids[first]) {
                ++second;
            } else {
                ++shared;
                ++first;
                ++second;
            }
        }
        const auto first_count = first_end
            - support.destination_offsets[pair.first_destination_index];
        const auto second_count = second_end
            - support.destination_offsets[pair.second_destination_index];
        output[pair_index] = {
            pair.first_destination_index, pair.second_destination_index,
            shared, first_count + second_count - shared,
            first_count, second_count};
    }
    return {exact_support_rescan_code_v1::rescanned,
            {output, candidates.pair_count, support.relation_identity,
             support.relation_generation},
            candidates.pair_count};
}

[[nodiscard]] constexpr bool authorizes_execution(
    exact_support_pair_score_view_v1) noexcept {
    return false;
}

} // namespace cellshard::compiler::discovery::support_signature
