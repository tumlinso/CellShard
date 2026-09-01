#pragma once

#include <CellShard/compiler/discovery/motif/canonical_encoding_v1.hh>

#include <cstdint>
#include <limits>

namespace cellshard::compiler::discovery::motif {

struct observed_typed_fragment_v1 {
    const std::uint64_t *canonical_words = nullptr;
    std::uint64_t canonical_word_count = 0;
    atom::atom_persistent_identity_v1 graph_identity{};
    std::uint64_t graph_generation = 0;
    std::uint64_t occurrence_count = 0;
    atom::atom_persistent_identity_v1 graph_family_identity{};
    evidence::evidence_identity_v1 stratum_identity{};
    std::uint64_t stratum_selection_generation = 0;
};

struct frequent_typed_fragment_v1 {
    const std::uint64_t *canonical_words = nullptr;
    std::uint64_t canonical_word_count = 0;
    atom::atom_persistent_identity_v1 proposal_identity{};
    std::uint64_t supporting_graph_count = 0;
    std::uint64_t occurrence_count = 0;
};

struct frequent_fragment_limits_v1 {
    std::uint64_t maximum_observations = 0;
    std::uint64_t maximum_words_per_fragment = 0;
    std::uint64_t minimum_supporting_graphs = 0;
};

struct frequent_fragment_output_v1 {
    frequent_typed_fragment_v1 *fragments = nullptr;
    std::uint64_t fragment_capacity = 0;
    std::uint64_t fragment_count = 0;
};

enum class frequent_fragment_code_v1 : std::uint32_t {
    mined = 0,
    truncated_output,
    empty_observations,
    missing_observations,
    invalid_limits,
    observation_bound_exceeded,
    missing_encoding,
    invalid_encoding,
    word_bound_exceeded,
    invalid_graph_identity,
    missing_graph_generation,
    empty_occurrences,
    invalid_graph_family,
    invalid_stratum,
    missing_stratum_generation,
    context_mismatch,
    count_overflow,
    missing_proposal_identities,
    insufficient_proposal_identities,
    invalid_proposal_identity,
    unordered_or_duplicate_proposal_identity,
    missing_output,
};

struct frequent_fragment_result_v1 {
    frequent_fragment_code_v1 code = frequent_fragment_code_v1::mined;
    std::uint64_t index = 0;
    std::uint64_t qualifying_fragment_count = 0;
    [[nodiscard]] constexpr bool complete() const noexcept {
        return code == frequent_fragment_code_v1::mined;
    }
};

namespace detail {

[[nodiscard]] inline bool same_encoding_v1(
    const observed_typed_fragment_v1 &lhs,
    const observed_typed_fragment_v1 &rhs) noexcept {
    if (lhs.canonical_word_count != rhs.canonical_word_count) return false;
    for (std::uint64_t index = 0; index < lhs.canonical_word_count; ++index) {
        if (lhs.canonical_words[index] != rhs.canonical_words[index]) {
            return false;
        }
    }
    return true;
}

} // namespace detail

// Experimental O(N^3 * W) reference miner. All bounds are explicit and all
// matching uses complete canonical words, never a digest. It emits proposals,
// not certificates or executable atoms.
[[nodiscard]] inline frequent_fragment_result_v1 mine_frequent_fragments_v1(
    const observed_typed_fragment_v1 *observations,
    std::uint64_t observation_count,
    frequent_fragment_limits_v1 limits,
    const atom::atom_persistent_identity_v1 *proposal_identities,
    std::uint64_t proposal_identity_count,
    frequent_fragment_output_v1 *output) noexcept {
    if (observation_count == 0) {
        return {frequent_fragment_code_v1::empty_observations};
    }
    if (observations == nullptr) {
        return {frequent_fragment_code_v1::missing_observations};
    }
    if (limits.maximum_observations == 0
        || limits.maximum_words_per_fragment < 3
        || limits.minimum_supporting_graphs == 0) {
        return {frequent_fragment_code_v1::invalid_limits};
    }
    if (observation_count > limits.maximum_observations) {
        return {frequent_fragment_code_v1::observation_bound_exceeded};
    }
    for (std::uint64_t index = 0; index < observation_count; ++index) {
        const auto &observation = observations[index];
        if (observation.canonical_words == nullptr) {
            return {frequent_fragment_code_v1::missing_encoding, index};
        }
        if (observation.canonical_word_count < 3
            || observation.canonical_words[0]
                   != canonical_motif_encoding_version_v1) {
            return {frequent_fragment_code_v1::invalid_encoding, index};
        }
        if (observation.canonical_word_count
            > limits.maximum_words_per_fragment) {
            return {frequent_fragment_code_v1::word_bound_exceeded, index};
        }
        if (!atom::validate_atom_persistent_identity_v1(
                 observation.graph_identity).valid()) {
            return {frequent_fragment_code_v1::invalid_graph_identity, index};
        }
        if (observation.graph_generation == 0) {
            return {frequent_fragment_code_v1::missing_graph_generation, index};
        }
        if (observation.occurrence_count == 0) {
            return {frequent_fragment_code_v1::empty_occurrences, index};
        }
        if (!atom::validate_atom_persistent_identity_v1(
                 observation.graph_family_identity).valid()) {
            return {frequent_fragment_code_v1::invalid_graph_family, index};
        }
        if (!evidence::valid_evidence_identity_v1(
                observation.stratum_identity)) {
            return {frequent_fragment_code_v1::invalid_stratum, index};
        }
        if (observation.stratum_selection_generation == 0) {
            return {frequent_fragment_code_v1::missing_stratum_generation,
                    index};
        }
        if (index != 0
            && (observation.graph_family_identity
                    != observations[0].graph_family_identity
                || !(observation.stratum_identity
                    == observations[0].stratum_identity)
                || observation.stratum_selection_generation
                    != observations[0].stratum_selection_generation)) {
            return {frequent_fragment_code_v1::context_mismatch, index};
        }
    }
    if (proposal_identities == nullptr) {
        return {frequent_fragment_code_v1::missing_proposal_identities};
    }
    if (output == nullptr || output->fragments == nullptr) {
        return {frequent_fragment_code_v1::missing_output};
    }
    output->fragment_count = 0;
    std::uint64_t qualifying = 0;
    for (std::uint64_t index = 0; index < observation_count; ++index) {
        bool previously_seen = false;
        for (std::uint64_t earlier = 0; earlier < index; ++earlier) {
            if (detail::same_encoding_v1(observations[earlier],
                                         observations[index])) {
                previously_seen = true;
                break;
            }
        }
        if (previously_seen) continue;
        std::uint64_t support = 0;
        std::uint64_t occurrences = 0;
        for (std::uint64_t candidate = index;
             candidate < observation_count;
             ++candidate) {
            if (!detail::same_encoding_v1(observations[index],
                                          observations[candidate])) {
                continue;
            }
            if (occurrences
                > std::numeric_limits<std::uint64_t>::max()
                      - observations[candidate].occurrence_count) {
                return {frequent_fragment_code_v1::count_overflow, candidate,
                        qualifying};
            }
            occurrences += observations[candidate].occurrence_count;
            bool graph_seen = false;
            for (std::uint64_t earlier = index;
                 earlier < candidate;
                 ++earlier) {
                if (detail::same_encoding_v1(observations[index],
                                             observations[earlier])
                    && observations[earlier].graph_identity
                           == observations[candidate].graph_identity
                    && observations[earlier].graph_generation
                           == observations[candidate].graph_generation) {
                    graph_seen = true;
                    break;
                }
            }
            if (!graph_seen) ++support;
        }
        if (support < limits.minimum_supporting_graphs) continue;
        if (qualifying >= proposal_identity_count) {
            return {frequent_fragment_code_v1::
                        insufficient_proposal_identities,
                    index, qualifying + 1};
        }
        if (!atom::validate_atom_persistent_identity_v1(
                 proposal_identities[qualifying]).valid()) {
            return {frequent_fragment_code_v1::invalid_proposal_identity,
                    qualifying, qualifying};
        }
        if (qualifying != 0
            && !atom::atom_persistent_identity_less_v1(
                proposal_identities[qualifying - 1],
                proposal_identities[qualifying])) {
            return {frequent_fragment_code_v1::
                        unordered_or_duplicate_proposal_identity,
                    qualifying, qualifying};
        }
        if (output->fragment_count < output->fragment_capacity) {
            output->fragments[output->fragment_count++] = {
                observations[index].canonical_words,
                observations[index].canonical_word_count,
                proposal_identities[qualifying], support, occurrences};
        }
        ++qualifying;
    }
    return {qualifying > output->fragment_capacity
                ? frequent_fragment_code_v1::truncated_output
                : frequent_fragment_code_v1::mined,
            observation_count, qualifying};
}

[[nodiscard]] constexpr bool authorizes_execution(
    const frequent_fragment_output_v1 &) noexcept {
    return false;
}

} // namespace cellshard::compiler::discovery::motif
