#pragma once

#include <CellShard/compiler/discovery/motif/typed_vocabulary_v1.hh>

#include <algorithm>
#include <cstdint>
#include <limits>

namespace cellshard::compiler::discovery::motif {

inline constexpr std::uint32_t canonical_motif_encoding_version_v1 = 1;

struct canonical_motif_encoding_workspace_v1 {
    std::uint32_t *permutation = nullptr;
    std::uint64_t permutation_capacity = 0;
    std::uint64_t *candidate_words = nullptr;
    std::uint64_t candidate_word_capacity = 0;
};

struct canonical_motif_encoding_output_v1 {
    std::uint64_t *words = nullptr;
    std::uint64_t word_capacity = 0;
    std::uint64_t word_count = 0;
};

enum class canonical_motif_encoding_code_v1 : std::uint32_t {
    encoded = 0,
    invalid_motif,
    missing_permutation,
    insufficient_permutation,
    missing_candidate_words,
    missing_output_words,
    word_count_overflow,
    insufficient_candidate_words,
    insufficient_output_words,
};

struct canonical_motif_encoding_result_v1 {
    canonical_motif_encoding_code_v1 code =
        canonical_motif_encoding_code_v1::encoded;
    std::uint64_t required_words = 0;
    std::uint64_t permutations_examined = 0;

    [[nodiscard]] constexpr bool encoded() const noexcept {
        return code == canonical_motif_encoding_code_v1::encoded;
    }
};

namespace detail {

[[nodiscard]] inline std::uint32_t canonical_node_for_old_v1(
    const std::uint32_t *permutation,
    std::uint32_t count,
    std::uint32_t old_node) noexcept {
    for (std::uint32_t canonical = 0; canonical < count; ++canonical) {
        if (permutation[canonical] == old_node) {
            return canonical;
        }
    }
    return count;
}

struct canonical_edge_words_v1 {
    std::uint64_t source = 0;
    std::uint64_t destination = 0;
    std::uint64_t direction = 0;
    std::uint64_t relation_namespace = 0;
    std::uint64_t relation_local = 0;
    std::uint64_t role = 0;
};

[[nodiscard]] inline canonical_edge_words_v1 remap_edge_v1(
    const typed_motif_edge_v1 &edge,
    const std::uint32_t *permutation,
    std::uint32_t node_count) noexcept {
    auto source = canonical_node_for_old_v1(
        permutation, node_count, edge.source_node);
    auto destination = canonical_node_for_old_v1(
        permutation, node_count, edge.destination_node);
    if (edge.direction == motif_edge_direction_v1::undirected
        && destination < source) {
        std::swap(source, destination);
    }
    return {source, destination, static_cast<std::uint64_t>(edge.direction),
            edge.relation_type.producer_namespace,
            edge.relation_type.local_identity, edge.role};
}

[[nodiscard]] constexpr bool edge_words_less_v1(
    const canonical_edge_words_v1 &lhs,
    const canonical_edge_words_v1 &rhs) noexcept {
    if (lhs.source != rhs.source) return lhs.source < rhs.source;
    if (lhs.destination != rhs.destination)
        return lhs.destination < rhs.destination;
    if (lhs.direction != rhs.direction) return lhs.direction < rhs.direction;
    if (lhs.relation_namespace != rhs.relation_namespace)
        return lhs.relation_namespace < rhs.relation_namespace;
    if (lhs.relation_local != rhs.relation_local)
        return lhs.relation_local < rhs.relation_local;
    return lhs.role < rhs.role;
}

[[nodiscard]] inline canonical_edge_words_v1 edge_rank_v1(
    typed_motif_vocabulary_view_v1 view,
    const std::uint32_t *permutation,
    std::uint32_t rank) noexcept {
    canonical_edge_words_v1 selected{};
    bool have_selected = false;
    for (std::uint32_t step = 0; step <= rank; ++step) {
        canonical_edge_words_v1 next{};
        bool have_next = false;
        for (std::uint32_t edge_index = 0;
             edge_index < view.edge_count;
             ++edge_index) {
            const auto value = remap_edge_v1(
                view.edges[edge_index], permutation, view.node_count);
            if (have_selected && !edge_words_less_v1(selected, value)) {
                continue;
            }
            if (!have_next || edge_words_less_v1(value, next)) {
                next = value;
                have_next = true;
            }
        }
        selected = next;
        have_selected = have_next;
    }
    return selected;
}

inline void encode_permutation_v1(
    typed_motif_vocabulary_view_v1 view,
    const std::uint32_t *permutation,
    std::uint64_t *words) noexcept {
    std::uint64_t cursor = 0;
    words[cursor++] = canonical_motif_encoding_version_v1;
    words[cursor++] = view.node_count;
    words[cursor++] = view.edge_count;
    for (std::uint32_t canonical = 0;
         canonical < view.node_count;
         ++canonical) {
        const auto &node = view.nodes[permutation[canonical]];
        words[cursor++] = node.node_type.producer_namespace;
        words[cursor++] = node.node_type.local_identity;
        words[cursor++] = node.role;
    }
    for (std::uint32_t rank = 0; rank < view.edge_count; ++rank) {
        const auto edge = edge_rank_v1(view, permutation, rank);
        words[cursor++] = edge.source;
        words[cursor++] = edge.destination;
        words[cursor++] = edge.direction;
        words[cursor++] = edge.relation_namespace;
        words[cursor++] = edge.relation_local;
        words[cursor++] = edge.role;
    }
}

} // namespace detail

// Exact canonicalization enumerates all node permutations. Its n! time is
// intentional and visible; callers bound motif size before invoking it.
[[nodiscard]] inline canonical_motif_encoding_result_v1
encode_canonical_motif_v1(
    typed_motif_vocabulary_view_v1 view,
    canonical_motif_encoding_workspace_v1 workspace,
    canonical_motif_encoding_output_v1 *output) noexcept {
    if (!validate_typed_motif_vocabulary_v1(view).valid()) {
        return {canonical_motif_encoding_code_v1::invalid_motif};
    }
    if (workspace.permutation == nullptr) {
        return {canonical_motif_encoding_code_v1::missing_permutation};
    }
    if (workspace.permutation_capacity < view.node_count) {
        return {canonical_motif_encoding_code_v1::insufficient_permutation};
    }
    if (view.edge_count
        > (std::numeric_limits<std::uint64_t>::max() - 3
           - UINT64_C(3) * view.node_count) / 6) {
        return {canonical_motif_encoding_code_v1::word_count_overflow};
    }
    const auto required = UINT64_C(3) + UINT64_C(3) * view.node_count
        + UINT64_C(6) * view.edge_count;
    if (workspace.candidate_words == nullptr) {
        return {canonical_motif_encoding_code_v1::missing_candidate_words,
                required};
    }
    if (workspace.candidate_word_capacity < required) {
        return {canonical_motif_encoding_code_v1::insufficient_candidate_words,
                required};
    }
    if (output == nullptr || output->words == nullptr) {
        return {canonical_motif_encoding_code_v1::missing_output_words,
                required};
    }
    output->word_count = 0;
    if (output->word_capacity < required) {
        return {canonical_motif_encoding_code_v1::insufficient_output_words,
                required};
    }
    for (std::uint32_t index = 0; index < view.node_count; ++index) {
        workspace.permutation[index] = index;
    }
    std::uint64_t examined = 0;
    do {
        detail::encode_permutation_v1(
            view, workspace.permutation, workspace.candidate_words);
        ++examined;
        if (examined == 1
            || std::lexicographical_compare(
                workspace.candidate_words,
                workspace.candidate_words + required,
                output->words, output->words + required)) {
            std::copy_n(workspace.candidate_words, required, output->words);
        }
    } while (std::next_permutation(
        workspace.permutation,
        workspace.permutation + view.node_count));
    output->word_count = required;
    return {canonical_motif_encoding_code_v1::encoded, required, examined};
}

} // namespace cellshard::compiler::discovery::motif
