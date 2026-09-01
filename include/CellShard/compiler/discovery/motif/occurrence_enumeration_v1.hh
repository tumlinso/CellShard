#pragma once

#include <CellShard/compiler/discovery/motif/typed_vocabulary_v1.hh>

#include <cstdint>
#include <limits>

namespace cellshard::compiler::discovery::motif {

struct typed_graph_node_v1 {
    std::uint64_t global_node_id = 0;
    atom::atom_persistent_identity_v1 node_type{};
};

struct typed_graph_edge_v1 {
    std::uint32_t source_node = 0;
    std::uint32_t destination_node = 0;
    atom::atom_persistent_identity_v1 relation_type{};
    motif_edge_direction_v1 direction = motif_edge_direction_v1::directed;
    std::uint8_t reserved[3]{};
};

struct typed_graph_view_v1 {
    const typed_graph_node_v1 *nodes = nullptr;
    const typed_graph_edge_v1 *edges = nullptr;
    std::uint32_t node_count = 0;
    std::uint32_t edge_count = 0;
    atom::atom_persistent_identity_v1 graph_family_identity{};
    std::uint64_t graph_generation = 0;
};

struct motif_occurrence_output_v1 {
    std::uint64_t *global_node_ids = nullptr;
    std::uint64_t node_id_capacity = 0;
    std::uint64_t occurrence_count = 0;
};

struct motif_occurrence_workspace_v1 {
    std::uint32_t *motif_to_graph = nullptr;
    std::uint64_t mapping_capacity = 0;
    std::uint8_t *graph_node_used = nullptr;
    std::uint64_t used_capacity = 0;
};

struct motif_occurrence_limits_v1 {
    std::uint64_t maximum_assignments_examined = 0;
    std::uint64_t maximum_occurrences = 0;
};

enum class motif_occurrence_code_v1 : std::uint32_t {
    enumerated = 0,
    truncated_assignment_limit,
    truncated_occurrence_limit,
    invalid_motif,
    invalid_graph,
    graph_identity_mismatch,
    graph_generation_mismatch,
    invalid_limits,
    missing_workspace,
    insufficient_workspace,
    missing_output,
    output_size_overflow,
    insufficient_output,
};

struct motif_occurrence_result_v1 {
    motif_occurrence_code_v1 code = motif_occurrence_code_v1::enumerated;
    std::uint64_t assignments_examined = 0;
    std::uint64_t occurrences = 0;

    [[nodiscard]] constexpr bool complete() const noexcept {
        return code == motif_occurrence_code_v1::enumerated;
    }
};

namespace detail {

[[nodiscard]] inline bool graph_has_motif_edge_v1(
    typed_graph_view_v1 graph,
    const typed_motif_edge_v1 &required,
    std::uint32_t graph_source,
    std::uint32_t graph_destination) noexcept {
    for (std::uint32_t index = 0; index < graph.edge_count; ++index) {
        const auto &edge = graph.edges[index];
        if (edge.relation_type != required.relation_type
            || edge.direction != required.direction) {
            continue;
        }
        if (edge.source_node == graph_source
            && edge.destination_node == graph_destination) {
            return true;
        }
        if (required.direction == motif_edge_direction_v1::undirected
            && edge.source_node == graph_destination
            && edge.destination_node == graph_source) {
            return true;
        }
    }
    return false;
}

[[nodiscard]] inline bool partial_edges_match_v1(
    typed_motif_vocabulary_view_v1 motif,
    typed_graph_view_v1 graph,
    const std::uint32_t *mapping,
    std::uint32_t assigned_count) noexcept {
    for (std::uint32_t index = 0; index < motif.edge_count; ++index) {
        const auto &edge = motif.edges[index];
        if (edge.source_node >= assigned_count
            || edge.destination_node >= assigned_count) {
            continue;
        }
        if (!graph_has_motif_edge_v1(
                graph, edge, mapping[edge.source_node],
                mapping[edge.destination_node])) {
            return false;
        }
    }
    return true;
}

struct enumeration_state_v1 {
    typed_motif_vocabulary_view_v1 motif{};
    typed_graph_view_v1 graph{};
    motif_occurrence_workspace_v1 workspace{};
    motif_occurrence_limits_v1 limits{};
    motif_occurrence_output_v1 *output = nullptr;
    motif_occurrence_result_v1 result{};
};

inline bool enumerate_depth_v1(
    enumeration_state_v1 *state,
    std::uint32_t depth) noexcept {
    if (depth == state->motif.node_count) {
        ++state->result.assignments_examined;
        if (state->result.assignments_examined
            > state->limits.maximum_assignments_examined) {
            state->result.code =
                motif_occurrence_code_v1::truncated_assignment_limit;
            return false;
        }
        if (!partial_edges_match_v1(
                state->motif, state->graph,
                state->workspace.motif_to_graph, depth)) {
            return true;
        }
        if (state->result.occurrences
            == state->limits.maximum_occurrences) {
            state->result.code =
                motif_occurrence_code_v1::truncated_occurrence_limit;
            return false;
        }
        const auto offset = state->result.occurrences
            * state->motif.node_count;
        for (std::uint32_t index = 0;
             index < state->motif.node_count;
             ++index) {
            state->output->global_node_ids[offset + index] =
                state->graph.nodes[
                    state->workspace.motif_to_graph[index]].global_node_id;
        }
        ++state->result.occurrences;
        return true;
    }
    for (std::uint32_t graph_node = 0;
         graph_node < state->graph.node_count;
         ++graph_node) {
        if (state->workspace.graph_node_used[graph_node] != 0
            || state->graph.nodes[graph_node].node_type
                   != state->motif.nodes[depth].node_type) {
            continue;
        }
        state->workspace.motif_to_graph[depth] = graph_node;
        state->workspace.graph_node_used[graph_node] = 1;
        if (partial_edges_match_v1(
                state->motif, state->graph,
                state->workspace.motif_to_graph, depth + 1)
            && !enumerate_depth_v1(state, depth + 1)) {
            state->workspace.graph_node_used[graph_node] = 0;
            return false;
        }
        state->workspace.graph_node_used[graph_node] = 0;
    }
    return true;
}

} // namespace detail

[[nodiscard]] inline motif_occurrence_result_v1 enumerate_motif_occurrences_v1(
    typed_motif_vocabulary_view_v1 motif,
    typed_graph_view_v1 graph,
    motif_occurrence_limits_v1 limits,
    motif_occurrence_workspace_v1 workspace,
    motif_occurrence_output_v1 *output) noexcept {
    if (!validate_typed_motif_vocabulary_v1(motif).valid()) {
        return {motif_occurrence_code_v1::invalid_motif};
    }
    if (graph.node_count == 0 || graph.nodes == nullptr
        || (graph.edge_count != 0 && graph.edges == nullptr)
        || !atom::validate_atom_persistent_identity_v1(
                graph.graph_family_identity).valid()
        || graph.graph_generation == 0) {
        return {motif_occurrence_code_v1::invalid_graph};
    }
    if (graph.graph_family_identity != motif.graph_family_identity) {
        return {motif_occurrence_code_v1::graph_identity_mismatch};
    }
    if (graph.graph_generation != motif.graph_generation) {
        return {motif_occurrence_code_v1::graph_generation_mismatch};
    }
    for (std::uint32_t index = 0; index < graph.node_count; ++index) {
        if (graph.nodes[index].global_node_id == 0
            || !atom::validate_atom_persistent_identity_v1(
                    graph.nodes[index].node_type).valid()) {
            return {motif_occurrence_code_v1::invalid_graph};
        }
    }
    for (std::uint32_t index = 0; index < graph.edge_count; ++index) {
        const auto &edge = graph.edges[index];
        if (edge.source_node >= graph.node_count
            || edge.destination_node >= graph.node_count
            || edge.source_node == edge.destination_node
            || !atom::validate_atom_persistent_identity_v1(
                    edge.relation_type).valid()
            || !valid_motif_edge_direction_v1(edge.direction)
            || edge.reserved[0] != 0 || edge.reserved[1] != 0
            || edge.reserved[2] != 0) {
            return {motif_occurrence_code_v1::invalid_graph};
        }
    }
    if (limits.maximum_assignments_examined == 0
        || limits.maximum_occurrences == 0) {
        return {motif_occurrence_code_v1::invalid_limits};
    }
    if (workspace.motif_to_graph == nullptr
        || workspace.graph_node_used == nullptr) {
        return {motif_occurrence_code_v1::missing_workspace};
    }
    if (workspace.mapping_capacity < motif.node_count
        || workspace.used_capacity < graph.node_count) {
        return {motif_occurrence_code_v1::insufficient_workspace};
    }
    if (output == nullptr || output->global_node_ids == nullptr) {
        return {motif_occurrence_code_v1::missing_output};
    }
    output->occurrence_count = 0;
    if (limits.maximum_occurrences
        > std::numeric_limits<std::uint64_t>::max() / motif.node_count) {
        return {motif_occurrence_code_v1::output_size_overflow};
    }
    if (output->node_id_capacity
        < limits.maximum_occurrences * motif.node_count) {
        return {motif_occurrence_code_v1::insufficient_output};
    }
    for (std::uint32_t index = 0; index < graph.node_count; ++index) {
        workspace.graph_node_used[index] = 0;
    }
    detail::enumeration_state_v1 state{
        motif, graph, workspace, limits, output, {}};
    detail::enumerate_depth_v1(&state, 0);
    output->occurrence_count = state.result.occurrences;
    return state.result;
}

[[nodiscard]] constexpr bool authorizes_execution(
    const motif_occurrence_output_v1 &) noexcept {
    return false;
}

} // namespace cellshard::compiler::discovery::motif
