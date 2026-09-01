#pragma once

#include <CellShard/compiler/composition/grammar_symbol_v1.hh>

#include <cstdint>
#include <limits>
#include <type_traits>

namespace cellshard::compiler::composition {

inline constexpr std::uint32_t max_derivation_nodes_v1 = 256;
inline constexpr std::uint32_t max_derivation_edges_v1 = 1024;

enum class composition_transform_kind_v1 : std::uint8_t {
    disjoint_union = 1,
    ordered_concatenation,
    support_union,
    coverage_intersection,
    coverage_difference,
    source_aligned_merge,
    destination_aligned_merge,
    identity_spine_join,
    multimodal_join,
    segment_alignment,
    halo_extension,
    overlay_application,
    value_plane_substitution,
    physical_view_addition,
    transpose_overlay,
    projection_packing,
    persistent_order_link,
    parameter_binding,
    partial_result_combination,
    relation_bundle,
    prefix_composition,
};

struct derivation_node_v1 {
    composition_production_id production{};
    composition_transform_kind_v1 transform =
        composition_transform_kind_v1::disjoint_union;
    std::uint8_t reserved[7]{};
    atom_interface_id input_interface{};
    atom_interface_id output_interface{};
};

struct derivation_edge_v1 {
    std::uint32_t producer_node = 0;
    std::uint32_t consumer_node = 0;
};

struct derivation_dag_view_v1 {
    composition_lineage_id lineage{};
    const derivation_node_v1 *nodes = nullptr;
    const derivation_edge_v1 *edges = nullptr;
    std::uint32_t node_count = 0;
    std::uint32_t edge_count = 0;
};

struct derivation_dag_workspace_v1 {
    std::uint32_t *indegrees = nullptr;
    std::uint32_t *edge_offsets = nullptr;
    std::uint32_t *queue = nullptr;
    std::uint32_t *topological_order = nullptr;
    std::uint32_t node_capacity = 0;
    std::uint32_t edge_offset_capacity = 0;
};

struct compiled_derivation_dag_v1 {
    composition_lineage_id lineage{};
    const derivation_node_v1 *nodes = nullptr;
    const derivation_edge_v1 *edges = nullptr;
    const std::uint32_t *topological_order = nullptr;
    std::uint32_t node_count = 0;
    std::uint32_t edge_count = 0;
};

enum class derivation_dag_code_v1 : std::uint32_t {
    compiled = 0,
    invalid_lineage,
    invalid_node_count,
    invalid_edge_count,
    missing_nodes,
    missing_edges,
    invalid_node,
    unordered_production_identity,
    edge_node_out_of_range,
    self_edge,
    unordered_edge,
    duplicate_edge,
    interface_mismatch,
    indegree_overflow,
    missing_workspace,
    insufficient_node_capacity,
    insufficient_edge_offset_capacity,
    cycle_detected,
    missing_output,
};

struct derivation_dag_result_v1 {
    derivation_dag_code_v1 code = derivation_dag_code_v1::compiled;
    std::uint32_t subject_index = 0;
    [[nodiscard]] constexpr bool compiled() const noexcept {
        return code == derivation_dag_code_v1::compiled;
    }
};

[[nodiscard]] constexpr bool valid_transform_kind_v1(
    composition_transform_kind_v1 kind) noexcept {
    return kind >= composition_transform_kind_v1::disjoint_union
        && kind <= composition_transform_kind_v1::prefix_composition;
}

[[nodiscard]] inline derivation_dag_result_v1 compile_derivation_dag_v1(
    const derivation_dag_view_v1 &dag,
    derivation_dag_workspace_v1 workspace,
    compiled_derivation_dag_v1 *output) noexcept {
    if (!dag.lineage.valid()) {
        return {derivation_dag_code_v1::invalid_lineage};
    }
    if (dag.node_count == 0 || dag.node_count > max_derivation_nodes_v1) {
        return {derivation_dag_code_v1::invalid_node_count};
    }
    if (dag.edge_count > max_derivation_edges_v1) {
        return {derivation_dag_code_v1::invalid_edge_count};
    }
    if (dag.nodes == nullptr) return {derivation_dag_code_v1::missing_nodes};
    if (dag.edge_count != 0 && dag.edges == nullptr) {
        return {derivation_dag_code_v1::missing_edges};
    }
    if (workspace.indegrees == nullptr || workspace.edge_offsets == nullptr
        || workspace.queue == nullptr
        || workspace.topological_order == nullptr) {
        return {derivation_dag_code_v1::missing_workspace};
    }
    if (workspace.node_capacity < dag.node_count) {
        return {derivation_dag_code_v1::insufficient_node_capacity};
    }
    if (workspace.edge_offset_capacity < dag.node_count + 1u) {
        return {derivation_dag_code_v1::insufficient_edge_offset_capacity};
    }
    if (output == nullptr) return {derivation_dag_code_v1::missing_output};
    *output = {};
    for (std::uint32_t node = 0; node < dag.node_count; ++node) {
        const auto &record = dag.nodes[node];
        if (!record.production.valid()
            || !valid_transform_kind_v1(record.transform)
            || !record.input_interface.valid()
            || !record.output_interface.valid()) {
            return {derivation_dag_code_v1::invalid_node, node};
        }
        for (const auto byte : record.reserved) {
            if (byte != 0) {
                return {derivation_dag_code_v1::invalid_node, node};
            }
        }
        if (node != 0
            && dag.nodes[node - 1].production >= record.production) {
            return {derivation_dag_code_v1::unordered_production_identity,
                    node};
        }
        workspace.indegrees[node] = 0;
    }
    for (std::uint32_t edge = 0; edge < dag.edge_count; ++edge) {
        const auto &record = dag.edges[edge];
        if (record.producer_node >= dag.node_count
            || record.consumer_node >= dag.node_count) {
            return {derivation_dag_code_v1::edge_node_out_of_range, edge};
        }
        if (record.producer_node == record.consumer_node) {
            return {derivation_dag_code_v1::self_edge, edge};
        }
        if (edge != 0) {
            const auto &previous = dag.edges[edge - 1];
            if (previous.producer_node > record.producer_node
                || (previous.producer_node == record.producer_node
                    && previous.consumer_node >= record.consumer_node)) {
                return {previous.producer_node == record.producer_node
                            && previous.consumer_node == record.consumer_node
                        ? derivation_dag_code_v1::duplicate_edge
                        : derivation_dag_code_v1::unordered_edge,
                        edge};
            }
        }
        if (dag.nodes[record.producer_node].output_interface
            != dag.nodes[record.consumer_node].input_interface) {
            return {derivation_dag_code_v1::interface_mismatch, edge};
        }
        if (workspace.indegrees[record.consumer_node]
            == std::numeric_limits<std::uint32_t>::max()) {
            return {derivation_dag_code_v1::indegree_overflow, edge};
        }
        ++workspace.indegrees[record.consumer_node];
    }
    std::uint32_t edge_cursor = 0;
    for (std::uint32_t node = 0; node < dag.node_count; ++node) {
        workspace.edge_offsets[node] = edge_cursor;
        while (edge_cursor < dag.edge_count
               && dag.edges[edge_cursor].producer_node == node) {
            ++edge_cursor;
        }
    }
    workspace.edge_offsets[dag.node_count] = edge_cursor;

    std::uint32_t queue_read = 0;
    std::uint32_t queue_write = 0;
    for (std::uint32_t node = 0; node < dag.node_count; ++node) {
        if (workspace.indegrees[node] == 0) {
            workspace.queue[queue_write++] = node;
        }
    }
    std::uint32_t produced = 0;
    while (queue_read < queue_write) {
        const auto node = workspace.queue[queue_read++];
        workspace.topological_order[produced++] = node;
        for (std::uint32_t edge = workspace.edge_offsets[node];
             edge < workspace.edge_offsets[node + 1u];
             ++edge) {
            const auto consumer = dag.edges[edge].consumer_node;
            --workspace.indegrees[consumer];
            if (workspace.indegrees[consumer] == 0) {
                workspace.queue[queue_write++] = consumer;
            }
        }
    }
    if (produced != dag.node_count) {
        return {derivation_dag_code_v1::cycle_detected, produced};
    }
    *output = {dag.lineage, dag.nodes, dag.edges,
               workspace.topological_order, dag.node_count, dag.edge_count};
    return {derivation_dag_code_v1::compiled, produced};
}

static_assert(std::is_trivially_copyable<derivation_node_v1>::value);
static_assert(std::is_trivially_copyable<derivation_edge_v1>::value);
static_assert(std::is_trivially_copyable<compiled_derivation_dag_v1>::value);

} // namespace cellshard::compiler::composition
