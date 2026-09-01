#pragma once

#include <CellShard/compiler/certification/multimodal_mapping_v1.hh>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace cellshard::compiler::certification {

inline constexpr std::uint32_t trajectory_lineage_contract_version_v1 = 1;

struct trajectory_lineage_edge_v1 {
    std::uint64_t parent_global_entity_id = 0;
    std::uint64_t child_global_entity_id = 0;
};

struct trajectory_lineage_mapping_view_v1 {
    const trajectory_lineage_edge_v1 *edges = nullptr;
    std::uint64_t edge_count = 0;
    atom::atom_persistent_identity_v1 mapping_identity{};
    atom::atom_persistent_identity_v1 node_domain_identity{};
    std::uint64_t mapping_generation = 0;
};

struct trajectory_lineage_workspace_v1 {
    std::uint64_t *parent_offsets = nullptr;
    std::uint64_t parent_offset_capacity = 0;
    std::uint64_t *indegrees = nullptr;
    std::uint64_t indegree_capacity = 0;
    std::uint64_t *queue = nullptr;
    std::uint64_t queue_capacity = 0;
};

enum class trajectory_lineage_validation_code_v1 : std::uint32_t {
    valid = 0,
    invalid_node_spine,
    empty_mapping,
    missing_edges,
    invalid_mapping_identity,
    node_domain_mismatch,
    missing_mapping_generation,
    missing_workspace,
    insufficient_workspace,
    zero_node_identity,
    self_edge,
    unordered_or_duplicate_edge,
    parent_not_canonical,
    child_not_canonical,
    cycle,
};

struct trajectory_lineage_validation_v1 {
    trajectory_lineage_validation_code_v1 code =
        trajectory_lineage_validation_code_v1::valid;
    std::uint64_t index = 0;
    std::uint64_t visited_node_count = 0;

    [[nodiscard]] constexpr bool valid() const noexcept {
        return code == trajectory_lineage_validation_code_v1::valid;
    }
};

static_assert(std::is_standard_layout<trajectory_lineage_edge_v1>::value);
static_assert(std::is_trivially_copyable<trajectory_lineage_edge_v1>::value);
static_assert(offsetof(trajectory_lineage_mapping_view_v1, edges) == 0);

[[nodiscard]] constexpr bool trajectory_lineage_edge_less_v1(
    trajectory_lineage_edge_v1 lhs,
    trajectory_lineage_edge_v1 rhs) noexcept {
    return lhs.parent_global_entity_id < rhs.parent_global_entity_id
        || (lhs.parent_global_entity_id == rhs.parent_global_entity_id
            && lhs.child_global_entity_id < rhs.child_global_entity_id);
}

[[nodiscard]] inline std::uint64_t canonical_entity_ordinal_v1(
    canonical_entity_spine_v1 nodes,
    std::uint64_t identity) noexcept {
    std::uint64_t begin = 0;
    std::uint64_t end = nodes.entity_count;
    while (begin < end) {
        const auto middle = begin + (end - begin) / 2;
        if (nodes.global_entity_ids[middle] < identity) {
            begin = middle + 1;
        } else {
            end = middle;
        }
    }
    return begin;
}

// Kahn validation uses caller-owned O(V) arrays. With edges sorted by global
// parent/child identity, validation is O(E log V + V + E), never O(V*E).
[[nodiscard]] inline trajectory_lineage_validation_v1
validate_trajectory_lineage_mapping_v1(
    canonical_entity_spine_v1 nodes,
    trajectory_lineage_mapping_view_v1 mapping,
    trajectory_lineage_workspace_v1 workspace) noexcept {
    const atom_entity_coverage_claim_v1 full_nodes{
        nodes.global_entity_ids, nodes.entity_count, nodes.domain_identity};
    if (!validate_exact_entity_coverage_v1(nodes, full_nodes).valid()) {
        return {trajectory_lineage_validation_code_v1::invalid_node_spine};
    }
    if (mapping.edge_count == 0) {
        return {trajectory_lineage_validation_code_v1::empty_mapping};
    }
    if (mapping.edges == nullptr) {
        return {trajectory_lineage_validation_code_v1::missing_edges};
    }
    if (!atom::validate_atom_persistent_identity_v1(mapping.mapping_identity)
             .valid()) {
        return {trajectory_lineage_validation_code_v1::
                    invalid_mapping_identity};
    }
    if (mapping.node_domain_identity != nodes.domain_identity) {
        return {trajectory_lineage_validation_code_v1::node_domain_mismatch};
    }
    if (mapping.mapping_generation == 0) {
        return {trajectory_lineage_validation_code_v1::
                    missing_mapping_generation};
    }
    if (workspace.parent_offsets == nullptr || workspace.indegrees == nullptr
        || workspace.queue == nullptr) {
        return {trajectory_lineage_validation_code_v1::missing_workspace};
    }
    if (workspace.parent_offset_capacity < nodes.entity_count + 1
        || workspace.indegree_capacity < nodes.entity_count
        || workspace.queue_capacity < nodes.entity_count) {
        return {trajectory_lineage_validation_code_v1::insufficient_workspace};
    }
    for (std::uint64_t index = 0; index <= nodes.entity_count; ++index) {
        workspace.parent_offsets[index] = mapping.edge_count;
    }
    for (std::uint64_t index = 0; index < nodes.entity_count; ++index) {
        workspace.indegrees[index] = 0;
    }
    for (std::uint64_t index = 0; index < mapping.edge_count; ++index) {
        const auto edge = mapping.edges[index];
        if (edge.parent_global_entity_id == 0
            || edge.child_global_entity_id == 0) {
            return {trajectory_lineage_validation_code_v1::zero_node_identity,
                    index};
        }
        if (edge.parent_global_entity_id == edge.child_global_entity_id) {
            return {trajectory_lineage_validation_code_v1::self_edge, index};
        }
        if (index != 0
            && !trajectory_lineage_edge_less_v1(
                mapping.edges[index - 1], edge)) {
            return {trajectory_lineage_validation_code_v1::
                        unordered_or_duplicate_edge,
                    index};
        }
        const auto parent = canonical_entity_ordinal_v1(
            nodes, edge.parent_global_entity_id);
        const auto child = canonical_entity_ordinal_v1(
            nodes, edge.child_global_entity_id);
        if (parent == nodes.entity_count
            || nodes.global_entity_ids[parent]
                   != edge.parent_global_entity_id) {
            return {trajectory_lineage_validation_code_v1::
                        parent_not_canonical,
                    index};
        }
        if (child == nodes.entity_count
            || nodes.global_entity_ids[child] != edge.child_global_entity_id) {
            return {trajectory_lineage_validation_code_v1::child_not_canonical,
                    index};
        }
        if (workspace.parent_offsets[parent] == mapping.edge_count) {
            workspace.parent_offsets[parent] = index;
        }
        ++workspace.indegrees[child];
    }
    workspace.parent_offsets[nodes.entity_count] = mapping.edge_count;
    for (std::uint64_t index = nodes.entity_count; index-- > 0;) {
        if (workspace.parent_offsets[index] == mapping.edge_count) {
            workspace.parent_offsets[index] = workspace.parent_offsets[index + 1];
        }
    }

    std::uint64_t queue_begin = 0;
    std::uint64_t queue_end = 0;
    for (std::uint64_t index = 0; index < nodes.entity_count; ++index) {
        if (workspace.indegrees[index] == 0) {
            workspace.queue[queue_end++] = index;
        }
    }
    while (queue_begin < queue_end) {
        const auto parent = workspace.queue[queue_begin++];
        for (std::uint64_t edge_index = workspace.parent_offsets[parent];
             edge_index < workspace.parent_offsets[parent + 1];
             ++edge_index) {
            const auto child = canonical_entity_ordinal_v1(
                nodes, mapping.edges[edge_index].child_global_entity_id);
            --workspace.indegrees[child];
            if (workspace.indegrees[child] == 0) {
                workspace.queue[queue_end++] = child;
            }
        }
    }
    if (queue_end != nodes.entity_count) {
        return {trajectory_lineage_validation_code_v1::cycle,
                mapping.edge_count,
                queue_end};
    }
    return {trajectory_lineage_validation_code_v1::valid,
            mapping.edge_count,
            queue_end};
}

} // namespace cellshard::compiler::certification
