#pragma once

#include <CellShard/compiler/atom/persistent_identity_v1.hh>
#include <CellShard/compiler/evidence/atom_evidence_record_v1.hh>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace cellshard::compiler::discovery::motif {

inline constexpr std::uint32_t typed_motif_contract_version_v1 = 1;

enum class motif_edge_direction_v1 : std::uint8_t {
    directed = 1,
    undirected = 2,
};

// Types are provider-qualified semantic identities supplied by the graph
// producer. A local node ordinal is only an endpoint inside this motif and is
// never promoted to biological identity.
struct typed_motif_node_v1 {
    atom::atom_persistent_identity_v1 node_type{};
    std::uint32_t role = 0;
    std::uint32_t reserved = 0;
};

struct typed_motif_edge_v1 {
    atom::atom_persistent_identity_v1 relation_type{};
    std::uint32_t source_node = 0;
    std::uint32_t destination_node = 0;
    std::uint32_t role = 0;
    motif_edge_direction_v1 direction = motif_edge_direction_v1::directed;
    std::uint8_t reserved[3]{};
};

// Pointer-first, allocation-free proposal vocabulary. The evidence identity
// names this vocabulary observation; graph_family identifies the producer's
// exact graph family. Neither validation nor any value in this view authorizes
// execution or claims exact coverage.
struct typed_motif_vocabulary_view_v1 {
    const typed_motif_node_v1 *nodes = nullptr;
    const typed_motif_edge_v1 *edges = nullptr;
    std::uint32_t node_count = 0;
    std::uint32_t edge_count = 0;
    std::uint32_t maximum_node_count = 0;
    std::uint32_t maximum_edge_count = 0;
    evidence::evidence_identity_v1 vocabulary_identity{};
    atom::atom_persistent_identity_v1 graph_family_identity{};
    std::uint64_t graph_generation = 0;
};

enum class typed_motif_validation_code_v1 : std::uint32_t {
    valid = 0,
    invalid_vocabulary_identity,
    invalid_graph_family_identity,
    missing_graph_generation,
    invalid_node_bound,
    invalid_edge_bound,
    empty_nodes,
    missing_nodes,
    empty_edges,
    missing_edges,
    node_bound_exceeded,
    edge_bound_exceeded,
    invalid_node_type,
    missing_node_role,
    nonzero_node_reserved,
    invalid_relation_type,
    endpoint_out_of_range,
    self_edge,
    missing_edge_role,
    invalid_direction,
    nonzero_edge_reserved,
    duplicate_typed_edge,
};

struct typed_motif_validation_v1 {
    typed_motif_validation_code_v1 code =
        typed_motif_validation_code_v1::valid;
    std::uint32_t index = 0;

    [[nodiscard]] constexpr bool valid() const noexcept {
        return code == typed_motif_validation_code_v1::valid;
    }
};

static_assert(std::is_standard_layout<typed_motif_node_v1>::value);
static_assert(std::is_trivially_copyable<typed_motif_node_v1>::value);
static_assert(std::is_standard_layout<typed_motif_edge_v1>::value);
static_assert(std::is_trivially_copyable<typed_motif_edge_v1>::value);
static_assert(offsetof(typed_motif_vocabulary_view_v1, nodes) == 0);
static_assert(std::is_standard_layout<typed_motif_vocabulary_view_v1>::value);
static_assert(
    std::is_trivially_copyable<typed_motif_vocabulary_view_v1>::value);

[[nodiscard]] constexpr bool valid_motif_edge_direction_v1(
    motif_edge_direction_v1 direction) noexcept {
    return direction == motif_edge_direction_v1::directed
        || direction == motif_edge_direction_v1::undirected;
}

[[nodiscard]] constexpr bool same_typed_edge_v1(
    const typed_motif_edge_v1 &lhs,
    const typed_motif_edge_v1 &rhs) noexcept {
    if (lhs.relation_type != rhs.relation_type || lhs.role != rhs.role
        || lhs.direction != rhs.direction) {
        return false;
    }
    if (lhs.direction == motif_edge_direction_v1::directed) {
        return lhs.source_node == rhs.source_node
            && lhs.destination_node == rhs.destination_node;
    }
    return (lhs.source_node == rhs.source_node
            && lhs.destination_node == rhs.destination_node)
        || (lhs.source_node == rhs.destination_node
            && lhs.destination_node == rhs.source_node);
}

// O(nodes + edges^2), O(1) storage. Motifs are deliberately small and bounded;
// the quadratic duplicate check avoids hidden allocation and makes the complete
// validation cost explicit to callers.
[[nodiscard]] constexpr typed_motif_validation_v1
validate_typed_motif_vocabulary_v1(
    typed_motif_vocabulary_view_v1 view) noexcept {
    if (!evidence::valid_evidence_identity_v1(view.vocabulary_identity)) {
        return {typed_motif_validation_code_v1::invalid_vocabulary_identity};
    }
    if (!atom::validate_atom_persistent_identity_v1(
             view.graph_family_identity).valid()) {
        return {typed_motif_validation_code_v1::invalid_graph_family_identity};
    }
    if (view.graph_generation == 0) {
        return {typed_motif_validation_code_v1::missing_graph_generation};
    }
    if (view.maximum_node_count == 0) {
        return {typed_motif_validation_code_v1::invalid_node_bound};
    }
    if (view.maximum_edge_count == 0) {
        return {typed_motif_validation_code_v1::invalid_edge_bound};
    }
    if (view.node_count == 0) {
        return {typed_motif_validation_code_v1::empty_nodes};
    }
    if (view.nodes == nullptr) {
        return {typed_motif_validation_code_v1::missing_nodes};
    }
    if (view.edge_count == 0) {
        return {typed_motif_validation_code_v1::empty_edges};
    }
    if (view.edges == nullptr) {
        return {typed_motif_validation_code_v1::missing_edges};
    }
    if (view.node_count > view.maximum_node_count) {
        return {typed_motif_validation_code_v1::node_bound_exceeded};
    }
    if (view.edge_count > view.maximum_edge_count) {
        return {typed_motif_validation_code_v1::edge_bound_exceeded};
    }
    for (std::uint32_t index = 0; index < view.node_count; ++index) {
        const auto &node = view.nodes[index];
        if (!atom::validate_atom_persistent_identity_v1(node.node_type).valid()) {
            return {typed_motif_validation_code_v1::invalid_node_type, index};
        }
        if (node.role == 0) {
            return {typed_motif_validation_code_v1::missing_node_role, index};
        }
        if (node.reserved != 0) {
            return {typed_motif_validation_code_v1::nonzero_node_reserved,
                    index};
        }
    }
    for (std::uint32_t index = 0; index < view.edge_count; ++index) {
        const auto &edge = view.edges[index];
        if (!atom::validate_atom_persistent_identity_v1(
                 edge.relation_type).valid()) {
            return {typed_motif_validation_code_v1::invalid_relation_type,
                    index};
        }
        if (edge.source_node >= view.node_count
            || edge.destination_node >= view.node_count) {
            return {typed_motif_validation_code_v1::endpoint_out_of_range,
                    index};
        }
        if (edge.source_node == edge.destination_node) {
            return {typed_motif_validation_code_v1::self_edge, index};
        }
        if (edge.role == 0) {
            return {typed_motif_validation_code_v1::missing_edge_role, index};
        }
        if (!valid_motif_edge_direction_v1(edge.direction)) {
            return {typed_motif_validation_code_v1::invalid_direction, index};
        }
        for (const auto byte : edge.reserved) {
            if (byte != 0) {
                return {typed_motif_validation_code_v1::nonzero_edge_reserved,
                        index};
            }
        }
        for (std::uint32_t earlier = 0; earlier < index; ++earlier) {
            if (same_typed_edge_v1(view.edges[earlier], edge)) {
                return {typed_motif_validation_code_v1::duplicate_typed_edge,
                        index};
            }
        }
    }
    return {typed_motif_validation_code_v1::valid, view.edge_count};
}

[[nodiscard]] constexpr bool authorizes_execution(
    typed_motif_vocabulary_view_v1) noexcept {
    return false;
}

} // namespace cellshard::compiler::discovery::motif
