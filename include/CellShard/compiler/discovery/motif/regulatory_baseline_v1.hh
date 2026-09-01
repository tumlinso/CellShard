#pragma once

#include <CellShard/compiler/discovery/motif/typed_vocabulary_v1.hh>

#include <cstdint>

namespace cellshard::compiler::discovery::motif {

enum class regulatory_motif_kind_v1 : std::uint32_t {
    feedback_pair = 1,
    feed_forward_loop = 2,
    fan_in = 3,
    fan_out = 4,
    bi_fan = 5,
};

struct regulatory_motif_baseline_request_v1 {
    regulatory_motif_kind_v1 kind = regulatory_motif_kind_v1::feedback_pair;
    const atom::atom_persistent_identity_v1 *node_types = nullptr;
    const atom::atom_persistent_identity_v1 *relation_types = nullptr;
    std::uint32_t node_type_count = 0;
    std::uint32_t relation_type_count = 0;
    evidence::evidence_identity_v1 vocabulary_identity{};
    atom::atom_persistent_identity_v1 graph_family_identity{};
    std::uint64_t graph_generation = 0;
};

struct regulatory_motif_baseline_buffers_v1 {
    typed_motif_node_v1 *nodes = nullptr;
    std::uint32_t node_capacity = 0;
    typed_motif_edge_v1 *edges = nullptr;
    std::uint32_t edge_capacity = 0;
};

enum class regulatory_motif_baseline_code_v1 : std::uint32_t {
    built = 0,
    invalid_kind,
    missing_types,
    incorrect_type_count,
    invalid_node_type,
    invalid_relation_type,
    missing_output,
    insufficient_output,
    invalid_result,
};

struct regulatory_motif_baseline_result_v1 {
    regulatory_motif_baseline_code_v1 code =
        regulatory_motif_baseline_code_v1::built;
    typed_motif_vocabulary_view_v1 motif{};
    std::uint32_t index = 0;
    [[nodiscard]] constexpr bool built() const noexcept {
        return code == regulatory_motif_baseline_code_v1::built;
    }
};

struct regulatory_motif_shape_v1 {
    std::uint32_t node_count = 0;
    std::uint32_t edge_count = 0;
};

[[nodiscard]] constexpr regulatory_motif_shape_v1 regulatory_motif_shape_v1_for(
    regulatory_motif_kind_v1 kind) noexcept {
    switch (kind) {
    case regulatory_motif_kind_v1::feedback_pair: return {2, 2};
    case regulatory_motif_kind_v1::feed_forward_loop: return {3, 3};
    case regulatory_motif_kind_v1::fan_in: return {3, 2};
    case regulatory_motif_kind_v1::fan_out: return {3, 2};
    case regulatory_motif_kind_v1::bi_fan: return {4, 4};
    }
    return {};
}

namespace detail {

inline void regulatory_endpoint_v1(
    regulatory_motif_kind_v1 kind,
    std::uint32_t edge,
    std::uint32_t *source,
    std::uint32_t *destination) noexcept {
    switch (kind) {
    case regulatory_motif_kind_v1::feedback_pair:
        *source = edge;
        *destination = 1 - edge;
        return;
    case regulatory_motif_kind_v1::feed_forward_loop:
        *source = edge == 2 ? 0 : edge;
        *destination = edge == 0 ? 1 : 2;
        return;
    case regulatory_motif_kind_v1::fan_in:
        *source = edge;
        *destination = 2;
        return;
    case regulatory_motif_kind_v1::fan_out:
        *source = 0;
        *destination = edge + 1;
        return;
    case regulatory_motif_kind_v1::bi_fan:
        *source = edge / 2;
        *destination = 2 + edge % 2;
        return;
    }
    *source = 0;
    *destination = 0;
}

} // namespace detail

// The library fixes only topology. Callers supply every biological node and
// relation type, so CellShard never infers regulatory semantics from shape.
[[nodiscard]] inline regulatory_motif_baseline_result_v1
build_regulatory_motif_baseline_v1(
    regulatory_motif_baseline_request_v1 request,
    regulatory_motif_baseline_buffers_v1 buffers) noexcept {
    const auto shape = regulatory_motif_shape_v1_for(request.kind);
    if (shape.node_count == 0) {
        return {regulatory_motif_baseline_code_v1::invalid_kind};
    }
    if (request.node_types == nullptr || request.relation_types == nullptr) {
        return {regulatory_motif_baseline_code_v1::missing_types};
    }
    if (request.node_type_count != shape.node_count
        || request.relation_type_count != shape.edge_count) {
        return {regulatory_motif_baseline_code_v1::incorrect_type_count};
    }
    for (std::uint32_t index = 0; index < shape.node_count; ++index) {
        if (!atom::validate_atom_persistent_identity_v1(
                 request.node_types[index]).valid()) {
            return {regulatory_motif_baseline_code_v1::invalid_node_type,
                    {}, index};
        }
    }
    for (std::uint32_t index = 0; index < shape.edge_count; ++index) {
        if (!atom::validate_atom_persistent_identity_v1(
                 request.relation_types[index]).valid()) {
            return {regulatory_motif_baseline_code_v1::invalid_relation_type,
                    {}, index};
        }
    }
    if (buffers.nodes == nullptr || buffers.edges == nullptr) {
        return {regulatory_motif_baseline_code_v1::missing_output};
    }
    if (buffers.node_capacity < shape.node_count
        || buffers.edge_capacity < shape.edge_count) {
        return {regulatory_motif_baseline_code_v1::insufficient_output};
    }
    for (std::uint32_t index = 0; index < shape.node_count; ++index) {
        buffers.nodes[index] = {request.node_types[index], index + 1, 0};
    }
    for (std::uint32_t index = 0; index < shape.edge_count; ++index) {
        std::uint32_t source = 0;
        std::uint32_t destination = 0;
        detail::regulatory_endpoint_v1(
            request.kind, index, &source, &destination);
        buffers.edges[index] = {
            request.relation_types[index], source, destination, index + 1,
            motif_edge_direction_v1::directed, {}};
    }
    const typed_motif_vocabulary_view_v1 motif{
        buffers.nodes, buffers.edges, shape.node_count, shape.edge_count,
        shape.node_count, shape.edge_count, request.vocabulary_identity,
        request.graph_family_identity, request.graph_generation};
    if (!validate_typed_motif_vocabulary_v1(motif).valid()) {
        return {regulatory_motif_baseline_code_v1::invalid_result};
    }
    return {regulatory_motif_baseline_code_v1::built, motif,
            shape.edge_count};
}

} // namespace cellshard::compiler::discovery::motif
