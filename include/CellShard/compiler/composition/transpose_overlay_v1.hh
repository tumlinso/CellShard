#pragma once

#include <CellShard/compiler/composition/destination_merge_v1.hh>
#include <CellShard/compiler/composition/production_identity_v1.hh>

#include <cstdint>
#include <type_traits>

namespace cellshard::compiler::composition {

struct transpose_overlay_composition_v1 {
    composition_production_id production{};
    structure_id forward_overlay{};
    structure_id transpose_overlay{};
    domain_id forward_source_domain{};
    order_id forward_source_order{};
    domain_id forward_destination_domain{};
    order_id forward_destination_order{};
    std::uint64_t logical_edge_count = 0;
};

enum class transpose_overlay_code_v1 : std::uint32_t {
    composed = 0,
    invalid_production,
    invalid_forward,
    invalid_transpose,
    axis_not_transposed,
    edge_count_mismatch,
    logical_edge_mismatch,
    endpoint_not_transposed,
    missing_output,
};

struct transpose_overlay_result_v1 {
    transpose_overlay_code_v1 code = transpose_overlay_code_v1::composed;
    std::uint64_t edge_index = 0;
    [[nodiscard]] constexpr bool composed() const noexcept {
        return code == transpose_overlay_code_v1::composed;
    }
};

[[nodiscard]] inline transpose_overlay_result_v1
compose_transpose_overlay_v1(
    composition_production_id production,
    const typed_relation_view_v1 &forward,
    const typed_relation_view_v1 &transpose,
    transpose_overlay_composition_v1 *output) noexcept {
    if (!production.valid()) {
        return {transpose_overlay_code_v1::invalid_production};
    }
    if (!validate_source_major_relation_v1(forward).composed()) {
        return {transpose_overlay_code_v1::invalid_forward};
    }
    if (!validate_destination_major_relation_v1(transpose).composed()) {
        return {transpose_overlay_code_v1::invalid_transpose};
    }
    if (forward.source_domain != transpose.destination_domain
        || forward.source_order != transpose.destination_order
        || forward.destination_domain != transpose.source_domain
        || forward.destination_order != transpose.source_order) {
        return {transpose_overlay_code_v1::axis_not_transposed};
    }
    if (forward.edge_count != transpose.edge_count) {
        return {transpose_overlay_code_v1::edge_count_mismatch};
    }
    for (std::uint64_t index = 0; index < forward.edge_count; ++index) {
        const auto &forward_edge = forward.edges[index];
        const auto &transpose_edge = transpose.edges[index];
        if (forward_edge.logical_edge_identity
            != transpose_edge.logical_edge_identity) {
            return {transpose_overlay_code_v1::logical_edge_mismatch, index};
        }
        if (forward_edge.source_identity != transpose_edge.destination_identity
            || forward_edge.destination_identity
                   != transpose_edge.source_identity) {
            return {transpose_overlay_code_v1::endpoint_not_transposed, index};
        }
    }
    if (output == nullptr) return {transpose_overlay_code_v1::missing_output};
    *output = {production, forward.identity, transpose.identity,
               forward.source_domain, forward.source_order,
               forward.destination_domain, forward.destination_order,
               forward.edge_count};
    return {transpose_overlay_code_v1::composed, forward.edge_count};
}

static_assert(
    std::is_trivially_copyable<transpose_overlay_composition_v1>::value);

} // namespace cellshard::compiler::composition
