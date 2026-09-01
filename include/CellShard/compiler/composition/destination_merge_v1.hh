#pragma once

#include <CellShard/compiler/composition/relation_merge_v1.hh>

namespace cellshard::compiler::composition {

[[nodiscard]] constexpr bool destination_major_less_v1(
    const typed_relation_edge_v1 &lhs,
    const typed_relation_edge_v1 &rhs) noexcept {
    return lhs.destination_identity < rhs.destination_identity
        || (lhs.destination_identity == rhs.destination_identity
            && (lhs.source_identity < rhs.source_identity
                || (lhs.source_identity == rhs.source_identity
                    && lhs.logical_edge_identity < rhs.logical_edge_identity)));
}

[[nodiscard]] inline relation_merge_result_v1
validate_destination_major_relation_v1(
    const typed_relation_view_v1 &relation) noexcept {
    if (!relation.identity.valid()) {
        return {relation_merge_code_v1::invalid_relation_identity};
    }
    if (!relation.source_domain.valid() || !relation.source_order.valid()
        || !relation.destination_domain.valid()
        || !relation.destination_order.valid()) {
        return {relation_merge_code_v1::invalid_axis_identity};
    }
    if (relation.edge_count != 0 && relation.edges == nullptr) {
        return {relation_merge_code_v1::missing_edges};
    }
    for (std::uint64_t index = 0; index < relation.edge_count; ++index) {
        if (relation.edges[index].source_identity == 0
            || relation.edges[index].destination_identity == 0
            || relation.edges[index].logical_edge_identity == 0) {
            return {relation_merge_code_v1::zero_logical_identity, index};
        }
        if (index != 0) {
            if (!destination_major_less_v1(
                    relation.edges[index - 1], relation.edges[index])) {
                return {relation_merge_code_v1::unordered_edges, index};
            }
            if (relation.edges[index - 1].logical_edge_identity
                >= relation.edges[index].logical_edge_identity) {
                return {relation_merge_code_v1::unordered_logical_identity,
                        index};
            }
        }
    }
    return {};
}

[[nodiscard]] inline relation_merge_result_v1
compose_destination_aligned_merge_v1(
    structure_id output_identity,
    const typed_relation_view_v1 &left,
    const typed_relation_view_v1 &right,
    relation_merge_storage_v1 storage,
    typed_relation_view_v1 *output) noexcept {
    if (!output_identity.valid()) {
        return {relation_merge_code_v1::invalid_output_identity};
    }
    const auto left_status = validate_destination_major_relation_v1(left);
    if (!left_status.composed()) return left_status;
    const auto right_status = validate_destination_major_relation_v1(right);
    if (!right_status.composed()) return right_status;
    if (left.destination_domain != right.destination_domain
        || left.destination_order != right.destination_order) {
        return {relation_merge_code_v1::destination_axis_mismatch};
    }
    if (left.source_domain != right.source_domain
        || left.source_order != right.source_order) {
        return {relation_merge_code_v1::source_axis_mismatch};
    }
    if (left.edge_count
        > std::numeric_limits<std::uint64_t>::max() - right.edge_count) {
        return {relation_merge_code_v1::count_overflow};
    }
    const auto edge_count = left.edge_count + right.edge_count;
    if (edge_count != 0 && storage.edges == nullptr) {
        return {relation_merge_code_v1::missing_storage};
    }
    if (storage.edge_capacity < edge_count) {
        return {relation_merge_code_v1::insufficient_capacity};
    }
    if (output == nullptr) return {relation_merge_code_v1::missing_output};
    *output = {};
    std::uint64_t left_index = 0;
    std::uint64_t right_index = 0;
    std::uint64_t output_index = 0;
    while (left_index < left.edge_count && right_index < right.edge_count) {
        const auto &left_edge = left.edges[left_index];
        const auto &right_edge = right.edges[right_index];
        if (left_edge.logical_edge_identity == right_edge.logical_edge_identity) {
            return {relation_merge_code_v1::duplicate_logical_edge,
                    left_edge.logical_edge_identity};
        }
        if (destination_major_less_v1(left_edge, right_edge)) {
            storage.edges[output_index++] = left_edge;
            ++left_index;
        } else {
            storage.edges[output_index++] = right_edge;
            ++right_index;
        }
    }
    while (left_index < left.edge_count) {
        storage.edges[output_index++] = left.edges[left_index++];
    }
    while (right_index < right.edge_count) {
        storage.edges[output_index++] = right.edges[right_index++];
    }
    *output = {output_identity, left.source_domain, left.source_order,
               left.destination_domain, left.destination_order,
               storage.edges, edge_count};
    return {relation_merge_code_v1::composed, edge_count};
}

} // namespace cellshard::compiler::composition
