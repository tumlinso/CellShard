#pragma once

#include <CellShard/compiler/composition/coverage_v1.hh>

namespace cellshard::compiler::composition {

[[nodiscard]] inline coverage_composition_result_v1
compose_sparse_support_union_v1(
    structure_id output_identity,
    const exact_coverage_view_v1 &left,
    const exact_coverage_view_v1 &right,
    mutable_coverage_storage_v1 storage,
    exact_coverage_view_v1 *output) noexcept {
    if (!output_identity.valid()) {
        return {coverage_composition_code_v1::invalid_output_identity};
    }
    const auto left_status = validate_exact_coverage_v1(left);
    if (!left_status.composed()) return left_status;
    const auto right_status = validate_exact_coverage_v1(right);
    if (!right_status.composed()) return right_status;
    if (left.domain != right.domain) {
        return {coverage_composition_code_v1::input_domain_mismatch};
    }
    if (left.order != right.order) {
        return {coverage_composition_code_v1::input_order_mismatch};
    }
    if (left.logical_item_count
        > std::numeric_limits<std::uint64_t>::max()
              - right.logical_item_count) {
        return {coverage_composition_code_v1::count_overflow};
    }
    const auto maximum_count =
        left.logical_item_count + right.logical_item_count;
    if (storage.logical_item_ids == nullptr) {
        return {coverage_composition_code_v1::missing_storage};
    }
    if (storage.capacity < maximum_count) {
        return {coverage_composition_code_v1::insufficient_capacity};
    }
    if (output == nullptr) {
        return {coverage_composition_code_v1::missing_output};
    }
    *output = {};
    std::uint64_t left_index = 0;
    std::uint64_t right_index = 0;
    std::uint64_t output_index = 0;
    while (left_index < left.logical_item_count
           && right_index < right.logical_item_count) {
        const auto left_item = left.logical_item_ids[left_index];
        const auto right_item = right.logical_item_ids[right_index];
        if (left_item <= right_item) {
            storage.logical_item_ids[output_index++] = left_item;
            ++left_index;
            if (left_item == right_item) ++right_index;
        } else {
            storage.logical_item_ids[output_index++] = right_item;
            ++right_index;
        }
    }
    while (left_index < left.logical_item_count) {
        storage.logical_item_ids[output_index++] =
            left.logical_item_ids[left_index++];
    }
    while (right_index < right.logical_item_count) {
        storage.logical_item_ids[output_index++] =
            right.logical_item_ids[right_index++];
    }
    *output = {output_identity, left.domain, left.order,
               storage.logical_item_ids, output_index};
    return {coverage_composition_code_v1::composed, output_index};
}

} // namespace cellshard::compiler::composition
