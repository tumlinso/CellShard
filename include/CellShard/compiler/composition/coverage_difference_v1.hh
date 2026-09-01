#pragma once

#include <CellShard/compiler/composition/coverage_v1.hh>

namespace cellshard::compiler::composition {

[[nodiscard]] inline coverage_composition_result_v1
compose_coverage_difference_v1(
    structure_id output_identity,
    const exact_coverage_view_v1 &minuend,
    const exact_coverage_view_v1 &subtrahend,
    mutable_coverage_storage_v1 storage,
    exact_coverage_view_v1 *output) noexcept {
    if (!output_identity.valid()) {
        return {coverage_composition_code_v1::invalid_output_identity};
    }
    const auto minuend_status = validate_exact_coverage_v1(minuend);
    if (!minuend_status.composed()) return minuend_status;
    const auto subtrahend_status = validate_exact_coverage_v1(subtrahend);
    if (!subtrahend_status.composed()) return subtrahend_status;
    if (minuend.domain != subtrahend.domain) {
        return {coverage_composition_code_v1::input_domain_mismatch};
    }
    if (minuend.order != subtrahend.order) {
        return {coverage_composition_code_v1::input_order_mismatch};
    }
    if (minuend.logical_item_count != 0
        && storage.logical_item_ids == nullptr) {
        return {coverage_composition_code_v1::missing_storage};
    }
    if (storage.capacity < minuend.logical_item_count) {
        return {coverage_composition_code_v1::insufficient_capacity};
    }
    if (output == nullptr) {
        return {coverage_composition_code_v1::missing_output};
    }
    *output = {};
    std::uint64_t minuend_index = 0;
    std::uint64_t subtrahend_index = 0;
    std::uint64_t output_index = 0;
    while (minuend_index < minuend.logical_item_count) {
        const auto minuend_item = minuend.logical_item_ids[minuend_index];
        while (subtrahend_index < subtrahend.logical_item_count
               && subtrahend.logical_item_ids[subtrahend_index]
                      < minuend_item) {
            ++subtrahend_index;
        }
        if (subtrahend_index == subtrahend.logical_item_count
            || subtrahend.logical_item_ids[subtrahend_index]
                   != minuend_item) {
            storage.logical_item_ids[output_index++] = minuend_item;
        }
        ++minuend_index;
    }
    *output = {output_identity, minuend.domain, minuend.order,
               storage.logical_item_ids, output_index};
    return {coverage_composition_code_v1::composed, output_index};
}

} // namespace cellshard::compiler::composition
