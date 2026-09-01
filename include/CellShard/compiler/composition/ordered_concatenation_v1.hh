#pragma once

#include <CellShard/compiler/composition/coverage_v1.hh>

#include <cstdint>
#include <type_traits>

namespace cellshard::compiler::composition {

struct ordered_concatenation_storage_v1 {
    std::uint64_t *logical_item_ids = nullptr;
    std::uint64_t logical_item_capacity = 0;
    std::uint64_t *component_offsets = nullptr;
    std::uint32_t component_offset_capacity = 0;
};

struct ordered_concatenation_view_v1 {
    structure_id identity{};
    domain_id domain{};
    order_id order{};
    const std::uint64_t *logical_item_ids = nullptr;
    const std::uint64_t *component_offsets = nullptr;
    std::uint64_t logical_item_count = 0;
    std::uint32_t component_count = 0;
    std::uint32_t reserved = 0;
};

enum class ordered_concatenation_code_v1 : std::uint32_t {
    composed = 0,
    invalid_output_identity,
    invalid_output_order,
    invalid_left,
    invalid_right,
    domain_mismatch,
    count_overflow,
    overlapping_inputs,
    missing_storage,
    insufficient_item_capacity,
    insufficient_offset_capacity,
    missing_output,
};

struct ordered_concatenation_result_v1 {
    ordered_concatenation_code_v1 code =
        ordered_concatenation_code_v1::composed;
    std::uint64_t item = 0;
    [[nodiscard]] constexpr bool composed() const noexcept {
        return code == ordered_concatenation_code_v1::composed;
    }
};

[[nodiscard]] inline ordered_concatenation_result_v1
compose_ordered_concatenation_v1(
    structure_id output_identity,
    order_id output_order,
    const exact_coverage_view_v1 &left,
    const exact_coverage_view_v1 &right,
    ordered_concatenation_storage_v1 storage,
    ordered_concatenation_view_v1 *output) noexcept {
    if (!output_identity.valid()) {
        return {ordered_concatenation_code_v1::invalid_output_identity};
    }
    if (!output_order.valid()) {
        return {ordered_concatenation_code_v1::invalid_output_order};
    }
    if (!validate_exact_coverage_v1(left).composed()) {
        return {ordered_concatenation_code_v1::invalid_left};
    }
    if (!validate_exact_coverage_v1(right).composed()) {
        return {ordered_concatenation_code_v1::invalid_right};
    }
    if (left.domain != right.domain) {
        return {ordered_concatenation_code_v1::domain_mismatch};
    }
    if (left.logical_item_count
        > std::numeric_limits<std::uint64_t>::max()
              - right.logical_item_count) {
        return {ordered_concatenation_code_v1::count_overflow};
    }
    std::uint64_t left_index = 0;
    std::uint64_t right_index = 0;
    while (left_index < left.logical_item_count
           && right_index < right.logical_item_count) {
        const auto left_item = left.logical_item_ids[left_index];
        const auto right_item = right.logical_item_ids[right_index];
        if (left_item == right_item) {
            return {ordered_concatenation_code_v1::overlapping_inputs,
                    left_item};
        }
        if (left_item < right_item) ++left_index;
        else ++right_index;
    }
    const auto item_count = left.logical_item_count + right.logical_item_count;
    if (storage.logical_item_ids == nullptr
        || storage.component_offsets == nullptr) {
        return {ordered_concatenation_code_v1::missing_storage};
    }
    if (storage.logical_item_capacity < item_count) {
        return {ordered_concatenation_code_v1::insufficient_item_capacity};
    }
    if (storage.component_offset_capacity < 3) {
        return {ordered_concatenation_code_v1::insufficient_offset_capacity};
    }
    if (output == nullptr) {
        return {ordered_concatenation_code_v1::missing_output};
    }
    *output = {};
    for (std::uint64_t index = 0; index < left.logical_item_count; ++index) {
        storage.logical_item_ids[index] = left.logical_item_ids[index];
    }
    for (std::uint64_t index = 0; index < right.logical_item_count; ++index) {
        storage.logical_item_ids[left.logical_item_count + index] =
            right.logical_item_ids[index];
    }
    storage.component_offsets[0] = 0;
    storage.component_offsets[1] = left.logical_item_count;
    storage.component_offsets[2] = item_count;
    *output = {output_identity, left.domain, output_order,
               storage.logical_item_ids, storage.component_offsets,
               item_count, 2, 0};
    return {ordered_concatenation_code_v1::composed, item_count};
}

static_assert(std::is_trivially_copyable<ordered_concatenation_view_v1>::value);

} // namespace cellshard::compiler::composition
