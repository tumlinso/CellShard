#pragma once

#include <CellShard/domain.hh>

#include <cstdint>
#include <limits>
#include <type_traits>

namespace cellshard::compiler::composition {

struct exact_coverage_view_v1 {
    structure_id identity{};
    domain_id domain{};
    order_id order{};
    const std::uint64_t *logical_item_ids = nullptr;
    std::uint64_t logical_item_count = 0;
};

struct mutable_coverage_storage_v1 {
    std::uint64_t *logical_item_ids = nullptr;
    std::uint64_t capacity = 0;
};

enum class coverage_composition_code_v1 : std::uint32_t {
    composed = 0,
    invalid_output_identity,
    invalid_input_identity,
    invalid_domain,
    invalid_order,
    input_domain_mismatch,
    input_order_mismatch,
    empty_input,
    missing_input_items,
    unordered_input,
    duplicate_input_item,
    count_overflow,
    missing_storage,
    insufficient_capacity,
    overlapping_inputs,
    missing_output,
};

struct coverage_composition_result_v1 {
    coverage_composition_code_v1 code =
        coverage_composition_code_v1::composed;
    std::uint64_t item_index = 0;
    [[nodiscard]] constexpr bool composed() const noexcept {
        return code == coverage_composition_code_v1::composed;
    }
};

[[nodiscard]] inline coverage_composition_result_v1
validate_exact_coverage_v1(const exact_coverage_view_v1 &coverage) noexcept {
    if (!coverage.identity.valid()) {
        return {coverage_composition_code_v1::invalid_input_identity};
    }
    if (!coverage.domain.valid()) {
        return {coverage_composition_code_v1::invalid_domain};
    }
    if (!coverage.order.valid()) {
        return {coverage_composition_code_v1::invalid_order};
    }
    if (coverage.logical_item_count == 0) {
        return {coverage_composition_code_v1::empty_input};
    }
    if (coverage.logical_item_ids == nullptr) {
        return {coverage_composition_code_v1::missing_input_items};
    }
    for (std::uint64_t index = 1;
         index < coverage.logical_item_count;
         ++index) {
        if (coverage.logical_item_ids[index - 1]
            >= coverage.logical_item_ids[index]) {
            return {coverage.logical_item_ids[index - 1]
                        == coverage.logical_item_ids[index]
                    ? coverage_composition_code_v1::duplicate_input_item
                    : coverage_composition_code_v1::unordered_input,
                    index};
        }
    }
    return {};
}

[[nodiscard]] inline coverage_composition_result_v1
compose_disjoint_union_v1(
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
    const auto output_count =
        left.logical_item_count + right.logical_item_count;
    if (storage.logical_item_ids == nullptr) {
        return {coverage_composition_code_v1::missing_storage};
    }
    if (storage.capacity < output_count) {
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
        if (left_item == right_item) {
            return {coverage_composition_code_v1::overlapping_inputs,
                    left_item};
        }
        if (left_item < right_item) {
            storage.logical_item_ids[output_index++] = left_item;
            ++left_index;
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
               storage.logical_item_ids, output_count};
    return {coverage_composition_code_v1::composed, output_count};
}

static_assert(std::is_trivially_copyable<exact_coverage_view_v1>::value);
static_assert(std::is_trivially_copyable<mutable_coverage_storage_v1>::value);

} // namespace cellshard::compiler::composition
