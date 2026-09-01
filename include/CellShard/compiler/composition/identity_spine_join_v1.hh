#pragma once

#include <CellShard/domain.hh>

#include <cstdint>
#include <limits>
#include <type_traits>

namespace cellshard::compiler::composition {

struct identity_spine_view_v1 {
    structure_id identity{};
    domain_id domain{};
    order_id order{};
    const std::uint64_t *logical_identities = nullptr;
    std::uint64_t identity_count = 0;
};

struct identity_spine_join_entry_v1 {
    std::uint64_t logical_identity = 0;
    std::uint32_t left_local_index = 0;
    std::uint32_t right_local_index = 0;
};

struct identity_spine_join_view_v1 {
    structure_id identity{};
    domain_id domain{};
    order_id left_order{};
    order_id right_order{};
    const identity_spine_join_entry_v1 *entries = nullptr;
    std::uint32_t entry_count = 0;
    std::uint32_t reserved = 0;
};

enum class identity_spine_join_code_v1 : std::uint32_t {
    joined = 0,
    invalid_output_identity,
    invalid_spine_identity,
    invalid_axis_identity,
    excessive_local_width,
    missing_identities,
    zero_logical_identity,
    unordered_identity,
    domain_mismatch,
    identity_count_mismatch,
    missing_left_identity,
    missing_right_identity,
    missing_storage,
    insufficient_capacity,
    missing_output,
};

struct identity_spine_join_result_v1 {
    identity_spine_join_code_v1 code = identity_spine_join_code_v1::joined;
    std::uint64_t identity = 0;
    [[nodiscard]] constexpr bool joined() const noexcept {
        return code == identity_spine_join_code_v1::joined;
    }
};

[[nodiscard]] inline identity_spine_join_result_v1 validate_identity_spine_v1(
    const identity_spine_view_v1 &spine) noexcept {
    if (!spine.identity.valid()) {
        return {identity_spine_join_code_v1::invalid_spine_identity};
    }
    if (!spine.domain.valid() || !spine.order.valid()) {
        return {identity_spine_join_code_v1::invalid_axis_identity};
    }
    if (spine.identity_count > std::numeric_limits<std::uint32_t>::max()) {
        return {identity_spine_join_code_v1::excessive_local_width};
    }
    if (spine.identity_count != 0 && spine.logical_identities == nullptr) {
        return {identity_spine_join_code_v1::missing_identities};
    }
    for (std::uint64_t index = 0; index < spine.identity_count; ++index) {
        if (spine.logical_identities[index] == 0) {
            return {identity_spine_join_code_v1::zero_logical_identity,
                    index};
        }
        if (index != 0
            && spine.logical_identities[index - 1]
                   >= spine.logical_identities[index]) {
            return {identity_spine_join_code_v1::unordered_identity, index};
        }
    }
    return {};
}

[[nodiscard]] inline identity_spine_join_result_v1
compose_identity_spine_join_v1(
    structure_id output_identity,
    const identity_spine_view_v1 &left,
    const identity_spine_view_v1 &right,
    identity_spine_join_entry_v1 *storage,
    std::uint32_t capacity,
    identity_spine_join_view_v1 *output) noexcept {
    if (!output_identity.valid()) {
        return {identity_spine_join_code_v1::invalid_output_identity};
    }
    const auto left_status = validate_identity_spine_v1(left);
    if (!left_status.joined()) return left_status;
    const auto right_status = validate_identity_spine_v1(right);
    if (!right_status.joined()) return right_status;
    if (left.domain != right.domain) {
        return {identity_spine_join_code_v1::domain_mismatch};
    }
    if (left.identity_count != right.identity_count) {
        return {identity_spine_join_code_v1::identity_count_mismatch};
    }
    if (left.identity_count != 0 && storage == nullptr) {
        return {identity_spine_join_code_v1::missing_storage};
    }
    if (capacity < left.identity_count) {
        return {identity_spine_join_code_v1::insufficient_capacity};
    }
    if (output == nullptr) return {identity_spine_join_code_v1::missing_output};
    *output = {};
    std::uint32_t left_index = 0;
    std::uint32_t right_index = 0;
    std::uint32_t output_index = 0;
    while (left_index < left.identity_count
           && right_index < right.identity_count) {
        const auto left_id = left.logical_identities[left_index];
        const auto right_id = right.logical_identities[right_index];
        if (left_id < right_id) {
            return {identity_spine_join_code_v1::missing_right_identity,
                    left_id};
        }
        if (right_id < left_id) {
            return {identity_spine_join_code_v1::missing_left_identity,
                    right_id};
        }
        storage[output_index++] = {left_id, left_index, right_index};
        ++left_index;
        ++right_index;
    }
    *output = {output_identity, left.domain, left.order, right.order,
               storage, output_index, 0};
    return {identity_spine_join_code_v1::joined, output_index};
}

static_assert(std::is_trivially_copyable<identity_spine_join_entry_v1>::value);
static_assert(std::is_trivially_copyable<identity_spine_join_view_v1>::value);

} // namespace cellshard::compiler::composition
