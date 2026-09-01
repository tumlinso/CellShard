#pragma once

#include <CellShard/domain.hh>

#include <cstdint>
#include <type_traits>

namespace cellshard::compiler::composition {

struct logical_to_local_order_entry_v1 {
    std::uint64_t logical_identity = 0;
    std::uint32_t local_index = 0;
    std::uint32_t reserved = 0;
};

struct persistent_order_index_v1 {
    domain_id domain{};
    order_id order{};
    const logical_to_local_order_entry_v1 *entries = nullptr;
    std::uint32_t entry_count = 0;
    std::uint32_t reserved = 0;
};

struct persistent_order_link_workspace_v1 {
    std::uint8_t *left_local_marks = nullptr;
    std::uint8_t *right_local_marks = nullptr;
    std::uint32_t mark_capacity = 0;
    std::uint32_t *left_to_right = nullptr;
    std::uint32_t mapping_capacity = 0;
};

struct persistent_order_link_v1 {
    domain_id domain{};
    order_id left_order{};
    order_id right_order{};
    const std::uint32_t *left_to_right = nullptr;
    std::uint32_t element_count = 0;
    std::uint32_t reserved = 0;
};

enum class persistent_order_link_code_v1 : std::uint32_t {
    linked = 0,
    invalid_axis,
    domain_mismatch,
    count_mismatch,
    missing_entries,
    zero_logical_identity,
    unordered_logical_identity,
    local_index_out_of_range,
    duplicate_local_index,
    logical_identity_mismatch,
    missing_workspace,
    insufficient_mark_capacity,
    insufficient_mapping_capacity,
    missing_output,
};

struct persistent_order_link_result_v1 {
    persistent_order_link_code_v1 code =
        persistent_order_link_code_v1::linked;
    std::uint32_t entry_index = 0;
    [[nodiscard]] constexpr bool linked() const noexcept {
        return code == persistent_order_link_code_v1::linked;
    }
};

[[nodiscard]] inline persistent_order_link_result_v1
compose_persistent_order_link_v1(
    const persistent_order_index_v1 &left,
    const persistent_order_index_v1 &right,
    persistent_order_link_workspace_v1 workspace,
    persistent_order_link_v1 *output) noexcept {
    if (!left.domain.valid() || !left.order.valid() || !right.domain.valid()
        || !right.order.valid()) {
        return {persistent_order_link_code_v1::invalid_axis};
    }
    if (left.domain != right.domain) {
        return {persistent_order_link_code_v1::domain_mismatch};
    }
    if (left.entry_count != right.entry_count) {
        return {persistent_order_link_code_v1::count_mismatch};
    }
    if (left.entry_count != 0
        && (left.entries == nullptr || right.entries == nullptr)) {
        return {persistent_order_link_code_v1::missing_entries};
    }
    if (left.entry_count != 0
        && (workspace.left_local_marks == nullptr
            || workspace.right_local_marks == nullptr
            || workspace.left_to_right == nullptr)) {
        return {persistent_order_link_code_v1::missing_workspace};
    }
    if (workspace.mark_capacity < left.entry_count) {
        return {persistent_order_link_code_v1::insufficient_mark_capacity};
    }
    if (workspace.mapping_capacity < left.entry_count) {
        return {persistent_order_link_code_v1::insufficient_mapping_capacity};
    }
    if (output == nullptr) return {persistent_order_link_code_v1::missing_output};
    *output = {};
    for (std::uint32_t index = 0; index < left.entry_count; ++index) {
        workspace.left_local_marks[index] = 0;
        workspace.right_local_marks[index] = 0;
    }
    for (std::uint32_t index = 0; index < left.entry_count; ++index) {
        const auto &left_entry = left.entries[index];
        const auto &right_entry = right.entries[index];
        if (left_entry.logical_identity == 0
            || right_entry.logical_identity == 0) {
            return {persistent_order_link_code_v1::zero_logical_identity,
                    index};
        }
        if (index != 0
            && (left.entries[index - 1].logical_identity
                    >= left_entry.logical_identity
                || right.entries[index - 1].logical_identity
                    >= right_entry.logical_identity)) {
            return {persistent_order_link_code_v1::
                        unordered_logical_identity,
                    index};
        }
        if (left_entry.logical_identity != right_entry.logical_identity) {
            return {persistent_order_link_code_v1::logical_identity_mismatch,
                    index};
        }
        if (left_entry.local_index >= left.entry_count
            || right_entry.local_index >= right.entry_count) {
            return {persistent_order_link_code_v1::local_index_out_of_range,
                    index};
        }
        if (workspace.left_local_marks[left_entry.local_index] != 0
            || workspace.right_local_marks[right_entry.local_index] != 0) {
            return {persistent_order_link_code_v1::duplicate_local_index,
                    index};
        }
        workspace.left_local_marks[left_entry.local_index] = 1;
        workspace.right_local_marks[right_entry.local_index] = 1;
        workspace.left_to_right[left_entry.local_index] = right_entry.local_index;
    }
    *output = {left.domain, left.order, right.order, workspace.left_to_right,
               left.entry_count, 0};
    return {persistent_order_link_code_v1::linked, left.entry_count};
}

static_assert(
    std::is_trivially_copyable<logical_to_local_order_entry_v1>::value);
static_assert(std::is_trivially_copyable<persistent_order_link_v1>::value);

} // namespace cellshard::compiler::composition
