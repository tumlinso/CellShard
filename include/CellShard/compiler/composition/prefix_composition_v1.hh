#pragma once

#include <CellShard/compiler/composition/production_identity_v1.hh>

#include <cstdint>
#include <type_traits>

namespace cellshard::compiler::composition {

enum class prefix_sequence_kind_v1 : std::uint8_t {
    graph = 1,
    trajectory = 2,
};

struct prefix_sequence_entry_v1 {
    std::uint64_t logical_identity = 0;
    std::uint32_t sequence_position = 0;
    std::uint32_t reserved = 0;
};

struct prefix_sequence_view_v1 {
    structure_id identity{};
    domain_id domain{};
    order_id order{};
    prefix_sequence_kind_v1 kind = prefix_sequence_kind_v1::graph;
    std::uint8_t reserved[3]{};
    const prefix_sequence_entry_v1 *entries = nullptr;
    std::uint32_t entry_count = 0;
    std::uint32_t reserved2 = 0;
};

struct prefix_composition_v1 {
    composition_production_id production{};
    structure_id prefix{};
    structure_id full{};
    domain_id domain{};
    order_id order{};
    prefix_sequence_kind_v1 kind = prefix_sequence_kind_v1::graph;
    std::uint8_t reserved[3]{};
    std::uint32_t prefix_length = 0;
    std::uint32_t full_length = 0;
};

enum class prefix_composition_code_v1 : std::uint32_t {
    composed = 0,
    invalid_production,
    invalid_sequence,
    missing_entries,
    zero_logical_identity,
    unordered_logical_identity,
    position_out_of_range,
    duplicate_position,
    axis_mismatch,
    kind_mismatch,
    prefix_longer_than_full,
    missing_prefix_identity,
    prefix_position_mismatch,
    missing_workspace,
    insufficient_workspace,
    missing_output,
};

struct prefix_composition_result_v1 {
    prefix_composition_code_v1 code = prefix_composition_code_v1::composed;
    std::uint64_t subject = 0;
    [[nodiscard]] constexpr bool composed() const noexcept {
        return code == prefix_composition_code_v1::composed;
    }
};

[[nodiscard]] constexpr bool valid_prefix_kind_v1(
    prefix_sequence_kind_v1 kind) noexcept {
    return kind == prefix_sequence_kind_v1::graph
        || kind == prefix_sequence_kind_v1::trajectory;
}

[[nodiscard]] inline prefix_composition_result_v1 validate_prefix_sequence_v1(
    const prefix_sequence_view_v1 &sequence,
    std::uint8_t *position_marks,
    std::uint32_t mark_capacity) noexcept {
    if (!sequence.identity.valid() || !sequence.domain.valid()
        || !sequence.order.valid() || !valid_prefix_kind_v1(sequence.kind)) {
        return {prefix_composition_code_v1::invalid_sequence};
    }
    if (sequence.entry_count != 0 && sequence.entries == nullptr) {
        return {prefix_composition_code_v1::missing_entries};
    }
    if (sequence.entry_count != 0 && position_marks == nullptr) {
        return {prefix_composition_code_v1::missing_workspace};
    }
    if (mark_capacity < sequence.entry_count) {
        return {prefix_composition_code_v1::insufficient_workspace};
    }
    for (std::uint32_t index = 0; index < sequence.entry_count; ++index) {
        position_marks[index] = 0;
    }
    for (std::uint32_t index = 0; index < sequence.entry_count; ++index) {
        const auto &entry = sequence.entries[index];
        if (entry.logical_identity == 0 || entry.reserved != 0) {
            return {prefix_composition_code_v1::zero_logical_identity, index};
        }
        if (index != 0
            && sequence.entries[index - 1].logical_identity
                   >= entry.logical_identity) {
            return {prefix_composition_code_v1::unordered_logical_identity,
                    index};
        }
        if (entry.sequence_position >= sequence.entry_count) {
            return {prefix_composition_code_v1::position_out_of_range, index};
        }
        if (position_marks[entry.sequence_position] != 0) {
            return {prefix_composition_code_v1::duplicate_position, index};
        }
        position_marks[entry.sequence_position] = 1;
    }
    return {};
}

[[nodiscard]] inline prefix_composition_result_v1 compose_prefix_v1(
    composition_production_id production,
    const prefix_sequence_view_v1 &prefix,
    const prefix_sequence_view_v1 &full,
    std::uint8_t *prefix_position_marks,
    std::uint8_t *full_position_marks,
    std::uint32_t mark_capacity,
    prefix_composition_v1 *output) noexcept {
    if (!production.valid()) {
        return {prefix_composition_code_v1::invalid_production};
    }
    const auto prefix_status = validate_prefix_sequence_v1(
        prefix, prefix_position_marks, mark_capacity);
    if (!prefix_status.composed()) return prefix_status;
    const auto full_status = validate_prefix_sequence_v1(
        full, full_position_marks, mark_capacity);
    if (!full_status.composed()) return full_status;
    if (prefix.domain != full.domain || prefix.order != full.order) {
        return {prefix_composition_code_v1::axis_mismatch};
    }
    if (prefix.kind != full.kind) {
        return {prefix_composition_code_v1::kind_mismatch};
    }
    if (prefix.entry_count > full.entry_count) {
        return {prefix_composition_code_v1::prefix_longer_than_full};
    }
    std::uint32_t full_index = 0;
    for (std::uint32_t prefix_index = 0;
         prefix_index < prefix.entry_count;
         ++prefix_index) {
        const auto &prefix_entry = prefix.entries[prefix_index];
        while (full_index < full.entry_count
               && full.entries[full_index].logical_identity
                      < prefix_entry.logical_identity) {
            ++full_index;
        }
        if (full_index == full.entry_count
            || full.entries[full_index].logical_identity
                   != prefix_entry.logical_identity) {
            return {prefix_composition_code_v1::missing_prefix_identity,
                    prefix_entry.logical_identity};
        }
        if (full.entries[full_index].sequence_position
                != prefix_entry.sequence_position
            || prefix_entry.sequence_position >= prefix.entry_count) {
            return {prefix_composition_code_v1::prefix_position_mismatch,
                    prefix_entry.logical_identity};
        }
    }
    if (output == nullptr) return {prefix_composition_code_v1::missing_output};
    *output = {production, prefix.identity, full.identity, prefix.domain,
               prefix.order, prefix.kind, {}, prefix.entry_count,
               full.entry_count};
    return {prefix_composition_code_v1::composed, prefix.entry_count};
}

static_assert(std::is_trivially_copyable<prefix_sequence_entry_v1>::value);
static_assert(std::is_trivially_copyable<prefix_composition_v1>::value);

} // namespace cellshard::compiler::composition
