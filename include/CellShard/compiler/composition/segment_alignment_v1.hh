#pragma once

#include <CellShard/domain.hh>

#include <cstdint>
#include <limits>
#include <type_traits>

namespace cellshard::compiler::composition {

struct exact_segment_v1 {
    std::uint64_t segment_identity = 0;
    std::uint64_t logical_item_count = 0;
};

struct segment_partition_view_v1 {
    structure_id identity{};
    domain_id domain{};
    order_id order{};
    const exact_segment_v1 *segments = nullptr;
    std::uint64_t segment_count = 0;
};

struct segment_alignment_entry_v1 {
    std::uint64_t segment_identity = 0;
    std::uint64_t logical_item_count = 0;
    std::uint32_t left_local_segment = 0;
    std::uint32_t right_local_segment = 0;
};

enum class segment_alignment_code_v1 : std::uint32_t {
    aligned = 0,
    invalid_output_identity,
    invalid_partition_identity,
    invalid_axis_identity,
    excessive_local_width,
    missing_segments,
    invalid_segment,
    unordered_segment,
    domain_mismatch,
    segment_count_mismatch,
    segment_identity_mismatch,
    segment_extent_mismatch,
    missing_storage,
    insufficient_capacity,
    missing_output,
};

struct segment_alignment_result_v1 {
    segment_alignment_code_v1 code = segment_alignment_code_v1::aligned;
    std::uint64_t segment = 0;
    [[nodiscard]] constexpr bool aligned() const noexcept {
        return code == segment_alignment_code_v1::aligned;
    }
};

struct segment_alignment_view_v1 {
    structure_id identity{};
    domain_id domain{};
    order_id left_order{};
    order_id right_order{};
    const segment_alignment_entry_v1 *entries = nullptr;
    std::uint32_t segment_count = 0;
    std::uint32_t reserved = 0;
};

[[nodiscard]] inline segment_alignment_result_v1
validate_segment_partition_v1(const segment_partition_view_v1 &partition) noexcept {
    if (!partition.identity.valid()) {
        return {segment_alignment_code_v1::invalid_partition_identity};
    }
    if (!partition.domain.valid() || !partition.order.valid()) {
        return {segment_alignment_code_v1::invalid_axis_identity};
    }
    if (partition.segment_count > std::numeric_limits<std::uint32_t>::max()) {
        return {segment_alignment_code_v1::excessive_local_width};
    }
    if (partition.segment_count != 0 && partition.segments == nullptr) {
        return {segment_alignment_code_v1::missing_segments};
    }
    for (std::uint64_t index = 0; index < partition.segment_count; ++index) {
        if (partition.segments[index].segment_identity == 0
            || partition.segments[index].logical_item_count == 0) {
            return {segment_alignment_code_v1::invalid_segment, index};
        }
        if (index != 0
            && partition.segments[index - 1].segment_identity
                   >= partition.segments[index].segment_identity) {
            return {segment_alignment_code_v1::unordered_segment, index};
        }
    }
    return {};
}

[[nodiscard]] inline segment_alignment_result_v1
compose_segment_alignment_v1(
    structure_id output_identity,
    const segment_partition_view_v1 &left,
    const segment_partition_view_v1 &right,
    segment_alignment_entry_v1 *storage,
    std::uint32_t capacity,
    segment_alignment_view_v1 *output) noexcept {
    if (!output_identity.valid()) {
        return {segment_alignment_code_v1::invalid_output_identity};
    }
    const auto left_status = validate_segment_partition_v1(left);
    if (!left_status.aligned()) return left_status;
    const auto right_status = validate_segment_partition_v1(right);
    if (!right_status.aligned()) return right_status;
    if (left.domain != right.domain) {
        return {segment_alignment_code_v1::domain_mismatch};
    }
    if (left.segment_count != right.segment_count) {
        return {segment_alignment_code_v1::segment_count_mismatch};
    }
    if (left.segment_count != 0 && storage == nullptr) {
        return {segment_alignment_code_v1::missing_storage};
    }
    if (capacity < left.segment_count) {
        return {segment_alignment_code_v1::insufficient_capacity};
    }
    if (output == nullptr) return {segment_alignment_code_v1::missing_output};
    *output = {};
    for (std::uint32_t index = 0; index < left.segment_count; ++index) {
        const auto &left_segment = left.segments[index];
        const auto &right_segment = right.segments[index];
        if (left_segment.segment_identity != right_segment.segment_identity) {
            return {segment_alignment_code_v1::segment_identity_mismatch,
                    left_segment.segment_identity};
        }
        if (left_segment.logical_item_count
            != right_segment.logical_item_count) {
            return {segment_alignment_code_v1::segment_extent_mismatch,
                    left_segment.segment_identity};
        }
        storage[index] = {left_segment.segment_identity,
                          left_segment.logical_item_count, index, index};
    }
    *output = {output_identity, left.domain, left.order, right.order,
               storage, static_cast<std::uint32_t>(left.segment_count), 0};
    return {segment_alignment_code_v1::aligned, left.segment_count};
}

static_assert(std::is_trivially_copyable<exact_segment_v1>::value);
static_assert(std::is_trivially_copyable<segment_alignment_entry_v1>::value);
static_assert(std::is_trivially_copyable<segment_alignment_view_v1>::value);

} // namespace cellshard::compiler::composition
