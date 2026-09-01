#pragma once

#include <CellShard/compiler/partial/partial_atom_v1.hh>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace cellshard::compiler::partial {

inline constexpr std::uint32_t gathered_panel_schema_version_v1 = 1;

struct gathered_axis_item_v1 {
    atom_persistent_identity_v1 biological_identity{};
    std::uint64_t canonical_ordinal = 0;
};

struct gathered_panel_view_v1 {
    const double *values = nullptr;
    std::uint64_t value_count = 0;
    const gathered_axis_item_v1 *rows = nullptr;
    std::uint64_t row_count = 0;
    const gathered_axis_item_v1 *columns = nullptr;
    std::uint64_t column_count = 0;
    atom_persistent_identity_v1 row_order_identity{};
    atom_persistent_identity_v1 column_order_identity{};
    atom_persistent_identity_v1 numerical_policy_identity{};
    std::uint64_t structure_generation = 0;
    std::uint64_t value_generation = 0;
    std::uint32_t schema_version = gathered_panel_schema_version_v1;
    std::uint32_t reserved = 0;
};

enum class gathered_panel_code_v1 : std::uint32_t {
    valid = 0,
    unsupported_schema,
    invalid_identity,
    missing_generation,
    missing_axis,
    extent_overflow,
    value_count_mismatch,
    missing_values,
    unordered_or_duplicate_axis,
    nonfinite_value,
    nonzero_reserved,
    source_out_of_bounds,
    capacity_overflow,
};

struct gathered_panel_result_v1 {
    gathered_panel_code_v1 code = gathered_panel_code_v1::valid;
    std::uint64_t index = 0;
    [[nodiscard]] constexpr bool valid() const noexcept {
        return code == gathered_panel_code_v1::valid;
    }
};

static_assert(offsetof(gathered_panel_view_v1, values) == 0);
static_assert(std::is_standard_layout<gathered_axis_item_v1>::value);
static_assert(std::is_trivially_copyable<gathered_axis_item_v1>::value);

[[nodiscard]] constexpr bool gathered_axis_item_less_v1(
    const gathered_axis_item_v1 &left,
    const gathered_axis_item_v1 &right) noexcept {
    return atom::atom_persistent_identity_less_v1(
               left.biological_identity, right.biological_identity)
        || (left.biological_identity == right.biological_identity
            && left.canonical_ordinal < right.canonical_ordinal);
}

[[nodiscard]] inline gathered_panel_result_v1 validate_gathered_panel_v1(
    const gathered_panel_view_v1 &panel) noexcept {
    if (panel.schema_version != gathered_panel_schema_version_v1) {
        return {gathered_panel_code_v1::unsupported_schema, 0};
    }
    if (!atom::validate_atom_persistent_identity_v1(panel.row_order_identity)
             .valid()
        || !atom::validate_atom_persistent_identity_v1(
                panel.column_order_identity)
                .valid()
        || !atom::validate_atom_persistent_identity_v1(
                panel.numerical_policy_identity)
                .valid()) {
        return {gathered_panel_code_v1::invalid_identity, 0};
    }
    if (panel.structure_generation == 0 || panel.value_generation == 0) {
        return {gathered_panel_code_v1::missing_generation, 0};
    }
    if (panel.row_count == 0 || panel.column_count == 0
        || panel.rows == nullptr || panel.columns == nullptr) {
        return {gathered_panel_code_v1::missing_axis, 0};
    }
    if (panel.row_count
        > std::numeric_limits<std::uint64_t>::max() / panel.column_count) {
        return {gathered_panel_code_v1::extent_overflow, 0};
    }
    if (panel.value_count != panel.row_count * panel.column_count) {
        return {gathered_panel_code_v1::value_count_mismatch, 0};
    }
    if (panel.values == nullptr) return {gathered_panel_code_v1::missing_values, 0};
    for (std::uint64_t index = 0; index < panel.row_count; ++index) {
        if (!atom::validate_atom_persistent_identity_v1(
                 panel.rows[index].biological_identity)
                 .valid()) return {gathered_panel_code_v1::invalid_identity, index};
        if (index != 0 && !gathered_axis_item_less_v1(panel.rows[index - 1], panel.rows[index]))
            return {gathered_panel_code_v1::unordered_or_duplicate_axis, index};
    }
    for (std::uint64_t index = 0; index < panel.column_count; ++index) {
        if (!atom::validate_atom_persistent_identity_v1(
                 panel.columns[index].biological_identity)
                 .valid()) return {gathered_panel_code_v1::invalid_identity, index};
        if (index != 0 && !gathered_axis_item_less_v1(panel.columns[index - 1], panel.columns[index]))
            return {gathered_panel_code_v1::unordered_or_duplicate_axis, index};
    }
    for (std::uint64_t index = 0; index < panel.value_count; ++index) {
        if (!std::isfinite(panel.values[index])) {
            return {gathered_panel_code_v1::nonfinite_value, index};
        }
    }
    return panel.reserved == 0
        ? gathered_panel_result_v1{gathered_panel_code_v1::valid, panel.value_count}
        : gathered_panel_result_v1{gathered_panel_code_v1::nonzero_reserved, 0};
}

// Cold exact gather into caller-owned row-major storage.
[[nodiscard]] inline gathered_panel_result_v1 gather_panel_values_v1(
    const double *source, std::uint64_t source_rows,
    std::uint64_t source_columns, std::uint64_t source_row_stride,
    const gathered_axis_item_v1 *rows, std::uint64_t row_count,
    const gathered_axis_item_v1 *columns, std::uint64_t column_count,
    double *output, std::uint64_t output_capacity) noexcept {
    if (source == nullptr || rows == nullptr || columns == nullptr
        || output == nullptr || row_count == 0 || column_count == 0
        || row_count > std::numeric_limits<std::uint64_t>::max() / column_count
        || output_capacity < row_count * column_count) {
        return {gathered_panel_code_v1::capacity_overflow, 0};
    }
    for (std::uint64_t row = 0; row < row_count; ++row) {
        if (rows[row].canonical_ordinal >= source_rows) {
            return {gathered_panel_code_v1::source_out_of_bounds, row};
        }
        for (std::uint64_t column = 0; column < column_count; ++column) {
            if (columns[column].canonical_ordinal >= source_columns) {
                return {gathered_panel_code_v1::source_out_of_bounds, column};
            }
            output[row * column_count + column] =
                source[rows[row].canonical_ordinal * source_row_stride
                       + columns[column].canonical_ordinal];
        }
    }
    return {gathered_panel_code_v1::valid, row_count * column_count};
}

} // namespace cellshard::compiler::partial
