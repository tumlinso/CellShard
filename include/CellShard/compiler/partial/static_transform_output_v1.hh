#pragma once

#include <CellShard/compiler/partial/transform_partial_v1.hh>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace cellshard::compiler::partial {

inline constexpr std::uint32_t static_transform_output_schema_version_v1 = 1;

struct static_transform_output_view_v1 {
    const double *values = nullptr;
    std::uint64_t value_count = 0;
    atom_persistent_identity_v1 source_content_identity{};
    atom_persistent_identity_v1 transform_identity{};
    atom_persistent_identity_v1 source_order_identity{};
    atom_persistent_identity_v1 destination_order_identity{};
    atom_persistent_identity_v1 numerical_policy_identity{};
    std::uint64_t structure_generation = 0;
    std::uint64_t source_value_generation = 0;
    std::uint64_t output_generation = 0;
    std::uint32_t schema_version = static_transform_output_schema_version_v1;
    std::uint32_t reserved = 0;
};

enum class static_transform_output_code_v1 : std::uint32_t {
    valid = 0,
    unsupported_schema,
    invalid_identity,
    missing_generation,
    missing_values,
    value_count_mismatch,
    nonfinite_value,
    nonzero_reserved,
    invalid_transform,
    transform_binding_mismatch,
    source_extent_mismatch,
    capacity_overflow,
};

struct static_transform_output_result_v1 {
    static_transform_output_code_v1 code =
        static_transform_output_code_v1::valid;
    std::uint64_t index = 0;
    [[nodiscard]] constexpr bool valid() const noexcept {
        return code == static_transform_output_code_v1::valid;
    }
};

static_assert(offsetof(static_transform_output_view_v1, values) == 0);
static_assert(std::is_standard_layout<static_transform_output_view_v1>::value);
static_assert(std::is_trivially_copyable<static_transform_output_view_v1>::value);

[[nodiscard]] inline static_transform_output_result_v1
validate_static_transform_output_v1(
    const static_transform_output_view_v1 &output,
    const transform_partial_view_v1 &transform,
    std::uint8_t *destination_marks,
    std::uint64_t mark_capacity) noexcept {
    if (output.schema_version != static_transform_output_schema_version_v1) {
        return {static_transform_output_code_v1::unsupported_schema, 0};
    }
    if (!atom::validate_atom_persistent_identity_v1(output.source_content_identity)
             .valid()
        || !atom::validate_atom_persistent_identity_v1(output.transform_identity)
                .valid()
        || !atom::validate_atom_persistent_identity_v1(output.source_order_identity)
                .valid()
        || !atom::validate_atom_persistent_identity_v1(
                output.destination_order_identity)
                .valid()
        || !atom::validate_atom_persistent_identity_v1(
                output.numerical_policy_identity)
                .valid()) {
        return {static_transform_output_code_v1::invalid_identity, 0};
    }
    if (output.structure_generation == 0 || output.source_value_generation == 0
        || output.output_generation == 0) {
        return {static_transform_output_code_v1::missing_generation, 0};
    }
    const auto transform_result = validate_transform_partial_v1(
        transform, destination_marks, mark_capacity);
    if (!transform_result.valid()) {
        return {static_transform_output_code_v1::invalid_transform,
                transform_result.index};
    }
    if (output.transform_identity != transform.transform_identity
        || output.source_order_identity != transform.source_order_identity
        || output.destination_order_identity != transform.destination_order_identity
        || output.structure_generation != transform.structure_generation) {
        return {static_transform_output_code_v1::transform_binding_mismatch, 0};
    }
    if (output.values == nullptr) return {static_transform_output_code_v1::missing_values, 0};
    if (output.value_count != transform.destination_extent) {
        return {static_transform_output_code_v1::value_count_mismatch, 0};
    }
    for (std::uint64_t index = 0; index < output.value_count; ++index) {
        if (!std::isfinite(output.values[index])) {
            return {static_transform_output_code_v1::nonfinite_value, index};
        }
    }
    return output.reserved == 0
        ? static_transform_output_result_v1{static_transform_output_code_v1::valid,
                                             output.value_count}
        : static_transform_output_result_v1{
              static_transform_output_code_v1::nonzero_reserved, 0};
}

[[nodiscard]] inline static_transform_output_result_v1
materialize_static_transform_output_v1(
    const double *source_values, std::uint64_t source_count,
    const transform_partial_view_v1 &transform, double *output,
    std::uint64_t output_capacity) noexcept {
    if (source_values == nullptr || source_count != transform.source_extent) {
        return {static_transform_output_code_v1::source_extent_mismatch, 0};
    }
    if (output == nullptr || output_capacity < transform.destination_extent) {
        return {static_transform_output_code_v1::capacity_overflow, 0};
    }
    for (std::uint64_t index = 0; index < transform.index_count; ++index) {
        const auto mapping = transform.indices[index];
        if (mapping.source_ordinal >= source_count
            || mapping.destination_ordinal >= transform.destination_extent) {
            return {static_transform_output_code_v1::invalid_transform, index};
        }
        output[mapping.destination_ordinal] = source_values[mapping.source_ordinal];
    }
    return {static_transform_output_code_v1::valid,
            transform.destination_extent};
}

} // namespace cellshard::compiler::partial
