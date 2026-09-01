#pragma once

#include <CellShard/compiler/partial/partial_atom_v1.hh>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace cellshard::compiler::partial {

inline constexpr std::uint32_t parameterized_function_schema_version_v1 = 1;

struct parameterized_partial_function_view_v1 {
    const void *function_state = nullptr;
    std::uint64_t function_state_bytes = 0;
    std::uint32_t function_state_alignment = 0;
    std::uint32_t reserved = 0;
    atom_persistent_identity_v1 function_identity{};
    atom_persistent_identity_v1 function_state_schema_identity{};
    atom_persistent_identity_v1 parameter_set_identity{};
    atom_persistent_identity_v1 parameter_content_identity{};
    atom_persistent_identity_v1 input_space_identity{};
    atom_persistent_identity_v1 output_space_identity{};
    atom_persistent_identity_v1 algebra_identity{};
    atom_persistent_identity_v1 numerical_policy_identity{};
    std::uint64_t structure_generation = 0;
    std::uint64_t input_value_generation = 0;
    std::uint64_t parameter_generation = 0;
    std::uint64_t function_state_generation = 0;
    std::uint32_t schema_version = parameterized_function_schema_version_v1;
    std::uint32_t trailing_reserved = 0;
};

struct parameter_binding_v1 {
    atom_persistent_identity_v1 parameter_set_identity{};
    atom_persistent_identity_v1 parameter_content_identity{};
    std::uint64_t parameter_generation = 0;
};

enum class parameterized_function_code_v1 : std::uint32_t {
    valid = 0,
    unsupported_schema,
    invalid_identity,
    missing_generation,
    missing_state,
    invalid_alignment,
    misaligned_state,
    nonzero_reserved,
    parameter_set_mismatch,
    parameter_content_mismatch,
    parameter_generation_mismatch,
};

struct parameterized_function_result_v1 {
    parameterized_function_code_v1 code =
        parameterized_function_code_v1::valid;
    [[nodiscard]] constexpr bool valid() const noexcept {
        return code == parameterized_function_code_v1::valid;
    }
};

static_assert(offsetof(parameterized_partial_function_view_v1, function_state)
              == 0);
static_assert(std::is_standard_layout<parameterized_partial_function_view_v1>::value);
static_assert(std::is_trivially_copyable<parameterized_partial_function_view_v1>::value);

[[nodiscard]] inline parameterized_function_result_v1
validate_parameterized_partial_function_v1(
    const parameterized_partial_function_view_v1 &function) noexcept {
    if (function.schema_version != parameterized_function_schema_version_v1) {
        return {parameterized_function_code_v1::unsupported_schema};
    }
#define CELLSHARD_PARAMETER_FUNCTION_ID(field) \
    if (!atom::validate_atom_persistent_identity_v1(function.field).valid()) \
        return {parameterized_function_code_v1::invalid_identity}
    CELLSHARD_PARAMETER_FUNCTION_ID(function_identity);
    CELLSHARD_PARAMETER_FUNCTION_ID(function_state_schema_identity);
    CELLSHARD_PARAMETER_FUNCTION_ID(parameter_set_identity);
    CELLSHARD_PARAMETER_FUNCTION_ID(parameter_content_identity);
    CELLSHARD_PARAMETER_FUNCTION_ID(input_space_identity);
    CELLSHARD_PARAMETER_FUNCTION_ID(output_space_identity);
    CELLSHARD_PARAMETER_FUNCTION_ID(algebra_identity);
    CELLSHARD_PARAMETER_FUNCTION_ID(numerical_policy_identity);
#undef CELLSHARD_PARAMETER_FUNCTION_ID
    if (function.structure_generation == 0
        || function.input_value_generation == 0
        || function.parameter_generation == 0
        || function.function_state_generation == 0) {
        return {parameterized_function_code_v1::missing_generation};
    }
    if (function.function_state == nullptr || function.function_state_bytes == 0) {
        return {parameterized_function_code_v1::missing_state};
    }
    if (function.function_state_alignment == 0
        || (function.function_state_alignment
            & (function.function_state_alignment - 1)) != 0) {
        return {parameterized_function_code_v1::invalid_alignment};
    }
    if (reinterpret_cast<std::uintptr_t>(function.function_state)
        % function.function_state_alignment != 0) {
        return {parameterized_function_code_v1::misaligned_state};
    }
    return function.reserved == 0 && function.trailing_reserved == 0
        ? parameterized_function_result_v1{}
        : parameterized_function_result_v1{
              parameterized_function_code_v1::nonzero_reserved};
}

// Only an exact identity, content and generation triple authorizes reuse.
[[nodiscard]] inline parameterized_function_result_v1
validate_parameter_binding_v1(
    const parameterized_partial_function_view_v1 &function,
    const parameter_binding_v1 &binding) noexcept {
    const auto result = validate_parameterized_partial_function_v1(function);
    if (!result.valid()) return result;
    if (binding.parameter_set_identity != function.parameter_set_identity) {
        return {parameterized_function_code_v1::parameter_set_mismatch};
    }
    if (binding.parameter_content_identity != function.parameter_content_identity) {
        return {parameterized_function_code_v1::parameter_content_mismatch};
    }
    if (binding.parameter_generation != function.parameter_generation
        || binding.parameter_generation == 0) {
        return {parameterized_function_code_v1::parameter_generation_mismatch};
    }
    return {};
}

} // namespace cellshard::compiler::partial
