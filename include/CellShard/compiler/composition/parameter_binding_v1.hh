#pragma once

#include <CellShard/compiler/composition/production_identity_v1.hh>

#include <cstdint>
#include <type_traits>

namespace cellshard::compiler::composition {

inline constexpr std::uint32_t max_composition_parameters_v1 = 64;

enum class parameter_kind_v1 : std::uint8_t {
    scalar = 1,
    value_plane = 2,
    state_plane = 3,
};

struct parameter_signature_v1 {
    std::uint64_t parameter_identity = 0;
    parameter_kind_v1 kind = parameter_kind_v1::scalar;
    std::uint8_t reserved[7]{};
    scalar_encoding_id encoding{};
    std::uint64_t element_count = 0;
};

struct parameter_binding_v1 {
    std::uint64_t parameter_identity = 0;
    std::uint64_t binding_identity = 0;
    parameter_kind_v1 kind = parameter_kind_v1::scalar;
    std::uint8_t reserved[7]{};
    scalar_encoding_id encoding{};
    std::uint64_t generation = 0;
    std::uint64_t element_count = 0;
};

struct parameter_binding_composition_v1 {
    composition_production_id production{};
    const parameter_binding_v1 *bindings = nullptr;
    std::uint32_t binding_count = 0;
    std::uint32_t reserved = 0;
};

enum class parameter_binding_code_v1 : std::uint32_t {
    bound = 0,
    invalid_production,
    invalid_count,
    missing_signatures,
    missing_bindings,
    invalid_signature,
    invalid_binding,
    unordered_parameter_identity,
    parameter_identity_mismatch,
    kind_mismatch,
    encoding_mismatch,
    element_count_mismatch,
    missing_output,
};

struct parameter_binding_result_v1 {
    parameter_binding_code_v1 code = parameter_binding_code_v1::bound;
    std::uint32_t parameter_index = 0;
    [[nodiscard]] constexpr bool bound() const noexcept {
        return code == parameter_binding_code_v1::bound;
    }
};

[[nodiscard]] constexpr bool valid_parameter_kind_v1(
    parameter_kind_v1 kind) noexcept {
    return kind == parameter_kind_v1::scalar
        || kind == parameter_kind_v1::value_plane
        || kind == parameter_kind_v1::state_plane;
}

[[nodiscard]] inline parameter_binding_result_v1
compose_parameter_bindings_v1(
    composition_production_id production,
    const parameter_signature_v1 *signatures,
    const parameter_binding_v1 *bindings,
    std::uint32_t parameter_count,
    parameter_binding_composition_v1 *output) noexcept {
    if (!production.valid()) {
        return {parameter_binding_code_v1::invalid_production};
    }
    if (parameter_count == 0
        || parameter_count > max_composition_parameters_v1) {
        return {parameter_binding_code_v1::invalid_count};
    }
    if (signatures == nullptr) {
        return {parameter_binding_code_v1::missing_signatures};
    }
    if (bindings == nullptr) {
        return {parameter_binding_code_v1::missing_bindings};
    }
    for (std::uint32_t index = 0; index < parameter_count; ++index) {
        const auto &signature = signatures[index];
        const auto &binding = bindings[index];
        if (signature.parameter_identity == 0
            || !valid_parameter_kind_v1(signature.kind)
            || !signature.encoding.valid() || signature.element_count == 0) {
            return {parameter_binding_code_v1::invalid_signature, index};
        }
        if (binding.parameter_identity == 0 || binding.binding_identity == 0
            || !valid_parameter_kind_v1(binding.kind)
            || !binding.encoding.valid() || binding.generation == 0
            || binding.element_count == 0) {
            return {parameter_binding_code_v1::invalid_binding, index};
        }
        if (index != 0
            && (signatures[index - 1].parameter_identity
                    >= signature.parameter_identity
                || bindings[index - 1].parameter_identity
                    >= binding.parameter_identity)) {
            return {parameter_binding_code_v1::unordered_parameter_identity,
                    index};
        }
        if (signature.parameter_identity != binding.parameter_identity) {
            return {parameter_binding_code_v1::parameter_identity_mismatch,
                    index};
        }
        if (signature.kind != binding.kind) {
            return {parameter_binding_code_v1::kind_mismatch, index};
        }
        if (signature.encoding != binding.encoding) {
            return {parameter_binding_code_v1::encoding_mismatch, index};
        }
        if (signature.element_count != binding.element_count) {
            return {parameter_binding_code_v1::element_count_mismatch, index};
        }
        for (const auto byte : signature.reserved) {
            if (byte != 0) {
                return {parameter_binding_code_v1::invalid_signature, index};
            }
        }
        for (const auto byte : binding.reserved) {
            if (byte != 0) {
                return {parameter_binding_code_v1::invalid_binding, index};
            }
        }
    }
    if (output == nullptr) return {parameter_binding_code_v1::missing_output};
    *output = {production, bindings, parameter_count, 0};
    return {parameter_binding_code_v1::bound, parameter_count};
}

static_assert(std::is_trivially_copyable<parameter_signature_v1>::value);
static_assert(std::is_trivially_copyable<parameter_binding_v1>::value);
static_assert(
    std::is_trivially_copyable<parameter_binding_composition_v1>::value);

} // namespace cellshard::compiler::composition
