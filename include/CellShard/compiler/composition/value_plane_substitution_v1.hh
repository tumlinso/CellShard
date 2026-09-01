#pragma once

#include <CellShard/compiler/composition/production_identity_v1.hh>

#include <cstdint>
#include <type_traits>

namespace cellshard::compiler::composition {

struct value_plane_tag {};
using value_plane_id = strong_id<value_plane_tag>;

struct value_plane_identity_v1 {
    value_plane_id identity{};
    structure_id structure{};
    order_id logical_order{};
    scalar_encoding_id encoding{};
    std::uint64_t generation = 0;
    std::uint64_t logical_value_count = 0;
};

struct value_plane_substitution_v1 {
    composition_production_id production{};
    value_plane_identity_v1 previous{};
    value_plane_identity_v1 replacement{};
};

enum class value_plane_substitution_code_v1 : std::uint32_t {
    substituted = 0,
    invalid_production,
    invalid_plane_identity,
    invalid_structure,
    invalid_order,
    invalid_encoding,
    missing_generation,
    empty_value_plane,
    structure_mismatch,
    order_mismatch,
    encoding_mismatch,
    value_count_mismatch,
    stale_replacement_generation,
    same_plane_identity,
    missing_output,
};

struct value_plane_substitution_result_v1 {
    value_plane_substitution_code_v1 code =
        value_plane_substitution_code_v1::substituted;
    [[nodiscard]] constexpr bool substituted() const noexcept {
        return code == value_plane_substitution_code_v1::substituted;
    }
};

[[nodiscard]] constexpr value_plane_substitution_result_v1
validate_value_plane_identity_v1(
    const value_plane_identity_v1 &plane) noexcept {
    if (!plane.identity.valid()) {
        return {value_plane_substitution_code_v1::invalid_plane_identity};
    }
    if (!plane.structure.valid()) {
        return {value_plane_substitution_code_v1::invalid_structure};
    }
    if (!plane.logical_order.valid()) {
        return {value_plane_substitution_code_v1::invalid_order};
    }
    if (!plane.encoding.valid()) {
        return {value_plane_substitution_code_v1::invalid_encoding};
    }
    if (plane.generation == 0) {
        return {value_plane_substitution_code_v1::missing_generation};
    }
    if (plane.logical_value_count == 0) {
        return {value_plane_substitution_code_v1::empty_value_plane};
    }
    return {};
}

[[nodiscard]] constexpr value_plane_substitution_result_v1
compose_value_plane_substitution_v1(
    composition_production_id production,
    const value_plane_identity_v1 &previous,
    const value_plane_identity_v1 &replacement,
    value_plane_substitution_v1 *output) noexcept {
    if (!production.valid()) {
        return {value_plane_substitution_code_v1::invalid_production};
    }
    const auto previous_status = validate_value_plane_identity_v1(previous);
    if (!previous_status.substituted()) return previous_status;
    const auto replacement_status = validate_value_plane_identity_v1(replacement);
    if (!replacement_status.substituted()) return replacement_status;
    if (previous.structure != replacement.structure) {
        return {value_plane_substitution_code_v1::structure_mismatch};
    }
    if (previous.logical_order != replacement.logical_order) {
        return {value_plane_substitution_code_v1::order_mismatch};
    }
    if (previous.encoding != replacement.encoding) {
        return {value_plane_substitution_code_v1::encoding_mismatch};
    }
    if (previous.logical_value_count != replacement.logical_value_count) {
        return {value_plane_substitution_code_v1::value_count_mismatch};
    }
    if (replacement.generation <= previous.generation) {
        return {value_plane_substitution_code_v1::
                    stale_replacement_generation};
    }
    if (previous.identity == replacement.identity) {
        return {value_plane_substitution_code_v1::same_plane_identity};
    }
    if (output == nullptr) {
        return {value_plane_substitution_code_v1::missing_output};
    }
    *output = {production, previous, replacement};
    return {};
}

static_assert(std::is_trivially_copyable<value_plane_identity_v1>::value);
static_assert(std::is_trivially_copyable<value_plane_substitution_v1>::value);

} // namespace cellshard::compiler::composition
