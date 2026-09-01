#pragma once

#include <CellShard/compiler/composition/production_identity_v1.hh>
#include <CellShard/domain.hh>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace cellshard::compiler::composition {

inline constexpr std::uint32_t max_atom_interface_ports_v1 = 64;

struct grammar_symbol_tag {};
struct atom_interface_tag {};
using grammar_symbol_id = strong_id<grammar_symbol_tag>;
using atom_interface_id = strong_id<atom_interface_tag>;

enum class grammar_symbol_kind_v1 : std::uint8_t {
    terminal_atom = 1,
    nonterminal_composition = 2,
};

enum class atom_port_direction_v1 : std::uint8_t {
    input = 1,
    output = 2,
    input_output = 3,
};

enum class atom_port_kind_v1 : std::uint8_t {
    immutable_structure = 1,
    mutable_value = 2,
    mutable_state = 3,
    materialization = 4,
    partial_result = 5,
};

struct atom_port_signature_v1 {
    std::uint64_t port_identity = 0;
    atom_port_direction_v1 direction = atom_port_direction_v1::input;
    atom_port_kind_v1 kind = atom_port_kind_v1::immutable_structure;
    std::uint16_t reserved0 = 0;
    domain_id domain{};
    order_id order{};
    structure_id relation{};
    scalar_encoding_id encoding{};
};

// Ports are sorted by globally stable port identity within each array. Arrays
// are caller-owned cold metadata; runtime pointers and locations are absent.
struct atom_interface_signature_v1 {
    atom_interface_id identity{};
    const atom_port_signature_v1 *inputs = nullptr;
    const atom_port_signature_v1 *outputs = nullptr;
    std::uint32_t input_count = 0;
    std::uint32_t output_count = 0;
};

struct typed_grammar_symbol_v1 {
    grammar_symbol_id identity{};
    grammar_symbol_kind_v1 kind = grammar_symbol_kind_v1::terminal_atom;
    std::uint8_t reserved[7]{};
    atom_interface_id interface_identity{};
    composition_lineage_id lineage{};
};

enum class grammar_signature_code_v1 : std::uint32_t {
    valid = 0,
    invalid_interface_identity,
    excessive_port_count,
    missing_inputs,
    missing_outputs,
    invalid_port_identity,
    unordered_port_identity,
    duplicate_port_identity,
    invalid_port_direction,
    direction_array_mismatch,
    invalid_port_kind,
    invalid_domain,
    invalid_order,
    invalid_relation,
    invalid_encoding,
    unexpected_encoding,
    invalid_symbol_identity,
    invalid_symbol_kind,
    nonzero_reserved,
    invalid_symbol_interface,
    invalid_symbol_lineage,
};

struct grammar_signature_status_v1 {
    grammar_signature_code_v1 code = grammar_signature_code_v1::valid;
    std::uint32_t port_index = 0;
    [[nodiscard]] constexpr bool valid() const noexcept {
        return code == grammar_signature_code_v1::valid;
    }
};

[[nodiscard]] constexpr bool valid_atom_port_kind_v1(
    atom_port_kind_v1 kind) noexcept {
    return kind == atom_port_kind_v1::immutable_structure
        || kind == atom_port_kind_v1::mutable_value
        || kind == atom_port_kind_v1::mutable_state
        || kind == atom_port_kind_v1::materialization
        || kind == atom_port_kind_v1::partial_result;
}

[[nodiscard]] constexpr bool valid_atom_port_direction_v1(
    atom_port_direction_v1 direction) noexcept {
    return direction == atom_port_direction_v1::input
        || direction == atom_port_direction_v1::output
        || direction == atom_port_direction_v1::input_output;
}

[[nodiscard]] constexpr grammar_signature_status_v1 validate_port_v1(
    const atom_port_signature_v1 &port,
    bool input_array,
    std::uint32_t index) noexcept {
    if (port.port_identity == 0) {
        return {grammar_signature_code_v1::invalid_port_identity, index};
    }
    if (!valid_atom_port_direction_v1(port.direction)) {
        return {grammar_signature_code_v1::invalid_port_direction, index};
    }
    if ((input_array && port.direction == atom_port_direction_v1::output)
        || (!input_array && port.direction == atom_port_direction_v1::input)) {
        return {grammar_signature_code_v1::direction_array_mismatch, index};
    }
    if (!valid_atom_port_kind_v1(port.kind)) {
        return {grammar_signature_code_v1::invalid_port_kind, index};
    }
    if (!port.domain.valid()) {
        return {grammar_signature_code_v1::invalid_domain, index};
    }
    if (!port.order.valid()) {
        return {grammar_signature_code_v1::invalid_order, index};
    }
    if (!port.relation.valid()) {
        return {grammar_signature_code_v1::invalid_relation, index};
    }
    const bool numerical = port.kind == atom_port_kind_v1::mutable_value
        || port.kind == atom_port_kind_v1::mutable_state
        || port.kind == atom_port_kind_v1::partial_result;
    if (numerical && !port.encoding.valid()) {
        return {grammar_signature_code_v1::invalid_encoding, index};
    }
    if (!numerical && port.encoding.valid()) {
        return {grammar_signature_code_v1::unexpected_encoding, index};
    }
    if (port.reserved0 != 0) {
        return {grammar_signature_code_v1::nonzero_reserved, index};
    }
    return {};
}

[[nodiscard]] inline grammar_signature_status_v1
validate_atom_interface_signature_v1(
    const atom_interface_signature_v1 &signature) noexcept {
    if (!signature.identity.valid()) {
        return {grammar_signature_code_v1::invalid_interface_identity};
    }
    if (signature.input_count > max_atom_interface_ports_v1
        || signature.output_count > max_atom_interface_ports_v1) {
        return {grammar_signature_code_v1::excessive_port_count};
    }
    if (signature.input_count != 0 && signature.inputs == nullptr) {
        return {grammar_signature_code_v1::missing_inputs};
    }
    if (signature.output_count == 0 || signature.outputs == nullptr) {
        return {grammar_signature_code_v1::missing_outputs};
    }
    for (std::uint32_t index = 0; index < signature.input_count; ++index) {
        const auto status = validate_port_v1(signature.inputs[index], true, index);
        if (!status.valid()) return status;
        if (index != 0
            && signature.inputs[index - 1].port_identity
                   >= signature.inputs[index].port_identity) {
            return {grammar_signature_code_v1::unordered_port_identity, index};
        }
    }
    for (std::uint32_t index = 0; index < signature.output_count; ++index) {
        const auto status =
            validate_port_v1(signature.outputs[index], false, index);
        if (!status.valid()) return status;
        if (index != 0
            && signature.outputs[index - 1].port_identity
                   >= signature.outputs[index].port_identity) {
            return {grammar_signature_code_v1::unordered_port_identity, index};
        }
    }
    std::uint32_t input = 0;
    std::uint32_t output = 0;
    while (input < signature.input_count && output < signature.output_count) {
        const auto input_id = signature.inputs[input].port_identity;
        const auto output_id = signature.outputs[output].port_identity;
        if (input_id == output_id) {
            return {grammar_signature_code_v1::duplicate_port_identity, input};
        }
        if (input_id < output_id) ++input;
        else ++output;
    }
    return {};
}

[[nodiscard]] constexpr grammar_signature_status_v1
validate_typed_grammar_symbol_v1(
    const typed_grammar_symbol_v1 &symbol) noexcept {
    if (!symbol.identity.valid()) {
        return {grammar_signature_code_v1::invalid_symbol_identity};
    }
    if (symbol.kind != grammar_symbol_kind_v1::terminal_atom
        && symbol.kind != grammar_symbol_kind_v1::nonterminal_composition) {
        return {grammar_signature_code_v1::invalid_symbol_kind};
    }
    for (const auto byte : symbol.reserved) {
        if (byte != 0) {
            return {grammar_signature_code_v1::nonzero_reserved};
        }
    }
    if (!symbol.interface_identity.valid()) {
        return {grammar_signature_code_v1::invalid_symbol_interface};
    }
    if (!symbol.lineage.valid()) {
        return {grammar_signature_code_v1::invalid_symbol_lineage};
    }
    return {};
}

static_assert(std::is_trivially_copyable<atom_port_signature_v1>::value);
static_assert(std::is_trivially_copyable<atom_interface_signature_v1>::value);
static_assert(std::is_trivially_copyable<typed_grammar_symbol_v1>::value);

} // namespace cellshard::compiler::composition
