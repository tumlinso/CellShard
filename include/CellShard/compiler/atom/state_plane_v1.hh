#pragma once

#include <CellShard/compiler/atom/port_v1.hh>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace cellshard::compiler::atom {

inline constexpr std::uint32_t atom_state_plane_schema_version_v1 = 1;

enum class atom_state_kind_v1 : std::uint32_t {
    cell_state = 1,
    biological_state = 2,
    embedding = 3,
    provider_defined = 4,
};

// Mutable biological state is distinct from relation/feature values. The
// producer and consumers are portable affordance identities, never callbacks.
struct atom_state_plane_v1 {
    void *state = nullptr;
    std::uint64_t state_bytes = 0;
    std::uint64_t axis_element_count = 0;
    std::uint64_t component_count = 0;
    std::uint64_t axis_stride_bytes = 0;
    std::uint64_t component_stride_bytes = 0;
    std::uint32_t element_bytes = 0;
    std::uint32_t state_alignment = 0;
    atom_persistent_identity_v1 plane_identity{};
    atom_persistent_identity_v1 domain_identity{};
    atom_persistent_identity_v1 axis_identity{};
    atom_persistent_identity_v1 persistent_order_identity{};
    atom_port_numeric_v1 numeric{};
    std::uint64_t structure_epoch = 0;
    std::uint64_t state_generation = 0;
    atom_persistent_identity_v1 producer_affordance{};
    const atom_persistent_identity_v1 *consumer_affordances = nullptr;
    std::uint64_t consumer_affordance_count = 0;
    atom_state_kind_v1 kind = atom_state_kind_v1::cell_state;
    std::uint32_t reserved = 0;
};

enum class atom_state_plane_validation_code_v1 : std::uint32_t {
    valid = 0,
    missing_state,
    empty_state,
    invalid_shape,
    invalid_element_bytes,
    invalid_component_stride,
    invalid_axis_stride,
    state_bytes_overflow,
    insufficient_state_bytes,
    invalid_alignment,
    misaligned_state,
    invalid_plane_identity,
    invalid_domain_identity,
    invalid_axis_identity,
    invalid_persistent_order,
    invalid_numeric,
    missing_structure_epoch,
    missing_state_generation,
    invalid_producer_affordance,
    missing_consumer_affordances,
    invalid_consumer_affordance,
    unordered_or_duplicate_consumer,
    producer_consumer_cycle,
    invalid_state_kind,
    nonzero_reserved,
};

struct atom_state_plane_validation_v1 {
    atom_state_plane_validation_code_v1 code =
        atom_state_plane_validation_code_v1::valid;
    std::uint64_t index = 0;

    [[nodiscard]] constexpr bool valid() const noexcept {
        return code == atom_state_plane_validation_code_v1::valid;
    }
};

static_assert(offsetof(atom_state_plane_v1, state) == 0,
              "state planes must remain pointer-first");
static_assert(std::is_standard_layout<atom_state_plane_v1>::value);
static_assert(std::is_trivially_copyable<atom_state_plane_v1>::value);

[[nodiscard]] constexpr bool valid_atom_state_kind_v1(
    atom_state_kind_v1 kind) noexcept {
    const auto value = static_cast<std::uint32_t>(kind);
    return value >= 1 && value <= 4;
}

[[nodiscard]] constexpr bool valid_atom_state_numeric_v1(
    const atom_port_numeric_v1 &numeric) noexcept {
    return validate_atom_persistent_identity_v1(numeric.storage_type).valid()
        && validate_atom_persistent_identity_v1(numeric.logical_type).valid()
        && validate_atom_persistent_identity_v1(numeric.accumulation_type)
               .valid();
}

// O(consumer_count) time and O(1) storage. Shape byte requirements are explicit
// and no allocation, packing, callback discovery, or synchronization occurs.
[[nodiscard]] inline atom_state_plane_validation_v1
validate_atom_state_plane_v1(const atom_state_plane_v1 &plane) noexcept {
    if (plane.state == nullptr) {
        return {atom_state_plane_validation_code_v1::missing_state, 0};
    }
    if (plane.state_bytes == 0) {
        return {atom_state_plane_validation_code_v1::empty_state, 0};
    }
    if (plane.axis_element_count == 0 || plane.component_count == 0) {
        return {atom_state_plane_validation_code_v1::invalid_shape, 0};
    }
    if (plane.element_bytes == 0) {
        return {atom_state_plane_validation_code_v1::invalid_element_bytes, 0};
    }
    if (plane.component_stride_bytes < plane.element_bytes) {
        return {atom_state_plane_validation_code_v1::invalid_component_stride,
                0};
    }
    if (plane.component_count - 1
        > (std::numeric_limits<std::uint64_t>::max() - plane.element_bytes)
              / plane.component_stride_bytes) {
        return {atom_state_plane_validation_code_v1::state_bytes_overflow, 0};
    }
    const auto row_bytes = (plane.component_count - 1)
            * plane.component_stride_bytes
        + plane.element_bytes;
    if (plane.axis_stride_bytes < row_bytes) {
        return {atom_state_plane_validation_code_v1::invalid_axis_stride, 0};
    }
    if (plane.axis_element_count - 1
        > (std::numeric_limits<std::uint64_t>::max() - row_bytes)
              / plane.axis_stride_bytes) {
        return {atom_state_plane_validation_code_v1::state_bytes_overflow, 0};
    }
    const auto required_bytes = (plane.axis_element_count - 1)
            * plane.axis_stride_bytes
        + row_bytes;
    if (plane.state_bytes < required_bytes) {
        return {atom_state_plane_validation_code_v1::insufficient_state_bytes,
                required_bytes};
    }
    if (plane.state_alignment == 0
        || (plane.state_alignment & (plane.state_alignment - 1)) != 0) {
        return {atom_state_plane_validation_code_v1::invalid_alignment, 0};
    }
    if (reinterpret_cast<std::uintptr_t>(plane.state)
        % plane.state_alignment != 0) {
        return {atom_state_plane_validation_code_v1::misaligned_state, 0};
    }
#define CELLSHARD_ATOM_STATE_CHECK_ID(field, code) \
    if (!validate_atom_persistent_identity_v1(plane.field).valid()) { \
        return {atom_state_plane_validation_code_v1::code, 0}; \
    }
    CELLSHARD_ATOM_STATE_CHECK_ID(plane_identity, invalid_plane_identity)
    CELLSHARD_ATOM_STATE_CHECK_ID(domain_identity, invalid_domain_identity)
    CELLSHARD_ATOM_STATE_CHECK_ID(axis_identity, invalid_axis_identity)
    CELLSHARD_ATOM_STATE_CHECK_ID(persistent_order_identity,
                                  invalid_persistent_order)
#undef CELLSHARD_ATOM_STATE_CHECK_ID
    if (!valid_atom_state_numeric_v1(plane.numeric)) {
        return {atom_state_plane_validation_code_v1::invalid_numeric, 0};
    }
    if (plane.structure_epoch == 0) {
        return {atom_state_plane_validation_code_v1::missing_structure_epoch,
                0};
    }
    if (plane.state_generation == 0) {
        return {atom_state_plane_validation_code_v1::missing_state_generation,
                0};
    }
    if (!validate_atom_persistent_identity_v1(plane.producer_affordance)
             .valid()) {
        return {atom_state_plane_validation_code_v1::
                    invalid_producer_affordance,
                0};
    }
    if (plane.consumer_affordance_count == 0
        || plane.consumer_affordances == nullptr) {
        return {atom_state_plane_validation_code_v1::
                    missing_consumer_affordances,
                0};
    }
    for (std::uint64_t index = 0; index < plane.consumer_affordance_count;
         ++index) {
        const auto identity = plane.consumer_affordances[index];
        if (!validate_atom_persistent_identity_v1(identity).valid()) {
            return {atom_state_plane_validation_code_v1::
                        invalid_consumer_affordance,
                    index};
        }
        if (index != 0
            && !atom_persistent_identity_less_v1(
                plane.consumer_affordances[index - 1], identity)) {
            return {atom_state_plane_validation_code_v1::
                        unordered_or_duplicate_consumer,
                    index};
        }
        if (identity == plane.producer_affordance) {
            return {atom_state_plane_validation_code_v1::
                        producer_consumer_cycle,
                    index};
        }
    }
    if (!valid_atom_state_kind_v1(plane.kind)) {
        return {atom_state_plane_validation_code_v1::invalid_state_kind, 0};
    }
    if (plane.reserved != 0) {
        return {atom_state_plane_validation_code_v1::nonzero_reserved, 0};
    }
    return {atom_state_plane_validation_code_v1::valid,
            plane.consumer_affordance_count};
}

} // namespace cellshard::compiler::atom
