#pragma once

#include <CellShard/compiler/partial/partial_atom_v1.hh>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace cellshard::compiler::partial {

inline constexpr std::uint32_t transform_partial_schema_version_v1 = 1;

enum class transform_partial_kind_v1 : std::uint32_t {
    permutation = 1,
    injective_subset = 2,
};

struct transform_index_v1 {
    std::uint64_t source_ordinal = 0;
    std::uint64_t destination_ordinal = 0;
};

struct transform_partial_view_v1 {
    const transform_index_v1 *indices = nullptr;
    std::uint64_t index_count = 0;
    atom_persistent_identity_v1 transform_identity{};
    atom_persistent_identity_v1 source_order_identity{};
    atom_persistent_identity_v1 destination_order_identity{};
    std::uint64_t source_extent = 0;
    std::uint64_t destination_extent = 0;
    std::uint64_t structure_generation = 0;
    transform_partial_kind_v1 kind = transform_partial_kind_v1::permutation;
    std::uint32_t schema_version = transform_partial_schema_version_v1;
};

enum class transform_partial_validation_code_v1 : std::uint32_t {
    valid = 0,
    unsupported_schema,
    invalid_kind,
    invalid_identity,
    missing_generation,
    missing_extent,
    missing_indices,
    permutation_extent_mismatch,
    source_out_of_bounds,
    destination_out_of_bounds,
    unordered_or_duplicate_source,
    duplicate_destination,
};

struct transform_partial_validation_v1 {
    transform_partial_validation_code_v1 code =
        transform_partial_validation_code_v1::valid;
    std::uint64_t index = 0;
    [[nodiscard]] constexpr bool valid() const noexcept {
        return code == transform_partial_validation_code_v1::valid;
    }
};

static_assert(offsetof(transform_partial_view_v1, indices) == 0);
static_assert(std::is_standard_layout<transform_index_v1>::value);
static_assert(std::is_trivially_copyable<transform_index_v1>::value);
static_assert(std::is_standard_layout<transform_partial_view_v1>::value);
static_assert(std::is_trivially_copyable<transform_partial_view_v1>::value);

// Caller marks make destination injectivity explicit without hidden allocation.
[[nodiscard]] inline transform_partial_validation_v1
validate_transform_partial_v1(const transform_partial_view_v1 &partial,
                              std::uint8_t *destination_marks,
                              std::uint64_t mark_capacity) noexcept {
    if (partial.schema_version != transform_partial_schema_version_v1) {
        return {transform_partial_validation_code_v1::unsupported_schema, 0};
    }
    if (partial.kind != transform_partial_kind_v1::permutation
        && partial.kind != transform_partial_kind_v1::injective_subset) {
        return {transform_partial_validation_code_v1::invalid_kind, 0};
    }
    if (!atom::validate_atom_persistent_identity_v1(partial.transform_identity)
             .valid()
        || !atom::validate_atom_persistent_identity_v1(
                partial.source_order_identity)
                .valid()
        || !atom::validate_atom_persistent_identity_v1(
                partial.destination_order_identity)
                .valid()) {
        return {transform_partial_validation_code_v1::invalid_identity, 0};
    }
    if (partial.structure_generation == 0) {
        return {transform_partial_validation_code_v1::missing_generation, 0};
    }
    if (partial.source_extent == 0 || partial.destination_extent == 0
        || destination_marks == nullptr
        || mark_capacity < partial.destination_extent) {
        return {transform_partial_validation_code_v1::missing_extent, 0};
    }
    if (partial.index_count == 0 || partial.indices == nullptr) {
        return {transform_partial_validation_code_v1::missing_indices, 0};
    }
    if (partial.kind == transform_partial_kind_v1::permutation
        && (partial.index_count != partial.source_extent
            || partial.index_count != partial.destination_extent)) {
        return {transform_partial_validation_code_v1::
                    permutation_extent_mismatch,
                0};
    }
    for (std::uint64_t index = 0; index < partial.destination_extent; ++index) {
        destination_marks[index] = 0;
    }
    for (std::uint64_t index = 0; index < partial.index_count; ++index) {
        const auto &entry = partial.indices[index];
        if (entry.source_ordinal >= partial.source_extent) {
            return {transform_partial_validation_code_v1::source_out_of_bounds,
                    index};
        }
        if (entry.destination_ordinal >= partial.destination_extent) {
            return {transform_partial_validation_code_v1::
                        destination_out_of_bounds,
                    index};
        }
        if (index != 0
            && partial.indices[index - 1].source_ordinal
                >= entry.source_ordinal) {
            return {transform_partial_validation_code_v1::
                        unordered_or_duplicate_source,
                    index};
        }
        if (destination_marks[entry.destination_ordinal] != 0) {
            return {transform_partial_validation_code_v1::duplicate_destination,
                    index};
        }
        destination_marks[entry.destination_ordinal] = 1;
    }
    return {transform_partial_validation_code_v1::valid, partial.index_count};
}

// Applies only the structural index transform. Numerical transforms remain
// Cellerator-owned; this helper writes an explicit destination-to-source gather.
[[nodiscard]] inline bool materialize_transform_gather_v1(
    const transform_partial_view_v1 &partial,
    std::uint64_t *destination_to_source,
    std::uint64_t output_capacity) noexcept {
    if (destination_to_source == nullptr
        || output_capacity < partial.destination_extent) {
        return false;
    }
    for (std::uint64_t index = 0; index < partial.destination_extent; ++index) {
        destination_to_source[index] = UINT64_MAX;
    }
    for (std::uint64_t index = 0; index < partial.index_count; ++index) {
        destination_to_source[partial.indices[index].destination_ordinal] =
            partial.indices[index].source_ordinal;
    }
    return true;
}

} // namespace cellshard::compiler::partial
