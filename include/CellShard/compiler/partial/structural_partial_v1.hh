#pragma once

#include <CellShard/compiler/partial/partial_atom_v1.hh>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace cellshard::compiler::partial {

inline constexpr std::uint32_t structural_partial_schema_version_v1 = 1;

enum class structural_partial_kind_v1 : std::uint32_t {
    node_membership = 1,
    relation_edge = 2,
};

// Exact structural facts use persistent biological identities plus canonical
// ordinals. Ordinal equality alone never establishes identity.
struct structural_partial_record_v1 {
    atom_persistent_identity_v1 subject_identity{};
    atom_persistent_identity_v1 object_identity{};
    std::uint64_t subject_canonical_ordinal = 0;
    std::uint64_t object_canonical_ordinal = 0;
};

struct structural_partial_view_v1 {
    const structural_partial_record_v1 *records = nullptr;
    std::uint64_t record_count = 0;
    atom_persistent_identity_v1 structure_identity{};
    atom_persistent_identity_v1 persistent_order_identity{};
    std::uint64_t structure_generation = 0;
    structural_partial_kind_v1 kind =
        structural_partial_kind_v1::node_membership;
    std::uint32_t schema_version = structural_partial_schema_version_v1;
};

enum class structural_partial_validation_code_v1 : std::uint32_t {
    valid = 0,
    unsupported_schema,
    invalid_kind,
    invalid_structure_identity,
    invalid_order_identity,
    missing_structure_generation,
    missing_records,
    invalid_subject_identity,
    unexpected_object_identity,
    invalid_object_identity,
    unexpected_object_ordinal,
    unordered_or_duplicate_record,
};

struct structural_partial_validation_v1 {
    structural_partial_validation_code_v1 code =
        structural_partial_validation_code_v1::valid;
    std::uint64_t index = 0;

    [[nodiscard]] constexpr bool valid() const noexcept {
        return code == structural_partial_validation_code_v1::valid;
    }
};

enum class structural_partial_merge_code_v1 : std::uint32_t {
    merged = 0,
    invalid_left,
    invalid_right,
    incompatible_contract,
    capacity_overflow,
    duplicate_contribution,
    invalid_output,
};

struct structural_partial_merge_result_v1 {
    structural_partial_merge_code_v1 code =
        structural_partial_merge_code_v1::merged;
    std::uint64_t output_count = 0;
    std::uint64_t index = 0;

    [[nodiscard]] constexpr bool merged() const noexcept {
        return code == structural_partial_merge_code_v1::merged;
    }
};

static_assert(offsetof(structural_partial_view_v1, records) == 0);
static_assert(std::is_standard_layout<structural_partial_record_v1>::value);
static_assert(std::is_trivially_copyable<structural_partial_record_v1>::value);
static_assert(std::is_standard_layout<structural_partial_view_v1>::value);
static_assert(std::is_trivially_copyable<structural_partial_view_v1>::value);

[[nodiscard]] constexpr bool valid_structural_partial_kind_v1(
    structural_partial_kind_v1 kind) noexcept {
    return kind == structural_partial_kind_v1::node_membership
        || kind == structural_partial_kind_v1::relation_edge;
}

[[nodiscard]] constexpr bool structural_partial_record_less_v1(
    const structural_partial_record_v1 &lhs,
    const structural_partial_record_v1 &rhs) noexcept {
    return atom::atom_persistent_identity_less_v1(
               lhs.subject_identity, rhs.subject_identity)
        || (lhs.subject_identity == rhs.subject_identity
            && (lhs.subject_canonical_ordinal < rhs.subject_canonical_ordinal
                || (lhs.subject_canonical_ordinal
                        == rhs.subject_canonical_ordinal
                    && (atom::atom_persistent_identity_less_v1(
                            lhs.object_identity, rhs.object_identity)
                        || (lhs.object_identity == rhs.object_identity
                            && lhs.object_canonical_ordinal
                                < rhs.object_canonical_ordinal)))));
}

[[nodiscard]] inline structural_partial_validation_v1
validate_structural_partial_v1(
    const structural_partial_view_v1 &partial) noexcept {
    if (partial.schema_version != structural_partial_schema_version_v1) {
        return {structural_partial_validation_code_v1::unsupported_schema, 0};
    }
    if (!valid_structural_partial_kind_v1(partial.kind)) {
        return {structural_partial_validation_code_v1::invalid_kind, 0};
    }
    if (!atom::validate_atom_persistent_identity_v1(partial.structure_identity)
             .valid()) {
        return {structural_partial_validation_code_v1::
                    invalid_structure_identity,
                0};
    }
    if (!atom::validate_atom_persistent_identity_v1(
             partial.persistent_order_identity)
             .valid()) {
        return {structural_partial_validation_code_v1::invalid_order_identity,
                0};
    }
    if (partial.structure_generation == 0) {
        return {structural_partial_validation_code_v1::
                    missing_structure_generation,
                0};
    }
    if (partial.record_count == 0 || partial.records == nullptr) {
        return {structural_partial_validation_code_v1::missing_records, 0};
    }
    for (std::uint64_t index = 0; index < partial.record_count; ++index) {
        const auto &record = partial.records[index];
        if (!atom::validate_atom_persistent_identity_v1(
                 record.subject_identity)
                 .valid()) {
            return {structural_partial_validation_code_v1::
                        invalid_subject_identity,
                    index};
        }
        const bool object_valid = atom::validate_atom_persistent_identity_v1(
            record.object_identity).valid();
        if (partial.kind == structural_partial_kind_v1::node_membership) {
            if (object_valid) {
                return {structural_partial_validation_code_v1::
                            unexpected_object_identity,
                        index};
            }
            if (record.object_canonical_ordinal != 0) {
                return {structural_partial_validation_code_v1::
                            unexpected_object_ordinal,
                        index};
            }
        } else if (!object_valid) {
            return {structural_partial_validation_code_v1::
                        invalid_object_identity,
                    index};
        }
        if (index != 0
            && !structural_partial_record_less_v1(
                partial.records[index - 1], record)) {
            return {structural_partial_validation_code_v1::
                        unordered_or_duplicate_record,
                    index};
        }
    }
    return {structural_partial_validation_code_v1::valid,
            partial.record_count};
}

// Exact union of two sorted, disjoint structural contributions. The caller
// owns output storage; duplicate contribution is an error rather than a silent
// set-union because exact ownership must remain visible.
[[nodiscard]] inline structural_partial_merge_result_v1
merge_structural_partials_v1(
    const structural_partial_view_v1 &left,
    const structural_partial_view_v1 &right,
    structural_partial_record_v1 *output,
    std::uint64_t output_capacity) noexcept {
    const auto left_validation = validate_structural_partial_v1(left);
    if (!left_validation.valid()) {
        return {structural_partial_merge_code_v1::invalid_left, 0,
                left_validation.index};
    }
    const auto right_validation = validate_structural_partial_v1(right);
    if (!right_validation.valid()) {
        return {structural_partial_merge_code_v1::invalid_right, 0,
                right_validation.index};
    }
    if (left.structure_identity != right.structure_identity
        || left.persistent_order_identity != right.persistent_order_identity
        || left.structure_generation != right.structure_generation
        || left.kind != right.kind) {
        return {structural_partial_merge_code_v1::incompatible_contract, 0, 0};
    }
    if (output == nullptr
        || output_capacity < left.record_count + right.record_count) {
        return {structural_partial_merge_code_v1::capacity_overflow, 0,
                left.record_count + right.record_count};
    }
    std::uint64_t left_index = 0;
    std::uint64_t right_index = 0;
    std::uint64_t output_index = 0;
    while (left_index < left.record_count
           && right_index < right.record_count) {
        const auto &lhs = left.records[left_index];
        const auto &rhs = right.records[right_index];
        if (structural_partial_record_less_v1(lhs, rhs)) {
            output[output_index++] = lhs;
            ++left_index;
        } else if (structural_partial_record_less_v1(rhs, lhs)) {
            output[output_index++] = rhs;
            ++right_index;
        } else {
            return {structural_partial_merge_code_v1::duplicate_contribution,
                    output_index, left_index};
        }
    }
    while (left_index < left.record_count) {
        output[output_index++] = left.records[left_index++];
    }
    while (right_index < right.record_count) {
        output[output_index++] = right.records[right_index++];
    }
    const structural_partial_view_v1 merged{
        output, output_index, left.structure_identity,
        left.persistent_order_identity, left.structure_generation, left.kind,
        structural_partial_schema_version_v1};
    if (!validate_structural_partial_v1(merged).valid()) {
        return {structural_partial_merge_code_v1::invalid_output,
                output_index, 0};
    }
    return {structural_partial_merge_code_v1::merged, output_index,
            output_index};
}

} // namespace cellshard::compiler::partial
