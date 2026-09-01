#pragma once

#include <CellShard/compiler/partial/partial_atom_v1.hh>

#include <cstdint>
#include <limits>
#include <type_traits>

namespace cellshard::compiler::partial {

inline constexpr std::uint32_t segment_summary_schema_version_v1 = 1;

struct segment_endpoint_v1 {
    atom_persistent_identity_v1 label_identity{};
    std::uint64_t canonical_ordinal = 0;
};

struct segment_summary_v1 {
    segment_endpoint_v1 first{};
    segment_endpoint_v1 last{};
    std::uint64_t element_count = 0;
    std::uint64_t segment_count = 0;
    atom_persistent_identity_v1 algebra_identity{};
    atom_persistent_identity_v1 persistent_order_identity{};
    std::uint64_t structure_generation = 0;
    std::uint32_t schema_version = segment_summary_schema_version_v1;
    std::uint32_t reserved = 0;
};

enum class segment_summary_code_v1 : std::uint32_t {
    valid = 0,
    unsupported_schema,
    invalid_identity,
    missing_generation,
    empty_summary,
    invalid_range,
    impossible_segment_count,
    nonzero_reserved,
    incompatible_contract,
    noncontiguous_ranges,
    count_overflow,
};

struct segment_summary_result_v1 {
    segment_summary_code_v1 code = segment_summary_code_v1::valid;
    segment_summary_v1 summary{};
    [[nodiscard]] constexpr bool valid() const noexcept {
        return code == segment_summary_code_v1::valid;
    }
};

static_assert(std::is_standard_layout<segment_endpoint_v1>::value);
static_assert(std::is_trivially_copyable<segment_endpoint_v1>::value);
static_assert(std::is_standard_layout<segment_summary_v1>::value);
static_assert(std::is_trivially_copyable<segment_summary_v1>::value);

[[nodiscard]] inline segment_summary_code_v1 validate_segment_summary_v1(
    const segment_summary_v1 &summary) noexcept {
    if (summary.schema_version != segment_summary_schema_version_v1) {
        return segment_summary_code_v1::unsupported_schema;
    }
    if (!atom::validate_atom_persistent_identity_v1(summary.first.label_identity)
             .valid()
        || !atom::validate_atom_persistent_identity_v1(
                summary.last.label_identity)
                .valid()
        || !atom::validate_atom_persistent_identity_v1(summary.algebra_identity)
                .valid()
        || !atom::validate_atom_persistent_identity_v1(
                summary.persistent_order_identity)
                .valid()) {
        return segment_summary_code_v1::invalid_identity;
    }
    if (summary.structure_generation == 0) {
        return segment_summary_code_v1::missing_generation;
    }
    if (summary.element_count == 0 || summary.segment_count == 0) {
        return segment_summary_code_v1::empty_summary;
    }
    if (summary.first.canonical_ordinal > summary.last.canonical_ordinal
        || summary.last.canonical_ordinal - summary.first.canonical_ordinal
                + 1
            != summary.element_count) {
        return segment_summary_code_v1::invalid_range;
    }
    if (summary.segment_count > summary.element_count) {
        return segment_summary_code_v1::impossible_segment_count;
    }
    return summary.reserved == 0 ? segment_summary_code_v1::valid
                                 : segment_summary_code_v1::nonzero_reserved;
}

[[nodiscard]] inline segment_summary_result_v1 merge_segment_summaries_v1(
    const segment_summary_v1 &left, const segment_summary_v1 &right) noexcept {
    const auto left_code = validate_segment_summary_v1(left);
    if (left_code != segment_summary_code_v1::valid) return {left_code, {}};
    const auto right_code = validate_segment_summary_v1(right);
    if (right_code != segment_summary_code_v1::valid) return {right_code, {}};
    if (left.algebra_identity != right.algebra_identity
        || left.persistent_order_identity != right.persistent_order_identity
        || left.structure_generation != right.structure_generation) {
        return {segment_summary_code_v1::incompatible_contract, {}};
    }
    if (left.last.canonical_ordinal
            == std::numeric_limits<std::uint64_t>::max()
        || left.last.canonical_ordinal + 1
            != right.first.canonical_ordinal) {
        return {segment_summary_code_v1::noncontiguous_ranges, {}};
    }
    if (left.element_count
            > std::numeric_limits<std::uint64_t>::max() - right.element_count
        || left.segment_count
            > std::numeric_limits<std::uint64_t>::max() - right.segment_count) {
        return {segment_summary_code_v1::count_overflow, {}};
    }
    auto merged = left;
    merged.last = right.last;
    merged.element_count += right.element_count;
    merged.segment_count += right.segment_count;
    if (left.last.label_identity == right.first.label_identity) {
        --merged.segment_count;
    }
    return {validate_segment_summary_v1(merged), merged};
}

} // namespace cellshard::compiler::partial
