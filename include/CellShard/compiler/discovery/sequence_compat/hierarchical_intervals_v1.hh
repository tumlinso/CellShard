#pragma once

#include <CellShard/compiler/discovery/sequence_compat/provider_coordinate_coverage_v1.hh>
#include <CellShard/compiler/discovery/sequence_compat/reference_strand_identity_v1.hh>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace cellshard::compiler::discovery::sequence_compat {

inline constexpr std::uint32_t hierarchical_intervals_schema_version_v1 = 1;

struct hierarchical_interval_parent_v1 {
    atom::atom_persistent_identity_v1 parent_identity{};
    std::uint64_t parent_index = 0;
};

struct hierarchical_interval_v1 {
    atom::atom_persistent_identity_v1 interval_identity{};
    std::uint64_t begin = 0;
    std::uint64_t end = 0;
    const hierarchical_interval_parent_v1 *parents = nullptr;
    std::uint64_t parent_count = 0;
};

// Nodes are source-linked in stable identity order. Parent indices point only
// backward and carry the matching persistent ID, making the view a validated
// DAG while allowing multi-parent composition without choosing one tree.
struct hierarchical_interval_dag_v1 {
    const hierarchical_interval_v1 *intervals = nullptr;
    std::uint64_t interval_count = 0;
    const provider_coordinate_coverage_v1 *coordinate_coverage = nullptr;
    const reference_strand_identity_v1 *reference = nullptr;
    std::uint32_t schema_version = hierarchical_intervals_schema_version_v1;
    std::uint32_t record_bytes = sizeof(hierarchical_interval_dag_v1);
};

enum class hierarchical_intervals_validation_code_v1 : std::uint32_t {
    valid = 0,
    unsupported_schema,
    invalid_record_bytes,
    missing_intervals,
    invalid_coordinate_coverage,
    invalid_reference,
    invalid_interval_identity,
    duplicate_or_unordered_interval_identity,
    invalid_interval,
    interval_outside_coverage,
    inconsistent_parent_pointer,
    invalid_parent_index,
    invalid_parent_identity,
    duplicate_or_unordered_parent,
    parent_does_not_contain_child,
    missing_root,
};

struct hierarchical_intervals_validation_v1 {
    hierarchical_intervals_validation_code_v1 code =
        hierarchical_intervals_validation_code_v1::valid;
    std::uint64_t interval_index = 0;
    std::uint64_t parent_index = 0;
    std::uint32_t nested_code = 0;

    [[nodiscard]] constexpr bool valid() const noexcept {
        return code == hierarchical_intervals_validation_code_v1::valid;
    }
};

static_assert(offsetof(hierarchical_interval_dag_v1, intervals) == 0,
              "hierarchical interval DAGs must remain pointer-first");
static_assert(std::is_standard_layout<hierarchical_interval_parent_v1>::value);
static_assert(
    std::is_trivially_copyable<hierarchical_interval_parent_v1>::value);
static_assert(std::is_standard_layout<hierarchical_interval_v1>::value);
static_assert(std::is_trivially_copyable<hierarchical_interval_v1>::value);
static_assert(std::is_standard_layout<hierarchical_interval_dag_v1>::value);
static_assert(std::is_trivially_copyable<hierarchical_interval_dag_v1>::value);

[[nodiscard]] inline hierarchical_intervals_validation_v1
validate_hierarchical_interval_dag_v1(
    const hierarchical_interval_dag_v1 &dag,
    std::uint32_t exact_coverage_source_validation) noexcept {
    if (dag.schema_version != hierarchical_intervals_schema_version_v1) {
        return {hierarchical_intervals_validation_code_v1::unsupported_schema,
                0, 0, 0};
    }
    if (dag.record_bytes != sizeof(hierarchical_interval_dag_v1)) {
        return {hierarchical_intervals_validation_code_v1::invalid_record_bytes,
                0, 0, 0};
    }
    if (dag.interval_count == 0 || dag.intervals == nullptr) {
        return {hierarchical_intervals_validation_code_v1::missing_intervals,
                0, 0, 0};
    }
    if (dag.coordinate_coverage == nullptr) {
        return {hierarchical_intervals_validation_code_v1::
                    invalid_coordinate_coverage,
                0, 0, 0};
    }
    const auto coverage_result = validate_provider_coordinate_coverage_v1(
        *dag.coordinate_coverage, exact_coverage_source_validation);
    if (!coverage_result.valid()) {
        return {hierarchical_intervals_validation_code_v1::
                    invalid_coordinate_coverage,
                0, 0, static_cast<std::uint32_t>(coverage_result.code)};
    }
    if (dag.reference == nullptr) {
        return {hierarchical_intervals_validation_code_v1::invalid_reference,
                0, 0, 0};
    }
    const auto reference_result =
        validate_reference_strand_identity_v1(*dag.reference);
    if (!reference_result.valid()) {
        return {hierarchical_intervals_validation_code_v1::invalid_reference,
                0, 0, static_cast<std::uint32_t>(reference_result.code)};
    }

    bool found_root = false;
    for (std::uint64_t index = 0; index < dag.interval_count; ++index) {
        const auto &interval = dag.intervals[index];
        if (!atom::validate_atom_persistent_identity_v1(
                 interval.interval_identity)
                 .valid()) {
            return {hierarchical_intervals_validation_code_v1::
                        invalid_interval_identity,
                    index, 0, 0};
        }
        if (index != 0
            && !atom::atom_persistent_identity_less_v1(
                dag.intervals[index - 1].interval_identity,
                interval.interval_identity)) {
            return {hierarchical_intervals_validation_code_v1::
                        duplicate_or_unordered_interval_identity,
                    index, 0, 0};
        }
        if (interval.begin >= interval.end) {
            return {hierarchical_intervals_validation_code_v1::invalid_interval,
                    index, 0, 0};
        }
        if (interval.begin < dag.coordinate_coverage->coordinate_begin
            || interval.end > dag.coordinate_coverage->coordinate_end) {
            return {hierarchical_intervals_validation_code_v1::
                        interval_outside_coverage,
                    index, 0, 0};
        }
        if ((interval.parent_count == 0) != (interval.parents == nullptr)) {
            return {hierarchical_intervals_validation_code_v1::
                        inconsistent_parent_pointer,
                    index, 0, 0};
        }
        found_root = found_root || interval.parent_count == 0;
        for (std::uint64_t parent_index = 0;
             parent_index < interval.parent_count; ++parent_index) {
            const auto &parent = interval.parents[parent_index];
            if (parent.parent_index >= index) {
                return {hierarchical_intervals_validation_code_v1::
                            invalid_parent_index,
                        index, parent_index, 0};
            }
            if (parent.parent_identity
                != dag.intervals[parent.parent_index].interval_identity) {
                return {hierarchical_intervals_validation_code_v1::
                            invalid_parent_identity,
                        index, parent_index, 0};
            }
            if (parent_index != 0
                && interval.parents[parent_index - 1].parent_index
                    >= parent.parent_index) {
                return {hierarchical_intervals_validation_code_v1::
                            duplicate_or_unordered_parent,
                        index, parent_index, 0};
            }
            const auto &parent_interval = dag.intervals[parent.parent_index];
            if (parent_interval.begin > interval.begin
                || parent_interval.end < interval.end) {
                return {hierarchical_intervals_validation_code_v1::
                            parent_does_not_contain_child,
                        index, parent_index, 0};
            }
        }
    }
    if (!found_root) {
        return {hierarchical_intervals_validation_code_v1::missing_root,
                0, 0, 0};
    }
    return {};
}

} // namespace cellshard::compiler::discovery::sequence_compat
