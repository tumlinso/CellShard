#pragma once

#include <CellShard/compiler/discovery/sequence_compat/provider_coordinate_coverage_v1.hh>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace cellshard::compiler::discovery::sequence_compat {

inline constexpr std::uint32_t owned_halo_intervals_schema_version_v1 = 1;

enum class coordinate_interval_role_v1 : std::uint8_t {
    owned = 1,
    read_only_halo = 2,
};

struct coordinate_interval_role_record_v1 {
    std::uint64_t begin = 0;
    std::uint64_t end = 0;
    coordinate_interval_role_v1 role = coordinate_interval_role_v1::owned;
    bool contribution_allowed = true;
    std::uint8_t reserved[6]{};
};

// Non-owning cold view. Intervals are sorted, disjoint, and contained by the
// provider coordinate envelope. Halo intervals may be arbitrary, but only
// contiguous coverage adjacent to the outer owned span satisfies left/right
// provider requirements.
struct owned_halo_intervals_v1 {
    const coordinate_interval_role_record_v1 *intervals = nullptr;
    std::uint64_t interval_count = 0;
    const provider_coordinate_coverage_v1 *coordinate_coverage = nullptr;
    std::uint64_t required_left_halo = 0;
    std::uint64_t required_right_halo = 0;
    std::uint32_t schema_version = owned_halo_intervals_schema_version_v1;
    std::uint32_t record_bytes = sizeof(owned_halo_intervals_v1);
};

enum class owned_halo_intervals_validation_code_v1 : std::uint32_t {
    valid = 0,
    unsupported_schema,
    invalid_record_bytes,
    missing_intervals,
    invalid_coordinate_coverage,
    invalid_interval,
    unordered_or_overlapping_interval,
    interval_outside_coverage,
    invalid_role,
    invalid_contribution_permission,
    nonzero_reserved,
    owned_count_overflow,
    owned_count_mismatch,
    insufficient_left_halo,
    insufficient_right_halo,
};

struct owned_halo_intervals_validation_v1 {
    owned_halo_intervals_validation_code_v1 code =
        owned_halo_intervals_validation_code_v1::valid;
    std::uint64_t index = 0;
    std::uint32_t nested_code = 0;

    [[nodiscard]] constexpr bool valid() const noexcept {
        return code == owned_halo_intervals_validation_code_v1::valid;
    }
};

static_assert(offsetof(owned_halo_intervals_v1, intervals) == 0,
              "owned/halo interval views must remain pointer-first");
static_assert(std::is_standard_layout<coordinate_interval_role_record_v1>::value);
static_assert(
    std::is_trivially_copyable<coordinate_interval_role_record_v1>::value);
static_assert(std::is_standard_layout<owned_halo_intervals_v1>::value);
static_assert(std::is_trivially_copyable<owned_halo_intervals_v1>::value);

[[nodiscard]] inline owned_halo_intervals_validation_v1
validate_owned_halo_intervals_v1(
    const owned_halo_intervals_v1 &view,
    std::uint32_t exact_coverage_source_validation) noexcept {
    if (view.schema_version != owned_halo_intervals_schema_version_v1) {
        return {owned_halo_intervals_validation_code_v1::unsupported_schema,
                0, 0};
    }
    if (view.record_bytes != sizeof(owned_halo_intervals_v1)) {
        return {owned_halo_intervals_validation_code_v1::invalid_record_bytes,
                0, 0};
    }
    if (view.interval_count == 0 || view.intervals == nullptr) {
        return {owned_halo_intervals_validation_code_v1::missing_intervals,
                0, 0};
    }
    if (view.coordinate_coverage == nullptr) {
        return {owned_halo_intervals_validation_code_v1::
                    invalid_coordinate_coverage,
                0, 0};
    }
    const auto coverage_validation = validate_provider_coordinate_coverage_v1(
        *view.coordinate_coverage, exact_coverage_source_validation);
    if (!coverage_validation.valid()) {
        return {owned_halo_intervals_validation_code_v1::
                    invalid_coordinate_coverage,
                0, static_cast<std::uint32_t>(coverage_validation.code)};
    }

    std::uint64_t owned_count = 0;
    std::uint64_t first_owned_begin = std::numeric_limits<std::uint64_t>::max();
    std::uint64_t last_owned_end = 0;
    for (std::uint64_t index = 0; index < view.interval_count; ++index) {
        const auto &interval = view.intervals[index];
        if (interval.begin >= interval.end) {
            return {owned_halo_intervals_validation_code_v1::invalid_interval,
                    index, 0};
        }
        if (index != 0 && view.intervals[index - 1].end > interval.begin) {
            return {owned_halo_intervals_validation_code_v1::
                        unordered_or_overlapping_interval,
                    index, 0};
        }
        if (interval.begin < view.coordinate_coverage->coordinate_begin
            || interval.end > view.coordinate_coverage->coordinate_end) {
            return {owned_halo_intervals_validation_code_v1::
                        interval_outside_coverage,
                    index, 0};
        }
        const auto role = static_cast<std::uint8_t>(interval.role);
        if (role < static_cast<std::uint8_t>(coordinate_interval_role_v1::owned)
            || role > static_cast<std::uint8_t>(
                coordinate_interval_role_v1::read_only_halo)) {
            return {owned_halo_intervals_validation_code_v1::invalid_role,
                    index, 0};
        }
        const bool owned = interval.role == coordinate_interval_role_v1::owned;
        if (interval.contribution_allowed != owned) {
            return {owned_halo_intervals_validation_code_v1::
                        invalid_contribution_permission,
                    index, 0};
        }
        for (const auto item : interval.reserved) {
            if (item != 0) {
                return {owned_halo_intervals_validation_code_v1::
                            nonzero_reserved,
                        index, 0};
            }
        }
        if (owned) {
            const std::uint64_t length = interval.end - interval.begin;
            if (owned_count > std::numeric_limits<std::uint64_t>::max() - length) {
                return {owned_halo_intervals_validation_code_v1::
                            owned_count_overflow,
                        index, 0};
            }
            owned_count += length;
            if (first_owned_begin == std::numeric_limits<std::uint64_t>::max()) {
                first_owned_begin = interval.begin;
            }
            last_owned_end = interval.end;
        }
    }
    if (owned_count != view.coordinate_coverage->owned_count) {
        return {owned_halo_intervals_validation_code_v1::owned_count_mismatch,
                0, 0};
    }

    std::uint64_t left_extent = 0;
    std::uint64_t cursor = first_owned_begin;
    for (std::uint64_t index = view.interval_count; index != 0; --index) {
        const auto &interval = view.intervals[index - 1];
        if (interval.end > first_owned_begin) {
            continue;
        }
        if (interval.end != cursor
            || interval.role != coordinate_interval_role_v1::read_only_halo) {
            break;
        }
        left_extent += interval.end - interval.begin;
        cursor = interval.begin;
    }
    if (left_extent < view.required_left_halo) {
        return {owned_halo_intervals_validation_code_v1::
                    insufficient_left_halo,
                0, 0};
    }

    std::uint64_t right_extent = 0;
    cursor = last_owned_end;
    for (std::uint64_t index = 0; index < view.interval_count; ++index) {
        const auto &interval = view.intervals[index];
        if (interval.begin < last_owned_end) {
            continue;
        }
        if (interval.begin != cursor
            || interval.role != coordinate_interval_role_v1::read_only_halo) {
            break;
        }
        right_extent += interval.end - interval.begin;
        cursor = interval.end;
    }
    if (right_extent < view.required_right_halo) {
        return {owned_halo_intervals_validation_code_v1::
                    insufficient_right_halo,
                0, 0};
    }
    return {};
}

} // namespace cellshard::compiler::discovery::sequence_compat
