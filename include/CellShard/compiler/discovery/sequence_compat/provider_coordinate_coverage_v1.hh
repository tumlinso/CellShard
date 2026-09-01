#pragma once

#include <CellShard/compiler/atom/logical_coverage_v1.hh>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace cellshard::compiler::discovery::sequence_compat {

inline constexpr std::uint32_t provider_coordinate_coverage_schema_version_v1 =
    1;

// A provider-defined coordinate interval remains an extension of exact atom
// coverage, not a replacement membership encoding. The provider owns payload
// bytes and validation; CellShard records only their stable schema and result.
// Coordinates use a half-open [begin, end) interval in the named reference.
struct provider_coordinate_coverage_v1 {
    const void *provider_payload = nullptr;
    atom::atom_logical_coverage_ref_v1 exact_coverage{};
    atom::atom_persistent_identity_v1 reference_identity{};
    atom::atom_persistent_identity_v1 payload_schema{};
    std::uint64_t coordinate_begin = 0;
    std::uint64_t coordinate_end = 0;
    std::uint64_t owned_count = 0;
    std::uint32_t schema_version =
        provider_coordinate_coverage_schema_version_v1;
    std::uint32_t record_bytes = sizeof(provider_coordinate_coverage_v1);
    std::uint32_t provider_validation_code = 0;
    std::uint32_t reserved = 0;
};

enum class provider_coordinate_coverage_validation_code_v1 : std::uint32_t {
    valid = 0,
    unsupported_schema,
    invalid_record_bytes,
    missing_provider_payload,
    invalid_exact_coverage,
    non_provider_defined_coverage,
    invalid_reference_identity,
    invalid_payload_schema,
    invalid_coordinate_interval,
    invalid_owned_count,
    provider_validation_failed,
    nonzero_reserved,
};

struct provider_coordinate_coverage_validation_v1 {
    provider_coordinate_coverage_validation_code_v1 code =
        provider_coordinate_coverage_validation_code_v1::valid;
    std::uint32_t nested_code = 0;

    [[nodiscard]] constexpr bool valid() const noexcept {
        return code
            == provider_coordinate_coverage_validation_code_v1::valid;
    }
};

static_assert(offsetof(provider_coordinate_coverage_v1, provider_payload) == 0,
              "provider coordinate coverage must remain pointer-first");
static_assert(std::is_standard_layout<provider_coordinate_coverage_v1>::value);
static_assert(
    std::is_trivially_copyable<provider_coordinate_coverage_v1>::value);

// O(1) validation allocates nothing and never interprets provider payload
// bytes. Exact member validation stays with the independently supplied atom
// coverage result and provider-specific validation stays with the provider.
[[nodiscard]] inline provider_coordinate_coverage_validation_v1
validate_provider_coordinate_coverage_v1(
    const provider_coordinate_coverage_v1 &coverage,
    std::uint32_t exact_coverage_source_validation) noexcept {
    if (coverage.schema_version
        != provider_coordinate_coverage_schema_version_v1) {
        return {provider_coordinate_coverage_validation_code_v1::
                    unsupported_schema,
                0};
    }
    if (coverage.record_bytes != sizeof(provider_coordinate_coverage_v1)) {
        return {provider_coordinate_coverage_validation_code_v1::
                    invalid_record_bytes,
                0};
    }
    if (coverage.provider_payload == nullptr) {
        return {provider_coordinate_coverage_validation_code_v1::
                    missing_provider_payload,
                0};
    }
    const auto exact_result = atom::validate_atom_logical_coverage_ref_v1(
        coverage.exact_coverage, exact_coverage_source_validation);
    if (!exact_result.valid()) {
        return {provider_coordinate_coverage_validation_code_v1::
                    invalid_exact_coverage,
                static_cast<std::uint32_t>(exact_result.code)};
    }
    if (coverage.exact_coverage.kind
        != atom::atom_logical_coverage_kind_v1::provider_defined) {
        return {provider_coordinate_coverage_validation_code_v1::
                    non_provider_defined_coverage,
                0};
    }
    if (!atom::validate_atom_persistent_identity_v1(
             coverage.reference_identity)
             .valid()) {
        return {provider_coordinate_coverage_validation_code_v1::
                    invalid_reference_identity,
                0};
    }
    if (!atom::validate_atom_persistent_identity_v1(coverage.payload_schema)
             .valid()) {
        return {provider_coordinate_coverage_validation_code_v1::
                    invalid_payload_schema,
                0};
    }
    if (coverage.coordinate_begin >= coverage.coordinate_end) {
        return {provider_coordinate_coverage_validation_code_v1::
                    invalid_coordinate_interval,
                0};
    }
    const std::uint64_t span =
        coverage.coordinate_end - coverage.coordinate_begin;
    if (coverage.owned_count == 0 || coverage.owned_count > span
        || coverage.owned_count != coverage.exact_coverage.logical_count) {
        return {provider_coordinate_coverage_validation_code_v1::
                    invalid_owned_count,
                0};
    }
    if (coverage.provider_validation_code != 0) {
        return {provider_coordinate_coverage_validation_code_v1::
                    provider_validation_failed,
                coverage.provider_validation_code};
    }
    if (coverage.reserved != 0) {
        return {provider_coordinate_coverage_validation_code_v1::
                    nonzero_reserved,
                0};
    }
    return {};
}

} // namespace cellshard::compiler::discovery::sequence_compat
