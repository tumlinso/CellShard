#pragma once

#include <CellShard/identity/digest.hh>
#include <CellShard/identity/strong_id.hh>

#include <cstdint>
#include <limits>
#include <type_traits>

namespace cellshard::compiler::composition {

inline constexpr std::uint32_t composition_production_schema_v1 = 1;

struct composition_production_tag {};
struct composition_lineage_tag {};
using composition_production_id = strong_id<composition_production_tag>;
using composition_lineage_id = strong_id<composition_lineage_tag>;

struct composition_version_v1 {
    std::uint16_t major = 1;
    std::uint16_t minor = 0;
    std::uint16_t patch = 0;
    std::uint16_t reserved = 0;
    std::uint64_t revision = 1;
};

// Stable cold-compiler identity. Runtime location, replica, placement, device,
// pointer, and service state are intentionally absent.
struct composition_production_identity_v1 {
    std::uint32_t schema_version = composition_production_schema_v1;
    std::uint32_t record_bytes = sizeof(composition_production_identity_v1);
    composition_production_id production{};
    composition_lineage_id lineage{};
    composition_version_v1 version{};
    content_digest definition_digest{};
};

enum class production_identity_code_v1 : std::uint32_t {
    valid = 0,
    unsupported_schema,
    invalid_record_bytes,
    invalid_production,
    invalid_lineage,
    invalid_version,
    nonzero_reserved,
    invalid_definition_digest,
    revision_overflow,
    missing_output,
};

struct production_identity_status_v1 {
    production_identity_code_v1 code = production_identity_code_v1::valid;
    [[nodiscard]] constexpr bool valid() const noexcept {
        return code == production_identity_code_v1::valid;
    }
};

[[nodiscard]] constexpr bool valid_composition_version_v1(
    const composition_version_v1 &version) noexcept {
    return version.major != 0 && version.revision != 0
        && version.reserved == 0;
}

[[nodiscard]] inline production_identity_status_v1
validate_composition_production_identity_v1(
    const composition_production_identity_v1 &identity) noexcept {
    if (identity.schema_version != composition_production_schema_v1) {
        return {production_identity_code_v1::unsupported_schema};
    }
    if (identity.record_bytes != sizeof(composition_production_identity_v1)) {
        return {production_identity_code_v1::invalid_record_bytes};
    }
    if (!identity.production.valid()) {
        return {production_identity_code_v1::invalid_production};
    }
    if (!identity.lineage.valid()) {
        return {production_identity_code_v1::invalid_lineage};
    }
    if (identity.version.major == 0 || identity.version.revision == 0) {
        return {production_identity_code_v1::invalid_version};
    }
    if (identity.version.reserved != 0) {
        return {production_identity_code_v1::nonzero_reserved};
    }
    if (identity.definition_digest.algorithm == digest_algorithm::none
        || !valid_content_digest(identity.definition_digest)) {
        return {production_identity_code_v1::invalid_definition_digest};
    }
    return {};
}

[[nodiscard]] constexpr bool same_composition_production_v1(
    const composition_production_identity_v1 &lhs,
    const composition_production_identity_v1 &rhs) noexcept {
    return lhs.production == rhs.production && lhs.lineage == rhs.lineage;
}

[[nodiscard]] inline bool same_composition_version_v1(
    const composition_production_identity_v1 &lhs,
    const composition_production_identity_v1 &rhs) noexcept {
    return same_composition_production_v1(lhs, rhs)
        && lhs.version.major == rhs.version.major
        && lhs.version.minor == rhs.version.minor
        && lhs.version.patch == rhs.version.patch
        && lhs.version.revision == rhs.version.revision
        && lhs.definition_digest == rhs.definition_digest;
}

[[nodiscard]] inline production_identity_status_v1
next_composition_revision_v1(
    const composition_production_identity_v1 &current,
    content_digest next_definition_digest,
    composition_production_identity_v1 *output) noexcept {
    const auto status = validate_composition_production_identity_v1(current);
    if (!status.valid()) return status;
    if (next_definition_digest.algorithm == digest_algorithm::none
        || !valid_content_digest(next_definition_digest)) {
        return {production_identity_code_v1::invalid_definition_digest};
    }
    if (current.version.revision
        == std::numeric_limits<std::uint64_t>::max()) {
        return {production_identity_code_v1::revision_overflow};
    }
    if (output == nullptr) {
        return {production_identity_code_v1::missing_output};
    }
    *output = current;
    ++output->version.revision;
    output->definition_digest = next_definition_digest;
    return {};
}

static_assert(sizeof(composition_production_id) == sizeof(std::uint64_t));
static_assert(sizeof(composition_lineage_id) == sizeof(std::uint64_t));
static_assert(std::is_standard_layout<composition_version_v1>::value);
static_assert(std::is_trivially_copyable<composition_version_v1>::value);
static_assert(std::is_standard_layout<composition_production_identity_v1>::value);
static_assert(
    std::is_trivially_copyable<composition_production_identity_v1>::value);

} // namespace cellshard::compiler::composition
