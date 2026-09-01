#pragma once

#include <CellShard/compiler/evidence/atom_evidence_record_v1.hh>

#include <cstdint>
#include <type_traits>

namespace cellshard::compiler::evidence {

inline constexpr std::uint32_t biological_stratum_schema_version_v1 = 1;

struct biological_stratum_ref_v1 {
    std::uint32_t schema_version = biological_stratum_schema_version_v1;
    std::uint32_t record_bytes = sizeof(biological_stratum_ref_v1);
    evidence_identity_v1 stratum_identity{};
    evidence_identity_v1 domain_identity{};
    evidence_identity_v1 order_identity{};
    evidence_identity_v1 selection_identity{};
    std::uint64_t domain_generation = 0;
    std::uint64_t selection_generation = 0;
    std::uint64_t selected_element_count = 0;
};

enum class biological_stratum_validation_code_v1 : std::uint32_t {
    valid = 0,
    unsupported_schema,
    invalid_record_bytes,
    invalid_stratum_identity,
    invalid_domain_identity,
    invalid_order_identity,
    invalid_selection_identity,
    missing_domain_generation,
    missing_selection_generation,
    empty_selection,
};

struct biological_stratum_validation_v1 {
    biological_stratum_validation_code_v1 code =
        biological_stratum_validation_code_v1::valid;
    [[nodiscard]] constexpr bool valid() const noexcept {
        return code == biological_stratum_validation_code_v1::valid;
    }
};

[[nodiscard]] constexpr biological_stratum_validation_v1
validate_biological_stratum_v1(
    const biological_stratum_ref_v1 &record) noexcept {
    if (record.schema_version != biological_stratum_schema_version_v1)
        return {biological_stratum_validation_code_v1::unsupported_schema};
    if (record.record_bytes != sizeof(biological_stratum_ref_v1))
        return {biological_stratum_validation_code_v1::invalid_record_bytes};
    if (!valid_evidence_identity_v1(record.stratum_identity))
        return {biological_stratum_validation_code_v1::invalid_stratum_identity};
    if (!valid_evidence_identity_v1(record.domain_identity))
        return {biological_stratum_validation_code_v1::invalid_domain_identity};
    if (!valid_evidence_identity_v1(record.order_identity))
        return {biological_stratum_validation_code_v1::invalid_order_identity};
    if (!valid_evidence_identity_v1(record.selection_identity))
        return {biological_stratum_validation_code_v1::invalid_selection_identity};
    if (record.domain_generation == 0)
        return {biological_stratum_validation_code_v1::missing_domain_generation};
    if (record.selection_generation == 0)
        return {biological_stratum_validation_code_v1::missing_selection_generation};
    if (record.selected_element_count == 0)
        return {biological_stratum_validation_code_v1::empty_selection};
    return {};
}

static_assert(std::is_standard_layout<biological_stratum_ref_v1>::value);
static_assert(std::is_trivially_copyable<biological_stratum_ref_v1>::value);

} // namespace cellshard::compiler::evidence
