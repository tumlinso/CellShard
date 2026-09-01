#pragma once

#include <CellShard/compiler/evidence/atom_evidence_record_v1.hh>

#include <cstdint>
#include <type_traits>

namespace cellshard::compiler::discovery::factor_topic {

inline constexpr std::uint32_t external_factor_topic_evidence_schema_version_v1 = 1;

enum class external_factor_topic_kind_v1 : std::uint32_t {
    factor = 1,
    topic = 2,
};

struct external_factor_topic_evidence_v1 {
    std::uint32_t schema_version = external_factor_topic_evidence_schema_version_v1;
    std::uint32_t record_bytes = sizeof(external_factor_topic_evidence_v1);
    evidence::evidence_identity_v1 evidence_identity{};
    evidence::evidence_identity_v1 subject_atom_identity{};
    evidence::evidence_identity_v1 source_identity{};
    std::uint64_t observation_generation = 0;
    std::uint64_t observation_count = 0;
    external_factor_topic_kind_v1 kind = external_factor_topic_kind_v1::factor;
    std::uint32_t reserved = 0;
};

enum class external_factor_topic_adapter_code_v1 : std::uint32_t {
    adapted = 0,
    null_destination,
    unsupported_schema,
    invalid_record_bytes,
    invalid_kind,
    nonzero_reserved,
    invalid_evidence_identity,
    invalid_subject_atom_identity,
    invalid_source_identity,
    missing_observation_generation,
    empty_observations,
};

struct external_factor_topic_adapter_result_v1 {
    external_factor_topic_adapter_code_v1 code =
        external_factor_topic_adapter_code_v1::adapted;

    [[nodiscard]] constexpr bool adapted() const noexcept {
        return code == external_factor_topic_adapter_code_v1::adapted;
    }
};

[[nodiscard]] constexpr bool valid_external_factor_topic_kind_v1(
    external_factor_topic_kind_v1 kind) noexcept {
    return kind == external_factor_topic_kind_v1::factor
        || kind == external_factor_topic_kind_v1::topic;
}

// The adapter deliberately preserves the provider's explicit identities. A
// factor/topic label, ordinal, shape, or payload location is never promoted to
// biological identity. The result remains proposal-only evidence.
[[nodiscard]] inline external_factor_topic_adapter_result_v1
adapt_external_factor_topic_evidence_v1(
    const external_factor_topic_evidence_v1 &source,
    evidence::atom_evidence_record_v1 *destination) noexcept {
    if (destination == nullptr) {
        return {external_factor_topic_adapter_code_v1::null_destination};
    }
    *destination = {};
    if (source.schema_version
        != external_factor_topic_evidence_schema_version_v1) {
        return {external_factor_topic_adapter_code_v1::unsupported_schema};
    }
    if (source.record_bytes != sizeof(external_factor_topic_evidence_v1)) {
        return {external_factor_topic_adapter_code_v1::invalid_record_bytes};
    }
    if (!valid_external_factor_topic_kind_v1(source.kind)) {
        return {external_factor_topic_adapter_code_v1::invalid_kind};
    }
    if (source.reserved != 0) {
        return {external_factor_topic_adapter_code_v1::nonzero_reserved};
    }
    if (!evidence::valid_evidence_identity_v1(source.evidence_identity)) {
        return {external_factor_topic_adapter_code_v1::invalid_evidence_identity};
    }
    if (!evidence::valid_evidence_identity_v1(source.subject_atom_identity)) {
        return {external_factor_topic_adapter_code_v1::invalid_subject_atom_identity};
    }
    if (!evidence::valid_evidence_identity_v1(source.source_identity)) {
        return {external_factor_topic_adapter_code_v1::invalid_source_identity};
    }
    if (source.observation_generation == 0) {
        return {external_factor_topic_adapter_code_v1::missing_observation_generation};
    }
    if (source.observation_count == 0) {
        return {external_factor_topic_adapter_code_v1::empty_observations};
    }

    *destination = {evidence::atom_evidence_record_schema_version_v1,
                    sizeof(evidence::atom_evidence_record_v1),
                    source.evidence_identity,
                    source.subject_atom_identity,
                    source.source_identity,
                    source.observation_generation,
                    source.observation_count,
                    evidence::evidence_kind::factor_membership,
                    evidence::evidence_disposition_v1::proposal_only};
    return {};
}

[[nodiscard]] constexpr bool authorizes_execution(
    const external_factor_topic_evidence_v1 &) noexcept {
    return false;
}

static_assert(std::is_standard_layout<external_factor_topic_evidence_v1>::value);
static_assert(std::is_trivially_copyable<external_factor_topic_evidence_v1>::value);

} // namespace cellshard::compiler::discovery::factor_topic
