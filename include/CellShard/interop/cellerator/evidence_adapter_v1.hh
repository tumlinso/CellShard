#pragma once

#include <CellShard/compiler/evidence/atom_evidence_record_v1.hh>

#include <Cellerator/geometry/support_atlas.hh>

#include <cstdint>
#include <limits>

namespace cellshard::interop::cellerator {

namespace evidence = ::cellshard::compiler::evidence;
namespace geometry = ::cellerator::geometry;

struct support_atlas_adapter_identity_v1 {
    std::uint64_t evidence_producer_namespace = 0;
    std::uint64_t first_record_local_identity = 0;
    evidence::evidence_identity_v1 subject_atom_identity{};
    std::uint64_t source_producer_namespace = 0;
};

enum class support_atlas_adapter_code_v1 : std::uint32_t {
    success = 0,
    unsupported_schema,
    invalid_source_identity,
    missing_structure_epoch,
    inconsistent_section_pointer,
    empty_atlas,
    invalid_adapter_identity,
    identity_overflow,
    missing_output,
    insufficient_capacity,
};

struct support_atlas_adapter_result_v1 {
    support_atlas_adapter_code_v1 code = support_atlas_adapter_code_v1::success;
    std::uint64_t required_records = 0;
    std::uint64_t section_index = 0;
    [[nodiscard]] constexpr bool ok() const noexcept {
        return code == support_atlas_adapter_code_v1::success;
    }
};

[[nodiscard]] constexpr evidence::evidence_kind adapt_support_evidence_kind_v1(
    geometry::support_evidence_kind_v1 kind) noexcept {
    switch (kind) {
    case geometry::support_evidence_kind_v1::sampled_co_support:
        return evidence::evidence_kind::co_support;
    case geometry::support_evidence_kind_v1::weighted_co_support:
        return evidence::evidence_kind::weighted_co_support;
    case geometry::support_evidence_kind_v1::normalized_association:
    case geometry::support_evidence_kind_v1::sparse_top_l_affinity:
        return evidence::evidence_kind::normalized_affinity;
    case geometry::support_evidence_kind_v1::community_assignment:
        return evidence::evidence_kind::community_membership;
    case geometry::support_evidence_kind_v1::prevalence:
    case geometry::support_evidence_kind_v1::destination_degree:
    case geometry::support_evidence_kind_v1::work_signature:
    case geometry::support_evidence_kind_v1::biological_stratum:
    case geometry::support_evidence_kind_v1::resampling_stability:
    case geometry::support_evidence_kind_v1::exact_rescan_summary:
    case geometry::support_evidence_kind_v1::deterministic_provenance:
    case geometry::support_evidence_kind_v1::validation_summary:
        return evidence::evidence_kind::support_signature;
    case geometry::support_evidence_kind_v1::none:
        return evidence::evidence_kind::invalid;
    }
    return evidence::evidence_kind::invalid;
}

[[nodiscard]] inline support_atlas_adapter_result_v1
support_atlas_adapter_requirements_v1(
    const geometry::support_atlas_view_v1 &source,
    support_atlas_adapter_identity_v1 identity) noexcept {
    if (source.schema_version != geometry::support_atlas_schema_version_v1)
        return {support_atlas_adapter_code_v1::unsupported_schema};
    if (source.evidence_identity == 0 || source.relation_identity == 0
        || source.structure_identity == 0)
        return {support_atlas_adapter_code_v1::invalid_source_identity};
    if (source.structure_epoch == 0)
        return {support_atlas_adapter_code_v1::missing_structure_epoch};
    const void *pointers[] = {source.prevalence, source.destination_degrees,
        source.co_support, source.affinity, source.communities,
        source.work_signatures, source.strata, source.stability,
        source.exact_rescans, source.validation_summaries};
    const std::uint64_t counts[] = {source.prevalence_count,
        source.destination_degree_count, source.co_support_count,
        source.affinity_count, source.community_count,
        source.work_signature_count, source.stratum_count,
        source.stability_count, source.exact_rescan_count,
        source.validation_summary_count};
    std::uint64_t required = 0;
    for (std::uint64_t index = 0; index < 10; ++index) {
        if ((counts[index] == 0) != (pointers[index] == nullptr))
            return {support_atlas_adapter_code_v1::inconsistent_section_pointer,
                    required, index};
        if (counts[index] != 0) ++required;
    }
    if (required == 0) return {support_atlas_adapter_code_v1::empty_atlas};
    if (identity.evidence_producer_namespace == 0
        || identity.first_record_local_identity == 0
        || identity.source_producer_namespace == 0
        || !evidence::valid_evidence_identity_v1(identity.subject_atom_identity))
        return {support_atlas_adapter_code_v1::invalid_adapter_identity, required};
    if (required - 1
        > std::numeric_limits<std::uint64_t>::max()
            - identity.first_record_local_identity)
        return {support_atlas_adapter_code_v1::identity_overflow, required};
    return {support_atlas_adapter_code_v1::success, required};
}

[[nodiscard]] inline support_atlas_adapter_result_v1 adapt_support_atlas_v1(
    const geometry::support_atlas_view_v1 &source,
    support_atlas_adapter_identity_v1 identity,
    evidence::atom_evidence_record_v1 *output,
    std::uint64_t output_capacity) noexcept {
    const auto requirement = support_atlas_adapter_requirements_v1(source, identity);
    if (!requirement.ok()) return requirement;
    if (output == nullptr)
        return {support_atlas_adapter_code_v1::missing_output,
                requirement.required_records};
    if (output_capacity < requirement.required_records)
        return {support_atlas_adapter_code_v1::insufficient_capacity,
                requirement.required_records};

    const std::uint64_t counts[] = {source.prevalence_count,
        source.destination_degree_count, source.co_support_count,
        source.affinity_count, source.community_count,
        source.work_signature_count, source.stratum_count,
        source.stability_count, source.exact_rescan_count,
        source.validation_summary_count};
    const geometry::support_evidence_kind_v1 kinds[] = {
        geometry::support_evidence_kind_v1::prevalence,
        geometry::support_evidence_kind_v1::destination_degree,
        (source.flags & geometry::support_atlas_flag_weighted) != 0
            ? geometry::support_evidence_kind_v1::weighted_co_support
            : geometry::support_evidence_kind_v1::sampled_co_support,
        geometry::support_evidence_kind_v1::sparse_top_l_affinity,
        geometry::support_evidence_kind_v1::community_assignment,
        geometry::support_evidence_kind_v1::work_signature,
        geometry::support_evidence_kind_v1::biological_stratum,
        geometry::support_evidence_kind_v1::resampling_stability,
        geometry::support_evidence_kind_v1::exact_rescan_summary,
        geometry::support_evidence_kind_v1::validation_summary};
    std::uint64_t written = 0;
    for (std::uint64_t section = 0; section < 10; ++section) {
        if (counts[section] == 0) continue;
        auto &record = output[written];
        record = {};
        record.evidence_identity = {identity.evidence_producer_namespace,
            identity.first_record_local_identity + written};
        record.subject_atom_identity = identity.subject_atom_identity;
        record.source_identity = {identity.source_producer_namespace,
                                  source.evidence_identity};
        record.observation_generation = source.structure_epoch;
        record.observation_count = counts[section];
        record.kind = adapt_support_evidence_kind_v1(kinds[section]);
        ++written;
    }
    return {support_atlas_adapter_code_v1::success, written};
}

// Cellerator exact-rescan summaries remain source-linked proposal evidence.
// The independent CellShard certification lane owns exact coverage.
[[nodiscard]] constexpr bool adapted_exact_rescan_certifies_coverage_v1() noexcept {
    return false;
}

} // namespace cellshard::interop::cellerator
