#pragma once

#include <CellShard/compiler/discovery/operation_trace/negative_trace_summary_v1.hh>

#include <cstdint>
#include <type_traits>

namespace cellshard::compiler::discovery::operation_trace {

inline constexpr std::uint32_t provenance_comparison_schema_version_v1 = 1;

enum class discovery_provenance_basis_v1 : std::uint8_t {
    trace_only = 1,
    biology_derived = 2,
};

struct discovery_provenance_v1 {
    evidence::evidence_identity_v1 evidence_identity{};
    evidence::evidence_identity_v1 candidate_identity{};
    evidence::evidence_identity_v1 source_identity{};
    evidence::evidence_identity_v1 algorithm_provenance_identity{};
    evidence::evidence_identity_v1 biological_stratum_identity{};
    std::uint64_t candidate_generation = 0;
    std::uint64_t observation_generation = 0;
    std::uint64_t observation_count = 0;
    std::uint64_t biological_stratum_generation = 0;
    discovery_provenance_basis_v1 basis =
        discovery_provenance_basis_v1::trace_only;
    std::uint8_t reserved8[3]{};
    std::uint32_t reserved32 = 0;
};

enum class provenance_support_agreement_v1 : std::uint8_t {
    concordant_support = 1,
    trace_only_support = 2,
    biology_only_support = 3,
    concordant_no_support = 4,
};

enum class provenance_source_relation_v1 : std::uint8_t {
    distinct_sources = 1,
    shared_source = 2,
};

struct provenance_comparison_v1 {
    std::uint32_t schema_version = provenance_comparison_schema_version_v1;
    std::uint32_t record_bytes = sizeof(provenance_comparison_v1);
    evidence::evidence_identity_v1 comparison_identity{};
    evidence::evidence_identity_v1 candidate_identity{};
    evidence::evidence_identity_v1 trace_evidence_identity{};
    evidence::evidence_identity_v1 biology_evidence_identity{};
    std::uint64_t candidate_generation = 0;
    std::uint64_t trace_observation_count = 0;
    std::uint64_t biology_observation_count = 0;
    std::uint64_t trace_support_threshold = 0;
    std::uint64_t biology_support_threshold = 0;
    provenance_support_agreement_v1 agreement =
        provenance_support_agreement_v1::concordant_no_support;
    provenance_source_relation_v1 source_relation =
        provenance_source_relation_v1::shared_source;
    std::uint16_t reserved16 = 0;
    std::uint32_t reserved32 = 0;
};

enum class provenance_comparison_code_v1 : std::uint32_t {
    compared = 0,
    invalid_comparison_identity,
    invalid_trace_provenance,
    invalid_biology_provenance,
    wrong_trace_basis,
    wrong_biology_basis,
    trace_has_biological_stratum,
    biology_missing_stratum,
    duplicate_evidence_identity,
    candidate_mismatch,
    candidate_generation_mismatch,
    invalid_threshold,
    missing_output,
};

struct provenance_comparison_result_v1 {
    provenance_comparison_code_v1 code =
        provenance_comparison_code_v1::compared;

    [[nodiscard]] constexpr bool compared() const noexcept {
        return code == provenance_comparison_code_v1::compared;
    }
};

[[nodiscard]] constexpr bool valid_discovery_provenance_identity_fields_v1(
    const discovery_provenance_v1 &value) noexcept {
    return evidence::valid_evidence_identity_v1(value.evidence_identity)
        && evidence::valid_evidence_identity_v1(value.candidate_identity)
        && evidence::valid_evidence_identity_v1(value.source_identity)
        && evidence::valid_evidence_identity_v1(
            value.algorithm_provenance_identity)
        && value.candidate_generation != 0
        && value.observation_generation != 0
        && value.observation_count != 0
        && value.reserved8[0] == 0
        && value.reserved8[1] == 0
        && value.reserved8[2] == 0
        && value.reserved32 == 0;
}

// Compares two explicitly different evidence records for one candidate. This
// records agreement and source sharing; it does not turn either proposal into
// exact coverage or treat a shared source as independent corroboration.
[[nodiscard]] constexpr provenance_comparison_result_v1
compare_trace_and_biology_provenance_v1(
    evidence::evidence_identity_v1 comparison_identity,
    const discovery_provenance_v1 &trace,
    const discovery_provenance_v1 &biology,
    std::uint64_t trace_support_threshold,
    std::uint64_t biology_support_threshold,
    provenance_comparison_v1 *output) noexcept {
    if (!evidence::valid_evidence_identity_v1(comparison_identity)) {
        return {provenance_comparison_code_v1::invalid_comparison_identity};
    }
    if (!valid_discovery_provenance_identity_fields_v1(trace)) {
        return {provenance_comparison_code_v1::invalid_trace_provenance};
    }
    if (!valid_discovery_provenance_identity_fields_v1(biology)) {
        return {provenance_comparison_code_v1::invalid_biology_provenance};
    }
    if (trace.basis != discovery_provenance_basis_v1::trace_only) {
        return {provenance_comparison_code_v1::wrong_trace_basis};
    }
    if (biology.basis != discovery_provenance_basis_v1::biology_derived) {
        return {provenance_comparison_code_v1::wrong_biology_basis};
    }
    if (evidence::valid_evidence_identity_v1(
            trace.biological_stratum_identity)
        || trace.biological_stratum_generation != 0) {
        return {provenance_comparison_code_v1::trace_has_biological_stratum};
    }
    if (!evidence::valid_evidence_identity_v1(
            biology.biological_stratum_identity)
        || biology.biological_stratum_generation == 0) {
        return {provenance_comparison_code_v1::biology_missing_stratum};
    }
    if (trace.evidence_identity == biology.evidence_identity) {
        return {provenance_comparison_code_v1::duplicate_evidence_identity};
    }
    if (!(trace.candidate_identity == biology.candidate_identity)) {
        return {provenance_comparison_code_v1::candidate_mismatch};
    }
    if (trace.candidate_generation != biology.candidate_generation) {
        return {provenance_comparison_code_v1::
                    candidate_generation_mismatch};
    }
    if (trace_support_threshold == 0 || biology_support_threshold == 0) {
        return {provenance_comparison_code_v1::invalid_threshold};
    }
    if (output == nullptr) {
        return {provenance_comparison_code_v1::missing_output};
    }
    *output = {};

    const bool trace_supported =
        trace.observation_count >= trace_support_threshold;
    const bool biology_supported =
        biology.observation_count >= biology_support_threshold;
    provenance_support_agreement_v1 agreement =
        provenance_support_agreement_v1::concordant_no_support;
    if (trace_supported && biology_supported) {
        agreement = provenance_support_agreement_v1::concordant_support;
    } else if (trace_supported) {
        agreement = provenance_support_agreement_v1::trace_only_support;
    } else if (biology_supported) {
        agreement = provenance_support_agreement_v1::biology_only_support;
    }
    *output = {
        provenance_comparison_schema_version_v1,
        sizeof(provenance_comparison_v1),
        comparison_identity,
        trace.candidate_identity,
        trace.evidence_identity,
        biology.evidence_identity,
        trace.candidate_generation,
        trace.observation_count,
        biology.observation_count,
        trace_support_threshold,
        biology_support_threshold,
        agreement,
        trace.source_identity == biology.source_identity
            ? provenance_source_relation_v1::shared_source
            : provenance_source_relation_v1::distinct_sources,
        0,
        0};
    return {};
}

[[nodiscard]] constexpr bool is_independent_corroboration_v1(
    const provenance_comparison_v1 &comparison) noexcept {
    return comparison.source_relation
            == provenance_source_relation_v1::distinct_sources
        && !(comparison.trace_evidence_identity
             == comparison.biology_evidence_identity);
}

[[nodiscard]] constexpr bool authorizes_execution(
    const provenance_comparison_v1 &) noexcept {
    return false;
}

static_assert(std::is_standard_layout<discovery_provenance_v1>::value);
static_assert(std::is_trivially_copyable<discovery_provenance_v1>::value);
static_assert(std::is_standard_layout<provenance_comparison_v1>::value);
static_assert(std::is_trivially_copyable<provenance_comparison_v1>::value);

} // namespace cellshard::compiler::discovery::operation_trace
