#include <CellShard/compiler/discovery/operation_trace/provenance_comparison_v1.hh>

#include <cassert>
#include <cstdint>

namespace trace = cellshard::compiler::discovery::operation_trace;

namespace {

trace::discovery_provenance_v1 trace_provenance() {
    trace::discovery_provenance_v1 value{};
    value.evidence_identity = {10, 1};
    value.candidate_identity = {10, UINT64_C(0xfffffffffffffff0)};
    value.source_identity = {10, 3};
    value.algorithm_provenance_identity = {10, 4};
    value.candidate_generation = UINT64_C(0x100000001);
    value.observation_generation = 2;
    value.observation_count = 100;
    value.basis = trace::discovery_provenance_basis_v1::trace_only;
    return value;
}

trace::discovery_provenance_v1 biology_provenance() {
    auto value = trace_provenance();
    value.evidence_identity = {20, 1};
    value.source_identity = {20, 3};
    value.algorithm_provenance_identity = {20, 4};
    value.biological_stratum_identity = {20, 5};
    value.biological_stratum_generation = 6;
    value.observation_count = 80;
    value.basis = trace::discovery_provenance_basis_v1::biology_derived;
    return value;
}

} // namespace

int main() {
    const auto trace_only = trace_provenance();
    auto biology = biology_provenance();
    trace::provenance_comparison_v1 comparison{};
    assert(trace::compare_trace_and_biology_provenance_v1(
               {30, 1}, trace_only, biology, 90, 70, &comparison)
               .compared());
    assert(comparison.agreement
           == trace::provenance_support_agreement_v1::concordant_support);
    assert(comparison.candidate_generation == UINT64_C(0x100000001));
    assert(trace::is_independent_corroboration_v1(comparison));
    assert(!trace::authorizes_execution(comparison));

    biology.observation_count = 60;
    assert(trace::compare_trace_and_biology_provenance_v1(
               {30, 2}, trace_only, biology, 90, 70, &comparison)
               .compared());
    assert(comparison.agreement
           == trace::provenance_support_agreement_v1::trace_only_support);
    biology.source_identity = trace_only.source_identity;
    assert(trace::compare_trace_and_biology_provenance_v1(
               {30, 3}, trace_only, biology, 101, 70, &comparison)
               .compared());
    assert(comparison.agreement
           == trace::provenance_support_agreement_v1::concordant_no_support);
    assert(!trace::is_independent_corroboration_v1(comparison));

    auto malformed = biology_provenance();
    malformed.biological_stratum_identity = {};
    assert(trace::compare_trace_and_biology_provenance_v1(
               {30, 4}, trace_only, malformed, 1, 1, &comparison)
               .code
           == trace::provenance_comparison_code_v1::biology_missing_stratum);
    malformed = biology_provenance();
    malformed.candidate_identity = {10, 99};
    assert(trace::compare_trace_and_biology_provenance_v1(
               {30, 5}, trace_only, malformed, 1, 1, &comparison)
               .code
           == trace::provenance_comparison_code_v1::candidate_mismatch);
    malformed = biology_provenance();
    malformed.candidate_generation += 1;
    assert(trace::compare_trace_and_biology_provenance_v1(
               {30, 6}, trace_only, malformed, 1, 1, &comparison)
               .code
           == trace::provenance_comparison_code_v1::
               candidate_generation_mismatch);
    malformed = biology_provenance();
    malformed.evidence_identity = trace_only.evidence_identity;
    assert(trace::compare_trace_and_biology_provenance_v1(
               {30, 7}, trace_only, malformed, 1, 1, &comparison)
               .code
           == trace::provenance_comparison_code_v1::
               duplicate_evidence_identity);
}
