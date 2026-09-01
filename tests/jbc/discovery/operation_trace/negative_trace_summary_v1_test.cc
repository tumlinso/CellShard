#include <CellShard/compiler/discovery/operation_trace/negative_trace_summary_v1.hh>

#include <cassert>
#include <cstdint>
#include <limits>

namespace trace = cellshard::compiler::discovery::operation_trace;
namespace evidence = cellshard::compiler::evidence;

namespace {

trace::negative_trace_summary_input_v1 input() {
    trace::negative_trace_summary_input_v1 value{};
    value.summary_identity = {10, 1};
    value.evidence_identity = {10, 2};
    value.trace_scope_identity = {10, 3};
    value.observation_generation = UINT64_C(0x100000001);
    value.attempted_observations[0] = 100;
    value.rejected_observations[0] = 20;
    value.attempted_observations[1] = 200;
    value.rejected_observations[1] = 40;
    value.attempted_observations[2] = 300;
    value.rejected_observations[2] = 60;
    value.attempted_observations[3] = 400;
    value.rejected_observations[3] = 80;
    value.candidate_capacities[0] = 8;
    value.candidate_capacities[1] = 8;
    value.candidate_capacities[2] = 8;
    value.candidate_capacities[3] = 8;
    value.maximum_sequence_gap = 32;
    value.maximum_fragment_events = 64;
    value.reason = evidence::negative_evidence_reason_v1::not_observed;
    return value;
}

} // namespace

int main() {
    const auto source = input();
    trace::negative_trace_summary_v1 summary{};
    assert(trace::build_negative_trace_summary_v1(source, &summary).built());
    assert(summary.schema_version
           == trace::negative_trace_summary_schema_version_v1);
    assert(summary.record_bytes == sizeof(summary));
    assert(summary.negative.attempted_observations == 1000);
    assert(summary.negative.contradictory_observations == 200);
    assert(!trace::certifies_absence(summary));
    assert(!trace::authorizes_execution(summary));

    auto malformed = source;
    malformed.retained_candidates[0] = 1;
    assert(trace::build_negative_trace_summary_v1(malformed, &summary).code
           == trace::negative_trace_summary_code_v1::
               inconsistent_no_observation_reason);
    malformed = source;
    malformed.reason =
        evidence::negative_evidence_reason_v1::candidate_cap_reached;
    assert(trace::build_negative_trace_summary_v1(malformed, &summary).code
           == trace::negative_trace_summary_code_v1::inconsistent_cap_reason);
    malformed.retained_candidates[2] = 8;
    assert(trace::build_negative_trace_summary_v1(malformed, &summary).built());
    malformed = source;
    malformed.rejected_observations[1] = 201;
    assert(trace::build_negative_trace_summary_v1(malformed, &summary).code
           == trace::negative_trace_summary_code_v1::rejection_overflow);
    malformed = source;
    malformed.retained_candidates[1] = 9;
    assert(trace::build_negative_trace_summary_v1(malformed, &summary).code
           == trace::negative_trace_summary_code_v1::
               candidate_capacity_overflow);
    malformed = source;
    malformed.attempted_observations[0] =
        std::numeric_limits<std::uint64_t>::max();
    assert(trace::build_negative_trace_summary_v1(malformed, &summary).code
           == trace::negative_trace_summary_code_v1::count_overflow);
    assert(trace::build_negative_trace_summary_v1(source, nullptr).code
           == trace::negative_trace_summary_code_v1::missing_output);
}
