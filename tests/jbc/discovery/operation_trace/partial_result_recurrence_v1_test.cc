#include <CellShard/compiler/discovery/operation_trace/partial_result_recurrence_v1.hh>

#include <array>
#include <cassert>
#include <cstdint>

namespace trace = cellshard::compiler::discovery::operation_trace;

namespace {

trace::atom_access_event_v1 event(std::uint64_t sequence,
                                  std::uint64_t atom_local) {
    trace::atom_access_event_v1 value{};
    value.event_identity = {10, sequence};
    value.trace_identity = {10, 100};
    value.source_identity = {10, 101};
    value.workload_identity = {20, 1};
    value.graph_identity = {20, 2};
    value.operation_identity = {20, 3};
    value.stage_identity = {20, 4};
    value.atom_identity = {20, atom_local};
    value.port_identity = {20, 1000 + atom_local};
    value.trace_generation = 1;
    value.graph_generation = 2;
    value.operation_generation = 3;
    value.stage_generation = 4;
    value.atom_generation = 5;
    value.sequence_number = sequence;
    value.logical_byte_count = UINT64_C(1) << 40;
    return value;
}

std::array<trace::atom_access_event_v1, 2> reuse(
    std::uint64_t sequence,
    std::uint64_t partial) {
    auto producer = event(sequence, partial);
    producer.mode = trace::atom_access_mode_v1::write;
    producer.role = trace::atom_access_role_v1::intermediate;
    auto consumer = event(sequence + 1, partial);
    consumer.operation_identity = {20, 30};
    consumer.stage_identity = {20, 40};
    consumer.operation_generation = 30;
    consumer.stage_generation = 40;
    consumer.mode = trace::atom_access_mode_v1::read;
    consumer.role = trace::atom_access_role_v1::input;
    return {producer, consumer};
}

} // namespace

int main() {
    std::array<trace::partial_result_counter_v1, 2> counters{};
    trace::partial_result_recurrence_state_v1 state{
        counters.data(), 2, 0, 4, 0, 0, {30, 1}, {30, 2}, 1};
    assert(trace::initialize_partial_result_recurrence_v1(&state).ok());
    assert(!trace::authorizes_execution(state));

    const auto first = reuse(1, UINT64_C(0xfffffffffffffff0));
    assert(trace::observe_partial_result_reuse_v1(
               &state, first[0], first[1])
               .ok());
    const auto repeated = reuse(10, UINT64_C(0xfffffffffffffff0));
    assert(trace::observe_partial_result_reuse_v1(
               &state, repeated[0], repeated[1])
               .ok());
    assert(counters[0].estimated_reuses == 2);
    assert(counters[0].logical_byte_count == (UINT64_C(1) << 40));

    auto malformed = first;
    malformed[1].atom_identity.local_identity = 99;
    assert(trace::observe_partial_result_reuse_v1(
               &state, malformed[0], malformed[1])
               .code
           == trace::partial_result_recurrence_code_v1::
               mismatched_partial_identity);
    malformed = first;
    malformed[1].atom_generation = 6;
    assert(trace::observe_partial_result_reuse_v1(
               &state, malformed[0], malformed[1])
               .code
           == trace::partial_result_recurrence_code_v1::
               stale_partial_generation);
    malformed = first;
    malformed[0].mode = trace::atom_access_mode_v1::read;
    assert(trace::observe_partial_result_reuse_v1(
               &state, malformed[0], malformed[1])
               .code
           == trace::partial_result_recurrence_code_v1::
               invalid_producer_access);
    malformed = first;
    malformed[1].operation_identity = malformed[0].operation_identity;
    malformed[1].stage_identity = malformed[0].stage_identity;
    assert(trace::observe_partial_result_reuse_v1(
               &state, malformed[0], malformed[1])
               .code
           == trace::partial_result_recurrence_code_v1::same_operation_stage);

    // Bounded replacement retains K without treating bytes as identity.
    for (std::uint64_t index = 1; index <= 20; ++index) {
        const auto observation = reuse(100 + index * 2, 1000 + index);
        assert(trace::observe_partial_result_reuse_v1(
                   &state, observation[0], observation[1])
                   .ok());
        assert(state.counter_count <= state.counter_capacity);
    }
    assert(state.observed_reuses == 22);
}
