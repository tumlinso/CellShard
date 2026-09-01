#include <CellShard/compiler/discovery/operation_trace/persistent_order_v1.hh>

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
    value.logical_byte_count = 8;
    return value;
}

} // namespace

int main() {
    std::array<trace::persistent_order_counter_v1, 2> counters{};
    trace::persistent_order_state_v1 state{
        counters.data(), 2, 0, 4, 0, 0, {30, 1}, {30, 2}, 1};
    assert(trace::initialize_persistent_order_v1(&state).ok());
    assert(!trace::authorizes_execution(state));

    assert(trace::observe_persistent_order_v1(
               &state, event(1, UINT64_C(0x100000001)), event(3, 9))
               .ok());
    assert(trace::observe_persistent_order_v1(
               &state, event(10, UINT64_C(0x100000001)), event(11, 9))
               .ok());
    assert(counters[0].estimated_occurrences == 2);
    assert(counters[0].total_sequence_distance == 3);
    assert(counters[0].maximum_sequence_distance == 2);

    // Direction is semantic: reverse order is a distinct candidate.
    assert(trace::observe_persistent_order_v1(
               &state, event(20, 9), event(21, UINT64_C(0x100000001)))
               .ok());
    assert(state.counter_count == 2);
    assert(counters[1].key.predecessor.local_identity == 9);

    // A third key deterministically replaces the minimum counter.
    assert(trace::observe_persistent_order_v1(
               &state, event(30, 7), event(31, 8))
               .ok());
    assert(state.counter_count == 2);
    assert(counters[1].estimated_occurrences == 2);
    assert(counters[1].maximum_overestimate == 1);

    assert(trace::observe_persistent_order_v1(
               &state, event(5, 1), event(5, 2))
               .code
           == trace::persistent_order_code_v1::non_increasing_sequence);
    assert(trace::observe_persistent_order_v1(
               &state, event(1, 1), event(6, 2))
               .code
           == trace::persistent_order_code_v1::sequence_gap_exceeded);
    assert(trace::observe_persistent_order_v1(
               &state, event(1, 1), event(2, 1))
               .code
           == trace::persistent_order_code_v1::self_order);
    auto other_trace = event(2, 2);
    other_trace.trace_identity = {10, 999};
    assert(trace::observe_persistent_order_v1(
               &state, event(1, 1), other_trace)
               .code
           == trace::persistent_order_code_v1::incompatible_event_context);

    // Randomized bounded-window observations preserve K and distance bounds.
    std::array<trace::persistent_order_counter_v1, 7> random_counters{};
    trace::persistent_order_state_v1 random_state{
        random_counters.data(), 7, 0, 8, 0, 0, {31, 1}, {31, 2}, 1};
    assert(trace::initialize_persistent_order_v1(&random_state).ok());
    std::uint64_t random = UINT64_C(0x9e3779b97f4a7c15);
    for (std::uint64_t index = 0; index < 5000; ++index) {
        random ^= random << 7;
        random ^= random >> 9;
        const auto first = 1 + random % 53;
        auto second = 1 + (random >> 8) % 53;
        if (first == second) second = 54;
        const auto distance = 1 + (random >> 16) % 8;
        assert(trace::observe_persistent_order_v1(
                   &random_state,
                   event(100 + index * 16, first),
                   event(100 + index * 16 + distance, second))
                   .ok());
        assert(random_state.counter_count <= random_state.counter_capacity);
    }
    assert(random_state.observed_pairs == 5000);
    for (std::uint32_t index = 0; index < random_state.counter_count; ++index) {
        assert(random_counters[index].maximum_sequence_distance <= 8);
        assert(random_counters[index].maximum_overestimate
               < random_counters[index].estimated_occurrences);
    }
}
