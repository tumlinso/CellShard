#include <CellShard/compiler/discovery/operation_trace/coaccess_heavy_hitters_v1.hh>

#include <array>
#include <cassert>
#include <cstdint>
#include <limits>

namespace trace = cellshard::compiler::discovery::operation_trace;
namespace atom = cellshard::compiler::atom;
namespace evidence = cellshard::compiler::evidence;

namespace {

constexpr atom::atom_persistent_identity_v1 atom_id(
    std::uint64_t local) noexcept {
    return {100, local};
}

trace::atom_access_event_v1 event(std::uint64_t sequence,
                                  std::uint64_t local_atom) {
    trace::atom_access_event_v1 value{};
    value.event_identity = {200, sequence};
    value.trace_identity = {200, 900};
    value.source_identity = {200, 901};
    value.workload_identity = atom_id(1);
    value.graph_identity = atom_id(2);
    value.operation_identity = atom_id(3);
    value.stage_identity = atom_id(4);
    value.atom_identity = atom_id(local_atom);
    value.port_identity = atom_id(1000 + local_atom);
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
    std::array<trace::atom_coaccess_counter_v1, 3> counters{};
    trace::coaccess_heavy_hitter_state_v1 state{
        counters.data(),
        static_cast<std::uint32_t>(counters.size()),
        0,
        0,
        evidence::evidence_identity_v1{300, 1},
        evidence::evidence_identity_v1{300, 2},
        1};
    assert(trace::initialize_coaccess_heavy_hitters_v1(&state).ok());
    assert(!trace::authorizes_execution(state));

    const auto a = event(1, UINT64_C(0x100000001));
    const auto b = event(2, UINT64_C(0xfffffffffffffff0));
    assert(trace::observe_atom_coaccess_v1(&state, b, a, 4).ok());
    assert(state.counter_count == 1);
    assert(counters[0].key.first_atom == a.atom_identity);
    assert(counters[0].key.second_atom == b.atom_identity);
    assert(counters[0].estimated_weight == 4);
    assert(counters[0].maximum_overestimate == 0);
    assert(trace::observe_atom_coaccess_v1(&state, a, b, 3).ok());
    assert(counters[0].estimated_weight == 7);

    assert(trace::observe_atom_coaccess_v1(&state, event(3, 11), event(4, 12))
               .ok());
    assert(trace::observe_atom_coaccess_v1(&state, event(5, 13), event(6, 14))
               .ok());
    assert(state.counter_count == counters.size());
    assert(trace::observe_atom_coaccess_v1(&state, event(7, 15), event(8, 16), 2)
               .ok());
    assert(state.counter_count == counters.size());
    assert(counters[1].estimated_weight == 3);
    assert(counters[1].maximum_overestimate == 1);

    auto incompatible = event(9, 20);
    incompatible.stage_generation = 99;
    assert(trace::observe_atom_coaccess_v1(
               &state, event(10, 21), incompatible)
               .code
           == trace::coaccess_heavy_hitter_code_v1::
               incompatible_event_context);
    assert(trace::observe_atom_coaccess_v1(&state, a, a).code
           == trace::coaccess_heavy_hitter_code_v1::duplicate_event);
    assert(trace::observe_atom_coaccess_v1(&state, event(11, 30), event(12, 30))
               .code
           == trace::coaccess_heavy_hitter_code_v1::self_coaccess);
    assert(trace::observe_atom_coaccess_v1(&state, a, b, 0).code
           == trace::coaccess_heavy_hitter_code_v1::empty_weight);

    state.observed_weight = std::numeric_limits<std::uint64_t>::max();
    assert(trace::observe_atom_coaccess_v1(&state, a, b).code
           == trace::coaccess_heavy_hitter_code_v1::weight_overflow);

    // Randomized bounded-stream invariant: storage never exceeds K and every
    // counter retains an ordered, non-self global pair.
    std::array<trace::atom_coaccess_counter_v1, 8> random_counters{};
    trace::coaccess_heavy_hitter_state_v1 random_state{
        random_counters.data(), 8, 0, 0, {400, 1}, {400, 2}, 1};
    assert(trace::initialize_coaccess_heavy_hitters_v1(&random_state).ok());
    std::uint64_t random = UINT64_C(0xd1b54a32d192ed03);
    for (std::uint64_t index = 0; index < 10000; ++index) {
        random ^= random << 13;
        random ^= random >> 7;
        random ^= random << 17;
        const auto left_atom = 1 + (random % 97);
        auto right_atom = 1 + ((random >> 8) % 97);
        if (right_atom == left_atom) right_atom = 98;
        assert(trace::observe_atom_coaccess_v1(
                   &random_state,
                   event(100 + index * 2, left_atom),
                   event(101 + index * 2, right_atom))
                   .ok());
        assert(random_state.counter_count <= random_state.counter_capacity);
    }
    assert(random_state.observed_weight == 10000);
    for (std::uint32_t index = 0; index < random_state.counter_count; ++index) {
        assert(atom::atom_persistent_identity_less_v1(
            random_counters[index].key.first_atom,
            random_counters[index].key.second_atom));
        assert(random_counters[index].maximum_overestimate
               < random_counters[index].estimated_weight);
    }
}
