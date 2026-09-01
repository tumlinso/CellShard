#include <CellShard/compiler/discovery/operation_trace/graph_family_fragment_v1.hh>

#include <array>
#include <cassert>
#include <cstdint>

namespace trace = cellshard::compiler::discovery::operation_trace;

namespace {

trace::atom_access_event_v1 event(std::uint64_t sequence,
                                  std::uint64_t graph,
                                  std::uint64_t atom_local) {
    trace::atom_access_event_v1 value{};
    value.event_identity = {10 + graph, sequence};
    value.trace_identity = {10 + graph, 100};
    value.source_identity = {10, 101};
    value.workload_identity = {20, 1};
    value.graph_identity = {20, graph};
    value.operation_identity = {20, 3};
    value.stage_identity = {20, 4};
    value.atom_identity = {20, atom_local};
    value.port_identity = {20, 1000 + atom_local};
    value.trace_generation = 1;
    value.graph_generation = graph;
    value.operation_generation = 3;
    value.stage_generation = 4;
    value.atom_generation = 5;
    value.sequence_number = sequence;
    value.logical_byte_count = 8;
    return value;
}

} // namespace

int main() {
    std::array<trace::atom_access_event_v1, 5> left{
        event(1, 1, 90), event(2, 1, 10), event(3, 1, 11),
        event(4, 1, 12), event(5, 1, 91)};
    std::array<trace::atom_access_event_v1, 5> right{
        event(1, 2, 80), event(2, 2, 10), event(3, 2, 11),
        event(4, 2, 12), event(5, 2, 81)};
    trace::graph_family_fragment_v1 fragment{};
    const auto result = trace::discover_graph_family_fragment_v1(
        {left.data(), left.size(), {20, 1}, 1},
        {right.data(), right.size(), {20, 2}, 2},
        {1, 1},
        8,
        {30, 1},
        {30, 2},
        &fragment);
    assert(result.discovered());
    assert(result.event_count == 3);
    assert(fragment.left_begin == 1);
    assert(fragment.right_begin == 1);
    assert(fragment.event_count == 3);
    assert(!trace::authorizes_execution(fragment));

    // The u32 local extension cap truncates a match without truncating global
    // offsets or identities.
    assert(trace::discover_graph_family_fragment_v1(
               {left.data(), left.size(), {20, 1}, 1},
               {right.data(), right.size(), {20, 2}, 2},
               {1, 1}, 2, {30, 3}, {30, 4}, &fragment)
               .event_count
           == 2);

    assert(trace::discover_graph_family_fragment_v1(
               {left.data(), left.size(), {20, 1}, 1},
               {left.data(), left.size(), {20, 1}, 1},
               {0, 0}, 2, {30, 3}, {30, 4}, &fragment)
               .code
           == trace::graph_family_fragment_code_v1::same_graph);
    assert(trace::discover_graph_family_fragment_v1(
               {left.data(), left.size(), {20, 1}, 1},
               {right.data(), right.size(), {20, 2}, 2},
               {5, 0}, 2, {30, 3}, {30, 4}, &fragment)
               .code
           == trace::graph_family_fragment_code_v1::seed_out_of_range);
    assert(trace::discover_graph_family_fragment_v1(
               {left.data(), left.size(), {20, 1}, 1},
               {right.data(), right.size(), {20, 2}, 2},
               {0, 0}, 2, {30, 3}, {30, 4}, &fragment)
               .code
           == trace::graph_family_fragment_code_v1::no_common_fragment);

    auto stale = right;
    stale[2].atom_generation = 99;
    assert(trace::discover_graph_family_fragment_v1(
               {left.data(), left.size(), {20, 1}, 1},
               {stale.data(), stale.size(), {20, 2}, 2},
               {1, 1}, 8, {30, 5}, {30, 6}, &fragment)
               .event_count
           == 1);
}
