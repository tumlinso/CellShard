#include <CellShard/compiler/discovery/motif/frequent_fragment_miner_v1.hh>

#include <array>
#include <cassert>

namespace motif = cellshard::compiler::discovery::motif;
namespace atom = cellshard::compiler::atom;

int main() {
    const std::uint64_t a[] = {1, 2, 1, 11, 12};
    const std::uint64_t b[] = {1, 2, 1, 21, 22};
    motif::observed_typed_fragment_v1 observations[] = {
        {a, 5, {1, 1}, 1, 2, {2, 1}, {3, 1}, 4},
        {a, 5, {1, 2}, 1, 3, {2, 1}, {3, 1}, 4},
        {a, 5, {1, 2}, 1, 1, {2, 1}, {3, 1}, 4},
        {b, 5, {1, 3}, 1, 7, {2, 1}, {3, 1}, 4}};
    atom::atom_persistent_identity_v1 ids[] = {{4, 1}, {4, 2}};
    std::array<motif::frequent_typed_fragment_v1, 2> fragments{};
    motif::frequent_fragment_output_v1 output{
        fragments.data(), fragments.size(), 0};
    auto result = motif::mine_frequent_fragments_v1(
        observations, 4, {4, 8, 2}, ids, 2, &output);
    assert(result.complete() && output.fragment_count == 1);
    assert(output.fragments[0].supporting_graph_count == 2);
    assert(output.fragments[0].occurrence_count == 6);
    assert(!motif::authorizes_execution(output));
    result = motif::mine_frequent_fragments_v1(
        observations, 4, {3, 8, 2}, ids, 2, &output);
    assert(result.code
           == motif::frequent_fragment_code_v1::observation_bound_exceeded);
    observations[3].stratum_selection_generation = 5;
    assert(motif::mine_frequent_fragments_v1(
               observations, 4, {4, 8, 2}, ids, 2, &output).code
           == motif::frequent_fragment_code_v1::context_mismatch);
}
