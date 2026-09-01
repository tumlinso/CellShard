#include <CellShard/compiler/discovery/motif/recurrence_measurement_v1.hh>

#include <cassert>
#include <limits>

namespace motif = cellshard::compiler::discovery::motif;

int main() {
    const motif::motif_recurrence_key_v1 key{
        {1, 1}, {2, 2}, {3, 3}, {4, 4}, 5, 6};
    motif::motif_recurrence_observation_v1 observations[] = {
        {key, 2, 3, 8}, {key, 3, 4, 12}};
    motif::motif_recurrence_measurement_v1 output{};
    auto result = motif::measure_motif_recurrence_v1(
        observations, 2, &output);
    assert(result.measured());
    assert(output.graph_count == 5);
    assert(output.occurrence_count == 7);
    assert(output.opportunity_count == 20);
    assert(!motif::authorizes_execution(output));
    observations[1].key.graph_generation = 9;
    assert(motif::measure_motif_recurrence_v1(
               observations, 2, &output).code
           == motif::motif_recurrence_code_v1::key_mismatch);
    observations[1] = {key, 1, 1, 1};
    observations[0].graph_count = std::numeric_limits<std::uint64_t>::max();
    assert(motif::measure_motif_recurrence_v1(
               observations, 2, &output).code
           == motif::motif_recurrence_code_v1::count_overflow);
}
