#include <CellShard/compiler/evidence/evidence_atlas_statistics_v1.hh>

#include <cassert>
#include <limits>
#include <vector>

namespace evidence = cellshard::compiler::evidence;

evidence::atom_evidence_record_v1 statistics_record(std::uint64_t id) {
    evidence::atom_evidence_record_v1 value{};
    value.evidence_identity = {1, id};
    value.subject_atom_identity = {2, id};
    value.source_identity = {3, id};
    value.observation_generation = 1;
    value.observation_count = id;
    value.kind = id % 5 == 0 ? evidence::evidence_kind::negative
                             : (id % 2 == 0
                                    ? evidence::evidence_kind::co_support
                                    : evidence::evidence_kind::cross_workload);
    return value;
}

int main() {
    std::vector<evidence::atom_evidence_record_v1> records;
    for (std::uint64_t id = 1; id <= 1000; ++id)
        records.push_back(statistics_record(id));
    evidence::evidence_atlas_view_v1 atlas{
        records.data(), records.size(), {9, 1}, 1};
    evidence::evidence_atlas_statistics_v1 statistics{};
    const auto result = evidence::validate_and_measure_evidence_atlas_v1(
        atlas, 1000, &statistics);
    assert(result.ok());
    assert(statistics.record_count == 1000);
    assert(statistics.total_observation_count == 500500);
    assert(statistics.maximum_observation_count == 1000);
    assert(statistics.negative_record_count == 200);
    assert(statistics.records_by_kind[
               static_cast<std::uint32_t>(evidence::evidence_kind::negative)]
           == 200);
    assert(!evidence::statistics_authorize_execution(statistics));

    records.resize(2);
    records[0].observation_count = std::numeric_limits<std::uint64_t>::max();
    records[1].observation_count = 1;
    atlas.record_count = records.size();
    assert(evidence::validate_and_measure_evidence_atlas_v1(
        atlas, 2, &statistics).code
           == evidence::evidence_atlas_statistics_code_v1::
               observation_count_overflow);
    assert(statistics.record_count == 0);
}
