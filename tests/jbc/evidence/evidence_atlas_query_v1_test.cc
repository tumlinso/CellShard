#include <CellShard/compiler/evidence/evidence_atlas_query_v1.hh>

#include <array>
#include <cassert>
#include <vector>

namespace evidence = cellshard::compiler::evidence;

evidence::atom_evidence_record_v1 query_record(std::uint64_t id) {
    evidence::atom_evidence_record_v1 value{};
    value.evidence_identity = {1, id};
    value.subject_atom_identity = {2, id % 3 + 1};
    value.source_identity = {3, id % 2 + 1};
    value.observation_generation = 1;
    value.observation_count = id;
    value.kind = id % 2 == 0 ? evidence::evidence_kind::co_support
                             : evidence::evidence_kind::negative;
    return value;
}

int main() {
    std::vector<evidence::atom_evidence_record_v1> records;
    for (std::uint64_t id = 1; id <= 101; ++id) records.push_back(query_record(id));
    evidence::evidence_atlas_view_v1 atlas{
        records.data(), records.size(), {9, 1}, 1};
    evidence::evidence_query_v1 query{};
    query.family = evidence::evidence_family::negative;
    query.minimum_observation_count = 50;
    const auto requirement = evidence::evidence_filter_requirements_v1(atlas, query, 200);
    assert(requirement.ok());
    std::vector<const evidence::atom_evidence_record_v1 *> output(
        requirement.match_count);
    const auto filtered = evidence::filter_evidence_atlas_v1(
        atlas, query, output.data(), output.size(), 200);
    assert(filtered.ok());
    for (const auto *record : output) {
        assert(record->kind == evidence::evidence_kind::negative);
        assert(record->observation_count >= 50);
    }

    const auto found = evidence::find_evidence_v1(atlas, {1, 73}, 200);
    assert(found.ok() && found.record->observation_count == 73);
    assert(evidence::find_evidence_v1(atlas, {1, 102}, 200).code
           == evidence::evidence_query_code_v1::not_found);
    assert(evidence::filter_evidence_atlas_v1(
               atlas, query, output.data(), 1, 200).code
           == evidence::evidence_query_code_v1::insufficient_capacity);
}
