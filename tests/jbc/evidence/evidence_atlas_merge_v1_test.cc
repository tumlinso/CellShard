#include <CellShard/compiler/evidence/evidence_atlas_merge_v1.hh>

#include <array>
#include <cassert>

namespace evidence = cellshard::compiler::evidence;

evidence::atom_evidence_record_v1 make_record(std::uint64_t id) {
    evidence::atom_evidence_record_v1 value{};
    value.evidence_identity = {1, id};
    value.subject_atom_identity = {2, id};
    value.source_identity = {3, id};
    value.observation_generation = 1;
    value.observation_count = id;
    value.kind = evidence::evidence_kind::co_support;
    return value;
}

int main() {
    std::array left_records{make_record(1), make_record(3), make_record(5)};
    std::array right_records{make_record(2), make_record(3), make_record(4)};
    evidence::evidence_atlas_view_v1 left{
        left_records.data(), left_records.size(), {8, 1}, 1};
    evidence::evidence_atlas_view_v1 right{
        right_records.data(), right_records.size(), {8, 2}, 1};
    evidence::evidence_atlas_builder_v1 output;
    auto result = evidence::merge_evidence_atlases_v1(
        left, right, {8, 3}, 2, 8, &output);
    assert(result.merged());
    assert(output.view().record_count == 5);
    for (std::uint64_t index = 0; index < 5; ++index)
        assert(output.view().records[index].evidence_identity.local_identity == index + 1);

    right_records[1].observation_count = 99;
    result = evidence::merge_evidence_atlases_v1(
        left, right, {8, 4}, 3, 8, &output);
    assert(result.code == evidence::evidence_atlas_merge_code_v1::conflicting_duplicate);
    assert(output.view().record_count == 0);
}
