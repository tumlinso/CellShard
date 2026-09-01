#include <CellShard/compiler/evidence/evidence_atlas_v1.hh>

#include <cassert>
#include <vector>

namespace evidence = cellshard::compiler::evidence;

evidence::atom_evidence_record_v1 record(std::uint64_t id) {
    evidence::atom_evidence_record_v1 value{};
    value.evidence_identity = {1, id};
    value.subject_atom_identity = {2, id};
    value.source_identity = {3, id};
    value.observation_generation = 1;
    value.observation_count = id;
    value.kind = evidence::evidence_kind::support_signature;
    return value;
}

int main() {
    std::vector<evidence::atom_evidence_record_v1> records;
    for (std::uint64_t id = 1; id <= 257; ++id) records.push_back(record(id));
    evidence::evidence_atlas_source_v1 source{
        records.data(), records.size(), {9, 1}, 2};
    const auto requirements = evidence::evidence_atlas_requirements(source, 300);
    assert(requirements.ok());

    evidence::evidence_atlas_builder_v1 builder;
    auto result = builder.fill(source, requirements.requirements, 300);
    assert(result.code == evidence::evidence_atlas_build_code_v1::built);
    assert(builder.view().record_count == records.size());
    assert(builder.view().records != records.data());

    auto duplicate = records;
    duplicate[8].evidence_identity = duplicate[7].evidence_identity;
    source.records = duplicate.data();
    result = builder.fill(source, requirements.requirements, 300);
    assert(result.code == evidence::evidence_atlas_build_code_v1::unordered_or_duplicate_record);
    assert(builder.view().record_count == 0);

    source.records = records.data();
    result = builder.fill(source, {1, 1}, 300);
    assert(result.code == evidence::evidence_atlas_build_code_v1::insufficient_capacity);
    assert(builder.view().records == nullptr);
}
