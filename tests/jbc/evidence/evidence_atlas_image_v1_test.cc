#include <CellShard/compiler/evidence/evidence_atlas_image_v1.hh>
#include <CellShard/compiler/evidence/evidence_atlas_merge_v1.hh>

#include <cassert>
#include <vector>

namespace evidence = cellshard::compiler::evidence;

evidence::atom_evidence_record_v1 image_record(std::uint64_t id) {
    evidence::atom_evidence_record_v1 value{};
    value.evidence_identity = {1, id};
    value.subject_atom_identity = {2, id};
    value.source_identity = {3, id};
    value.observation_generation = 4;
    value.observation_count = id;
    value.kind = evidence::evidence_kind::cross_dataset_template;
    return value;
}

int main() {
    std::vector<evidence::atom_evidence_record_v1> records;
    for (std::uint64_t id = 1; id <= 129; ++id) records.push_back(image_record(id));
    evidence::evidence_atlas_view_v1 atlas{
        records.data(), records.size(), {9, 1}, 7};
    const auto requirement = evidence::evidence_atlas_image_requirements_v1(atlas, 200);
    assert(requirement.ok());
    std::vector<unsigned char> first(requirement.required_bytes);
    std::vector<unsigned char> second(requirement.required_bytes);
    assert(evidence::encode_evidence_atlas_v1(atlas, first.data(), first.size(), 200).ok());
    assert(evidence::encode_evidence_atlas_v1(atlas, second.data(), second.size(), 200).ok());
    assert(first == second);

    evidence::evidence_atlas_builder_v1 decoded;
    assert(evidence::decode_evidence_atlas_v1(
        first.data(), first.size(), 200, &decoded).ok());
    assert(decoded.view().record_count == records.size());
    assert(decoded.view().atlas_identity == atlas.atlas_identity);
    for (std::uint64_t index = 0; index < records.size(); ++index)
        assert(evidence::atom_evidence_record_equal_v1(
            decoded.view().records[index], records[index]));

    first.back() ^= 1;
    assert(evidence::decode_evidence_atlas_v1(
        first.data(), first.size(), 200, &decoded).code
           == evidence::evidence_atlas_image_code_v1::checksum_mismatch);
    assert(decoded.view().record_count == 0);
    assert(evidence::decode_evidence_atlas_v1(
        second.data(), second.size() - 1, 200, &decoded).code
           == evidence::evidence_atlas_image_code_v1::invalid_total_size);
}
