#include <CellShard/compiler/discovery/multimodal/multi_payload_atom_v1.hh>

#include <cassert>

namespace multimodal = cellshard::compiler::discovery::multimodal;

int main() {
    const multimodal::modality_identity_binding_v1 bindings[] = {
        {10, 30, 31, 40, 41, 0, 1,
         multimodal::modality_kind_v1::transcriptome, 0},
        {11, 30, 31, 60, 61, 0, 1,
         multimodal::modality_kind_v1::spatial, 0},
    };
    const multimodal::multimodal_identity_spine_view_v1 spine{
        bindings, 2, 0, 1, 2, 30, 31, 4};
    multimodal::multimodal_payload_descriptor_v1 payloads[] = {
        {10, 100, 1, 0, 64, 8,
         multimodal::multimodal_payload_kind_v1::sparse_counts, 8},
        {11, 101, 1, 64, 32, 8,
         multimodal::multimodal_payload_kind_v1::spatial_coordinates, 8},
    };
    const multimodal::multi_payload_atom_v1 atom{
        200, 201, 1, 4, 0, 8, 0, 2, 0};
    assert(multimodal::validate_multi_payload_atom_v1(
        spine, atom, payloads, 2, 96).valid());
    payloads[1].modality_identity = 10;
    assert(multimodal::validate_multi_payload_atom_v1(
        spine, atom, payloads, 2, 96).code
        == multimodal::multi_payload_atom_code_v1::duplicate_modality);
    return 0;
}
