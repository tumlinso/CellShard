#include <CellShard/compiler/discovery/multimodal/cross_modal_atom_v1.hh>

#include <cassert>

namespace multimodal = cellshard::compiler::discovery::multimodal;

int main() {
    const multimodal::modality_identity_binding_v1 bindings[] = {
        {10, 30, 31, 40, 41, 0, 1,
         multimodal::modality_kind_v1::transcriptome, 0},
        {11, 30, 31, 60, 61, 0, 1,
         multimodal::modality_kind_v1::protein, 0},
    };
    const multimodal::multimodal_identity_spine_view_v1 spine{
        bindings, 2, 0, 1, 2, 30, 31, 4};
    const multimodal::cross_modal_relation_atom_v1 candidates[] = {
        {2, 70, 80, 10, 100, 11, 200, 3, 4,
         multimodal::cross_modal_relation_kind_v1::statistical_proposal, 0},
        {1, 71, 81, 11, 201, 10, 101, -1, 2,
         multimodal::cross_modal_relation_kind_v1::regulatory, 1},
    };
    multimodal::cross_modal_relation_atom_v1 atoms[2]{};
    auto result = multimodal::construct_cross_modal_atoms_v1(
        spine, candidates, 2, atoms, 2);
    assert(result.constructed());
    assert(atoms[0].atom_identity == 1);
    assert(atoms[1].atom_identity == 2);
    auto duplicate = candidates[1];
    duplicate.atom_identity = 2;
    const multimodal::cross_modal_relation_atom_v1 duplicates[] = {
        candidates[0], duplicate};
    assert(multimodal::construct_cross_modal_atoms_v1(
        spine, duplicates, 2, atoms, 2).code
        == multimodal::cross_modal_atom_code_v1::duplicate_atom_identity);
    return 0;
}
