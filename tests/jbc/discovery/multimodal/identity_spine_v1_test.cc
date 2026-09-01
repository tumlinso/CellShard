#include <CellShard/compiler/discovery/multimodal/identity_spine_v1.hh>

#include <cassert>

namespace multimodal = cellshard::compiler::discovery::multimodal;

int main() {
    multimodal::modality_identity_binding_v1 modalities[] = {
        {10, 30, 31, 40, 41, 0, 1,
         multimodal::modality_kind_v1::transcriptome, 0},
        {11, 50, 51, 60, 61, 70, 3,
         multimodal::modality_kind_v1::chromatin, 0},
    };
    const multimodal::multimodal_identity_spine_view_v1 spine{
        modalities, 2, 0, 1, 2, 30, 31, 4};
    assert(multimodal::validate_multimodal_identity_spine_v1(spine).valid());

    modalities[1].observation_to_subject_relation_identity = 0;
    auto result = multimodal::validate_multimodal_identity_spine_v1(spine);
    assert(result.code == multimodal::identity_spine_code_v1::
        missing_subject_relation);
    modalities[1].observation_to_subject_relation_identity = 70;
    modalities[1].modality_identity = 10;
    result = multimodal::validate_multimodal_identity_spine_v1(spine);
    assert(result.code == multimodal::identity_spine_code_v1::
        duplicate_modality_identity);
    return 0;
}
