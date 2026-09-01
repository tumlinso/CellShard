#include <CellShard/compiler/discovery/multimodal/entity_alignment_v1.hh>

#include <cassert>

namespace multimodal = cellshard::compiler::discovery::multimodal;

int main() {
    const multimodal::modality_identity_binding_v1 bindings[] = {
        {10, 30, 31, 40, 41, 0, 1,
         multimodal::modality_kind_v1::transcriptome, 0},
        {11, 50, 51, 60, 61, 70, 1,
         multimodal::modality_kind_v1::protein, 0},
    };
    const multimodal::multimodal_identity_spine_view_v1 spine{
        bindings, 2, 0, 1, 2, 30, 31, 4};
    const multimodal::modality_entity_alignment_v1 alignments[] = {
        {10, 0, 0, 100}, {10, 1, 1, 101},
        {11, 0, 1, 200}, {11, 1, multimodal::unmatched_subject_v1, 201},
    };
    multimodal::modality_entity_alignment_summary_v1 summaries[2]{};
    auto result = multimodal::summarize_entity_alignment_v1(
        spine, 2, alignments, 4, summaries, 2, 16);
    assert(result.summarized());
    assert(summaries[0].matched_entity_count == 2);
    assert(summaries[1].matched_entity_count == 1);
    assert(summaries[1].modality_only_entity_count == 1);

    const multimodal::modality_entity_alignment_v1 duplicate[] = {
        {10, 0, 0, 100}, {10, 0, 1, 101},
    };
    assert(multimodal::summarize_entity_alignment_v1(
        spine, 2, duplicate, 2, summaries, 2, 4).code
        == multimodal::entity_alignment_code_v1::duplicate_observation);
    return 0;
}
