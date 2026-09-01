#include <CellShard/compiler/discovery/multimodal/modality_missingness_v1.hh>

#include <cassert>

namespace multimodal = cellshard::compiler::discovery::multimodal;

int main() {
    const multimodal::modality_identity_binding_v1 bindings[] = {
        {10, 30, 31, 40, 41, 0, 1,
         multimodal::modality_kind_v1::transcriptome, 0},
        {11, 30, 31, 60, 61, 0, 3,
         multimodal::modality_kind_v1::protein, 0},
    };
    const multimodal::multimodal_identity_spine_view_v1 spine{
        bindings, 2, 0, 1, 2, 30, 31, 4};
    std::uint8_t status[] = {0, 1, 0, 0, 2, 3};
    const multimodal::modality_missingness_view_v1 missingness{
        status, 3, 2, 2, 0, 1, 30, 31, 1};
    multimodal::modality_missingness_summary_v1 summaries[2]{};
    auto result = multimodal::summarize_modality_missingness_v1(
        spine, missingness, summaries, 2, 6);
    assert(result.summarized());
    assert(summaries[0].observed_count == 2);
    assert(summaries[0].failed_quality_control_count == 1);
    assert(summaries[1].observed_count == 1);
    assert(summaries[1].not_assayed_count == 1);
    assert(summaries[1].below_detection_count == 1);
    status[5] = 9;
    assert(multimodal::summarize_modality_missingness_v1(
        spine, missingness, summaries, 2, 6).code
        == multimodal::modality_missingness_code_v1::invalid_status);
    return 0;
}
