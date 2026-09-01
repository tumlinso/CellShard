#include <CellShard/compiler/discovery/multimodal/domain_value_overlay_v1.hh>

#include <cassert>

namespace multimodal = cellshard::compiler::discovery::multimodal;

int main() {
    const multimodal::modality_identity_binding_v1 bindings[] = {
        {10, 30, 31, 40, 41, 0, 1,
         multimodal::modality_kind_v1::transcriptome, 0},
        {11, 50, 51, 60, 61, 70, 3,
         multimodal::modality_kind_v1::chromatin, 0},
    };
    const multimodal::multimodal_identity_spine_view_v1 spine{
        bindings, 2, 0, 1, 2, 30, 31, 4};
    const multimodal::modality_domain_overlay_v1 domains[] = {
        {10, 100, 40, 41, 101}, {11, 110, 60, 61, 111},
    };
    multimodal::modality_value_overlay_v1 values[] = {
        {10, 200, 1, 1, 1,
         multimodal::value_scalar_kind_v1::unsigned_integer,
         multimodal::missing_value_policy_v1::absent_is_zero},
        {11, 201, 3, 1, 1000,
         multimodal::value_scalar_kind_v1::floating_point,
         multimodal::missing_value_policy_v1::explicit_mask},
    };
    const multimodal::domain_value_overlays_view_v1 overlays{
        domains, values, 2, 0, 1, 4};
    assert(multimodal::validate_domain_value_overlays_v1(
        spine, overlays).valid());
    values[1].value_generation = 2;
    assert(multimodal::validate_domain_value_overlays_v1(spine, overlays).code
        == multimodal::domain_value_overlay_code_v1::stale_value_generation);
    return 0;
}
