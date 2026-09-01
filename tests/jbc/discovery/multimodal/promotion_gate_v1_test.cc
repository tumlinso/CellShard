#include <CellShard/compiler/discovery/multimodal/promotion_gate_v1.hh>

#include <cassert>

namespace multimodal = cellshard::compiler::discovery::multimodal;

int main() {
    const multimodal::multi_payload_atom_v1 atom{
        200, 201, 1, 4, 0, 8, 0, 2, 0};
    const multimodal::multimodal_exact_certificate_v1 certificate{
        500, 200, 201, 1, 4, 8, 2, 12, 0, 1, 0};
    std::uint64_t null_scores[99]{};
    for (std::uint64_t index = 0; index < 99; ++index)
        null_scores[index] = index % 7;
    auto result = multimodal::run_modality_map_null_promotion_gate_v1(
        10, null_scores, 99, certificate, atom, {1, 20, 8, 12, 2, 0});
    assert(result.promoted == 1);
    assert(result.null_summary.p_value_numerator == 1);
    assert(result.null_summary.p_value_denominator == 100);

    null_scores[0] = 10;
    result = multimodal::run_modality_map_null_promotion_gate_v1(
        10, null_scores, 99, certificate, atom, {1, 100, 8, 12, 2, 0});
    assert(result.promoted == 0);
    assert(result.reason
           == multimodal::multimodal_promotion_reason_v1::null_not_rejected);
    return 0;
}
