#include <CellShard/compiler/partial/gradient_partial_v1.hh>

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <random>

namespace {
using namespace cellshard::compiler::atom;
using namespace cellshard::compiler::partial;
alignas(16) float gradients[4]{};
alignas(8) std::byte coverage_record[248]{};
const std::array<std::byte, 4> identity_map{
    std::byte{0}, std::byte{1}, std::byte{2}, std::byte{3}};

atom_gradient_plane_v1 plane() {
    atom_gradient_plane_v1 result{};
    auto &v = result.value_layout;
    v.values = gradients; v.value_bytes = sizeof(gradients); v.element_count = 4;
    v.element_stride_bytes = sizeof(float); v.element_bytes = sizeof(float);
    v.value_alignment = 16; v.plane_identity = {1, 11};
    v.structure_plane_identity = {1, 12}; v.subject_space_identity = {1, 13};
    v.persistent_order_identity = {1, 14}; v.numeric = {{2, 1}, {2, 1}, {2, 2}};
    v.structure_epoch = 7; v.value_generation = 9; v.canonical_element_count = 4;
    v.local_to_canonical = {identity_map.data(), 4, 4,
                            compact_edge_index_width_v1::u8, {}};
    result.exact_target_coverage = {coverage_record, {3, 21}, 4, 1, 248,
        atom_certified_exact_coverage_role_v1,
        atom_logical_coverage_kind_v1::relation_edge_ids, 0};
    result.gradient_target_identity = {4, 31};
    result.accumulation_algebra_identity = {4, 32};
    return result;
}

gradient_partial_view_v1 partial(const atom_gradient_plane_v1 *gradient) {
    return {gradient, {5, 1}, {5, 2}, {5, 3}, {2, 2},
            7, 8, 9, 10, 11, gradient_legality_v1::exact_declared_vjp, 1, 0, 0};
}

void test_legality_and_fail_closed() {
    auto gradient = plane();
    std::array<std::uint8_t, 4> marks{};
    assert(validate_gradient_partial_v1(
               partial(&gradient), 0, marks.data(), marks.size()).valid());
    auto candidate = partial(&gradient);
    candidate.legality = static_cast<gradient_legality_v1>(2);
    assert(validate_gradient_partial_v1(candidate, 0, marks.data(), marks.size()).code
           == gradient_partial_code_v1::unproven_derivative);
    gradient.ownership = atom_gradient_ownership_v1::physical_mirror;
    gradient.primary_gradient_plane_identity = {9, 9};
    assert(validate_gradient_partial_v1(
               partial(&gradient), 0, marks.data(), marks.size()).code
           == gradient_partial_code_v1::mirror_cannot_own_contribution);
}

void test_randomized_generation_fail_closed() {
    const auto gradient = plane();
    std::array<std::uint8_t, 4> marks{};
    std::mt19937_64 generator(0xba7c9045f12c7f99ULL);
    for (std::uint32_t trial = 0; trial < 4096; ++trial) {
        auto candidate = partial(&gradient);
        const std::uint32_t field = generator() % 5;
        if ((generator() & 7U) != 0) {
            if (field == 0) candidate.forward_structure_generation = 0;
            if (field == 1) candidate.forward_value_generation = 0;
            if (field == 2) candidate.forward_state_generation = 0;
            if (field == 3) candidate.parameter_generation = 0;
            if (field == 4) candidate.adjoint_generation = 0;
        }
        const bool expected = candidate.forward_structure_generation != 0
            && candidate.forward_value_generation != 0
            && candidate.forward_state_generation != 0
            && candidate.parameter_generation != 0
            && candidate.adjoint_generation != 0;
        assert(validate_gradient_partial_v1(
                   candidate, 0, marks.data(), marks.size()).valid() == expected);
    }
}
}

int main() {
    test_legality_and_fail_closed();
    test_randomized_generation_fail_closed();
    return 0;
}
