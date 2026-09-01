#include <CellShard/compiler/atom/gradient_plane_v1.hh>

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>

namespace {

using namespace cellshard::compiler::atom;

alignas(16) float gradients[4]{};
alignas(8) std::byte coverage_record[248]{};
const std::array<std::byte, 4> identity_map{
    std::byte{0}, std::byte{1}, std::byte{2}, std::byte{3}};

atom_gradient_plane_v1 valid_plane() {
    atom_gradient_plane_v1 plane{};
    auto &values = plane.value_layout;
    values.values = gradients;
    values.value_bytes = sizeof(gradients);
    values.element_count = 4;
    values.element_stride_bytes = sizeof(float);
    values.element_bytes = sizeof(float);
    values.value_alignment = 16;
    values.plane_identity = {1, 11};
    values.structure_plane_identity = {1, 12};
    values.subject_space_identity = {1, 13};
    values.persistent_order_identity = {1, 14};
    values.numeric = {{2, 1}, {2, 1}, {2, 2}};
    values.structure_epoch = 7;
    values.value_generation = 9;
    values.canonical_element_count = 4;
    values.local_to_canonical = {
        identity_map.data(), 4, 4, compact_edge_index_width_v1::u8, {}};
    plane.exact_target_coverage = {
        coverage_record, {3, 21}, 4, 1, 248,
        atom_certified_exact_coverage_role_v1,
        atom_logical_coverage_kind_v1::relation_edge_ids, 0};
    plane.gradient_target_identity = {4, 31};
    plane.accumulation_algebra_identity = {4, 32};
    return plane;
}

void test_primary_and_mirror_gradient() {
    std::array<std::uint8_t, 4> marks{};
    auto plane = valid_plane();
    assert(validate_atom_gradient_plane_v1(
               plane, 0, marks.data(), marks.size())
               .valid());
    plane.ownership = atom_gradient_ownership_v1::physical_mirror;
    plane.primary_gradient_plane_identity = {1, 99};
    assert(validate_atom_gradient_plane_v1(
               plane, 0, marks.data(), marks.size())
               .valid());
}

void test_deterministic_rejections() {
    std::array<std::uint8_t, 4> marks{};
    auto plane = valid_plane();
    plane.value_layout.value_generation = 0;
    assert(validate_atom_gradient_plane_v1(
               plane, 0, marks.data(), marks.size())
               .code == atom_gradient_plane_validation_code_v1::
                            invalid_value_layout);
    plane = valid_plane();
    assert(validate_atom_gradient_plane_v1(
               plane, 17, marks.data(), marks.size())
               .code == atom_gradient_plane_validation_code_v1::
                            invalid_target_coverage);
    plane = valid_plane();
    plane.exact_target_coverage.logical_count = 3;
    assert(validate_atom_gradient_plane_v1(
               plane, 0, marks.data(), marks.size())
               .code == atom_gradient_plane_validation_code_v1::
                            target_count_mismatch);
    plane = valid_plane();
    plane.gradient_target_identity = {};
    assert(validate_atom_gradient_plane_v1(
               plane, 0, marks.data(), marks.size())
               .code == atom_gradient_plane_validation_code_v1::
                            invalid_gradient_target);
    plane = valid_plane();
    plane.accumulation_algebra_identity = {};
    assert(validate_atom_gradient_plane_v1(
               plane, 0, marks.data(), marks.size())
               .code == atom_gradient_plane_validation_code_v1::
                            invalid_accumulation_algebra);
    plane = valid_plane();
    plane.primary_gradient_plane_identity = {1, 99};
    assert(validate_atom_gradient_plane_v1(
               plane, 0, marks.data(), marks.size())
               .code == atom_gradient_plane_validation_code_v1::
                            unexpected_primary_reference);
    plane = valid_plane();
    plane.ownership = atom_gradient_ownership_v1::physical_mirror;
    assert(validate_atom_gradient_plane_v1(
               plane, 0, marks.data(), marks.size())
               .code == atom_gradient_plane_validation_code_v1::
                            missing_primary_reference);
    plane.primary_gradient_plane_identity = plane.value_layout.plane_identity;
    assert(validate_atom_gradient_plane_v1(
               plane, 0, marks.data(), marks.size())
               .code == atom_gradient_plane_validation_code_v1::
                            mirror_self_reference);
}

void test_generation_and_order_are_independent() {
    std::array<std::uint8_t, 4> marks{};
    auto first = valid_plane();
    auto next_generation = first;
    next_generation.value_layout.value_generation += 1;
    assert(next_generation.value_layout.structure_epoch
           == first.value_layout.structure_epoch);
    assert(validate_atom_gradient_plane_v1(
               next_generation, 0, marks.data(), marks.size())
               .valid());
}

} // namespace

int main() {
    test_primary_and_mirror_gradient();
    test_deterministic_rejections();
    test_generation_and_order_are_independent();
    return 0;
}
