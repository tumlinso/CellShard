#include <CellShard/compiler/atom/partial_result_plane_v1.hh>

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>

namespace {

using namespace cellshard::compiler::atom;

alignas(16) float partials[4]{};
alignas(8) std::byte coverage_record[248]{};
const std::array<std::byte, 4> identity_map{
    std::byte{0}, std::byte{1}, std::byte{2}, std::byte{3}};
const std::array<atom_dependency_generation_v1, 2> dependencies{
    atom_dependency_generation_v1{{7, 1}, {8, 1}, 11},
    atom_dependency_generation_v1{{7, 2}, {8, 2}, 12}};

atom_partial_result_plane_v1 valid_plane() {
    atom_partial_result_plane_v1 plane{};
    auto &layout = plane.partial_layout;
    layout.values = partials;
    layout.value_bytes = sizeof(partials);
    layout.element_count = 4;
    layout.element_stride_bytes = sizeof(float);
    layout.element_bytes = sizeof(float);
    layout.value_alignment = 16;
    layout.plane_identity = {1, 11};
    layout.structure_plane_identity = {1, 12};
    layout.subject_space_identity = {1, 13};
    layout.persistent_order_identity = {1, 14};
    layout.numeric = {{2, 1}, {2, 1}, {2, 2}};
    layout.structure_epoch = 7;
    layout.value_generation = 9;
    layout.canonical_element_count = 4;
    layout.local_to_canonical = {
        identity_map.data(), 4, 4, compact_edge_index_width_v1::u8, {}};
    plane.exact_contribution_coverage = {
        coverage_record, {3, 21}, 4, 1, 248,
        atom_certified_exact_coverage_role_v1
            | atom_partial_contribution_owner_role_v1,
        atom_logical_coverage_kind_v1::relation_edge_ids, 0};
    plane.contribution_owner_identity = {4, 31};
    plane.reconstruction_algebra_identity = {4, 32};
    plane.numerical_policy_identity = {4, 33};
    plane.dependencies = dependencies.data();
    plane.dependency_count = dependencies.size();
    return plane;
}

void test_status_progression() {
    std::array<std::uint8_t, 4> marks{};
    auto plane = valid_plane();
    assert(validate_atom_partial_result_plane_v1(
               plane, 0, marks.data(), marks.size())
               .valid());
    plane.status = atom_partial_result_status_v1::ready_to_merge;
    assert(validate_atom_partial_result_plane_v1(
               plane, 0, marks.data(), marks.size())
               .valid());
    plane.status = atom_partial_result_status_v1::merged;
    plane.merge_generation = 10;
    assert(validate_atom_partial_result_plane_v1(
               plane, 0, marks.data(), marks.size())
               .valid());
    plane.status = atom_partial_result_status_v1::finalized;
    plane.finalized_output_identity = {5, 41};
    assert(validate_atom_partial_result_plane_v1(
               plane, 0, marks.data(), marks.size())
               .valid());
}

void test_deterministic_rejections() {
    std::array<std::uint8_t, 4> marks{};
    auto plane = valid_plane();
    plane.exact_contribution_coverage.logical_count = 3;
    assert(validate_atom_partial_result_plane_v1(
               plane, 0, marks.data(), marks.size())
               .code == atom_partial_result_validation_code_v1::
                            contribution_count_mismatch);
    plane = valid_plane();
    plane.reconstruction_algebra_identity = {};
    assert(validate_atom_partial_result_plane_v1(
               plane, 0, marks.data(), marks.size())
               .code == atom_partial_result_validation_code_v1::
                            invalid_reconstruction_algebra);
    plane = valid_plane();
    plane.dependency_count = 0;
    assert(validate_atom_partial_result_plane_v1(
               plane, 0, marks.data(), marks.size())
               .code == atom_partial_result_validation_code_v1::
                            missing_dependencies);
    auto malformed_dependencies = dependencies;
    malformed_dependencies[1].dependency_identity =
        malformed_dependencies[0].dependency_identity;
    plane = valid_plane();
    plane.dependencies = malformed_dependencies.data();
    assert(validate_atom_partial_result_plane_v1(
               plane, 0, marks.data(), marks.size())
               .code == atom_partial_result_validation_code_v1::
                            unordered_or_duplicate_dependency);
    plane = valid_plane();
    plane.merge_generation = 1;
    assert(validate_atom_partial_result_plane_v1(
               plane, 0, marks.data(), marks.size())
               .code == atom_partial_result_validation_code_v1::
                            premature_merge_generation);
    plane = valid_plane();
    plane.status = atom_partial_result_status_v1::merged;
    assert(validate_atom_partial_result_plane_v1(
               plane, 0, marks.data(), marks.size())
               .code == atom_partial_result_validation_code_v1::
                            missing_merge_generation);
    plane = valid_plane();
    plane.status = atom_partial_result_status_v1::finalized;
    plane.merge_generation = 1;
    assert(validate_atom_partial_result_plane_v1(
               plane, 0, marks.data(), marks.size())
               .code == atom_partial_result_validation_code_v1::
                            missing_finalized_output);
}

} // namespace

int main() {
    test_status_progression();
    test_deterministic_rejections();
    return 0;
}
