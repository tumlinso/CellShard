#include <CellShard/compiler/atom/physical_view_plane_v1.hh>

#include <array>
#include <cassert>
#include <cstddef>

namespace {

using namespace cellshard::compiler::atom;

alignas(32) std::array<std::byte, 128> payload{};
const std::array<atom_physical_extent_v1, 2> extents{
    atom_physical_extent_v1{{1, 1}, 7, 8, 32},
    atom_physical_extent_v1{{1, 2}, 11, 16, 4}};

atom_physical_view_plane_v1 valid_plane() {
    atom_physical_view_plane_v1 plane{};
    plane.payload = payload.data();
    plane.payload_bytes = payload.size();
    plane.extents = extents.data();
    plane.extent_count = extents.size();
    plane.semantic_family = {{2, 1}};
    plane.materialization = {{2, 2}};
    plane.physical_view_identity = {2, 3};
    plane.encoding_identity = {2, 4};
    plane.persistent_order_identity = {2, 5};
    plane.projection_abi_identity = {2, 6};
    plane.materialization_generation = 3;
    plane.payload_alignment = 32;
    return plane;
}

void test_neutral_and_specific_views() {
    auto plane = valid_plane();
    assert(validate_atom_physical_view_plane_v1(plane).valid());

    plane.target_kind = atom_physical_target_kind_v1::target_specific;
    plane.target_identity = {3, 1};
    assert(validate_atom_physical_view_plane_v1(plane).valid());
    assert(plane.semantic_family.persistent
           == valid_plane().semantic_family.persistent);
}

void test_layout_and_identity_are_independent() {
    auto plane = valid_plane();
    plane.encoding_identity = {4, 1};
    plane.materialization = {{4, 2}};
    assert(validate_atom_physical_view_plane_v1(plane).valid());
    const atom_persistent_identity_v1 expected_semantic_identity{2, 1};
    assert(plane.semantic_family.persistent == expected_semantic_identity);
}

void test_deterministic_rejections() {
    auto plane = valid_plane();
    plane.payload_alignment = 3;
    assert(validate_atom_physical_view_plane_v1(plane).code
           == atom_physical_view_validation_code_v1::
                  invalid_payload_alignment);

    auto malformed_extents = extents;
    malformed_extents[1].physical_extent = 10;
    plane = valid_plane();
    plane.extents = malformed_extents.data();
    auto result = validate_atom_physical_view_plane_v1(plane);
    assert(result.code
           == atom_physical_view_validation_code_v1::
                  physical_extent_smaller_than_logical);
    assert(result.index == 1);

    plane = valid_plane();
    plane.materialization = {};
    assert(validate_atom_physical_view_plane_v1(plane).code
           == atom_physical_view_validation_code_v1::invalid_materialization);

    plane = valid_plane();
    plane.target_identity = {5, 1};
    assert(validate_atom_physical_view_plane_v1(plane).code
           == atom_physical_view_validation_code_v1::
                  unexpected_target_identity);

    plane = valid_plane();
    plane.target_kind = atom_physical_target_kind_v1::target_specific;
    assert(validate_atom_physical_view_plane_v1(plane).code
           == atom_physical_view_validation_code_v1::missing_target_identity);

    plane = valid_plane();
    plane.materialization_generation = 0;
    assert(validate_atom_physical_view_plane_v1(plane).code
           == atom_physical_view_validation_code_v1::
                  missing_materialization_generation);
}

} // namespace

int main() {
    test_neutral_and_specific_views();
    test_layout_and_identity_are_independent();
    test_deterministic_rejections();
    return 0;
}
