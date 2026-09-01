#include <CellShard/compiler/atom/structural_plane_v1.hh>

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>

namespace {

using namespace cellshard::compiler::atom;

alignas(16) std::byte payload[64]{};
alignas(8) std::byte coverage_record[248]{};
const std::uint64_t edge_ids[]{11, 22, 33};

atom_structural_component_ref_v1 component(
    atom_structural_component_kind_v1 kind, std::uint64_t id) {
    return {payload, sizeof(payload), {7, id}, {8, id}, 19, kind, 16};
}

atom_structural_plane_v1 valid_plane(
    const atom_structural_component_ref_v1 *components,
    std::uint64_t component_count) {
    atom_structural_plane_v1 plane{};
    plane.components = components;
    plane.component_count = component_count;
    plane.plane_identity = {1, 100};
    plane.persistent_order_identity = {2, 200};
    plane.structure_epoch = 19;
    plane.edge_spine = {edge_ids, 3, {3, 300}, 19};
    plane.exact_coverage = {
        coverage_record, {4, 400}, 3, 1, 248,
        atom_certified_exact_coverage_role_v1,
        atom_logical_coverage_kind_v1::relation_edge_ids, 0};
    return plane;
}

void test_immutable_structural_plane() {
    const std::array<atom_structural_component_ref_v1, 4> components{
        component(atom_structural_component_kind_v1::support, 1),
        component(atom_structural_component_kind_v1::hierarchy, 2),
        component(atom_structural_component_kind_v1::relation_map, 3),
        component(atom_structural_component_kind_v1::segment_definition, 4)};
    const auto plane = valid_plane(components.data(), components.size());
    const auto result = validate_atom_structural_plane_v1(plane, 0);
    assert(result.valid());
    assert(result.index == components.size());
}

void test_deterministic_rejections() {
    std::array<atom_structural_component_ref_v1, 2> components{
        component(atom_structural_component_kind_v1::support, 1),
        component(atom_structural_component_kind_v1::relation_map, 2)};
    auto plane = valid_plane(components.data(), components.size());
    plane.plane_identity = {};
    assert(validate_atom_structural_plane_v1(plane, 0).code
           == atom_structural_plane_validation_code_v1::invalid_plane_identity);
    plane = valid_plane(components.data(), components.size());
    plane.structure_epoch = 20;
    assert(validate_atom_structural_plane_v1(plane, 0).code
           == atom_structural_plane_validation_code_v1::
                  edge_spine_epoch_mismatch);
    plane = valid_plane(components.data(), components.size());
    assert(validate_atom_structural_plane_v1(plane, 9).code
           == atom_structural_plane_validation_code_v1::
                  invalid_exact_coverage);
    plane = valid_plane(nullptr, 0);
    assert(validate_atom_structural_plane_v1(plane, 0).code
           == atom_structural_plane_validation_code_v1::empty_components);

    auto malformed = components;
    malformed[0].descriptor = nullptr;
    plane = valid_plane(malformed.data(), malformed.size());
    assert(validate_atom_structural_plane_v1(plane, 0).code
           == atom_structural_plane_validation_code_v1::missing_descriptor);
    malformed = components;
    malformed[0].structure_epoch = 18;
    plane = valid_plane(malformed.data(), malformed.size());
    assert(validate_atom_structural_plane_v1(plane, 0).code
           == atom_structural_plane_validation_code_v1::stale_component_epoch);
    malformed = components;
    malformed[1] = malformed[0];
    plane = valid_plane(malformed.data(), malformed.size());
    assert(validate_atom_structural_plane_v1(plane, 0).code
           == atom_structural_plane_validation_code_v1::
                  unordered_or_duplicate_component);

    std::array<atom_structural_component_ref_v1, 1> only_relation{
        component(atom_structural_component_kind_v1::relation_map, 1)};
    plane = valid_plane(only_relation.data(), only_relation.size());
    assert(validate_atom_structural_plane_v1(plane, 0).code
           == atom_structural_plane_validation_code_v1::missing_support);
    std::array<atom_structural_component_ref_v1, 1> only_support{
        component(atom_structural_component_kind_v1::support, 1)};
    plane = valid_plane(only_support.data(), only_support.size());
    assert(validate_atom_structural_plane_v1(plane, 0).code
           == atom_structural_plane_validation_code_v1::missing_relation_map);
}

void test_randomized_component_tables() {
    std::uint64_t state = UINT64_C(0xa10c0de);
    for (std::size_t iteration = 0; iteration < 4096; ++iteration) {
        state = state * UINT64_C(6364136223846793005) + 1;
        std::array<atom_structural_component_ref_v1, 4> components{
            component(atom_structural_component_kind_v1::support,
                      1 + state % 1000),
            component(atom_structural_component_kind_v1::hierarchy,
                      1 + state % 1000),
            component(atom_structural_component_kind_v1::relation_map,
                      1 + state % 1000),
            component(atom_structural_component_kind_v1::segment_definition,
                      1 + state % 1000)};
        const auto plane = valid_plane(components.data(), components.size());
        assert(validate_atom_structural_plane_v1(plane, 0).valid());
    }
}

} // namespace

int main() {
    test_immutable_structural_plane();
    test_deterministic_rejections();
    test_randomized_component_tables();
    return 0;
}
