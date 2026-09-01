#include <CellShard/compiler/atom/common_atom_v1.hh>

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>

namespace {

using namespace cellshard;
using namespace cellshard::compiler::atom;

alignas(8) std::array<std::byte, 248> coverage_record{};
alignas(16) std::array<std::byte, 32> plane_descriptor{};
alignas(16) std::array<std::byte, 32> evidence_payload{};

content_digest digest_with(std::uint64_t value) {
    content_digest digest{};
    digest.algorithm = digest_algorithm::legacy_fnv1a64;
    digest.used_bytes = sizeof(value);
    for (std::size_t index = 0; index < sizeof(value); ++index) {
        digest.bytes[index] = static_cast<std::byte>(
            (value >> (index * 8)) & UINT64_C(0xff));
    }
    return digest;
}

atom_port_v1 make_port() {
    atom_port_v1 port{};
    port.port_identity = {8, 1};
    port.domain_identity = {8, 2};
    port.axis_identity = {8, 3};
    port.order_identity = {8, 4};
    port.plane_kind = {8, 5};
    port.numeric = {{9, 1}, {9, 2}, {9, 3}};
    port.generation = 4;
    port.accepted_extent_forms = atom_single_contiguous_extent_v1;
    port.minimum_extent_count = 1;
    port.maximum_extent_count = 1;
    port.direction = atom_port_direction_v1::inout;
    port.mutability = atom_port_mutability_v1::mutable_value;
    return port;
}

struct fixture_v1 {
    std::array<atom_level_v1, 3> levels{
        atom_level_v1::semantic,
        atom_level_v1::materialized,
        atom_level_v1::executable};
    std::array<atom_parent_ref_v1, 1> parents{
        atom_parent_ref_v1{{{10, 1}}, {10, 2}, 3}};
    std::array<atom_port_v1, 1> ports{make_port()};
    std::array<atom_plane_descriptor_v1, 1> planes{};
    std::array<atom_dependency_requirement_v1, 1> dependencies{
        atom_dependency_requirement_v1{
            {11, 1}, {11, 2}, 5, 5,
            atom_dependency_generation_kind_v1::structure,
            atom_dependency_effect_v1::correctness}};
    std::array<atom_evidence_record_ref_v1, 1> evidence{};
    std::array<atom_persistent_identity_v1, 1> required_ports{
        atom_persistent_identity_v1{8, 1}};
    std::array<atom_executable_affordance_v1, 1> affordances{};
    std::array<atom_overlap_role_record_v1, 1> overlap_roles{};
    common_atom_view_v1 atom{};

    fixture_v1() {
        planes[0].plane_kind = {12, 1};
        planes[0].plane_identity = {12, 2};
        planes[0].descriptor_schema = {12, 3};
        planes[0].descriptor = plane_descriptor.data();
        planes[0].descriptor_bytes = plane_descriptor.size();
        planes[0].descriptor_alignment = 16;

        evidence[0].record = evidence_payload.data();
        evidence[0].record_bytes = evidence_payload.size();
        evidence[0].record_identity = {13, 1};
        evidence[0].provenance_identity = {13, 2};
        evidence[0].provenance_schema = {13, 3};
        evidence[0].method_identity = {13, 4};
        evidence[0].subject_identity = {13, 5};
        evidence[0].record_digest = digest_with(7);
        evidence[0].observation_generation = 6;
        evidence[0].confidence_numerator = 9;
        evidence[0].confidence_denominator = 10;
        evidence[0].record_alignment = 16;

        affordances[0].operation_identity = {14, 1};
        affordances[0].lowering_entry_identity = {14, 2};
        affordances[0].required_mutable_ports = required_ports.data();
        affordances[0].required_mutable_port_count = required_ports.size();
        affordances[0].output_affordances =
            atom_complete_output_affordance_v1;

        overlap_roles[0].member_identity = {15, 1};
        overlap_roles[0].membership_identity = {15, 2};
        overlap_roles[0].role =
            atom_overlap_role_v1::exclusive_contribution_owner;

        atom.levels = levels.data();
        atom.level_count = levels.size();
        atom.parents = parents.data();
        atom.parent_count = parents.size();
        atom.identities = {
            {{16, 1}},
            {digest_with(8)},
            {{16, 2}},
            {{16, 3}},
            {16, 4}};
        atom.species = core_atom_species_id_v1(
            core_atom_species_v1::executable);
        atom.atomicity = atom_atomicity_capability_v1::semantic
            | atom_atomicity_capability_v1::executable;
        atom.exact_coverage = {
            coverage_record.data(), {17, 1}, 12, 1, 248,
            atom_certified_exact_coverage_role_v1
                | atom_exclusive_output_owner_role_v1,
            atom_logical_coverage_kind_v1::relation_edge_ids, 0};
        atom.ports = {ports.data(), ports.size()};
        atom.planes = {planes.data(), planes.size()};
        atom.dependencies = {
            dependencies.data(), dependencies.size(), {18, 1}, 5};
        atom.evidence = {evidence.data(), evidence.size(), {19, 1}, 6};
        atom.affordances = {
            affordances.data(), affordances.size(), atom.ports,
            {20, 1}, {20, 2}, 7};
        atom.overlap_roles = {
            overlap_roles.data(), overlap_roles.size(), {21, 1}};
        atom.lineage_identity = {22, 1};
        atom.lineage_generation = 8;
    }
};

void test_complete_nonowning_view() {
    fixture_v1 fixture;
    const auto result = validate_common_atom_v1(fixture.atom, 0);
    assert(result.valid());
    assert(result.index == fixture.parents.size());
}

void test_cold_builder_owns_tables() {
    fixture_v1 fixture;
    common_atom_builder_v1 builder;
    const auto result = builder.build(fixture.atom, 0);
    assert(result.built());
    assert(validate_common_atom_v1(builder.view(), 0).valid());
    assert(builder.view().ports.ports != fixture.atom.ports.ports);
    assert(builder.view().planes.planes != fixture.atom.planes.planes);
    assert(builder.view().affordances.affordances
           != fixture.atom.affordances.affordances);
    assert(builder.view().affordances.affordances[0].required_mutable_ports
           != fixture.required_ports.data());

    fixture.ports[0].port_identity = {};
    fixture.required_ports[0] = {};
    assert(validate_common_atom_v1(builder.view(), 0).valid());
}

void test_deterministic_nested_rejections() {
    fixture_v1 invalid_species;
    invalid_species.atom.species = {};
    assert(validate_common_atom_v1(invalid_species.atom, 0).code
           == common_atom_validation_code_v1::invalid_species);

    fixture_v1 invalid_parent;
    invalid_parent.parents[0].parent_generation = 0;
    assert(validate_common_atom_v1(invalid_parent.atom, 0).code
           == common_atom_validation_code_v1::missing_parent_generation);

    fixture_v1 invalid_affordance;
    invalid_affordance.atom.affordances.ports = {};
    assert(validate_common_atom_v1(invalid_affordance.atom, 0).code
           == common_atom_validation_code_v1::invalid_affordances);

    fixture_v1 invalid_lineage;
    invalid_lineage.atom.lineage_generation = 0;
    common_atom_builder_v1 builder;
    const auto build_result = builder.build(invalid_lineage.atom, 0);
    assert(build_result.code == common_atom_build_code_v1::invalid_input);
    assert(build_result.validation.code
           == common_atom_validation_code_v1::missing_lineage_generation);
}

void test_randomized_generation_copying() {
    std::uint64_t state = UINT64_C(0xa20c0de);
    for (std::size_t iteration = 0; iteration < 512; ++iteration) {
        state = state * UINT64_C(6364136223846793005) + 1;
        fixture_v1 fixture;
        const auto generation = state | 1;
        fixture.parents[0].parent_generation = generation;
        fixture.dependencies[0].required_generation = generation;
        fixture.dependencies[0].observed_generation = generation;
        fixture.atom.lineage_generation = generation;
        common_atom_builder_v1 builder;
        assert(builder.build(fixture.atom, 0).built());
        assert(builder.view().lineage_generation == generation);
        assert(builder.view().parents[0].parent_generation == generation);
    }
}

} // namespace

int main() {
    test_complete_nonowning_view();
    test_cold_builder_owns_tables();
    test_deterministic_nested_rejections();
    test_randomized_generation_copying();
    return 0;
}
