#include <CellShard/compiler/atom/executable_affordance_plane_v1.hh>

#include <array>
#include <cassert>

namespace {

using namespace cellshard::compiler::atom;

atom_port_v1 make_port(std::uint64_t local_identity) {
    atom_port_v1 port{};
    port.port_identity = {1, local_identity};
    port.domain_identity = {2, 1};
    port.axis_identity = {2, 2};
    port.order_identity = {2, 3};
    port.plane_kind = {2, 4};
    port.numeric = {{3, 1}, {3, 2}, {3, 3}};
    port.generation = 5;
    port.accepted_extent_forms = atom_single_contiguous_extent_v1;
    port.minimum_extent_count = 1;
    port.maximum_extent_count = 1;
    port.direction = atom_port_direction_v1::inout;
    port.mutability = atom_port_mutability_v1::mutable_value;
    return port;
}

std::array<atom_port_v1, 2> ports{make_port(1), make_port(2)};
const std::array<atom_persistent_identity_v1, 2> required_ports{
    atom_persistent_identity_v1{1, 1}, atom_persistent_identity_v1{1, 2}};

atom_executable_affordance_plane_v1 valid_plane(
    atom_executable_affordance_v1 *affordances) {
    affordances[0] = {};
    affordances[1] = {};
    affordances[0].operation_identity = {4, 1};
    affordances[0].lowering_entry_identity = {4, 2};
    affordances[0].required_mutable_ports = required_ports.data();
    affordances[0].required_mutable_port_count = required_ports.size();
    affordances[0].output_affordances = atom_complete_output_affordance_v1
        | atom_partial_output_affordance_v1;
    affordances[1] = affordances[0];
    affordances[1].operation_identity = {4, 3};
    affordances[1].lowering_entry_identity = {4, 4};
    affordances[1].lowering_kind = atom_lowering_entry_kind_v1::external;
    affordances[1].target_restriction =
        atom_target_restriction_kind_v1::exact_target;
    affordances[1].target_identity = {5, 1};

    atom_executable_affordance_plane_v1 plane{};
    plane.affordances = affordances;
    plane.affordance_count = 2;
    plane.ports = {ports.data(), ports.size()};
    plane.plane_identity = {6, 1};
    plane.preparation_identity = {6, 2};
    plane.preparation_generation = 7;
    return plane;
}

void test_native_and_external_lowerings() {
    std::array<atom_executable_affordance_v1, 2> affordances{};
    const auto plane = valid_plane(affordances.data());
    const auto result = validate_atom_executable_affordance_plane_v1(plane);
    assert(result.valid());
    assert(result.affordance_index == affordances.size());
}

void test_deterministic_rejections() {
    std::array<atom_executable_affordance_v1, 2> affordances{};
    auto plane = valid_plane(affordances.data());
    affordances[1].operation_identity = affordances[0].operation_identity;
    assert(validate_atom_executable_affordance_plane_v1(plane).code
           == atom_executable_affordance_validation_code_v1::
                  unordered_or_duplicate_operation);

    plane = valid_plane(affordances.data());
    affordances[0].target_identity = {7, 1};
    assert(validate_atom_executable_affordance_plane_v1(plane).code
           == atom_executable_affordance_validation_code_v1::
                  unexpected_target_identity);

    plane = valid_plane(affordances.data());
    const std::array<atom_persistent_identity_v1, 1> unknown{{{1, 9}}};
    affordances[0].required_mutable_ports = unknown.data();
    affordances[0].required_mutable_port_count = unknown.size();
    assert(validate_atom_executable_affordance_plane_v1(plane).code
           == atom_executable_affordance_validation_code_v1::
                  unknown_required_mutable_port);

    plane = valid_plane(affordances.data());
    ports[0].mutability = atom_port_mutability_v1::immutable;
    ports[0].direction = atom_port_direction_v1::input;
    assert(validate_atom_executable_affordance_plane_v1(plane).code
           == atom_executable_affordance_validation_code_v1::
                  referenced_port_not_required_mutable);
    ports[0] = make_port(1);

    plane = valid_plane(affordances.data());
    affordances[0].output_affordances = 0;
    assert(validate_atom_executable_affordance_plane_v1(plane).code
           == atom_executable_affordance_validation_code_v1::
                  invalid_output_affordance);

    plane = valid_plane(affordances.data());
    plane.preparation_generation = 0;
    assert(validate_atom_executable_affordance_plane_v1(plane).code
           == atom_executable_affordance_validation_code_v1::
                  missing_preparation_generation);
}

} // namespace

int main() {
    test_native_and_external_lowerings();
    test_deterministic_rejections();
    return 0;
}
