#include <CellShard/compiler/atom/port_v1.hh>

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace {

using namespace cellshard::compiler::atom;

atom_port_v1 make_port(std::uint64_t local_id) {
    atom_port_v1 port{};
    port.port_identity = {1, local_id};
    port.domain_identity = {2, 11};
    port.axis_identity = {2, 12};
    port.order_identity = {2, 13};
    port.plane_kind = {3, 1};
    port.numeric = {{4, 1}, {4, 2}, {4, 3}};
    port.generation = 7;
    port.accepted_extent_forms = atom_single_contiguous_extent_v1;
    port.minimum_extent_count = 1;
    port.maximum_extent_count = 1;
    port.direction = atom_port_direction_v1::input;
    port.axis_role = atom_port_axis_role_v1::source;
    port.mutability = atom_port_mutability_v1::immutable;
    port.requirement = atom_port_requirement_v1::required;
    return port;
}

void test_typed_ports() {
    std::array<atom_port_v1, 3> ports{
        make_port(1), make_port(2), make_port(3)};
    ports[1].direction = atom_port_direction_v1::output;
    ports[1].axis_role = atom_port_axis_role_v1::destination;
    ports[1].mutability = atom_port_mutability_v1::mutable_value;
    ports[2].direction = atom_port_direction_v1::inout;
    ports[2].mutability = atom_port_mutability_v1::mutable_state;
    ports[2].requirement = atom_port_requirement_v1::optional;
    ports[2].accepted_extent_forms = atom_multiple_contiguous_extents_v1
        | atom_segmented_extent_v1;
    ports[2].maximum_extent_count = 8;
    const auto result = validate_atom_port_table_v1(
        {ports.data(), ports.size()});
    assert(result.valid());
    assert(result.index == ports.size());
}

void test_deterministic_rejections() {
    assert(validate_atom_port_table_v1({nullptr, 0}).code
           == atom_port_validation_code_v1::empty_table);
    assert(validate_atom_port_table_v1({nullptr, 1}).code
           == atom_port_validation_code_v1::missing_ports);

    auto base = make_port(1);
    auto expect = [&](atom_port_v1 port, atom_port_validation_code_v1 code) {
        const auto result = validate_atom_port_table_v1({&port, 1});
        assert(result.code == code);
        assert(result.index == 0);
    };
    auto malformed = base;
    malformed.port_identity = {};
    expect(malformed, atom_port_validation_code_v1::invalid_port_identity);
    malformed = base;
    malformed.domain_identity = {};
    expect(malformed, atom_port_validation_code_v1::invalid_domain_identity);
    malformed = base;
    malformed.axis_identity = {};
    expect(malformed, atom_port_validation_code_v1::invalid_axis_identity);
    malformed = base;
    malformed.order_identity = {};
    expect(malformed, atom_port_validation_code_v1::invalid_order_identity);
    malformed = base;
    malformed.plane_kind = {};
    expect(malformed, atom_port_validation_code_v1::invalid_plane_kind);
    malformed = base;
    malformed.numeric.storage_type = {};
    expect(malformed, atom_port_validation_code_v1::invalid_storage_type);
    malformed = base;
    malformed.numeric.logical_type = {};
    expect(malformed, atom_port_validation_code_v1::invalid_logical_type);
    malformed = base;
    malformed.numeric.accumulation_type = {};
    expect(malformed, atom_port_validation_code_v1::invalid_accumulation_type);
    malformed = base;
    malformed.generation = 0;
    expect(malformed, atom_port_validation_code_v1::missing_generation);
    malformed = base;
    malformed.accepted_extent_forms = 1u << 31u;
    expect(malformed, atom_port_validation_code_v1::invalid_extent_form);
    malformed = base;
    malformed.maximum_extent_count = 2;
    expect(malformed, atom_port_validation_code_v1::invalid_extent_count);
    malformed = base;
    malformed.direction = static_cast<atom_port_direction_v1>(0);
    expect(malformed, atom_port_validation_code_v1::invalid_direction);
    malformed = base;
    malformed.axis_role = static_cast<atom_port_axis_role_v1>(0);
    expect(malformed, atom_port_validation_code_v1::invalid_axis_role);
    malformed = base;
    malformed.mutability = static_cast<atom_port_mutability_v1>(0);
    expect(malformed, atom_port_validation_code_v1::invalid_mutability);
    malformed = base;
    malformed.requirement = static_cast<atom_port_requirement_v1>(0);
    expect(malformed, atom_port_validation_code_v1::invalid_requirement);
    malformed = base;
    malformed.direction = atom_port_direction_v1::inout;
    expect(malformed, atom_port_validation_code_v1::immutable_inout);
    malformed = base;
    malformed.reserved = 1;
    expect(malformed, atom_port_validation_code_v1::nonzero_reserved);

    std::array<atom_port_v1, 2> duplicate{base, base};
    assert(validate_atom_port_table_v1({duplicate.data(), duplicate.size()}).code
           == atom_port_validation_code_v1::unordered_or_duplicate_port);
}

void test_randomized_sorted_tables() {
    std::uint64_t state = UINT64_C(0xa08c0de);
    for (std::size_t iteration = 0; iteration < 4096; ++iteration) {
        state = state * UINT64_C(6364136223846793005) + 1;
        const std::size_t count = 1 + state % 32;
        std::vector<atom_port_v1> ports;
        ports.reserve(count);
        std::uint64_t id = 0;
        for (std::size_t index = 0; index < count; ++index) {
            state = state * UINT64_C(6364136223846793005) + 1;
            id += 1 + state % 5;
            ports.push_back(make_port(id));
            if ((state & 1) != 0) {
                ports.back().direction = atom_port_direction_v1::output;
                ports.back().axis_role = atom_port_axis_role_v1::destination;
                ports.back().mutability = atom_port_mutability_v1::mutable_value;
            }
        }
        assert(validate_atom_port_table_v1({ports.data(), ports.size()}).valid());
    }
}

} // namespace

int main() {
    test_typed_ports();
    test_deterministic_rejections();
    test_randomized_sorted_tables();
    return 0;
}
