#include <CellShard/compiler/atom/state_plane_v1.hh>

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace {

using namespace cellshard::compiler::atom;

alignas(16) float state_values[12]{};
const std::array<atom_persistent_identity_v1, 2> consumers{
    atom_persistent_identity_v1{9, 2}, atom_persistent_identity_v1{9, 3}};

atom_state_plane_v1 valid_plane() {
    atom_state_plane_v1 plane{};
    plane.state = state_values;
    plane.state_bytes = sizeof(state_values);
    plane.axis_element_count = 3;
    plane.component_count = 4;
    plane.axis_stride_bytes = 4 * sizeof(float);
    plane.component_stride_bytes = sizeof(float);
    plane.element_bytes = sizeof(float);
    plane.state_alignment = 16;
    plane.plane_identity = {1, 11};
    plane.domain_identity = {2, 12};
    plane.axis_identity = {2, 13};
    plane.persistent_order_identity = {2, 14};
    plane.numeric = {{3, 1}, {3, 1}, {3, 2}};
    plane.structure_epoch = 7;
    plane.state_generation = 8;
    plane.producer_affordance = {9, 1};
    plane.consumer_affordances = consumers.data();
    plane.consumer_affordance_count = consumers.size();
    return plane;
}

void test_typed_mutable_state() {
    auto plane = valid_plane();
    assert(validate_atom_state_plane_v1(plane).valid());
    plane.kind = atom_state_kind_v1::embedding;
    assert(validate_atom_state_plane_v1(plane).valid());
}

void test_deterministic_rejections() {
    auto plane = valid_plane();
    plane.state = nullptr;
    assert(validate_atom_state_plane_v1(plane).code
           == atom_state_plane_validation_code_v1::missing_state);
    plane = valid_plane();
    plane.state_bytes -= 1;
    assert(validate_atom_state_plane_v1(plane).code
           == atom_state_plane_validation_code_v1::insufficient_state_bytes);
    plane = valid_plane();
    plane.axis_stride_bytes = 15;
    assert(validate_atom_state_plane_v1(plane).code
           == atom_state_plane_validation_code_v1::invalid_axis_stride);
    plane = valid_plane();
    plane.axis_identity = {};
    assert(validate_atom_state_plane_v1(plane).code
           == atom_state_plane_validation_code_v1::invalid_axis_identity);
    plane = valid_plane();
    plane.state_generation = 0;
    assert(validate_atom_state_plane_v1(plane).code
           == atom_state_plane_validation_code_v1::missing_state_generation);
    plane = valid_plane();
    plane.consumer_affordance_count = 0;
    assert(validate_atom_state_plane_v1(plane).code
           == atom_state_plane_validation_code_v1::
                  missing_consumer_affordances);

    std::array<atom_persistent_identity_v1, 2> malformed{
        atom_persistent_identity_v1{9, 2}, atom_persistent_identity_v1{9, 2}};
    plane = valid_plane();
    plane.consumer_affordances = malformed.data();
    assert(validate_atom_state_plane_v1(plane).code
           == atom_state_plane_validation_code_v1::
                  unordered_or_duplicate_consumer);
    malformed[0] = {9, 1};
    malformed[1] = {9, 3};
    plane.consumer_affordances = malformed.data();
    assert(validate_atom_state_plane_v1(plane).code
           == atom_state_plane_validation_code_v1::producer_consumer_cycle);
}

void test_randomized_consumer_tables() {
    std::uint64_t state = UINT64_C(0xa12c0de);
    for (std::size_t iteration = 0; iteration < 4096; ++iteration) {
        state = state * UINT64_C(6364136223846793005) + 1;
        const std::size_t count = 1 + state % 32;
        std::vector<atom_persistent_identity_v1> table;
        table.reserve(count);
        std::uint64_t local = 10;
        for (std::size_t index = 0; index < count; ++index) {
            state = state * UINT64_C(6364136223846793005) + 1;
            local += 1 + state % 4;
            table.push_back({9, local});
        }
        auto plane = valid_plane();
        plane.consumer_affordances = table.data();
        plane.consumer_affordance_count = table.size();
        assert(validate_atom_state_plane_v1(plane).valid());
    }
}

} // namespace

int main() {
    test_typed_mutable_state();
    test_deterministic_rejections();
    test_randomized_consumer_tables();
    return 0;
}
