#include <CellShard/compiler/atom/value_plane_v1.hh>

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace {

using namespace cellshard::compiler::atom;

alignas(16) float values[4]{};

atom_value_plane_v1 valid_plane(const std::byte *mapping,
                                const std::byte *dirty = nullptr,
                                std::uint64_t dirty_count = 0) {
    atom_value_plane_v1 plane{};
    plane.values = values;
    plane.value_bytes = sizeof(values);
    plane.element_count = 4;
    plane.element_stride_bytes = sizeof(float);
    plane.element_bytes = sizeof(float);
    plane.value_alignment = 16;
    plane.plane_identity = {1, 11};
    plane.structure_plane_identity = {1, 12};
    plane.subject_space_identity = {1, 13};
    plane.persistent_order_identity = {1, 14};
    plane.numeric = {{2, 1}, {2, 1}, {2, 2}};
    plane.structure_epoch = 7;
    plane.value_generation = 9;
    plane.canonical_element_count = 4;
    plane.local_to_canonical = {
        mapping, 4, 4, compact_edge_index_width_v1::u8, {}};
    plane.dirty_local_indices = {
        dirty, dirty_count, dirty_count, compact_edge_index_width_v1::u8, {}};
    return plane;
}

void test_logical_and_projection_primary() {
    const std::array<std::byte, 4> identity{
        std::byte{0}, std::byte{1}, std::byte{2}, std::byte{3}};
    const std::array<std::byte, 4> projection{
        std::byte{3}, std::byte{1}, std::byte{0}, std::byte{2}};
    const std::array<std::byte, 2> dirty{std::byte{1}, std::byte{3}};
    std::array<std::uint8_t, 4> marks{};
    auto plane = valid_plane(identity.data(), dirty.data(), dirty.size());
    assert(validate_atom_value_plane_v1(
               plane, marks.data(), marks.size())
               .valid());
    plane = valid_plane(projection.data());
    plane.ownership = atom_value_ownership_v1::projection_primary;
    assert(validate_atom_value_plane_v1(
               plane, marks.data(), marks.size())
               .valid());
}

void test_deterministic_rejections() {
    std::array<std::byte, 4> mapping{
        std::byte{0}, std::byte{1}, std::byte{2}, std::byte{3}};
    std::array<std::uint8_t, 4> marks{};
    auto plane = valid_plane(mapping.data());
    plane.values = nullptr;
    assert(validate_atom_value_plane_v1(plane, marks.data(), marks.size()).code
           == atom_value_plane_validation_code_v1::missing_values);
    plane = valid_plane(mapping.data());
    plane.value_generation = 0;
    assert(validate_atom_value_plane_v1(plane, marks.data(), marks.size()).code
           == atom_value_plane_validation_code_v1::missing_value_generation);
    plane = valid_plane(mapping.data());
    plane.value_bytes = 15;
    assert(validate_atom_value_plane_v1(plane, marks.data(), marks.size()).code
           == atom_value_plane_validation_code_v1::insufficient_value_bytes);
    plane = valid_plane(mapping.data());
    plane.local_to_canonical.index_bytes = 3;
    assert(validate_atom_value_plane_v1(plane, marks.data(), marks.size()).code
           == atom_value_plane_validation_code_v1::
                  invalid_canonical_map_bytes);
    mapping[3] = std::byte{2};
    plane = valid_plane(mapping.data());
    plane.ownership = atom_value_ownership_v1::projection_primary;
    assert(validate_atom_value_plane_v1(plane, marks.data(), marks.size()).code
           == atom_value_plane_validation_code_v1::duplicate_canonical_index);
    mapping[3] = std::byte{0};
    plane = valid_plane(mapping.data());
    plane.ownership = atom_value_ownership_v1::projection_primary;
    assert(validate_atom_value_plane_v1(plane, marks.data(), marks.size()).code
           == atom_value_plane_validation_code_v1::duplicate_canonical_index);
    mapping = {std::byte{3}, std::byte{1}, std::byte{0}, std::byte{2}};
    plane = valid_plane(mapping.data());
    assert(validate_atom_value_plane_v1(plane, marks.data(), marks.size()).code
           == atom_value_plane_validation_code_v1::
                  nonidentity_logical_primary);
    const std::array<std::byte, 2> duplicate_dirty{
        std::byte{2}, std::byte{2}};
    mapping = {std::byte{0}, std::byte{1}, std::byte{2}, std::byte{3}};
    plane = valid_plane(mapping.data(), duplicate_dirty.data(), 2);
    assert(validate_atom_value_plane_v1(plane, marks.data(), marks.size()).code
           == atom_value_plane_validation_code_v1::
                  unordered_or_duplicate_dirty_index);
}

void test_randomized_projection_orders() {
    std::uint64_t state = UINT64_C(0xa11c0de);
    constexpr std::size_t count = 251;
    std::vector<float> storage(count);
    std::vector<std::byte> mapping(count);
    std::vector<std::uint8_t> marks(count);
    for (std::size_t iteration = 0; iteration < 4096; ++iteration) {
        for (std::size_t index = 0; index < count; ++index) {
            mapping[index] = static_cast<std::byte>(index);
        }
        for (std::size_t index = count - 1; index != 0; --index) {
            state = state * UINT64_C(6364136223846793005) + 1;
            const auto other = state % (index + 1);
            const auto temporary = mapping[index];
            mapping[index] = mapping[other];
            mapping[other] = temporary;
        }
        auto plane = valid_plane(mapping.data());
        plane.values = storage.data();
        plane.value_bytes = storage.size() * sizeof(float);
        plane.element_count = count;
        plane.canonical_element_count = count;
        plane.local_to_canonical.index_count = count;
        plane.local_to_canonical.index_bytes = count;
        plane.value_alignment = alignof(float);
        plane.ownership = atom_value_ownership_v1::projection_primary;
        assert(validate_atom_value_plane_v1(
                   plane, marks.data(), marks.size())
                   .valid());
    }
}

} // namespace

int main() {
    test_logical_and_projection_primary();
    test_deterministic_rejections();
    test_randomized_projection_orders();
    return 0;
}
