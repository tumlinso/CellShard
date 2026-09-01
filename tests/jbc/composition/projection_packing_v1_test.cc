#include <CellShard/compiler/composition/projection_packing_v1.hh>

#include <array>
#include <cassert>
#include <cstdint>

namespace composition = cellshard::compiler::composition;

namespace {

composition::physical_view_identity_v1 view(
    std::uint64_t identity, std::uint64_t bytes, std::uint32_t alignment) {
    return {cellshard::image_id{identity}, cellshard::structure_id{2},
            cellshard::geometry_id{3}, cellshard::operator_class_id{4},
            cellshard::scalar_encoding_id{5}, cellshard::order_id{6},
            bytes, bytes * 2, alignment, 0};
}

} // namespace

int main() {
    const std::array<composition::physical_view_identity_v1, 3> views{{
        view(10, 13, 8), view(20, 20, 16), view(30, 7, 64)}};
    const composition::physical_view_family_v1 family{
        cellshard::structure_id{2}, views.data(), views.size(), 0};
    std::array<composition::packed_projection_extent_v1, 3> extents{};
    composition::projection_pack_composition_v1 output{};
    assert(composition::compose_projection_packing_v1(
               cellshard::image_id{40}, family, extents.data(), extents.size(),
               &output).packed());
    assert(extents[0].byte_offset == 0);
    assert(extents[1].byte_offset == 16);
    assert(extents[2].byte_offset == 64);
    assert(output.packed_bytes == 71);

    auto malformed_views = views;
    malformed_views[1].required_alignment = 3;
    auto malformed = family;
    malformed.views = malformed_views.data();
    assert(composition::compose_projection_packing_v1(
               cellshard::image_id{40}, malformed, extents.data(),
               extents.size(), &output).code
           == composition::projection_packing_code_v1::invalid_view);
}
