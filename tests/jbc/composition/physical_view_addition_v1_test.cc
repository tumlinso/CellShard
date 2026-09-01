#include <CellShard/compiler/composition/physical_view_addition_v1.hh>

#include <array>
#include <cassert>

namespace composition = cellshard::compiler::composition;

namespace {

composition::physical_view_identity_v1 view(std::uint64_t identity) {
    return {cellshard::image_id{identity}, cellshard::structure_id{2},
            cellshard::geometry_id{3}, cellshard::operator_class_id{4},
            cellshard::scalar_encoding_id{5}, cellshard::order_id{6},
            100, 200, 64, 0};
}

} // namespace

int main() {
    const std::array<composition::physical_view_identity_v1, 2> current{{
        view(10), view(30)}};
    const composition::physical_view_family_v1 family{
        cellshard::structure_id{2}, current.data(), current.size(), 0};
    std::array<composition::physical_view_identity_v1, 3> storage{};
    composition::physical_view_family_v1 output{};
    assert(composition::compose_physical_view_addition_v1(
               family, view(20), storage.data(), storage.size(), &output)
               .added());
    assert(storage[0].identity == cellshard::image_id{10});
    assert(storage[1].identity == cellshard::image_id{20});
    assert(storage[2].identity == cellshard::image_id{30});

    assert(composition::compose_physical_view_addition_v1(
               family, view(30), storage.data(), storage.size(), &output).code
           == composition::physical_view_addition_code_v1::
                  duplicate_view_identity);
}
