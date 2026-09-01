#include <CellShard/compiler/composition/value_plane_substitution_v1.hh>

#include <cassert>
#include <cstdint>

namespace composition = cellshard::compiler::composition;

int main() {
    const composition::value_plane_identity_v1 previous{
        composition::value_plane_id{1}, cellshard::structure_id{2},
        cellshard::order_id{3}, cellshard::scalar_encoding_id{4}, 7,
        (std::uint64_t{1} << 40u)};
    const composition::value_plane_identity_v1 replacement{
        composition::value_plane_id{5}, cellshard::structure_id{2},
        cellshard::order_id{3}, cellshard::scalar_encoding_id{4}, 8,
        (std::uint64_t{1} << 40u)};
    composition::value_plane_substitution_v1 output{};
    assert(composition::compose_value_plane_substitution_v1(
               composition::composition_production_id{6}, previous,
               replacement, &output).substituted());
    assert(output.previous.generation == 7);
    assert(output.replacement.generation == 8);

    auto stale = replacement;
    stale.generation = 7;
    assert(composition::compose_value_plane_substitution_v1(
               composition::composition_production_id{6}, previous, stale,
               &output).code
           == composition::value_plane_substitution_code_v1::
                  stale_replacement_generation);

    auto wrong_structure = replacement;
    wrong_structure.structure = cellshard::structure_id{9};
    assert(composition::compose_value_plane_substitution_v1(
               composition::composition_production_id{6}, previous,
               wrong_structure, &output).code
           == composition::value_plane_substitution_code_v1::
                  structure_mismatch);
}
