#include <CellShard/compiler/composition/parameter_binding_v1.hh>

#include <array>
#include <cassert>

namespace composition = cellshard::compiler::composition;

int main() {
    const std::array<composition::parameter_signature_v1, 2> signatures{{
        {1, composition::parameter_kind_v1::scalar, {},
         cellshard::scalar_encoding_id{2}, 1},
        {3, composition::parameter_kind_v1::value_plane, {},
         cellshard::scalar_encoding_id{4}, (std::uint64_t{1} << 40u)}}};
    const std::array<composition::parameter_binding_v1, 2> bindings{{
        {1, 10, composition::parameter_kind_v1::scalar, {},
         cellshard::scalar_encoding_id{2}, 7, 1},
        {3, 11, composition::parameter_kind_v1::value_plane, {},
         cellshard::scalar_encoding_id{4}, 8,
         (std::uint64_t{1} << 40u)}}};
    composition::parameter_binding_composition_v1 output{};
    assert(composition::compose_parameter_bindings_v1(
               composition::composition_production_id{12}, signatures.data(),
               bindings.data(), bindings.size(), &output).bound());
    assert(output.binding_count == 2);

    auto stale_shape = bindings;
    --stale_shape[1].element_count;
    assert(composition::compose_parameter_bindings_v1(
               composition::composition_production_id{12}, signatures.data(),
               stale_shape.data(), stale_shape.size(), &output).code
           == composition::parameter_binding_code_v1::element_count_mismatch);
}
