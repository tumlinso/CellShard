#pragma once
#include <CellShard/artifact/image.hh>
#include <CellShard/compiler/graph/graph_family.hh>
#include <type_traits>
namespace cellshard::compiler::graph {
struct physical_realization_id_tag;
using physical_realization_id=strong_id<physical_realization_id_tag>;
struct physical_node_binding { operation_node_id logical_node{}; producer_abi_id provider{}; image_id projection{}; std::uint64_t physical_operator_identity=0; };
struct physical_graph_realization {
    physical_realization_id id{}; graph_family_id family{}; target_capabilities target{};
    std::uint64_t binding_offset=0; std::uint64_t binding_count=0;
    std::uint64_t preparation_bytes=0; std::uint64_t persistent_bytes=0;
    std::uint64_t transient_bytes=0; std::uint64_t estimated_launches=0;
};
[[nodiscard]] constexpr bool valid_physical_node_binding(const physical_node_binding&b)noexcept{return b.logical_node.valid()&&b.provider.valid()&&b.projection.valid()&&b.physical_operator_identity!=0;}
[[nodiscard]] constexpr bool valid_physical_graph_realization(const physical_graph_realization&r)noexcept{return r.id.valid()&&r.family.valid()&&valid_target_capabilities(r.target)&&r.binding_count!=0&&r.binding_offset<=UINT64_MAX-r.binding_count&&r.estimated_launches!=0;}
static_assert(std::is_trivially_copyable<physical_node_binding>::value);
static_assert(std::is_trivially_copyable<physical_graph_realization>::value);
}
