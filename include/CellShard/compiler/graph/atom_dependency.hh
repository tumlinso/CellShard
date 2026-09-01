#pragma once
#include <CellShard/compiler/graph/operation_node.hh>
#include <cstdint>
#include <type_traits>
namespace cellshard::compiler::graph {
struct graph_atom_id_tag;
using graph_atom_id=strong_id<graph_atom_id_tag>;
enum class dependency_kind : std::uint32_t { exact_atom=1, coverage=2, value_generation=3, control=4 };
struct atom_dependency_edge {
    graph_atom_id atom{}; operation_node_id producer_node{}; operation_port_id producer_port{};
    operation_node_id consumer_node{}; operation_port_id consumer_port{};
    std::uint64_t required_structure_epoch=0; std::uint64_t required_value_generation=0;
    dependency_kind kind{}; std::uint32_t reserved=0;
};
[[nodiscard]] constexpr bool valid_atom_dependency_edge(const atom_dependency_edge&e)noexcept{
    const bool kind=e.kind==dependency_kind::exact_atom||e.kind==dependency_kind::coverage||e.kind==dependency_kind::value_generation||e.kind==dependency_kind::control;
    if(!e.atom.valid()||!e.producer_node.valid()||!e.producer_port.valid()||!e.consumer_node.valid()||!e.consumer_port.valid()||e.producer_node==e.consumer_node||e.required_structure_epoch==0||!kind)return false;
    return e.kind!=dependency_kind::value_generation||e.required_value_generation!=0;
}
static_assert(std::is_trivially_copyable<atom_dependency_edge>::value);
}
