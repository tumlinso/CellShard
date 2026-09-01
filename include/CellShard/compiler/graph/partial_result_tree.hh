#pragma once
#include <CellShard/compiler/graph/atom_dependency.hh>
#include <cstddef>
#include <cstdint>
namespace cellshard::compiler::graph {
struct partial_result_node { graph_atom_id result{}; std::uint64_t algebra_identity=0; std::uint64_t first_child=0; std::uint32_t child_count=0; std::uint32_t level=0; };
enum class partial_tree_status : std::uint32_t { success, invalid_input, insufficient_output, identity_overflow };
[[nodiscard]] std::size_t partial_result_tree_node_count(std::size_t leaf_count)noexcept;
[[nodiscard]] partial_tree_status compile_partial_result_tree(const graph_atom_id*leaves,std::size_t leaf_count,std::uint64_t algebra_identity,std::uint64_t generated_identity_base,partial_result_node*output,std::size_t capacity,std::size_t*node_count,std::size_t*root_index)noexcept;
}
