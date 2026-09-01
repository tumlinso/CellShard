#pragma once
#include <CellShard/compiler/graph/graph_family.hh>
#include <cstddef>
#include <cstdint>
#include <type_traits>
namespace cellshard::compiler::graph {
enum class recipe_opcode : std::uint32_t { emit_node=1, emit_edge=2, repeat_begin=3, repeat_end=4 };
struct recipe_parameter { std::uint64_t identity=0; std::uint64_t minimum=0; std::uint64_t maximum=0; std::uint64_t value=0; };
struct recipe_instruction { recipe_opcode opcode{}; std::uint32_t parameter_index=0; std::uint64_t operand0=0; std::uint64_t operand1=0; };
struct graph_recipe { graph_family_id family{}; content_digest recipe_digest{}; std::uint64_t instruction_count=0; std::uint64_t parameter_count=0; std::uint64_t maximum_expanded_nodes=0; std::uint64_t maximum_expanded_edges=0; };
[[nodiscard]] bool valid_graph_recipe(const graph_recipe&recipe,const recipe_instruction*instructions,std::size_t instruction_count,const recipe_parameter*parameters,std::size_t parameter_count)noexcept;
static_assert(std::is_trivially_copyable<recipe_instruction>::value);
static_assert(std::is_trivially_copyable<recipe_parameter>::value);
}
