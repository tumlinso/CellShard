#include <CellShard/compiler/graph/graph_recipe.hh>
#include <array>
#include <cassert>
using namespace cellshard::compiler::graph;
int main(){graph_recipe r{};r.family=graph_family_id{1};r.recipe_digest.algorithm=cellshard::digest_algorithm::legacy_fnv1a64;r.recipe_digest.used_bytes=8;r.recipe_digest.bytes[0]=std::byte{1};r.instruction_count=4;r.parameter_count=1;r.maximum_expanded_nodes=8;r.maximum_expanded_edges=7;recipe_parameter p{1,1,8,4};std::array<recipe_instruction,4>i{{{recipe_opcode::repeat_begin,0,0,0},{recipe_opcode::emit_node,0,2,0},{recipe_opcode::emit_edge,0,2,3},{recipe_opcode::repeat_end,0,0,0}}};assert(valid_graph_recipe(r,i.data(),i.size(),&p,1));i[3].opcode=recipe_opcode::emit_node;assert(!valid_graph_recipe(r,i.data(),i.size(),&p,1));}
