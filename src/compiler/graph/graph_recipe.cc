#include <CellShard/compiler/graph/graph_recipe.hh>
namespace cellshard::compiler::graph {
bool valid_graph_recipe(const graph_recipe&r,const recipe_instruction*ins,std::size_t n,const recipe_parameter*p,std::size_t pn)noexcept{
    if(!r.family.valid()||!valid_content_digest(r.recipe_digest)||r.recipe_digest.algorithm==digest_algorithm::none||r.instruction_count!=n||r.parameter_count!=pn||n==0||p==nullptr||ins==nullptr||r.maximum_expanded_nodes==0)return false;
    for(std::size_t i=0;i<pn;++i)if(p[i].identity==0||p[i].minimum>p[i].value||p[i].value>p[i].maximum)return false;
    bool repeat=false;for(std::size_t i=0;i<n;++i){const auto&x=ins[i];if(x.opcode==recipe_opcode::repeat_begin){if(repeat||x.parameter_index>=pn)return false;repeat=true;}else if(x.opcode==recipe_opcode::repeat_end){if(!repeat)return false;repeat=false;}else if(x.opcode==recipe_opcode::emit_node){if(x.operand0==0)return false;}else if(x.opcode==recipe_opcode::emit_edge){if(x.operand0==0||x.operand1==0)return false;}else return false;}return !repeat;
}
}
