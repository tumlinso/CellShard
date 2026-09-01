#include <CellShard/compiler/grammar/explicit_production_registry_v1.hh>
#include <cassert>
namespace g=cellshard::compiler::grammar;
int main(){
 g::typed_grammar_symbol_v1 symbols[]={{{1,1},{2,1},{2,2},{2,3},{},1,0,g::grammar_symbol_kind_v1::terminal_atom,g::grammar_value_kind_v1::immutable_structure},{{1,2},{2,1},{2,2},{2,3},{2,4},1,1,g::grammar_symbol_kind_v1::nonterminal,g::grammar_value_kind_v1::partial_result}};
 g::typed_symbol_table_v1 table{symbols,2,2,{3,1},4};g::grammar_identity_v1 rhs[]={{1,1}};
 g::explicit_production_v1 p{{4,1},{1,2},rhs,1,1,5,g::production_algebra_v1::ordered_composition};g::explicit_production_registry_v1 r{&p,1,1,{5,1},{3,1},6,4};
 assert(g::validate_explicit_production_registry_v1(r,table).valid()&&!g::authorizes_execution(r));rhs[0]={9,9};assert(g::validate_explicit_production_registry_v1(r,table).code==g::production_registry_code_v1::invalid_rhs);
}
