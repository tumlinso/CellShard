#include <compiler/grammar/induced/observation_input_v1.hh>
#include <cassert>
namespace ig=cellshard::compiler::grammar::induced;
int main(){ig::execution_observation_v1 x[]={{1,{1,1},{2,1},{3,1},{3,2},4,5,10,20,64,1},{2,{1,2},{2,1},{3,1},{3,2},4,6,20,35,96,2}};ig::induced_grammar_observation_view_v1 v{x,2,2,{4,1},{4,2},7};assert(ig::validate_induced_grammar_observations_v1(v).valid()&&!ig::authorizes_execution(v));x[1].begin_tick=19;assert(ig::validate_induced_grammar_observations_v1(v).code==ig::observation_input_code_v1::overlapping_time);}
