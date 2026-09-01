#include <compiler/grammar/induced/mdl_complexity_v1.hh>
#include <cassert>
namespace ig=cellshard::compiler::grammar::induced;
int main(){ig::canonical_symbol_v1 s[3]{};ig::canonical_induced_production_v1 p{s,3,{1,1},{2,1},{3,1},4,1};ig::frequency_stability_v1 f{{2,1},{4,1},10,2,2,2,2,4};auto r=ig::compute_mdl_complexity_v1(p,f,{8,16,0,{5,1}});assert(r.computed()&&r.complexity.model_bits==40&&r.complexity.encoded_data_bits==80&&r.complexity.flat_data_bits==240&&r.complexity.saved_bits==120&&r.complexity.compresses&&!ig::authorizes_execution(r.complexity));}
