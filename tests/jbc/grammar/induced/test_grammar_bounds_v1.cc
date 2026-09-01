#include <compiler/grammar/induced/grammar_bounds_v1.hh>
#include <cassert>
namespace ig=cellshard::compiler::grammar::induced;
int main(){std::uint64_t depths[]={1,3,2};ig::induced_production_shape_v1 x{{1,1},depths,3,3,4,7};auto r=ig::validate_induced_production_bounds_v1(x,{4,5});assert(r.valid()&&r.computed_depth==4&&!ig::authorizes_execution(x));x.arity=5;assert(ig::validate_induced_production_bounds_v1(x,{4,5}).code==ig::grammar_bounds_code_v1::capacity_overflow);x.arity=3;x.declared_depth=3;assert(ig::validate_induced_production_bounds_v1(x,{4,5}).code==ig::grammar_bounds_code_v1::depth_mismatch);}
