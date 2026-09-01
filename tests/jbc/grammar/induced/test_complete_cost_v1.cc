#include <compiler/grammar/induced/complete_cost_v1.hh>
#include <cassert>
#include <limits>
namespace ig=cellshard::compiler::grammar::induced;
int main(){auto r=ig::compute_induced_complete_cost_v1({{1,1},{2,1},3,100,20,1000,50,1,4,1,10});assert(r.computed()&&r.cost.storage_ns==100&&r.cost.expected_invalidation_ns==13&&r.cost.total_ns==233&&!ig::authorizes_execution(r.cost));assert(ig::compute_induced_complete_cost_v1({{1,1},{2,1},3,100,20,std::numeric_limits<std::uint64_t>::max(),50,1,4,2,1}).code==ig::complete_cost_code_v1::overflow);}
