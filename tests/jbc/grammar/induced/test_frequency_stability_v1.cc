#include <compiler/grammar/induced/frequency_stability_v1.hh>
#include <cassert>
namespace ig=cellshard::compiler::grammar::induced;
int main(){ig::stratum_occurrence_v1 x[]={{{1,1},{2,1},{3,1},5,7},{{1,1},{2,2},{3,1},2,7},{{1,1},{2,3},{3,2},8,7}};auto r=ig::compute_frequency_stability_v1({1,1},{4,1},x,3,4,7);assert(r.computed()&&r.evidence.total_occurrences==15&&r.evidence.stable_strata==2&&r.evidence.stability_denominator==3&&!ig::authorizes_execution(r.evidence));}
