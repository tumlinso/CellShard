#include <compiler/grammar/induced/promotion_evidence_v1.hh>
#include <cassert>
namespace ig=cellshard::compiler::grammar::induced;
int main(){ig::frequency_stability_v1 f{{1,1},{2,1},10,4,3,3,4,7};ig::induced_complete_cost_v1 c{{1,1},{3,1},7,10,10,10,10,40};ig::mdl_complexity_v1 m{{1,1},{4,1},7,10,10,20,40,20,true};auto r=ig::decide_grammar_promotion_v1({5,1},f,c,m,{3,4,50},false);assert(r.decided()&&r.evidence.disposition==ig::grammar_evidence_disposition_v1::promote&&ig::authorizes_execution(r.evidence));c.total_ns=60;r=ig::decide_grammar_promotion_v1({5,2},f,c,m,{3,4,50},true);assert(r.evidence.disposition==ig::grammar_evidence_disposition_v1::demote&&r.evidence.negative_reason==ig::negative_grammar_reason_v1::complete_cost_nonpromotion&&!ig::authorizes_execution(r.evidence));}
