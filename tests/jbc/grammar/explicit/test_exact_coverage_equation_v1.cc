#include <CellShard/compiler/grammar/exact_coverage_equation_v1.hh>
#include <cassert>
#include <limits>
namespace g=cellshard::compiler::grammar;
int main(){g::coverage_term_v1 t[]={{{1,1},{2,1},7,10,3,4},{{1,2},{2,2},3,10,3,4}};g::exact_coverage_equation_v1 e{t,2,2,10,10,{3,1},{3,2},{3,3},5};auto r=g::evaluate_exact_coverage_equation_v1(e);assert(r.exact()&&g::authorizes_execution(r));t[1].numerator=2;assert(g::evaluate_exact_coverage_equation_v1(e).code==g::coverage_equation_code_v1::not_exact);t[0].numerator=std::numeric_limits<std::int64_t>::max();assert(g::evaluate_exact_coverage_equation_v1(e).code==g::coverage_equation_code_v1::arithmetic_overflow);}
