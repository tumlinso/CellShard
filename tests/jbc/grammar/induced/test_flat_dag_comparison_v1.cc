#include <compiler/grammar/induced/flat_dag_comparison_v1.hh>
#include <cassert>
namespace ig=cellshard::compiler::grammar::induced;
int main(){auto planted=ig::compare_with_flat_dag_v1({{1,1},{2,1},{2,1},3,100,200,120,ig::fixture_kind_v1::planted_repetition});assert(planted.compared()&&planted.induced_wins&&planted.saved_ns==80&&!ig::authorizes_execution(planted));auto adversarial=ig::compare_with_flat_dag_v1({{1,2},{2,2},{2,2},3,100,100,140,ig::fixture_kind_v1::adversarial_unique});assert(adversarial.compared()&&!adversarial.induced_wins);}
