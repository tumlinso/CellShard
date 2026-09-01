#include <compiler/grammar/induced/flat_dag_comparison_v1.hh>
#include <iostream>
namespace ig=cellshard::compiler::grammar::induced;
int main(){const auto planted=ig::compare_with_flat_dag_v1({{1,1},{2,1},{2,1},3,1000,2000,1200,ig::fixture_kind_v1::planted_repetition});const auto adversarial=ig::compare_with_flat_dag_v1({{1,2},{2,2},{2,2},3,1000,1000,1400,ig::fixture_kind_v1::adversarial_unique});if(!planted.compared()||!adversarial.compared())return 2;std::cout<<"{\"planted_induced_wins\":"<<(planted.induced_wins?"true":"false")<<",\"planted_saved_ns\":"<<planted.saved_ns<<",\"adversarial_induced_wins\":"<<(adversarial.induced_wins?"true":"false")<<"}\n";return planted.induced_wins&&!adversarial.induced_wins?0:3;}
