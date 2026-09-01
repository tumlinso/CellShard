#include <CellShard/compiler/grammar/production_parameter_schema_v1.hh>
#include <cassert>
namespace g=cellshard::compiler::grammar;
int main(){g::explicit_production_v1 p{{1,1},{2,1},nullptr,1,1,3,g::production_algebra_v1::ordered_composition};g::explicit_production_registry_v1 r{&p,1,1,{3,1},{3,2},4,5};g::production_parameter_spec_v1 specs[]={{{4,1},g::production_parameter_kind_v1::rational,-2,2,10,true,{}}};g::production_parameter_schema_v1 s{{5,1},{1,1},specs,1,1,6,3};assert(g::validate_production_parameter_schema_v1(s,r).valid()&&!g::authorizes_execution(s));specs[0].denominator=0;assert(g::validate_production_parameter_schema_v1(s,r).code==g::parameter_schema_code_v1::invalid_range);}
