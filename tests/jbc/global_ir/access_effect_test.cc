#include <CellShard/compiler/graph/access_effect.hh>
#include <cassert>
using namespace cellshard::compiler::graph;
int main(){port_access_descriptor a{};a.port=operation_port_id{1};a.mode=access_mode::read;a.residency=residency_kind::device;a.structure_epoch=2;assert(valid_port_access_descriptor(a));a.effects=effect_advance_values;assert(!valid_port_access_descriptor(a));a.mode=access_mode::read_write;a.value_generation=3;assert(valid_port_access_descriptor(a));a.effects|=UINT64_C(1)<<63;assert(!valid_port_access_descriptor(a));}
