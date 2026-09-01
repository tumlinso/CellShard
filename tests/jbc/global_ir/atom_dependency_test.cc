#include <CellShard/compiler/graph/atom_dependency.hh>
#include <cassert>
using namespace cellshard::compiler::graph;
int main(){atom_dependency_edge e{graph_atom_id{1},operation_node_id{2},operation_port_id{3},operation_node_id{4},operation_port_id{5},6,0,dependency_kind::exact_atom,0};assert(valid_atom_dependency_edge(e));e.kind=dependency_kind::value_generation;assert(!valid_atom_dependency_edge(e));e.required_value_generation=7;assert(valid_atom_dependency_edge(e));e.consumer_node=e.producer_node;assert(!valid_atom_dependency_edge(e));}
