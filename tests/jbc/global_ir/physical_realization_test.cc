#include <CellShard/compiler/graph/physical_realization.hh>
#include <cassert>
using namespace cellshard::compiler::graph;
int main(){physical_node_binding b{operation_node_id{1},cellshard::producer_abi_id{2},cellshard::image_id{3},4};assert(valid_physical_node_binding(b));physical_graph_realization r{};r.id=physical_realization_id{1};r.family=graph_family_id{2};r.target.backend=cellshard::execution_backend::cuda;r.target.capability_major=7;r.binding_count=1;r.estimated_launches=2;assert(valid_physical_graph_realization(r));r.target.capability_major=0;assert(!valid_physical_graph_realization(r));}
