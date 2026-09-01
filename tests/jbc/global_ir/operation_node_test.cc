#include <CellShard/compiler/graph/operation_node.hh>
#include <cassert>
using namespace cellshard::compiler::graph;
int main(){operation_node_descriptor n{operation_node_id{1},cellshard::producer_abi_id{2},cellshard::operator_class_id{3},0,2,4};assert(valid_operation_node_descriptor(n));typed_port_descriptor p{};p.id=operation_port_id{1};p.node=n.id;p.domain=cellshard::domain_id{2};p.order=cellshard::order_id{3};p.encoding=cellshard::scalar_encoding_id{4};p.direction=port_direction::input;p.payload=port_payload_kind::value_plane;assert(valid_typed_port_descriptor(p));p.order={};assert(!valid_typed_port_descriptor(p));p.payload=port_payload_kind::control;assert(valid_typed_port_descriptor(p));}
