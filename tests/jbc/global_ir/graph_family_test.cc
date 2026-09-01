#include <CellShard/compiler/graph/graph_family.hh>
#include <cassert>
using namespace cellshard::compiler::graph;
int main(){graph_family_descriptor f{};f.id=graph_family_id{1};f.logical_graph_digest.algorithm=cellshard::digest_algorithm::legacy_fnv1a64;f.logical_graph_digest.used_bytes=8;f.logical_graph_digest.bytes[0]=std::byte{1};f.provider_count=2;f.node_class_count=3;f.structure_epoch=4;assert(valid_graph_family_descriptor(f));workload_distribution w{f.id,10,1,4,8,0,3,9,65536};assert(valid_workload_distribution(w));w.median_nodes=9;assert(!valid_workload_distribution(w));}
