#include <CellShard/compiler/graph/partial_result_tree.hh>
#include <array>
#include <cassert>
using namespace cellshard::compiler::graph;
int main(){std::array<graph_atom_id,5>leaves{{graph_atom_id{1},graph_atom_id{2},graph_atom_id{3},graph_atom_id{4},graph_atom_id{5}}};std::array<partial_result_node,11>nodes{};std::size_t count=0,root=0;assert(partial_result_tree_node_count(leaves.size())==11);assert(compile_partial_result_tree(leaves.data(),leaves.size(),9,100,nodes.data(),nodes.size(),&count,&root)==partial_tree_status::success);assert(count==11&&root==10&&nodes[root].child_count==2&&nodes[root].level==3);assert(compile_partial_result_tree(leaves.data(),leaves.size(),9,100,nodes.data(),10,&count,&root)==partial_tree_status::insufficient_output);}
