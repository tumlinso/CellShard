#include <CellShard/compiler/graph/partial_result_tree.hh>
namespace cellshard::compiler::graph {
std::size_t partial_result_tree_node_count(std::size_t n)noexcept{if(n==0)return 0;std::size_t total=n;while(n>1){n=(n+1)/2;if(total>SIZE_MAX-n)return 0;total+=n;}return total;}
partial_tree_status compile_partial_result_tree(const graph_atom_id*leaves,std::size_t n,std::uint64_t algebra,std::uint64_t base,partial_result_node*out,std::size_t cap,std::size_t*count,std::size_t*root)noexcept{
    if(leaves==nullptr||n==0||algebra==0||base==0||out==nullptr||count==nullptr||root==nullptr)return partial_tree_status::invalid_input;
    const auto needed=partial_result_tree_node_count(n);if(needed==0)return partial_tree_status::invalid_input;if(cap<needed)return partial_tree_status::insufficient_output;
    for(std::size_t i=0;i<n;++i){if(!leaves[i].valid())return partial_tree_status::invalid_input;out[i]={leaves[i],algebra,0,0,0};}
    std::size_t level_offset=0,level_count=n,next=n;std::uint32_t level=1;while(level_count>1){const auto parents=(level_count+1)/2;for(std::size_t i=0;i<parents;++i){if(base>UINT64_MAX-(next+i))return partial_tree_status::identity_overflow;const auto children=static_cast<std::uint32_t>((2*i+1<level_count)?2:1);out[next+i]={graph_atom_id{base+next+i},algebra,level_offset+2*i,children,level};}level_offset=next;level_count=parents;next+=parents;++level;}
    *count=next;*root=next-1;return partial_tree_status::success;
}
}
