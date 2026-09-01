#include <CellShard/artifact/atom_store/reachability_gc_v1.hh>
namespace cellshard::artifact::atom_store {
namespace {bool eq(const content_digest_v1&a,const content_digest_v1&b){for(std::size_t i=0;i<a.bytes.size();++i)if(a.bytes[i]!=b.bytes[i])return false;return a.algorithm==b.algorithm&&a.digest_bytes==b.digest_bytes;}bool zero(const content_digest_v1&d){for(auto b:d.bytes)if(b!=std::byte{0})return false;return true;}}
reachability_status_v1 mark_reachable_generations_v1(const generation_node_v1 *nodes,std::size_t n,const content_digest_v1&active,const snapshot_pin_v1*pins,std::size_t pn,std::uint64_t current,bool *reachable) noexcept {
    if(nodes==nullptr||reachable==nullptr||n==0||(pn!=0&&pins==nullptr))return reachability_status_v1::invalid_input;
    for(std::size_t i=0;i<n;++i)reachable[i]=false;
    auto walk=[&](const content_digest_v1&start){content_digest_v1 cursor=start;std::size_t steps=0;while(!zero(cursor)){std::size_t found=n;for(std::size_t i=0;i<n;++i)if(eq(nodes[i].root,cursor)){found=i;break;}if(found==n)return reachability_status_v1::missing_root;if(reachable[found])return reachability_status_v1::success;reachable[found]=true;if(++steps>n)return reachability_status_v1::cycle;cursor=nodes[found].parent;}return reachability_status_v1::success;};
    auto status=walk(active);if(status!=reachability_status_v1::success)return status;
    for(std::size_t i=0;i<pn;++i){const auto&p=pins[i];if(!p.snapshot.valid()||p.pin_generation==0||p.valid_through_generation<p.pin_generation)return reachability_status_v1::invalid_input;if(current>p.valid_through_generation)continue;status=walk(p.root);if(status!=reachability_status_v1::success)return status;}
    return reachability_status_v1::success;
}
}
