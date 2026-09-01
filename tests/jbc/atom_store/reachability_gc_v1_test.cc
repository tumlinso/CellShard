#include <CellShard/artifact/atom_store/reachability_gc_v1.hh>
#include <array>
#include <cassert>
using namespace cellshard::artifact::atom_store;
content_digest_v1 d(std::byte b){content_digest_v1 x{};x.bytes[0]=b;return x;}
int main(){std::array<generation_node_v1,4>n{{{d(std::byte{1}),{},1},{d(std::byte{2}),d(std::byte{1}),2},{d(std::byte{3}),d(std::byte{2}),3},{d(std::byte{9}),{},1}}};std::array<bool,4>r{};snapshot_pin_v1 pin{cellshard::snapshot_id{1},d(std::byte{9}),1,5};assert(mark_reachable_generations_v1(n.data(),n.size(),d(std::byte{3}),&pin,1,4,r.data())==reachability_status_v1::success);for(bool x:r)assert(x);pin.valid_through_generation=3;assert(mark_reachable_generations_v1(n.data(),n.size(),d(std::byte{3}),&pin,1,4,r.data())==reachability_status_v1::success);assert(!r[3]);}
