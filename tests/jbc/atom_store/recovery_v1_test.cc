#include <CellShard/artifact/atom_store/recovery_v1.hh>
#include <cassert>
using namespace cellshard::artifact::atom_store;
root_generation_manifest_v1 m(std::uint64_t g,std::byte d){root_generation_manifest_v1 x{};x.store_identity={1,2};x.generation=g;x.structure_epoch=1;x.root_content.bytes[0]=d;if(g>1)x.parent_root_content.bytes[0]=static_cast<std::byte>(static_cast<unsigned>(d)-1);return x;}
int main(){auto root=m(1,std::byte{1});recovery_candidate_v1 c{root,1,1,1,0};assert(classify_recovery_candidate_v1(root,c)==recovery_class_v1::active_root);c={m(2,std::byte{2}),1,1,0,0};assert(classify_recovery_candidate_v1(root,c)==recovery_class_v1::recoverable_successor);c.manifest.generation=3;assert(classify_recovery_candidate_v1(root,c)==recovery_class_v1::orphan);c.object_durable=0;assert(classify_recovery_candidate_v1(root,c)==recovery_class_v1::incomplete);}
