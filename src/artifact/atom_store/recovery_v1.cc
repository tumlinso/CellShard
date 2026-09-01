#include <CellShard/artifact/atom_store/recovery_v1.hh>
namespace cellshard::artifact::atom_store {
namespace { bool equal_digest(const content_digest_v1&a,const content_digest_v1&b){for(std::size_t i=0;i<a.bytes.size();++i)if(a.bytes[i]!=b.bytes[i])return false;return a.algorithm==b.algorithm&&a.digest_bytes==b.digest_bytes;} }
recovery_class_v1 classify_recovery_candidate_v1(const root_generation_manifest_v1 &root,const recovery_candidate_v1 &c) noexcept {
    if (!valid_root_generation_manifest_v1(c.manifest) || c.object_valid>1 || c.object_durable>1 || c.selected_by_root>1) return recovery_class_v1::corrupt;
    if (c.object_valid==0) return recovery_class_v1::corrupt;
    if (c.object_durable==0) return recovery_class_v1::incomplete;
    if (c.selected_by_root==1) return equal_digest(c.manifest.root_content,root.root_content) && c.manifest.generation==root.generation ? recovery_class_v1::active_root : recovery_class_v1::corrupt;
    if (c.manifest.generation==root.generation+1 && equal_digest(c.manifest.parent_root_content,root.root_content)) return recovery_class_v1::recoverable_successor;
    return recovery_class_v1::orphan;
}
}
