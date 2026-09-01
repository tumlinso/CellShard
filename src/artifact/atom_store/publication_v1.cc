#include <CellShard/artifact/atom_store/publication_v1.hh>
namespace cellshard::artifact::atom_store {
namespace { bool equal_digest(const content_digest_v1&a,const content_digest_v1&b){if(a.algorithm!=b.algorithm||a.digest_bytes!=b.digest_bytes)return false;for(std::size_t i=0;i<a.bytes.size();++i)if(a.bytes[i]!=b.bytes[i])return false;return true;} }
publication_status_v1 publish_root_generation_v1(const root_generation_manifest_v1 &current, const root_generation_manifest_v1 &next, const std::byte *image, std::size_t image_bytes, const publication_backend_v1 &backend) noexcept {
    if (!valid_root_generation_manifest_v1(current) || !valid_root_generation_manifest_v1(next)
        || next.generation!=current.generation+1 || next.structure_epoch<current.structure_epoch
        || !equal_digest(next.parent_root_content,current.root_content) || image==nullptr || image_bytes==0
        || backend.stage_immutable==nullptr || backend.sync_immutable==nullptr || backend.compare_exchange_root==nullptr || backend.sync_root==nullptr) return publication_status_v1::invalid_generation;
    if (!backend.stage_immutable(backend.context,next.root_content,image,image_bytes)) return publication_status_v1::stage_failed;
    if (!backend.sync_immutable(backend.context,next.root_content)) return publication_status_v1::object_sync_failed;
    if (!backend.compare_exchange_root(backend.context,current.root_content,next.root_content)) return publication_status_v1::root_conflict;
    if (!backend.sync_root(backend.context)) return publication_status_v1::root_sync_failed;
    return publication_status_v1::success;
}
}
