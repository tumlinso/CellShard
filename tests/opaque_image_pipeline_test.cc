#include <CellShard/artifact/snapshot.hh>
#include <CellShard/io/pack/image_envelope.hh>
#include <CellShard/runtime/residency/host.hh>
#include <CellShard/runtime/source/local_file_source.hh>

#include <array>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <unistd.h>

namespace {
void require(bool v, const char *m) { if (!v) { std::fprintf(stderr, "cellShardOpaqueImagePipelineTest: %s\n", m); std::exit(1); } }
cellshard::content_digest digest(const std::byte *p, std::size_t n) {
    std::uint64_t h=1469598103934665603ull; for(std::size_t i=0;i<n;++i){h^=std::to_integer<unsigned char>(p[i]);h*=1099511628211ull;} if(!h)h=1;
    cellshard::content_digest d{}; d.algorithm=cellshard::digest_algorithm::legacy_fnv1a64; d.used_bytes=8; for(unsigned s=0;s<64;s+=8)d.bytes[s/8]=std::byte((h>>s)&255u); return d;
}
}
int main() {
    using namespace cellshard;
    const std::array<std::byte,8> bytes{{std::byte{1},std::byte{3},std::byte{5},std::byte{7},std::byte{9},std::byte{11},std::byte{13},std::byte{15}}};
    image_descriptor image{}; image.id=image_id{40}; image.projection={producer_abi_id{41},structure_id{42},geometry_id{43},operator_class_id{44},scalar_encoding_id{45},{execution_backend::cuda,7,0,1}}; image.stored_bytes=bytes.size(); image.device_bytes=bytes.size(); image.required_alignment=64; image.reuse=image_reuse_class::durable_reuse; image.payload_digest=digest(bytes.data(),bytes.size()); image.domains={{domain_binding_role::primary,domain_id{10},partition_map_id{20},partition_id{30},order_id{46}}};
    const std::string path="/tmp/cellshard-opaque-pipeline-"+std::to_string((unsigned long long)::getpid())+".cspack";
    image_cspack_entry_source entry{extent_id{80},view_of(image),{bytes.data(),bytes.size()}}; published_image_cspack published{};
    require(store_image_cspack(path.c_str(),9,storage_object_id{70},&entry,1,&published)==status_code::success,"fake producer publication");
    image_cspack_inspection inspected{}; require(inspect_image_cspack_partition(path.c_str(),9,0,published.object.id,extent_id{80},&inspected)==status_code::success,"independent image inspection");
    artifact_catalog catalog{}; catalog.generation=catalog_generation_id{1}; catalog.domains={{domain_id{10},domain_kind::cells,archive_generation_id{2},100}}; catalog.partition_maps={{partition_map_id{20},domain_id{10},archive_generation_id{2},100,1}}; catalog.partitions={{partition_id{30},partition_map_id{20},domain_id{10},archive_generation_id{2},0,partition_selection::contiguous(0,100)}}; catalog.images={inspected.descriptor}; catalog.storage_objects={published.object}; catalog.extents={inspected.payload_extent}; catalog.image_extents={{image.id,{extent_id{80}}}};
    source_catalog sources{{{source_location_id{90},source_provider_id{91},published.object.id,capability_bit(source_capability::exact_range_read)|capability_bit(source_capability::stable_size),path}}};
    snapshot_manifest snapshot{snapshot_id{100},catalog.generation,archive_generation_id{2},{domain_id{10}},{partition_map_id{20}},{image.id},{}};
    require(valid_snapshot_manifest(snapshot,catalog,sources),"artifact/source snapshot composition");
    local_file_source source{}; require(open_local_file_source(path.c_str(),source_provider_id{91},source_location_id{90},published.object,&source)==status_code::success,"bind mutable local source");
    host_residency host{}; require(load_host_residency(source.ref(),published.object,inspected.payload_extent,view_of(inspected.descriptor),default_host_allocator(),&host)==status_code::success,"one-allocation host residency");
    const auto consumer=host.view(); require(consumer.image==image.id&&consumer.payload_bytes==bytes.size()&&std::equal(bytes.begin(),bytes.end(),consumer.payload),"fake consumer exact identity and bytes");
    host.reset(); source.reset(); std::remove(path.c_str()); std::puts("cellShardOpaqueImagePipelineTest: passed"); return 0;
}
