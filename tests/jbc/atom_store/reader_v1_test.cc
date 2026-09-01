#include <CellShard/artifact/atom_store/reader_v1.hh>
#include <CellShard/artifact/atom_store/writer_v1.hh>
#include <array>
#include <cassert>
#include <vector>
using namespace cellshard::artifact::atom_store;
int main() {
    std::array<std::byte,64> payload{}; payload[0]=std::byte{1};
    writer_section_source_v1 source{arena_section_kind_v1::atom_dictionary,arena_section_required_v1,64,payload.data(),payload.size(),32,2};
    writer_requirements_v1 req{}; assert(atom_store_writer_requirements_v1(&source,1,&req)==writer_status_v1::success);
    arena_header_v1 seed{}; seed.store_identity={1,1}; seed.catalog_identity={2,2}; seed.structure_identity={3,3}; seed.certification_identity={4,4};
    std::vector<std::byte> image(req.total_bytes); assert(fill_atom_store_v1(seed,&source,1,image.data(),image.size())==writer_status_v1::success);
    std::array<arena_directory_entry_v1,1> entries{}; metadata_inspection_v1 inspection{};
    const auto metadata_bytes=req.directory_offset+req.directory_bytes;
    assert(inspect_atom_store_metadata_v1(image.data(),metadata_bytes,entries.data(),entries.size(),&inspection)==reader_status_v1::success);
    assert(inspection.payload_verification_required && inspection.section_count==1);
    assert(inspect_atom_store_metadata_v1(image.data(),metadata_bytes-1,entries.data(),entries.size(),&inspection)==reader_status_v1::short_metadata);
}
