#include <CellShard/artifact/atom_store/writer_v1.hh>
#include <array>
#include <cassert>
#include <cstring>
#include <vector>
using namespace cellshard::artifact::atom_store;
int main() {
    std::array<std::byte,64> a{}; std::array<std::byte,128> b{}; a[0]=std::byte{1}; b[0]=std::byte{2};
    writer_section_source_v1 sections[2]{{arena_section_kind_v1::atom_dictionary,arena_section_required_v1,64,a.data(),a.size(),32,2},{arena_section_kind_v1::payload_bytes,arena_section_required_v1,128,b.data(),b.size(),0,0}};
    writer_requirements_v1 req{}; assert(atom_store_writer_requirements_v1(sections,2,&req)==writer_status_v1::success);
    std::vector<std::byte> bytes(req.total_bytes); arena_header_v1 seed{}; seed.store_identity={1,1}; seed.catalog_identity={2,2}; seed.structure_identity={3,3}; seed.certification_identity={4,4};
    assert(fill_atom_store_v1(seed,sections,2,bytes.data(),bytes.size())==writer_status_v1::success);
    arena_header_v1 header{}; std::memcpy(&header,bytes.data(),sizeof(header)); assert(valid_arena_header_shape_v1(header));
    arena_directory_entry_v1 entry{}; std::memcpy(&entry,bytes.data()+header.section_directory_offset,sizeof(entry));
    assert(valid_arena_directory_entry_shape_v1(entry,header.total_bytes)); assert(bytes[entry.offset]==std::byte{1});
    assert(fill_atom_store_v1(seed,sections,2,bytes.data(),bytes.size()-1)==writer_status_v1::insufficient_capacity);
}
