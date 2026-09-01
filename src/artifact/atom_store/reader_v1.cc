#include <CellShard/artifact/atom_store/reader_v1.hh>
#include <cstring>
namespace cellshard::artifact::atom_store {
namespace {
bool known_kind(arena_section_kind_v1 kind) {
    const auto value=static_cast<std::uint32_t>(kind);
    return value>=static_cast<std::uint32_t>(arena_section_kind_v1::atom_dictionary)
        && value<=static_cast<std::uint32_t>(arena_section_kind_v1::provenance);
}
bool nonzero_digest(const content_digest_v1 &digest) {
    for (auto byte:digest.bytes) if (byte!=std::byte{0}) return true;
    return false;
}
}
reader_status_v1 inspect_atom_store_metadata_v1(const std::byte *metadata, std::size_t metadata_bytes, arena_directory_entry_v1 *entries, std::size_t entry_capacity, metadata_inspection_v1 *out) noexcept {
    if (metadata==nullptr || out==nullptr || metadata_bytes<sizeof(arena_header_v1)) return reader_status_v1::short_metadata;
    arena_header_v1 header{}; std::memcpy(&header,metadata,sizeof(header));
    if (!valid_arena_header_shape_v1(header) || !header.store_identity.valid() || !header.catalog_identity.valid()
        || !header.structure_identity.valid() || !header.certification_identity.valid()
        || !valid_content_digest_v1(header.content_digest) || !nonzero_digest(header.content_digest)) return reader_status_v1::invalid_header;
    if (header.section_directory_offset>metadata_bytes || header.section_directory_bytes>metadata_bytes-header.section_directory_offset) return reader_status_v1::short_metadata;
    if (header.section_count>entry_capacity || entries==nullptr) return reader_status_v1::insufficient_entries;
    std::uint64_t previous_end=header.section_directory_offset+header.section_directory_bytes;
    for (std::size_t i=0;i<header.section_count;++i) {
        arena_directory_entry_v1 entry{}; std::memcpy(&entry,metadata+header.section_directory_offset+i*sizeof(entry),sizeof(entry));
        if (!valid_arena_directory_entry_shape_v1(entry,header.total_bytes) || !valid_content_digest_v1(entry.content_digest)
            || !nonzero_digest(entry.content_digest) || entry.offset<previous_end
            || (!known_kind(entry.kind) && (entry.flags&arena_section_required_v1)!=0)) return reader_status_v1::invalid_directory;
        for (std::size_t j=0;j<i;++j) if (entries[j].kind==entry.kind) return reader_status_v1::invalid_directory;
        entries[i]=entry; previous_end=entry.offset+entry.bytes;
    }
    out->header=header; out->section_count=header.section_count; out->payload_verification_required=true;
    return reader_status_v1::success;
}
}
