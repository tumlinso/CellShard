#include <CellShard/artifact/atom_store/writer_v1.hh>
#include <cstring>
#include <limits>
namespace cellshard::artifact::atom_store {
namespace {
bool align_up(std::uint64_t value, std::uint64_t alignment, std::uint64_t *out) {
    if (!is_power_of_two_v1(alignment) || value > UINT64_MAX - (alignment - 1)) return false;
    *out = (value + alignment - 1) & ~(alignment - 1); return true;
}
}
writer_status_v1 atom_store_writer_requirements_v1(const writer_section_source_v1 *sections, std::size_t count, writer_requirements_v1 *out) noexcept {
    if (sections == nullptr || out == nullptr || count == 0 || count > UINT64_MAX / arena_directory_entry_bytes_v1) return writer_status_v1::invalid_input;
    writer_requirements_v1 result{}; result.directory_offset = arena_header_bytes_v1;
    result.directory_bytes = static_cast<std::uint64_t>(count) * arena_directory_entry_bytes_v1;
    std::uint64_t cursor = result.directory_offset + result.directory_bytes;
    for (std::size_t i=0;i<count;++i) {
        const auto &s=sections[i]; std::uint64_t offset=0;
        if (s.data==nullptr || s.bytes==0 || !is_power_of_two_v1(s.alignment) || s.alignment<arena_min_alignment_v1
            || !align_up(cursor,s.alignment,&offset) || s.bytes>UINT64_MAX-offset) return writer_status_v1::invalid_input;
        if ((s.record_bytes==0)!=(s.record_count==0) || (s.record_bytes!=0 && (s.record_count>UINT64_MAX/s.record_bytes || s.record_count*s.record_bytes!=s.bytes))) return writer_status_v1::invalid_input;
        cursor=offset+s.bytes;
    }
    result.total_bytes=cursor; *out=result; return writer_status_v1::success;
}
writer_status_v1 fill_atom_store_v1(const arena_header_v1 &identity_header, const writer_section_source_v1 *sections, std::size_t count, std::byte *output, std::size_t capacity) noexcept {
    writer_requirements_v1 req{}; const auto status=atom_store_writer_requirements_v1(sections,count,&req);
    if (status!=writer_status_v1::success || output==nullptr) return writer_status_v1::invalid_input;
    if (req.total_bytes>capacity) return writer_status_v1::insufficient_capacity;
    std::memset(output,0,static_cast<std::size_t>(req.total_bytes));
    auto header=identity_header; header.magic=file_magic_v1; header.schema_version=schema_version_v1; header.endian_marker=endian_marker_v1;
    header.header_bytes=arena_header_bytes_v1; header.total_bytes=req.total_bytes; header.section_directory_offset=req.directory_offset;
    header.section_count=count; header.section_directory_bytes=req.directory_bytes; header.content_digest=content_digest_v1{};
    std::memcpy(output,&header,sizeof(header)); std::uint64_t cursor=req.directory_offset+req.directory_bytes;
    for (std::size_t i=0;i<count;++i) {
        const auto &s=sections[i]; std::uint64_t offset=0; (void)align_up(cursor,s.alignment,&offset);
        std::memcpy(output+offset,s.data,static_cast<std::size_t>(s.bytes));
        arena_directory_entry_v1 entry{}; entry.kind=s.kind; entry.flags=s.flags; entry.alignment=s.alignment;
        entry.offset=offset; entry.bytes=s.bytes; entry.record_bytes=s.record_bytes; entry.record_count=s.record_count;
        entry.content_digest=sha256_digest_v1(output+offset,static_cast<std::size_t>(s.bytes));
        std::memcpy(output+req.directory_offset+i*sizeof(entry),&entry,sizeof(entry)); cursor=offset+s.bytes;
    }
    header.content_digest=sha256_digest_v1(output,static_cast<std::size_t>(req.total_bytes));
    std::memcpy(output,&header,sizeof(header)); return writer_status_v1::success;
}
}
