#include <CellShard/artifact/atom_store/arena_v1.hh>

#include <cassert>

using namespace cellshard::artifact::atom_store;

int main() {
    arena_header_v1 header{};
    header.total_bytes = 1024;
    header.section_directory_offset = 256;
    header.section_count = 2;
    header.section_directory_bytes = 2 * arena_directory_entry_bytes_v1;
    assert(valid_arena_header_shape_v1(header));

    auto bad_header = header;
    bad_header.section_count = UINT64_MAX;
    assert(!valid_arena_header_shape_v1(bad_header));
    bad_header = header;
    bad_header.section_directory_bytes--;
    assert(!valid_arena_header_shape_v1(bad_header));

    arena_directory_entry_v1 entry{};
    entry.kind = arena_section_kind_v1::atom_dictionary;
    entry.flags = arena_section_required_v1;
    entry.alignment = 64;
    entry.offset = 512;
    entry.bytes = 128;
    entry.record_bytes = 32;
    entry.record_count = 4;
    assert(valid_arena_directory_entry_shape_v1(entry, header.total_bytes));

    auto bad_entry = entry;
    bad_entry.flags |= arena_section_optional_v1;
    assert(!valid_arena_directory_entry_shape_v1(bad_entry, header.total_bytes));
    bad_entry = entry;
    bad_entry.offset = 1000;
    assert(!valid_arena_directory_entry_shape_v1(bad_entry, header.total_bytes));
    bad_entry = entry;
    bad_entry.record_count = 5;
    assert(!valid_arena_directory_entry_shape_v1(bad_entry, header.total_bytes));
}
