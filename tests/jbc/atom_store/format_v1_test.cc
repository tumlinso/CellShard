#include <CellShard/artifact/atom_store/format_v1.hh>

#include <cassert>

namespace atom_store = cellshard::artifact::atom_store;

int main() {
    assert(atom_store::family_name_v1 == "CSATOM v1");
    assert(atom_store::file_suffix_v1 == ".csatom");
    assert(atom_store::file_magic_v1[0] == std::byte{'C'});
    assert(atom_store::file_magic_v1[7] == std::byte{'1'});
    assert(atom_store::schema_version_v1 == 1);
    return 0;
}
