#include <CellShard/artifact/atom_store/atom_dictionary_v1.hh>
#include <cassert>
namespace atom_store = cellshard::artifact::atom_store;
int main() {
    const std::byte bytes[] = {std::byte{3}};
    atom_store::atom_dictionary_record_v1 record{
        {1, 2}, atom_store::sha256_digest_v1(bytes, 1), 0, 2, 0, 0,
        0, 1, 4, atom_store::atom_kind_v1::relation, 1};
    assert(atom_store::valid_atom_dictionary_record_v1(record));
    record.certified = 0;
    assert(!atom_store::valid_atom_dictionary_record_v1(record));
    return 0;
}
