#include <CellShard/artifact/atom_store/basis_superatom_v1.hh>
#include <cassert>
namespace atom_store = cellshard::artifact::atom_store;
int main() {
    const std::byte bytes[] = {std::byte{6}};
    const auto digest = atom_store::sha256_digest_v1(bytes, 1);
    const atom_store::basis_record_v1 basis{1, 0, 2, 3, 4, 5, digest};
    const atom_store::superatom_record_v1 superatom{{1, 2}, 0, 2, 0, 2, digest};
    assert(atom_store::valid_basis_record_v1(basis));
    assert(atom_store::valid_superatom_record_v1(superatom));
    return 0;
}
