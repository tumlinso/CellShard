#include <CellShard/artifact/atom_store/root_manifest_v1.hh>
#include <cassert>
namespace atom_store = cellshard::artifact::atom_store;
int main() {
    const std::byte bytes[] = {std::byte{1}};
    atom_store::root_generation_manifest_v1 root{
        {1, 2}, 1, 1, atom_store::sha256_digest_v1(bytes, 1), {}, 4, 3, 2, 1};
    assert(atom_store::valid_root_generation_manifest_v1(root));
    root.generation = 2;
    assert(!atom_store::valid_root_generation_manifest_v1(root));
    const std::byte parent[] = {std::byte{2}};
    root.parent_root_content = atom_store::sha256_digest_v1(parent, 1);
    assert(atom_store::valid_root_generation_manifest_v1(root));
    return 0;
}
