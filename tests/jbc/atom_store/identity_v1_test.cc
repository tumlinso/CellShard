#include <CellShard/artifact/atom_store/identity_v1.hh>

#include <cassert>
#include <type_traits>

namespace atom_store = cellshard::artifact::atom_store;

int main() {
    const std::byte payload[] = {std::byte{1}};
    atom_store::atom_identity_bundle_v1 identity{
        {1, 1}, atom_store::sha256_digest_v1(payload, 1),
        {2, 2}, {3, 3}, {4, 4}};
    assert(atom_store::valid_atom_identity_bundle_v1(identity));
    static_assert(!std::is_same<atom_store::semantic_identity_v1,
                                atom_store::action_identity_v1>::value);
    identity.replica = {};
    assert(!atom_store::valid_atom_identity_bundle_v1(identity));
    return 0;
}
