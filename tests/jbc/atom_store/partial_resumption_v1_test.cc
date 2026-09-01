#include <CellShard/artifact/atom_store/partial_resumption_v1.hh>
#include <cassert>
namespace atom_store = cellshard::artifact::atom_store;
int main() {
    const std::byte bytes[] = {std::byte{7}};
    const auto digest = atom_store::sha256_digest_v1(bytes, 1);
    const atom_store::partial_record_v1 partial{{1, 1}, 2, 0, 4, 3, digest};
    const atom_store::lowering_resumption_record_v1 resume{
        {4, 4}, {5, 5}, 2, 0, 16, digest};
    assert(atom_store::valid_partial_record_v1(partial));
    assert(atom_store::valid_lowering_resumption_record_v1(resume));
    return 0;
}
