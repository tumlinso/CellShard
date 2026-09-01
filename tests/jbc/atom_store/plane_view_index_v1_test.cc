#include <CellShard/artifact/atom_store/plane_view_index_v1.hh>
#include <cassert>
namespace atom_store = cellshard::artifact::atom_store;
int main() {
    const std::byte bytes[] = {std::byte{4}};
    const atom_store::plane_index_record_v1 plane{{1, 1}, 2, 3, 4, 5};
    atom_store::physical_view_index_record_v1 view{
        2, {6, 7}, 8, 16, 32, 16, 0, atom_store::sha256_digest_v1(bytes, 1)};
    assert(atom_store::valid_plane_index_record_v1(plane));
    assert(atom_store::valid_physical_view_index_record_v1(view));
    view.byte_offset = 1;
    assert(!atom_store::valid_physical_view_index_record_v1(view));
    return 0;
}
