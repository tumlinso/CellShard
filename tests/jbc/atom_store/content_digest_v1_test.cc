#include <CellShard/artifact/atom_store/content_digest_v1.hh>

#include <cassert>

namespace atom_store = cellshard::artifact::atom_store;

int main() {
    const auto empty = atom_store::sha256_digest_v1(nullptr, 0);
    assert(atom_store::valid_content_digest_v1(empty));
    assert(empty.bytes[0] == std::byte{0xe3});
    assert(empty.bytes[31] == std::byte{0x55});
    const std::byte abc[] = {std::byte{'a'}, std::byte{'b'}, std::byte{'c'}};
    const auto digest = atom_store::sha256_digest_v1(abc, 3);
    assert(digest.bytes[0] == std::byte{0xba});
    assert(digest.bytes[1] == std::byte{0x78});
    assert(digest.bytes[30] == std::byte{0x15});
    assert(digest.bytes[31] == std::byte{0xad});
    return 0;
}
