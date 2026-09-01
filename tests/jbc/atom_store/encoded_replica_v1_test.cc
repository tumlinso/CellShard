#include <CellShard/artifact/atom_store/encoded_replica_v1.hh>

#include <cassert>

using namespace cellshard::artifact::atom_store;

int main() {
    encoded_replica_descriptor_v1 replica{};
    replica.replica = {1, 2};
    replica.atom = {3, 4};
    replica.materialization = {5, 6};
    replica.decoded_content.bytes[0] = std::byte{9};
    replica.encoded_content.bytes[0] = std::byte{9};
    replica.object = cellshard::storage_object_id{7};
    replica.encoded_bytes = 128;
    replica.decoded_bytes = 128;
    replica.extent_slice_count = 2;
    assert(valid_encoded_replica_descriptor_v1(replica));

    auto invalid = replica;
    invalid.encoded_bytes = 127;
    assert(!valid_encoded_replica_descriptor_v1(invalid));
    invalid = replica;
    invalid.encoded_content.bytes[0] = std::byte{8};
    assert(!valid_encoded_replica_descriptor_v1(invalid));

    auto compressed = replica;
    compressed.encoding = replica_encoding_v1::zstd;
    compressed.encoded_bytes = 32;
    compressed.encoded_content.bytes[0] = std::byte{8};
    assert(valid_encoded_replica_descriptor_v1(compressed));
}
